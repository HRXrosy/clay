"""Coordinate-free topology compiler for hierarchical group-route ports.

The source declaration names one physical ``@place_group`` and never lists a
net or coordinate.  This module derives the complete external-net denominator
from the canonical schematic.  It does not allocate geometry; its immutable
inventory is the authority shared by the later placement, router and final-GDS
consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Dict, Tuple

from clay.layout.intent import (
    PlaceGroup,
    RouteGroupIntent,
    replay_route_group_inventory,
)
from clay.layout.intent_closure import ConsumerProjection, IntentEntailment
from clay.layout.intent_facts import (
    EntityKind,
    EntityRef,
    FactAuthority,
    IntentFact,
    IntentFactKind,
    IntentFactSet,
    MINIMUM_REQUIRED_CONSUMERS,
    SourceInventory,
)
from clay.layout.place.intent_consumption import (
    _ObservedRelation,
    _bind_observed_relations,
    _relation_from_payload,
)


class GroupRoutePortError(ValueError):
    """A route-group declaration cannot be bound to exact topology."""


def _canonical_text(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    if not value or value != value.strip() or len(value) > 256:
        raise GroupRoutePortError(f"{label} must be canonical and non-empty")
    return value


def _canonical_terminal_pair_tuple(
    raw: object,
    *,
    label: str,
    require_body_terminal: bool = False,
) -> Tuple[Tuple[str, str], ...]:
    """Validate and return a canonically sorted unique tuple of terminal pairs."""
    if type(raw) is not tuple:
        raise TypeError(f"{label}s must be an exact tuple")
    canonical = []
    for row in raw:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError(f"{label} must be an exact pair")
        device = _canonical_text(row[0], f"{label} device")
        terminal = _canonical_text(row[1], f"{label} name")
        if require_body_terminal and terminal != "b":
            raise GroupRoutePortError(
                f"{label} must target 'b', got {(device, terminal)!r}"
            )
        canonical.append((device, terminal))
    canonical_tuple = tuple(canonical)
    if canonical_tuple != tuple(sorted(canonical_tuple)):
        raise GroupRoutePortError(f"{label}s must be canonical")
    if len(set(canonical_tuple)) != len(canonical_tuple):
        raise GroupRoutePortError(f"duplicate {label}s are forbidden")
    return canonical_tuple


@dataclass(frozen=True)
class GroupRoutePort:
    """Every terminal inside one group that carries one external net."""

    group: str
    net: str
    terminals: Tuple[Tuple[str, str], ...]

    def __post_init__(self) -> None:
        _canonical_text(self.group, "group-route port group")
        _canonical_text(self.net, "group-route port net")
        if type(self.terminals) is not tuple or not self.terminals:
            raise TypeError("group-route port terminals must be a non-empty exact tuple")
        _canonical_terminal_pair_tuple(self.terminals, label="group-route terminal")

    def canonical_data(self) -> Dict[str, object]:
        return {
            "group": self.group,
            "net": self.net,
            "terminals": [list(row) for row in self.terminals],
        }


@dataclass(frozen=True)
class GroupRoutePortInventory:
    """Exact declaration, placement-membership and derived-port denominator."""

    cell: str
    declarations: Tuple[RouteGroupIntent, ...]
    place_group_members: Tuple[Tuple[str, Tuple[str, ...]], ...]
    ports: Tuple[GroupRoutePort, ...]
    fact_projection: ConsumerProjection | None = None
    source_welded_body_terminals: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _canonical_text(self.cell, "group-route inventory cell")
        replayed = replay_route_group_inventory(self.declarations)
        if replayed != self.declarations:
            raise GroupRoutePortError("group-route declarations do not replay")
        if type(self.place_group_members) is not tuple:
            raise TypeError("group-route place-group inventory must be an exact tuple")
        normalized_groups = []
        for row in self.place_group_members:
            if type(row) is not tuple or len(row) != 2:
                raise TypeError("group-route place-group row must be an exact pair")
            group = _canonical_text(row[0], "group-route place-group name")
            members = row[1]
            if type(members) is not tuple or not members:
                raise TypeError("group-route place-group members must be a non-empty tuple")
            canonical_members = tuple(
                _canonical_text(member, "group-route place-group member")
                for member in members
            )
            if canonical_members != tuple(sorted(canonical_members)):
                raise GroupRoutePortError("group-route place-group members are not canonical")
            if len(set(canonical_members)) != len(canonical_members):
                raise GroupRoutePortError("duplicate group-route place-group members")
            normalized_groups.append((group, canonical_members))
        if tuple(normalized_groups) != tuple(sorted(normalized_groups)):
            raise GroupRoutePortError("group-route place-group rows are not canonical")
        if len({row[0] for row in normalized_groups}) != len(normalized_groups):
            raise GroupRoutePortError("duplicate group-route place-group rows")
        if type(self.ports) is not tuple or not self.ports:
            raise TypeError("group-route port inventory must be a non-empty exact tuple")
        if any(type(port) is not GroupRoutePort for port in self.ports):
            raise TypeError("group-route port inventory contains a foreign row")
        port_keys = tuple((port.group, port.net) for port in self.ports)
        if port_keys != tuple(sorted(port_keys)):
            raise GroupRoutePortError("group-route ports are not canonical")
        if len(set(port_keys)) != len(port_keys):
            raise GroupRoutePortError("duplicate group-route ports are forbidden")
        if self.fact_projection is not None:
            if type(self.fact_projection) is not ConsumerProjection:
                raise TypeError(
                    "group-route fact projection must be an exact ConsumerProjection"
                )
            if self.fact_projection.consumer != "topology":
                raise GroupRoutePortError(
                    "group-route inventory fact projection must bind topology"
                )
        if type(self.source_welded_body_terminals) is not tuple:
            raise TypeError(
                "group-route source welded body terminals must be an exact tuple"
            )
        normalized_welded = _canonical_terminal_pair_tuple(
            self.source_welded_body_terminals,
            label="group-route source welded body terminal",
            require_body_terminal=True,
        )
        port_terminals_by_group_net = [set(port.terminals) for port in self.ports]
        for device, _body in normalized_welded:
            if not any(
                (device, "b") in terms and (device, "s") in terms
                for terms in port_terminals_by_group_net
            ):
                raise GroupRoutePortError(
                    "group-route source welded body requires matching source and "
                    f"body terminals on one port: {(device, 'b')!r}"
                )

    def canonical_data(self) -> Dict[str, object]:
        data = {
            "cell": self.cell,
            "declarations": [item.canonical_data() for item in self.declarations],
            "place_group_members": [
                {"group": group, "members": list(members)}
                for group, members in self.place_group_members
            ],
            "ports": [port.canonical_data() for port in self.ports],
            "schema": "clay.group-route-port-inventory.v1",
        }
        if self.fact_projection is not None:
            data["fact_projection"] = self.fact_projection.canonical_data()
        return data

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class GroupRouteIntentFactFragment:
    """Coordinate-free route facts ready for a caller-owned combined FactSet.

    The fragment deliberately carries no source, topology, PDK, observation or
    reference identity.  The integration authority must replay the canonical
    source once, migrate any independently observed relations explicitly, and
    bind this fragment together with the floorplan and well facts under that
    one replay identity.
    """

    facts: Tuple[IntentFact, ...]
    entailments: Tuple[IntentEntailment, ...]
    source_entities: Tuple[EntityRef, ...]

    def __post_init__(self) -> None:
        if type(self.facts) is not tuple or any(
            type(fact) is not IntentFact for fact in self.facts
        ):
            raise TypeError("route fact fragment facts must be an exact tuple")
        if tuple(sorted(self.facts, key=lambda fact: fact.fact_id)) != self.facts:
            raise GroupRoutePortError("route fact fragment facts are not canonical")
        if len({fact.fact_id for fact in self.facts}) != len(self.facts):
            raise GroupRoutePortError("route fact fragment facts are not unique")
        if type(self.entailments) is not tuple or any(
            type(row) is not IntentEntailment for row in self.entailments
        ):
            raise TypeError("route fact fragment entailments must be an exact tuple")
        if tuple(sorted(self.entailments)) != self.entailments:
            raise GroupRoutePortError(
                "route fact fragment entailments are not canonical"
            )
        if len({row.atom_id for row in self.entailments}) != len(self.entailments):
            raise GroupRoutePortError("route fact fragment entailments are not unique")
        fact_ids = frozenset(fact.fact_id for fact in self.facts)
        if (
            frozenset(
                fact_id for row in self.entailments for fact_id in row.fact_ids
            )
            != fact_ids
        ):
            raise GroupRoutePortError(
                "route fact fragment entailments do not cover exact facts"
            )
        if type(self.source_entities) is not tuple or any(
            type(entity) is not EntityRef for entity in self.source_entities
        ):
            raise TypeError(
                "route fact fragment source entities must be an exact tuple"
            )
        if tuple(sorted(set(self.source_entities))) != self.source_entities:
            raise GroupRoutePortError(
                "route fact fragment source entities are not canonical"
            )


_BBoxNm = Tuple[int, int, int, int]


def _canonical_bbox_nm(value: object, label: str) -> _BBoxNm:
    if (
        type(value) is not tuple
        or len(value) != 4
        or any(type(coordinate) is not int for coordinate in value)
    ):
        raise TypeError(f"{label} must be one exact integer bbox tuple")
    bbox = value
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise GroupRoutePortError(f"{label} must have positive area")
    return bbox


@dataclass(frozen=True, order=True)
class GroupRoutePlacement:
    """Exact final member geometry and union envelope for one route group."""

    group: str
    members: Tuple[str, ...]
    member_bboxes_nm: Tuple[Tuple[str, _BBoxNm], ...]
    envelope_nm: _BBoxNm

    def __post_init__(self) -> None:
        _canonical_text(self.group, "group-route placement group")
        if type(self.members) is not tuple or not self.members:
            raise TypeError("group-route placement members must be a non-empty tuple")
        members = tuple(
            _canonical_text(member, "group-route placement member")
            for member in self.members
        )
        if members != tuple(sorted(members)) or len(set(members)) != len(members):
            raise GroupRoutePortError(
                "group-route placement members must be canonical and unique"
            )
        if type(self.member_bboxes_nm) is not tuple:
            raise TypeError("group-route placement bboxes must be an exact tuple")
        rows = []
        for row in self.member_bboxes_nm:
            if type(row) is not tuple or len(row) != 2:
                raise TypeError("group-route placement bbox row must be an exact pair")
            rows.append(
                (
                    _canonical_text(row[0], "group-route placement bbox member"),
                    _canonical_bbox_nm(row[1], "group-route placement member bbox"),
                )
            )
        canonical_rows = tuple(rows)
        if (
            canonical_rows != tuple(sorted(canonical_rows))
            or tuple(name for name, _bbox in canonical_rows) != members
        ):
            raise GroupRoutePortError(
                "group-route placement bboxes must exactly cover canonical members"
            )
        expected_envelope = (
            min(bbox[0] for _name, bbox in canonical_rows),
            min(bbox[1] for _name, bbox in canonical_rows),
            max(bbox[2] for _name, bbox in canonical_rows),
            max(bbox[3] for _name, bbox in canonical_rows),
        )
        if _canonical_bbox_nm(
            self.envelope_nm, "group-route placement envelope"
        ) != expected_envelope:
            raise GroupRoutePortError(
                "group-route placement envelope does not replay member union"
            )

    def canonical_data(self) -> Dict[str, object]:
        return {
            "envelope_nm": list(self.envelope_nm),
            "group": self.group,
            "member_bboxes_nm": [
                [member, list(bbox)] for member, bbox in self.member_bboxes_nm
            ],
            "members": list(self.members),
        }


@dataclass(frozen=True)
class GroupRoutePlacementCertificate:
    """Placement-owned binding of one topology inventory to final geometry."""

    topology_inventory_sha256: str
    groups: Tuple[GroupRoutePlacement, ...]

    def __post_init__(self) -> None:
        if (
            type(self.topology_inventory_sha256) is not str
            or len(self.topology_inventory_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.topology_inventory_sha256
            )
        ):
            raise ValueError("group-route topology inventory digest is malformed")
        if type(self.groups) is not tuple or not self.groups:
            raise TypeError("group-route placement groups must be a non-empty tuple")
        if any(type(group) is not GroupRoutePlacement for group in self.groups):
            raise TypeError("group-route placement contains a foreign group")
        if (
            self.groups != tuple(sorted(self.groups))
            or len({group.group for group in self.groups}) != len(self.groups)
        ):
            raise GroupRoutePortError(
                "group-route placement groups must be canonical and unique"
            )

    def canonical_data(self) -> Dict[str, object]:
        return {
            "groups": [group.canonical_data() for group in self.groups],
            "schema": "clay.group-route-placement-certificate.v1",
            "topology_inventory_sha256": self.topology_inventory_sha256,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


def _canonical_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is malformed")
    return value


@dataclass(frozen=True)
class GroupRouteTerminalAccess:
    """One exact physical Pin belonging to one logical group terminal."""

    device: str
    terminal: str
    xy_nm: Tuple[int, int]
    layer_key: Tuple[int, int]
    terminal_binding: object = None

    def __post_init__(self) -> None:
        from clay.layout.route.terminal_identity import pin_binding_payload

        _canonical_text(self.device, "group-route access device")
        _canonical_text(self.terminal, "group-route access terminal")
        if (
            type(self.xy_nm) is not tuple
            or len(self.xy_nm) != 2
            or any(type(value) is not int for value in self.xy_nm)
        ):
            raise TypeError("group-route access point must be an exact integer pair")
        if (
            type(self.layer_key) is not tuple
            or len(self.layer_key) != 2
            or any(type(value) is not int for value in self.layer_key)
        ):
            raise TypeError("group-route access layer must be an exact GDS key")
        pin_binding_payload(self)

    def __lt__(self, other):
        if type(other) is not GroupRouteTerminalAccess:
            return NotImplemented
        def key(value):
            binding = value.terminal_binding
            return (value.device, value.terminal, value.xy_nm, value.layer_key,
                    () if binding is None else binding.canonical_tuple())
        return key(self) < key(other)

    def canonical_data(self) -> Dict[str, object]:
        from clay.layout.route.terminal_identity import pin_binding_payload

        result = {
            "device": self.device,
            "layer_key": list(self.layer_key),
            "terminal": self.terminal,
            "xy_nm": list(self.xy_nm),
        }
        binding = pin_binding_payload(self)
        if binding is not None:
            result["terminal_binding"] = binding
        return result


@dataclass(frozen=True, order=True)
class GroupRoutePhysicalPort:
    """Complete physical access set for one topology-derived group port."""

    group: str
    net: str
    envelope_nm: _BBoxNm
    terminals: Tuple[Tuple[str, str], ...]
    accesses: Tuple[GroupRouteTerminalAccess, ...]
    welded_source_body_terminals: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _canonical_text(self.group, "group-route physical port group")
        _canonical_text(self.net, "group-route physical port net")
        _canonical_bbox_nm(self.envelope_nm, "group-route physical port envelope")
        if type(self.terminals) is not tuple or not self.terminals:
            raise TypeError("group-route physical terminals must be a non-empty tuple")
        canonical_terminals = list(
            _canonical_terminal_pair_tuple(
                self.terminals,
                label="group-route physical terminal",
            )
        )
        if type(self.accesses) is not tuple or not self.accesses:
            raise TypeError("group-route physical accesses must be a non-empty tuple")
        if any(
            type(access) is not GroupRouteTerminalAccess
            for access in self.accesses
        ):
            raise TypeError("group-route physical accesses contain a foreign row")
        for access in self.accesses:
            access.__post_init__()
        if (
            self.accesses != tuple(sorted(self.accesses))
            or len(set(self.accesses)) != len(self.accesses)
        ):
            raise GroupRoutePortError(
                "group-route physical accesses must be canonical and unique"
            )
        from clay.layout.route.terminal_identity import pin_source_aliases, pin_provenance

        access_terminals = {alias for access in self.accesses for alias in pin_source_aliases(access)}
        welded_terminals = set(
            _canonical_terminal_pair_tuple(
                self.welded_source_body_terminals,
                label="group-route welded source body terminal",
                require_body_terminal=True,
            )
        )
        terminal_set = set(canonical_terminals)
        if not welded_terminals <= terminal_set:
            raise GroupRoutePortError(
                "group-route welded source bodies must belong to port terminals"
            )
        if not welded_terminals.isdisjoint(access_terminals):
            raise GroupRoutePortError(
                "group-route welded source body already has a standalone access"
            )
        for device, _body in welded_terminals:
            if (device, "s") not in terminal_set or (device, "s") not in access_terminals:
                raise GroupRoutePortError(
                    "group-route welded source body lacks its same-net source "
                    f"access: {(device, 's')!r}"
                )
        if (access_terminals | welded_terminals) != terminal_set:
            raise GroupRoutePortError(
                "group-route physical accesses do not cover exact terminals"
            )
        typed_ids = [pin_provenance(access) for access in self.accesses
                     if access.terminal_binding is not None]
        if len(typed_ids) != len(set(typed_ids)):
            raise GroupRoutePortError("group-route physical component access must be unique")

    def canonical_data(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "accesses": [access.canonical_data() for access in self.accesses],
            "envelope_nm": list(self.envelope_nm),
            "group": self.group,
            "net": self.net,
            "terminals": [list(row) for row in self.terminals],
        }
        if self.welded_source_body_terminals:
            data["welded_source_body_terminals"] = [
                list(row) for row in self.welded_source_body_terminals
            ]
        return data


@dataclass(frozen=True)
class GroupRouteRoutingContract:
    """Router input binding topology, final placement and exact physical Pins."""

    topology_inventory_sha256: str
    placement_certificate_sha256: str
    ports: Tuple[GroupRoutePhysicalPort, ...]

    def __post_init__(self) -> None:
        _canonical_sha256(
            self.topology_inventory_sha256,
            "group-route routing topology digest",
        )
        _canonical_sha256(
            self.placement_certificate_sha256,
            "group-route routing placement digest",
        )
        if type(self.ports) is not tuple or not self.ports:
            raise TypeError("group-route routing ports must be a non-empty tuple")
        if any(type(port) is not GroupRoutePhysicalPort for port in self.ports):
            raise TypeError("group-route routing ports contain a foreign row")
        if (
            self.ports != tuple(sorted(self.ports))
            or len({(port.group, port.net) for port in self.ports})
            != len(self.ports)
        ):
            raise GroupRoutePortError(
                "group-route routing ports must be canonical and unique"
            )

    def canonical_data(self) -> Dict[str, object]:
        return {
            "placement_certificate_sha256": self.placement_certificate_sha256,
            "ports": [port.canonical_data() for port in self.ports],
            "schema": ("clay.group-route-routing-contract.v2"
                       if any(access.terminal_binding is not None
                              for port in self.ports for access in port.accesses)
                       else "clay.group-route-routing-contract.v1"),
            "topology_inventory_sha256": self.topology_inventory_sha256,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, order=True)
class GroupRouteGlobalWireWitness:
    """One router-emitted global wire, including its actual centerline."""

    net: str
    layer_key: Tuple[int, int]
    start_xy_nm: Tuple[int, int]
    end_xy_nm: Tuple[int, int]
    bbox_nm: Tuple[int, int, int, int]

    def __post_init__(self) -> None:
        _canonical_text(self.net, "group-route global wire net")
        for name, point in (
            ("layer", self.layer_key),
            ("start", self.start_xy_nm),
            ("end", self.end_xy_nm),
        ):
            if (
                type(point) is not tuple
                or len(point) != 2
                or any(type(x) is not int for x in point)
            ):
                raise TypeError(
                    f"group-route global wire {name} must be an integer pair"
                )
        a, b = self.start_xy_nm, self.end_xy_nm
        if a == b or (a[0] != b[0] and a[1] != b[1]):
            raise ValueError(
                "group-route global wire requires a nonzero Manhattan centerline"
            )
        box = _canonical_bbox_nm(self.bbox_nm, "group-route global wire bbox")
        # _emit_net emits a square end cap with the same exact half-width.
        half = (
            (box[3] - box[1]) // 2
            if a[1] == b[1]
            else (box[2] - box[0]) // 2
        )
        expected = (
            min(a[0], b[0]) - half,
            min(a[1], b[1]) - half,
            max(a[0], b[0]) + half,
            max(a[1], b[1]) + half,
        )
        if half <= 0 or box != expected:
            raise ValueError(
                "group-route global wire bbox differs from emitted centerline"
            )


@dataclass(frozen=True, order=True)
class GroupRouteLanding:
    """One local source interface and its sole boundary portal.

    ``interface_kind`` distinguishes a direct planar interface, a newly emitted
    via, and an already materialized literal PCell device stack.  The latter is
    not route-owned via geometry and therefore retains ``via_count == 0`` while
    changing layer; final GDS replay must rediscover its owned ``device_cut``
    path rather than trusting that declaration.
    """

    group: str
    net: str
    source_access: GroupRouteTerminalAccess
    xy_nm: Tuple[int, int]
    portal_xy_nm: Tuple[int, int]
    lower_layer_key: Tuple[int, int]
    upper_layer_key: Tuple[int, int]
    interface_kind: str
    via_count: int
    # None retains the exact v2 certificate and its original semantics.
    portal_layer_key: Tuple[int, int] | None = None

    def __post_init__(self) -> None:
        _canonical_text(self.group, "group-route landing group")
        _canonical_text(self.net, "group-route landing net")
        if type(self.source_access) is not GroupRouteTerminalAccess:
            raise TypeError("group-route landing source access has a foreign type")
        if (
            type(self.xy_nm) is not tuple
            or len(self.xy_nm) != 2
            or any(type(value) is not int for value in self.xy_nm)
        ):
            raise TypeError("group-route landing point must be an exact integer pair")
        if (
            type(self.portal_xy_nm) is not tuple
            or len(self.portal_xy_nm) != 2
            or any(type(value) is not int for value in self.portal_xy_nm)
        ):
            raise TypeError("group-route portal point must be an exact integer pair")
        for label, key in (
            ("lower", self.lower_layer_key),
            ("upper", self.upper_layer_key),
        ):
            if (
                type(key) is not tuple
                or len(key) != 2
                or any(type(value) is not int for value in key)
            ):
                raise TypeError(
                    f"group-route landing {label} layer must be an exact GDS key"
                )
        if self.portal_layer_key is not None and (
            type(self.portal_layer_key) is not tuple
            or len(self.portal_layer_key) != 2
            or any(type(value) is not int for value in self.portal_layer_key)
        ):
            raise TypeError("group-route portal layer must be an exact GDS key")
        if self.lower_layer_key != self.source_access.layer_key:
            raise GroupRoutePortError(
                "group-route landing lower layer differs from source access"
            )
        if type(self.via_count) is not int or self.via_count < 0:
            raise ValueError("group-route landing via count must be non-negative")
        if type(self.interface_kind) is not str:
            raise TypeError("group-route landing interface kind must be an exact string")
        if self.interface_kind not in (
            "direct",
            "emitted_via",
            "fixed_device_stack",
        ):
            raise ValueError("group-route landing interface kind is invalid")
        if self.interface_kind == "direct":
            if (
                self.via_count != 0
                or self.lower_layer_key != self.upper_layer_key
                or self.xy_nm != self.source_access.xy_nm
            ):
                raise GroupRoutePortError(
                    "direct group-route landing must preserve source point and layer"
                )
        elif self.interface_kind == "emitted_via":
            if self.via_count <= 0 or self.lower_layer_key == self.upper_layer_key:
                raise GroupRoutePortError(
                    "emitted-via group-route landing must change layer with a via"
                )
        elif self.via_count != 0 or self.lower_layer_key == self.upper_layer_key:
            raise GroupRoutePortError(
                "fixed-stack group-route landing must change layer without a new via"
            )

    def canonical_data(self) -> Dict[str, object]:
        data = {
            "group": self.group,
            "interface_kind": self.interface_kind,
            "lower_layer_key": list(self.lower_layer_key),
            "net": self.net,
            "portal_xy_nm": list(self.portal_xy_nm),
            "source_access": self.source_access.canonical_data(),
            "upper_layer_key": list(self.upper_layer_key),
            "via_count": self.via_count,
            "xy_nm": list(self.xy_nm),
        }
        if self.portal_layer_key is not None:
            data["portal_layer_key"] = list(self.portal_layer_key)
        return data


@dataclass(frozen=True)
class GroupRouteLandingCertificate:
    """Complete one-landing-per-port geometry selected before global routing."""

    routing_contract_sha256: str
    landings: Tuple[GroupRouteLanding, ...]
    fact_projection: ConsumerProjection | None = None

    def __post_init__(self) -> None:
        _canonical_sha256(
            self.routing_contract_sha256,
            "group-route landing contract digest",
        )
        if type(self.landings) is not tuple or not self.landings:
            raise TypeError("group-route landings must be a non-empty tuple")
        if any(type(row) is not GroupRouteLanding for row in self.landings):
            raise TypeError("group-route landings contain a foreign row")
        if (
            self.landings != tuple(sorted(self.landings))
            or len({(row.group, row.net) for row in self.landings})
            != len(self.landings)
        ):
            raise GroupRoutePortError(
                "group-route landings must be canonical and unique"
            )
        if self.fact_projection is not None:
            if type(self.fact_projection) is not ConsumerProjection:
                raise TypeError(
                    "group-route landing fact projection has a foreign type"
                )
            if self.fact_projection.consumer != "routing":
                raise GroupRoutePortError(
                    "group-route landing fact projection must bind routing"
                )

    def canonical_data(self) -> Dict[str, object]:
        has_portal_layer = any(
            row.portal_layer_key is not None for row in self.landings
        )
        data = {
            "landings": [row.canonical_data() for row in self.landings],
            "routing_contract_sha256": self.routing_contract_sha256,
            "schema": (
                "clay.group-route-landing-certificate.v4"
                if has_portal_layer
                else "clay.group-route-landing-certificate.v2"
            ),
        }
        if self.fact_projection is not None:
            data["fact_projection"] = self.fact_projection.canonical_data()
            if not has_portal_layer:
                data["schema"] = "clay.group-route-landing-certificate.v3"
        return data

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, order=True)
class GdsGroupRoutePortObligation:
    """Exact route-manifest replay of one emitted hierarchical group port."""

    group: str
    net: str
    status: str
    measured_on_gds: bool
    terminal_count: int
    connected_terminal_count: int
    boundary_crossing_count: int
    topology_inventory_sha256: str
    placement_certificate_sha256: str
    routing_contract_sha256: str
    landing_certificate_sha256: str
    gds_sha256: str
    reason: str

    def __post_init__(self) -> None:
        _canonical_text(self.group, "GDS group-route obligation group")
        _canonical_text(self.net, "GDS group-route obligation net")
        if self.status not in ("pass", "fail", "unmeasurable"):
            raise ValueError("GDS group-route obligation status is invalid")
        if type(self.measured_on_gds) is not bool:
            raise TypeError("GDS group-route measured flag must be exact")
        for label, value in (
            ("terminal_count", self.terminal_count),
            ("connected_terminal_count", self.connected_terminal_count),
            ("boundary_crossing_count", self.boundary_crossing_count),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"GDS group-route {label} must be non-negative")
        for label, value in (
            ("topology", self.topology_inventory_sha256),
            ("placement", self.placement_certificate_sha256),
            ("routing", self.routing_contract_sha256),
            ("landing", self.landing_certificate_sha256),
            ("GDS", self.gds_sha256),
        ):
            _canonical_sha256(value, f"GDS group-route {label} digest")
        if type(self.reason) is not str or not self.reason:
            raise ValueError("GDS group-route obligation requires a reason")


def verify_group_route_landing_certificate(
    contract: GroupRouteRoutingContract,
    certificate: object,
    *,
    expected_via_redundancy: int | None = None,
    allow_single_escape: bool = False,
) -> None:
    """Reject missing, duplicate, foreign or out-of-envelope landing rows."""

    if type(contract) is not GroupRouteRoutingContract:
        raise TypeError("group-route landing verification requires an exact contract")
    if type(certificate) is not GroupRouteLandingCertificate:
        raise TypeError("group-route landing verification requires an exact certificate")
    if type(allow_single_escape) is not bool:
        raise TypeError("allow_single_escape must be an explicit bool")
    if expected_via_redundancy is not None and (
        type(expected_via_redundancy) is not int or expected_via_redundancy < 1
    ):
        raise ValueError("expected group-route via redundancy must be a positive integer")
    if certificate.routing_contract_sha256 != contract.sha256:
        raise GroupRoutePortError("group-route landing contract digest is stale")
    ports = {(port.group, port.net): port for port in contract.ports}
    landings = {(row.group, row.net): row for row in certificate.landings}
    if set(landings) != set(ports):
        raise GroupRoutePortError(
            "group-route landing certificate does not cover exact ports"
        )
    allowed_via_counts = (
        None
        if expected_via_redundancy is None
        else (1, expected_via_redundancy)
        if allow_single_escape
        else (expected_via_redundancy,)
    )
    for key, landing in landings.items():
        if (allowed_via_counts is not None
                and landing.interface_kind == "emitted_via"
                and landing.via_count not in allowed_via_counts):
            raise GroupRoutePortError("group-route emitted via differs from declared redundancy")
        port = ports[key]
        if landing.source_access not in port.accesses:
            raise GroupRoutePortError(
                "group-route landing source is absent from physical port"
            )
        x, y = landing.xy_nm
        bbox = port.envelope_nm
        if not (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]):
            raise GroupRoutePortError(
                "group-route landing escapes exact placement envelope"
            )
        portal_x, portal_y = landing.portal_xy_nm
        if not (
            bbox[0] <= portal_x <= bbox[2]
            and bbox[1] <= portal_y <= bbox[3]
            and (
                portal_x in (bbox[0], bbox[2])
                or portal_y in (bbox[1], bbox[3])
            )
        ):
            raise GroupRoutePortError(
                "group-route portal is absent from the exact envelope boundary"
            )


def bind_group_route_routing_fact_projection(
    *,
    inventory: GroupRoutePortInventory,
    contract: GroupRouteRoutingContract,
    certificate: GroupRouteLandingCertificate,
    fact_set: IntentFactSet,
    source_inventory: SourceInventory,
) -> GroupRouteLandingCertificate:
    """Bind rebuilt terminal relations only after exact portal landing closes."""

    if type(inventory) is not GroupRoutePortInventory:
        raise TypeError("routing fact projection requires an exact topology inventory")
    if inventory.fact_projection is None:
        raise GroupRoutePortError(
            "routing fact projection lacks governed topology authority"
        )
    if type(contract) is not GroupRouteRoutingContract:
        raise TypeError("routing fact projection requires an exact routing contract")
    if type(certificate) is not GroupRouteLandingCertificate:
        raise TypeError("routing fact projection requires an exact landing certificate")
    verify_group_route_landing_certificate(contract, certificate)
    if contract.topology_inventory_sha256 != inventory.sha256:
        raise GroupRoutePortError("routing fact projection has stale topology authority")

    topology_ports = {
        (port.group, port.net): port for port in inventory.ports
    }
    physical_ports = {
        (port.group, port.net): port for port in contract.ports
    }
    if set(topology_ports) != set(physical_ports):
        raise GroupRoutePortError(
            "routing fact projection does not cover exact topology ports"
        )
    for key, topology_port in topology_ports.items():
        physical_port = physical_ports[key]
        if physical_port.terminals != topology_port.terminals:
            raise GroupRoutePortError(
                "routing fact projection terminal denominator differs"
            )
        from clay.layout.route.terminal_identity import pin_source_aliases

        covered_aliases = {
            alias
            for access in physical_port.accesses
            for alias in pin_source_aliases(access)
        }
        welded_aliases = set(physical_port.welded_source_body_terminals)
        if not welded_aliases <= set(inventory.source_welded_body_terminals):
            raise GroupRoutePortError(
                "routing fact projection declares non-source-welded body"
            )
        if not welded_aliases.isdisjoint(covered_aliases):
            raise GroupRoutePortError(
                "routing fact projection duplicates physical and welded body"
            )
        if any((device, "s") not in covered_aliases for device, _ in welded_aliases):
            raise GroupRoutePortError(
                "routing fact projection welded body lacks source access"
            )
        if (covered_aliases | welded_aliases) != set(topology_port.terminals):
            raise GroupRoutePortError(
                "routing fact projection lacks exact terminal accesses"
            )

    actual: set[_ObservedRelation] = set()
    for declaration in inventory.declarations:
        source_groups = declaration.place_groups or (declaration.group,)
        route_ref = EntityRef(EntityKind.ROUTE_GROUP, declaration.group)
        place_refs = tuple(
            EntityRef(EntityKind.PLACE_GROUP, name) for name in source_groups
        )
        nets = tuple(
            sorted(
                port.net
                for port in contract.ports
                if port.group == declaration.group
            )
        )
        if not nets:
            continue
        subjects = (
            route_ref,
            *place_refs,
            *(EntityRef(EntityKind.NET, net) for net in nets),
        )
        for kind in (
            IntentFactKind.EXTERNAL_PORT_DENOMINATOR,
            IntentFactKind.TOPOLOGY_EXTERNAL_NET_INVENTORY,
        ):
            relation = _relation_from_payload(kind, subjects, {"complete": True})
            if relation in actual:
                raise GroupRoutePortError(
                    "routing producer emitted a duplicate terminal relation"
                )
            actual.add(relation)

    projection = _bind_observed_relations(
        consumer="routing",
        fact_set=fact_set,
        source_inventory=source_inventory,
        observed_relations=tuple(sorted(actual)),
    )
    return replace(certificate, fact_projection=projection)


def measure_group_route_port_manifest(
    contract: GroupRouteRoutingContract,
    certificate: GroupRouteLandingCertificate,
    route_entries: object,
    tech: object,
    terminal_conductors: object,
    *,
    gds_sha256: str,
    global_wire_witnesses: Tuple[GroupRouteGlobalWireWitness, ...] = (),
) -> Tuple[GdsGroupRoutePortObligation, ...]:
    """Replay terminal contact and the sole boundary crossing from exact routes.

    ``route_entries`` is accepted only from the exhaustive, artifact-bound route
    ownership manifest at the final consumer.  The pure function form keeps the
    geometric proof independently mutation-testable without minting GDS truth.
    """

    verify_group_route_landing_certificate(contract, certificate)
    _canonical_sha256(gds_sha256, "group-route exact GDS digest")
    if type(route_entries) is not tuple:
        raise TypeError("group-route exact route manifest must be an exact tuple")
    if type(terminal_conductors) is not tuple:
        raise TypeError("group-route terminal conductors must be an exact tuple")
    if type(global_wire_witnesses) is not tuple or any(
        type(item) is not GroupRouteGlobalWireWitness
        for item in global_wire_witnesses
    ):
        raise TypeError(
            "group-route global wire witnesses must be an exact tuple"
        )
    from clay.layout.route.engine import Obstacle

    if any(type(item) is not Obstacle for item in terminal_conductors):
        raise TypeError("group-route terminal conductors contain a foreign row")
    canonical_entries = []
    for row in route_entries:
        if type(row) is not tuple or len(row) != 3:
            raise TypeError("group-route route entry must be an exact triple")
        net = _canonical_text(row[0], "group-route route-entry net")
        layer = row[1]
        bbox = row[2]
        if (
            type(layer) is not tuple
            or len(layer) != 2
            or any(type(value) is not int for value in layer)
        ):
            raise TypeError("group-route route-entry layer must be an exact GDS key")
        canonical_entries.append((
            net,
            tuple(layer),
            _canonical_bbox_nm(bbox, "group-route route-entry bbox"),
        ))
    entries = tuple(canonical_entries)
    if entries != tuple(sorted(entries)):
        raise GroupRoutePortError("group-route exact route manifest is not canonical")

    metal_keys = []
    for layer_n in range(1, 65):
        try:
            key = tuple(tech.metal_key(layer_n))
        except KeyError:
            break
        if len(key) != 2 or any(type(value) is not int for value in key):
            raise ValueError("group-route GDS replay lacks exact metal authority")
        metal_keys.append(key)
    if not metal_keys:
        raise ValueError("group-route GDS replay has an empty metal stack")
    via_keys = {}
    for lower_index in range(len(metal_keys) - 1):
        try:
            key = tuple(tech.via_key(lower_index + 1))
        except KeyError as exc:
            raise ValueError(
                "group-route GDS replay lacks a via-layer authority"
            ) from exc
        if len(key) != 2 or any(type(value) is not int for value in key):
            raise ValueError("group-route GDS replay has malformed via authority")
        via_keys[key] = lower_index
    metal_index = {key: index for index, key in enumerate(metal_keys)}
    landing_by_port = {
        (landing.group, landing.net): landing
        for landing in certificate.landings
    }

    def measured_port(port: GroupRoutePhysicalPort) -> GdsGroupRoutePortObligation:
        import klayout.db as kdb

        port_entries = tuple(row for row in entries if row[0] == port.net)
        common = dict(
            group=port.group,
            net=port.net,
            measured_on_gds=True,
            terminal_count=len(port.terminals),
            topology_inventory_sha256=contract.topology_inventory_sha256,
            placement_certificate_sha256=contract.placement_certificate_sha256,
            routing_contract_sha256=contract.sha256,
            landing_certificate_sha256=certificate.sha256,
            gds_sha256=gds_sha256,
        )
        if not port_entries:
            return GdsGroupRoutePortObligation(
                **common,
                status="fail",
                connected_terminal_count=0,
                boundary_crossing_count=0,
                reason="exact route manifest has no geometry for the port net",
            )

        metal_boxes: Dict[Tuple[int, int], list[_BBoxNm]] = {
            key: [] for key in metal_keys
        }
        cut_boxes: Dict[Tuple[int, int], list[_BBoxNm]] = {
            key: [] for key in via_keys
        }
        unknown_layers = []
        for _net, key, bbox in port_entries:
            if key in metal_boxes:
                metal_boxes[key].append(bbox)
            elif key in cut_boxes:
                cut_boxes[key].append(bbox)
            else:
                unknown_layers.append(key)
        route_metal_boxes = {
            key: tuple(boxes) for key, boxes in metal_boxes.items()
        }
        from clay.layout.model import ROUTING_ABSTRACTION_KINDS

        for obstacle in terminal_conductors:
            if obstacle.owner_net != port.net:
                continue
            layer = obstacle.layer
            if (
                type(layer) is not tuple
                or len(layer) != 2
                or any(type(value) is not int for value in layer)
            ):
                continue
            key = tuple(layer)
            if obstacle.kind in ROUTING_ABSTRACTION_KINDS:
                continue
            bbox = (
                int(round(float(obstacle.x0) * 1000.0)),
                int(round(float(obstacle.y0) * 1000.0)),
                int(round(float(obstacle.x1) * 1000.0)),
                int(round(float(obstacle.y1) * 1000.0)),
            )
            exact_bbox = _canonical_bbox_nm(
                bbox, "group-route terminal conductor bbox"
            )
            if key in metal_boxes:
                metal_boxes[key].append(exact_bbox)
            elif key in cut_boxes and obstacle.kind == "device_cut":
                cut_boxes[key].append(exact_bbox)
        if unknown_layers:
            return GdsGroupRoutePortObligation(
                **common,
                status="unmeasurable",
                connected_terminal_count=0,
                boundary_crossing_count=0,
                reason=(
                    "exact route manifest contains layers outside the PDK route "
                    f"stack: {tuple(sorted(set(unknown_layers)))!r}"
                ),
            )

        parents: list[int] = []

        def add_node() -> int:
            node = len(parents)
            parents.append(node)
            return node

        def find(node: int) -> int:
            while parents[node] != node:
                parents[node] = parents[parents[node]]
                node = parents[node]
            return node

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        components_by_key: Dict[
            Tuple[int, int], list[Tuple[int, object]]
        ] = {}
        box_nodes_by_key: Dict[
            Tuple[int, int], list[Tuple[_BBoxNm, int]]
        ] = {}
        route_box_nodes_by_key: Dict[
            Tuple[int, int], Tuple[Tuple[_BBoxNm, int], ...]
        ] = {}
        for key in metal_keys:
            region = kdb.Region()
            for bbox in metal_boxes[key]:
                region.insert(kdb.Box(*bbox))
            region.merge()
            components = []
            for polygon in region.each_merged():
                node = add_node()
                components.append((node, kdb.Region(kdb.Polygon(polygon))))
            components_by_key[key] = components
            mapped_boxes = []
            for bbox in metal_boxes[key]:
                witness = kdb.Region(kdb.Box(*bbox))
                matches = tuple(
                    node
                    for node, component in components
                    if not component.interacting(witness).is_empty()
                )
                if len(matches) != 1:
                    return GdsGroupRoutePortObligation(
                        **common,
                        status="unmeasurable",
                        connected_terminal_count=0,
                        boundary_crossing_count=0,
                        reason="route metal cannot be assigned to one merged component",
                    )
                mapped_boxes.append((bbox, matches[0]))
            box_nodes_by_key[key] = mapped_boxes
            route_box_nodes_by_key[key] = tuple(
                mapped_boxes[:len(route_metal_boxes[key])]
            )

        for via_key, lower_index in via_keys.items():
            lower_key = metal_keys[lower_index]
            upper_key = metal_keys[lower_index + 1]
            for bbox in cut_boxes[via_key]:
                cut = kdb.Region(kdb.Box(*bbox))
                lower = tuple(
                    node
                    for node, component in components_by_key[lower_key]
                    if not component.interacting(cut).is_empty()
                )
                upper = tuple(
                    node
                    for node, component in components_by_key[upper_key]
                    if not component.interacting(cut).is_empty()
                )
                if lower and upper:
                    for left in lower:
                        for right in upper:
                            union(left, right)

        def point_roots(
            point: Tuple[int, int], layer_key: Tuple[int, int]
        ) -> set[int]:
            x, y = point
            return {
                find(node)
                for bbox, node in box_nodes_by_key.get(layer_key, ())
                if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]
            }

        landing = landing_by_port[(port.group, port.net)]
        portal_roots = point_roots(
            landing.portal_xy_nm,
            landing.portal_layer_key or landing.upper_layer_key,
        )
        connected_terminals = 0
        terminal_root_sets = {}
        component_root_sets = {}
        from clay.layout.route.terminal_identity import pin_matches_source

        welded_terminals = set(port.welded_source_body_terminals)
        for terminal in port.terminals:
            target_terminal = (
                (terminal[0], "s") if terminal in welded_terminals else terminal
            )
            roots = set()
            required_components = []
            for access in port.accesses:
                if not pin_matches_source(access, *target_terminal):
                    continue
                access_roots = point_roots(access.xy_nm, access.layer_key)
                roots.update(access_roots)
                if access.terminal_binding is not None:
                    # Each typed component is an AND obligation. An absent
                    # component contributes an empty set, which cannot disappear
                    # into the union of another connected access of this alias.
                    required_components.append(access_roots)
            terminal_root_sets[terminal] = roots
            component_root_sets[terminal] = required_components
        if len(portal_roots) == 1:
            portal_root = next(iter(portal_roots))
            connected_terminals = sum(
                1 for terminal, roots in terminal_root_sets.items()
                if roots == {portal_root}
                and all(component_roots == {portal_root}
                        for component_roots in component_root_sets[terminal])
            )

        x0, y0, x1, y1 = port.envelope_nm
        raw_intervals = []
        for key in metal_keys:
            for (bx0, by0, bx1, by1), node in route_box_nodes_by_key[key]:
                component_root = find(node)
                if bx0 < x0 < bx1 and max(by0, y0) <= min(by1, y1):
                    raw_intervals.append((
                        component_root,
                        "left",
                        max(by0, y0),
                        min(by1, y1),
                    ))
                if bx0 < x1 < bx1 and max(by0, y0) <= min(by1, y1):
                    raw_intervals.append((
                        component_root,
                        "right",
                        max(by0, y0),
                        min(by1, y1),
                    ))
                if by0 < y0 < by1 and max(bx0, x0) <= min(bx1, x1):
                    raw_intervals.append((
                        component_root,
                        "bottom",
                        max(bx0, x0),
                        min(bx1, x1),
                    ))
                if by0 < y1 < by1 and max(bx0, x0) <= min(bx1, x1):
                    raw_intervals.append((
                        component_root,
                        "top",
                        max(bx0, x0),
                        min(bx1, x1),
                    ))
        merged_intervals = []
        for component_root in sorted({row[0] for row in raw_intervals}):
            for side in ("bottom", "left", "right", "top"):
                intervals = sorted(
                    (start, end)
                    for candidate_root, candidate_side, start, end in raw_intervals
                    if candidate_root == component_root and candidate_side == side
                )
                for start, end in intervals:
                    if (
                        merged_intervals
                        and merged_intervals[-1][:2] == (component_root, side)
                        and start <= merged_intervals[-1][3]
                    ):
                        prior_root, prior_side, prior_start, prior_end = (
                            merged_intervals[-1]
                        )
                        merged_intervals[-1] = (
                            prior_root,
                            prior_side,
                            prior_start,
                            max(prior_end, end),
                        )
                    else:
                        merged_intervals.append((
                            component_root,
                            side,
                            start,
                            end,
                        ))

        portal_x, portal_y = landing.portal_xy_nm
        portal_sides = set()
        if portal_x == x0:
            portal_sides.add(("left", portal_y))
        if portal_x == x1:
            portal_sides.add(("right", portal_y))
        if portal_y == y0:
            portal_sides.add(("bottom", portal_x))
        if portal_y == y1:
            portal_sides.add(("top", portal_x))
        boundary_parents = list(range(len(merged_intervals)))

        def boundary_find(index: int) -> int:
            while boundary_parents[index] != index:
                boundary_parents[index] = boundary_parents[
                    boundary_parents[index]
                ]
                index = boundary_parents[index]
            return index

        def boundary_union(left: int, right: int) -> None:
            left_root = boundary_find(left)
            right_root = boundary_find(right)
            if left_root != right_root:
                boundary_parents[right_root] = left_root

        corners = (
            (("left", y0), ("bottom", x0)),
            (("bottom", x1), ("right", y0)),
            (("right", y1), ("top", x1)),
            (("top", x0), ("left", y1)),
        )
        for first, second in corners:
            first_rows = tuple(
                index
                for index, row in enumerate(merged_intervals)
                if row[1] == first[0] and row[2] <= first[1] <= row[3]
            )
            second_rows = tuple(
                index
                for index, row in enumerate(merged_intervals)
                if row[1] == second[0] and row[2] <= second[1] <= row[3]
            )
            for left in first_rows:
                for right in second_rows:
                    if merged_intervals[left][0] == merged_intervals[right][0]:
                        boundary_union(left, right)
        boundary_roots = {
            boundary_find(index) for index in range(len(merged_intervals))
        }
        portal_hit_roots = {
            boundary_find(index)
            for index, row in enumerate(merged_intervals)
            if any(
                row[1] == side and row[2] <= coordinate <= row[3]
                for side, coordinate in portal_sides
            )
        }
        crossing_count = len(boundary_roots)
        connected = connected_terminals == len(port.terminals)
        portal_root = next(iter(portal_roots)) if len(portal_roots) == 1 else None
        crossing_component_roots = {row[0] for row in merged_intervals}
        boundary_clean = (
            crossing_count == 1
            and bool(portal_hit_roots)
            and portal_root is not None
            and crossing_component_roots == {portal_root}
        )
        if landing.portal_layer_key is not None:
            # A via pad can straddle the boundary; require an actual global
            # route centerline on the portal layer, bound to this exact route
            # manifest and its connected component.
            portal = landing.portal_xy_nm
            layer_crossing = False
            for witness in global_wire_witnesses:
                if (
                    witness.net != port.net
                    or witness.layer_key != landing.portal_layer_key
                    or portal not in (witness.start_xy_nm, witness.end_xy_nm)
                    or (witness.net, witness.layer_key, witness.bbox_nm) not in entries
                ):
                    continue
                ox, oy = (
                    witness.end_xy_nm
                    if witness.start_xy_nm == portal
                    else witness.start_xy_nm
                )
                outward = (
                    (portal_x == x0 and oy == portal_y and ox < x0)
                    or (portal_x == x1 and oy == portal_y and ox > x1)
                    or (portal_y == y0 and ox == portal_x and oy < y0)
                    or (portal_y == y1 and ox == portal_x and oy > y1)
                )
                if outward and any(
                    bbox == witness.bbox_nm and find(node) == portal_root
                    for bbox, node in route_box_nodes_by_key.get(witness.layer_key, ())
                ):
                    layer_crossing = True
                    break
            boundary_clean = boundary_clean and layer_crossing
        passed = connected and boundary_clean
        return GdsGroupRoutePortObligation(
            **common,
            status="pass" if passed else "fail",
            connected_terminal_count=connected_terminals,
            boundary_crossing_count=crossing_count,
            reason=(
                "all group terminals join one exact route component and that "
                "component has one certified boundary portal"
                if passed
                else (
                    "group-route exact replay failed: "
                    f"connected_terminals={connected_terminals}/"
                    f"{len(port.terminals)}, boundary_crossings={crossing_count}, "
                    f"portal_observed={bool(portal_hit_roots)}"
                )
            ),
        )

    return tuple(measured_port(port) for port in contract.ports)


def verify_group_route_ports_on_gds(
    *,
    source_cell: object,
    intent: object,
    final_placement_certificate: object,
    pins_by_net: object,
    obstacles: object,
    routed: object,
    exact_artifact: object,
    tech: object,
    fact_set: IntentFactSet | None = None,
    source_inventory: SourceInventory | None = None,
) -> Tuple[GdsGroupRoutePortObligation, ...]:
    """Rebuild every authority and measure the artifact-bound route manifest."""

    declarations = getattr(intent, "route_groups", None)
    if type(declarations) is not tuple:
        raise TypeError("final group-route declarations must be an exact tuple")
    landing_certificate = getattr(
        routed, "group_route_landing_certificate", None
    )
    if not declarations:
        if landing_certificate is not None:
            raise GroupRoutePortError(
                "undeclared group-route evidence cannot enter final GDS signoff"
            )
        return ()
    topology = build_group_route_port_inventory(
        source_cell,
        intent,
        fact_set=fact_set,
        source_inventory=source_inventory,
    )
    placement = getattr(
        final_placement_certificate,
        "group_route_placement_certificate",
        None,
    )
    if type(placement) is not GroupRoutePlacementCertificate:
        raise GroupRoutePortError(
            "final GDS group-route replay lacks its placement certificate"
        )
    verify_group_route_placement(
        topology, final_placement_certificate, placement
    )
    contract = build_group_route_routing_contract(
        topology, placement, pins_by_net
    )
    if type(landing_certificate) is not GroupRouteLandingCertificate:
        raise GroupRoutePortError(
            "final GDS group-route replay lacks its landing certificate"
        )
    verify_group_route_landing_certificate(
        contract,
        landing_certificate,
        expected_via_redundancy=getattr(routed, "router_via_redundancy", None),
        allow_single_escape=bool(getattr(routed, "router_via_allow_single_escape", False)),
    )
    from clay.layout.route.exact_artifact_evidence import ExactEmittedArtifact

    if type(exact_artifact) is not ExactEmittedArtifact:
        raise TypeError("final group-route replay requires ExactEmittedArtifact")
    exact_artifact.validate()
    if type(obstacles) is not list:
        raise TypeError("final group-route obstacle inventory must be an exact list")
    return measure_group_route_port_manifest(
        contract,
        landing_certificate,
        tuple(exact_artifact.route_manifest.entries),
        tech,
        tuple(obstacles),
        gds_sha256=exact_artifact.scope.exact_gds_sha256,
        global_wire_witnesses=getattr(
            routed, "group_route_global_wire_witnesses", ()
        ),
    )


def _canonical_devices(cell_def: Any) -> Dict[str, Any]:
    raw_devices = getattr(cell_def, "devices", None)
    if type(raw_devices) not in (tuple, list) or not raw_devices:
        raise GroupRoutePortError("group-route topology requires explicit devices")
    devices: Dict[str, Any] = {}
    for device in raw_devices:
        name = _canonical_text(
            getattr(device, "instance_name", None),
            "group-route device name",
        )
        pins = getattr(device, "pins", None)
        if type(pins) is not dict or not pins:
            raise GroupRoutePortError(
                f"group-route device {name!r} requires exact pin topology"
            )
        for terminal, net in pins.items():
            _canonical_text(terminal, "group-route terminal name")
            _canonical_text(net, "group-route terminal net")
        if name in devices:
            raise GroupRoutePortError(f"duplicate group-route device {name!r}")
        devices[name] = device
    return devices


def _deck_global_substrate_body_terminals(
    cell_def: Any,
    devices: Dict[str, Any],
) -> frozenset[Tuple[str, str]]:
    """Classify source-declared NMOS substrate bodies that have no route pin.

    The supported extraction decks bind one semantic ``ground``/``power_n``
    port to the global p-substrate.  An NMOS whose body is that substrate while
    its source is a different net is connected by the deck-global substrate,
    not by an independently drawn M1 landing.  This is topology/port-role data;
    it consumes no placement coordinate or reference geometry.
    """

    from clay.port_roles import declared_substrate_ports

    substrate_ports = declared_substrate_ports(
        getattr(cell_def, "port_roles", {})
    )
    if not substrate_ports:
        return frozenset()
    substrate = substrate_ports[0]
    nmos_classes = frozenset({"nmos", "nmos3v", "nch", "nch3", "nd"})
    implicit = set()
    for name, device in devices.items():
        device_class = str(getattr(device, "device_class", "") or "").casefold()
        pins = device.pins
        if (
            device_class in nmos_classes
            and pins.get("b") == substrate
            and pins.get("s") != substrate
        ):
            implicit.add((name, "b"))
    return frozenset(implicit)


def _source_welded_pmos_body_terminals(
    cell_def: Any,
    intent: Any,
    devices: Dict[str, Any],
    ports: Tuple[GroupRoutePort, ...],
) -> Tuple[Tuple[str, str], ...]:
    """Return non-isolated PMOS (device, 'b') terminals eligible for s==b welding.

    Private/isolated bodies (`s != b`, `@well_isolated`, and all NMOS/non-PMOS
    devices whose P+ substrate taps remain separate M1 islands) are excluded so
    missing physical `'b'` accesses on those devices always fail closed.
    """
    from clay.well_isolation import declared_well_isolated_devices

    isolated = set(declared_well_isolated_devices(cell_def))
    isolated.update(
        str(item)
        for item in (getattr(intent, "well_isolated_devices", ()) or ())
        if item
    )
    pmos_classes = frozenset({"pmos", "pmos3v", "pch", "pch3", "pd"})
    same_port_devices = [set(port.terminals) for port in ports]
    welded = []
    for name, device in sorted(devices.items()):
        if name in isolated:
            continue
        device_class = str(getattr(device, "device_class", "") or "").casefold()
        if device_class not in pmos_classes:
            continue
        pins = device.pins
        source_net = pins.get("s")
        body_net = pins.get("b")
        if (
            isinstance(source_net, str)
            and source_net
            and body_net == source_net
            and any(
                (name, "b") in terms and (name, "s") in terms
                for terms in same_port_devices
            )
        ):
            welded.append((name, "b"))
    return tuple(welded)


def build_explicit_route_group_envelope_constraints(
    intent: object,
    *,
    blocks: object,
    block_member_devices: object,
    known_device_names: object,
) -> tuple[object, ...]:
    """Project explicit multi-place route groups onto exact solver masters.

    The route-group declaration owns source devices through its named
    ``@place_group`` rows.  This seam joins that source ownership to the current
    materialized-master inventory without relying on block names or reference
    coordinates.  A rigid master that mixes one module device with any known
    foreign device is not separable and therefore fails closed.
    """

    from clay.layout.place_blocks_2d import FlexibleEnvelopeConstraint

    declarations = replay_route_group_inventory(
        getattr(intent, "route_groups", ())
    )
    explicit = tuple(item for item in declarations if item.place_groups)
    if not explicit:
        return ()

    raw_place_groups = getattr(intent, "place_groups", None)
    if type(raw_place_groups) is not tuple:
        raise TypeError("route-group envelope place_groups must be an exact tuple")
    place_groups: Dict[str, Tuple[str, ...]] = {}
    for row in raw_place_groups:
        if type(row) is not PlaceGroup:
            raise TypeError("route-group envelope place_groups contains a foreign row")
        name = _canonical_text(row.group, "route-group envelope place_group name")
        if name in place_groups:
            raise GroupRoutePortError(
                f"duplicate route-group envelope place_group {name!r}"
            )
        raw_devices = row.devices
        if type(raw_devices) is not tuple or not raw_devices:
            raise TypeError(
                "route-group envelope place_group devices must be a non-empty exact tuple"
            )
        devices = tuple(
            _canonical_text(device, "route-group envelope source device")
            for device in raw_devices
        )
        if len(set(devices)) != len(devices):
            raise GroupRoutePortError(
                f"route-group envelope place_group {name!r} has duplicate devices"
            )
        place_groups[name] = tuple(sorted(devices))

    if type(known_device_names) is not tuple or not known_device_names:
        raise TypeError(
            "route-group envelope known devices must be a non-empty exact tuple"
        )
    known_devices = tuple(
        _canonical_text(device, "route-group envelope known device")
        for device in known_device_names
    )
    if (
        known_devices != tuple(sorted(known_devices))
        or len(set(known_devices)) != len(known_devices)
    ):
        raise GroupRoutePortError(
            "route-group envelope known devices must be canonical and unique"
        )
    known_set = set(known_devices)

    if type(blocks) is not tuple or not blocks:
        raise TypeError("route-group envelope blocks must be a non-empty exact tuple")
    canonical_blocks: list[Tuple[str, _BBoxNm]] = []
    for row in blocks:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError("route-group envelope block row must be an exact pair")
        canonical_blocks.append(
            (
                _canonical_text(row[0], "route-group envelope block name"),
                _canonical_bbox_nm(row[1], "route-group envelope block bbox"),
            )
        )
    block_rows = tuple(canonical_blocks)
    if (
        block_rows != tuple(sorted(block_rows))
        or len({name for name, _bbox in block_rows}) != len(block_rows)
    ):
        raise GroupRoutePortError(
            "route-group envelope blocks must be canonical and unique"
        )

    if type(block_member_devices) is not dict:
        raise TypeError("route-group envelope block ownership must be an exact dict")
    block_names = tuple(name for name, _bbox in block_rows)
    if set(block_member_devices) != set(block_names):
        raise GroupRoutePortError(
            "route-group envelope block ownership must completely cover blocks"
        )
    owned_by_block: Dict[str, set[str]] = {}
    owner_by_known_device: Dict[str, str] = {}
    for block_name in block_names:
        raw_members = block_member_devices[block_name]
        if type(raw_members) is not set:
            raise TypeError(
                "route-group envelope block members must be exact sets"
            )
        members = {
            _canonical_text(member, "route-group envelope block member")
            for member in raw_members
        }
        if len(members) != len(raw_members):
            raise GroupRoutePortError(
                f"route-group envelope block {block_name!r} has duplicate members"
            )
        known_members = members.intersection(known_set)
        for device in sorted(known_members):
            prior_owner = owner_by_known_device.get(device)
            if prior_owner is not None:
                raise GroupRoutePortError(
                    "route-group envelope device ownership is not unique; "
                    f"device={device!r}, masters={(prior_owner, block_name)!r}"
                )
            owner_by_known_device[device] = block_name
        owned_by_block[block_name] = known_members

    dimensions = {
        name: (bbox[2] - bbox[0], bbox[3] - bbox[1])
        for name, bbox in block_rows
    }
    occupied_source_groups: set[str] = set()
    occupied_masters: set[str] = set()
    constraints = []
    for declaration in explicit:
        overlap_groups = tuple(
            sorted(occupied_source_groups.intersection(declaration.place_groups))
        )
        if overlap_groups:
            raise GroupRoutePortError(
                "explicit route-group envelope place-group ownership must be disjoint; "
                f"overlap={overlap_groups!r}"
            )
        occupied_source_groups.update(declaration.place_groups)
        source_members: set[str] = set()
        for group_name in declaration.place_groups:
            group_members = place_groups.get(group_name)
            if group_members is None:
                raise GroupRoutePortError(
                    f"explicit route_group {declaration.group!r} names unknown "
                    f"place_group {group_name!r}"
                )
            overlap = source_members.intersection(group_members)
            if overlap:
                raise GroupRoutePortError(
                    "explicit route-group source device ownership is not disjoint; "
                    f"overlap={tuple(sorted(overlap))!r}"
                )
            source_members.update(group_members)
        unknown = tuple(sorted(source_members - known_set))
        if unknown:
            raise GroupRoutePortError(
                f"explicit route_group {declaration.group!r} has unknown devices {unknown!r}"
            )

        masters = []
        consumed: set[str] = set()
        for block_name in block_names:
            block_devices = owned_by_block[block_name]
            module_devices = block_devices.intersection(source_members)
            if not module_devices:
                continue
            foreign_devices = block_devices - source_members
            if foreign_devices:
                raise GroupRoutePortError(
                    f"explicit route_group {declaration.group!r} is split with foreign "
                    f"devices in master {block_name!r}: "
                    f"{tuple(sorted(foreign_devices))!r}"
                )
            masters.append(block_name)
            consumed.update(module_devices)
        missing = tuple(sorted(source_members - consumed))
        if missing:
            raise GroupRoutePortError(
                f"explicit route_group {declaration.group!r} master ownership is not "
                f"complete; missing={missing!r}"
            )
        canonical_masters = tuple(sorted(masters))
        if len(canonical_masters) < 2:
            raise GroupRoutePortError(
                f"explicit route_group {declaration.group!r} requires at least "
                "two independent final masters"
            )
        master_overlap = tuple(sorted(occupied_masters.intersection(canonical_masters)))
        if master_overlap:
            raise GroupRoutePortError(
                "explicit route-group envelope master ownership must be disjoint; "
                f"overlap={master_overlap!r}"
            )
        occupied_masters.update(canonical_masters)
        offsets = tuple(
            (master, (0, 0, dimensions[master][0], dimensions[master][1]))
            for master in canonical_masters
        )
        constraints.append(
            FlexibleEnvelopeConstraint(
                group=declaration.group,
                members=canonical_masters,
                extension_nm=0,
                foreign_clearance_nm=0,
                member_physical_offsets_nm=offsets,
                member_adjacency_offsets_nm=offsets,
                foreign_exclusion_mode="envelope",
            )
        )
    return tuple(sorted(constraints))


def _topology_fact_projection(
    *,
    cell: str,
    place_groups: Dict[str, PlaceGroup],
    declarations: Tuple[RouteGroupIntent, ...],
    devices: Dict[str, Any],
    top_port_set: set[str],
    net_devices: Dict[str, set[str]],
    ports: Tuple[GroupRoutePort, ...],
    module_partitions: tuple[tuple[str, tuple[str, ...]], ...],
    fact_set: IntentFactSet,
    source_inventory: SourceInventory,
) -> ConsumerProjection:
    """Rebuild coordinate-free topology relations before binding fact labels."""

    actual: set[_ObservedRelation] = set()

    def add(
        kind: IntentFactKind,
        subjects: tuple[EntityRef, ...],
        payload: dict[str, object],
    ) -> None:
        relation = _relation_from_payload(kind, subjects, payload)
        if relation in actual:
            raise GroupRoutePortError(
                "topology producer emitted a duplicate relation"
            )
        actual.add(relation)

    def external_nets(members: tuple[str, ...]) -> tuple[str, ...]:
        member_set = set(members)
        nets = {
            net
            for member in members
            for net in devices[member].pins.values()
            if net in top_port_set or bool(net_devices.get(net, set()) - member_set)
        }
        return tuple(sorted(nets))

    if (
        type(module_partitions) is not tuple
        or module_partitions != tuple(sorted(module_partitions))
        or len({name for name, _members in module_partitions})
        != len(module_partitions)
    ):
        raise TypeError("topology module partitions must be canonical")
    for name, members in module_partitions:
        if (
            type(name) is not str
            or not name
            or type(members) is not tuple
            or not members
            or members != tuple(sorted(set(members)))
            or any(member not in devices for member in members)
        ):
            raise GroupRoutePortError(
                "topology physical macro partition is malformed"
            )
        add(
            IntentFactKind.MEMBER_PARTITION,
            (
                EntityRef(EntityKind.MODULE, name),
                *(EntityRef(EntityKind.DEVICE, member) for member in members),
            ),
            {"complete": True},
        )

    for name, group in sorted(place_groups.items()):
        members = tuple(sorted(group.devices))
        place_ref = EntityRef(EntityKind.PLACE_GROUP, name)
        device_refs = tuple(EntityRef(EntityKind.DEVICE, item) for item in members)
        add(
            IntentFactKind.MEMBER_PARTITION,
            (place_ref, *device_refs),
            {"complete": True},
        )
        nets = external_nets(members)
        if nets:
            subjects = (
                place_ref,
                *device_refs,
                *(EntityRef(EntityKind.NET, net) for net in nets),
            )
            for kind in (
                IntentFactKind.EXTERNAL_PORT_DENOMINATOR,
                IntentFactKind.TOPOLOGY_EXTERNAL_NET_INVENTORY,
            ):
                add(kind, subjects, {"complete": True})

    route_port_nets: Dict[str, tuple[str, ...]] = {}
    for declaration in declarations:
        route_port_nets[declaration.group] = tuple(
            sorted(port.net for port in ports if port.group == declaration.group)
        )
        source_groups = declaration.place_groups or (declaration.group,)
        route_ref = EntityRef(EntityKind.ROUTE_GROUP, declaration.group)
        place_refs = tuple(
            EntityRef(EntityKind.PLACE_GROUP, item) for item in source_groups
        )
        add(
            IntentFactKind.MEMBER_PARTITION,
            (route_ref, *place_refs),
            {"complete": True},
        )
        subjects = (
            route_ref,
            *place_refs,
            *(
                EntityRef(EntityKind.NET, net)
                for net in route_port_nets[declaration.group]
            ),
        )
        for kind in (
            IntentFactKind.EXTERNAL_PORT_DENOMINATOR,
            IntentFactKind.TOPOLOGY_EXTERNAL_NET_INVENTORY,
        ):
            add(kind, subjects, {"complete": True})

    return _bind_observed_relations(
        consumer="topology",
        fact_set=fact_set,
        source_inventory=source_inventory,
        observed_relations=tuple(sorted(actual)),
    )


def build_topology_fact_projection(
    cell_def: Any,
    intent: Any,
    *,
    fact_set: IntentFactSet,
    source_inventory: SourceInventory,
) -> ConsumerProjection:
    """Rebuild topology facts even when no hierarchical route group exists.

    Place-group membership is a topology obligation in its own right.  A
    design with no ``@route_group`` must therefore project its source partition
    without fabricating an empty route-port inventory (whose type correctly
    requires at least one physical port).
    """

    if type(fact_set) is not IntentFactSet:
        raise TypeError("fact_set must be an exact IntentFactSet")
    if type(source_inventory) is not SourceInventory:
        raise TypeError("source_inventory must be an exact SourceInventory")
    declarations = replay_route_group_inventory(
        getattr(intent, "route_groups", None)
    )
    if declarations:
        inventory = build_group_route_port_inventory(
            cell_def,
            intent,
            fact_set=fact_set,
            source_inventory=source_inventory,
        )
        if type(inventory.fact_projection) is not ConsumerProjection:
            raise GroupRoutePortError("topology inventory lacks its fact projection")
        return inventory.fact_projection

    from clay.layout.physical_macro_graph import physical_macro_graph_from_intent

    macro_graph = physical_macro_graph_from_intent(
        cell_def,
        intent,
        device_records={},
    )
    module_partitions = tuple(
        sorted((macro.macro_id, macro.members) for macro in macro_graph.macros)
    )

    raw_place_groups = getattr(intent, "place_groups", None)
    if type(raw_place_groups) is not tuple:
        raise TypeError("topology place_groups must be an exact tuple")
    place_groups: Dict[str, PlaceGroup] = {}
    for group in raw_place_groups:
        if type(group) is not PlaceGroup:
            raise TypeError("topology place_groups contains a foreign row")
        name = _canonical_text(group.group, "topology place_group name")
        if name in place_groups:
            raise GroupRoutePortError(f"duplicate place_group {name!r}")
        place_groups[name] = group
    devices = _canonical_devices(cell_def)
    top_ports = getattr(cell_def, "ports", None)
    if type(top_ports) not in (tuple, list):
        raise TypeError("topology top-cell ports must be an exact sequence")
    top_port_set = {
        _canonical_text(net, "topology top-cell port") for net in top_ports
    }
    net_devices: Dict[str, set[str]] = {}
    for device_name, device in devices.items():
        for net in device.pins.values():
            net_devices.setdefault(net, set()).add(device_name)
    return _topology_fact_projection(
        cell=_canonical_text(
            getattr(cell_def, "local_name", None),
            "topology cell name",
        ),
        place_groups=place_groups,
        declarations=(),
        devices=devices,
        top_port_set=top_port_set,
        net_devices=net_devices,
        ports=(),
        module_partitions=module_partitions,
        fact_set=fact_set,
        source_inventory=source_inventory,
    )


def build_group_route_port_inventory(
    cell_def: Any,
    intent: Any,
    *,
    fact_set: IntentFactSet | None = None,
    source_inventory: SourceInventory | None = None,
) -> GroupRoutePortInventory:
    """Derive every declared group's complete external port set from topology."""

    if (fact_set is None) != (source_inventory is None):
        raise TypeError("fact_set and source_inventory must be provided together")
    if fact_set is not None and type(fact_set) is not IntentFactSet:
        raise TypeError("fact_set must be an exact IntentFactSet")
    if source_inventory is not None and type(source_inventory) is not SourceInventory:
        raise TypeError("source_inventory must be an exact SourceInventory")

    declarations = replay_route_group_inventory(
        getattr(intent, "route_groups", None)
    )
    if not declarations:
        raise GroupRoutePortError("group-route declaration inventory is empty")
    raw_place_groups = getattr(intent, "place_groups", None)
    if type(raw_place_groups) is not tuple:
        raise TypeError("group-route place_groups must be an exact tuple")
    place_groups: Dict[str, PlaceGroup] = {}
    for group in raw_place_groups:
        if type(group) is not PlaceGroup:
            raise TypeError("group-route place_groups contains a foreign row")
        name = _canonical_text(group.group, "group-route place_group name")
        if name in place_groups:
            raise GroupRoutePortError(f"duplicate place_group {name!r}")
        place_groups[name] = group

    devices = _canonical_devices(cell_def)
    implicit_substrate_bodies = _deck_global_substrate_body_terminals(
        cell_def, devices
    )
    top_ports = getattr(cell_def, "ports", None)
    if type(top_ports) not in (tuple, list):
        raise TypeError("group-route top-cell ports must be an exact sequence")
    top_port_set = {
        _canonical_text(net, "group-route top-cell port") for net in top_ports
    }
    net_devices: Dict[str, set[str]] = {}
    for device_name, device in devices.items():
        for net in device.pins.values():
            net_devices.setdefault(net, set()).add(device_name)

    selected_groups = []
    occupied_members: set[str] = set()
    occupied_place_groups: set[str] = set()
    derived_ports = []
    for declaration in declarations:
        source_group_names = declaration.place_groups or (declaration.group,)
        repeated_place_groups = tuple(
            sorted(occupied_place_groups.intersection(source_group_names))
        )
        if repeated_place_groups:
            raise GroupRoutePortError(
                "route groups require disjoint place-group ownership; "
                f"overlap={repeated_place_groups!r}"
            )
        source_members: list[str] = []
        source_member_owners: set[str] = set()
        for source_group_name in source_group_names:
            source_group = place_groups.get(source_group_name)
            if source_group is None:
                raise GroupRoutePortError(
                    f"route_group {declaration.group!r} names an unknown "
                    f"place_group {source_group_name!r}"
                )
            group_members = tuple(sorted(source_group.devices))
            if not group_members:
                raise GroupRoutePortError(
                    f"route_group {declaration.group!r} has an empty source "
                    f"place_group {source_group_name!r}"
                )
            if len(set(group_members)) != len(group_members):
                raise GroupRoutePortError(
                    f"route_group {declaration.group!r} source place_group "
                    f"{source_group_name!r} has duplicate members"
                )
            group_overlap = tuple(
                sorted(source_member_owners.intersection(group_members))
            )
            if group_overlap:
                raise GroupRoutePortError(
                    "route_group source place groups require disjoint device "
                    f"ownership; overlap={group_overlap!r}"
                )
            source_member_owners.update(group_members)
            source_members.extend(group_members)
        occupied_place_groups.update(source_group_names)
        members = tuple(sorted(source_members))
        if not members:
            raise GroupRoutePortError(
                f"route_group {declaration.group!r} has no place-group members"
            )
        if len(set(members)) != len(members):
            raise GroupRoutePortError(
                f"route_group {declaration.group!r} has duplicate members"
            )
        missing = tuple(member for member in members if member not in devices)
        if missing:
            raise GroupRoutePortError(
                f"route_group {declaration.group!r} has unknown devices {missing!r}"
            )
        overlap = tuple(sorted(occupied_members.intersection(members)))
        if overlap:
            raise GroupRoutePortError(
                f"route groups require disjoint physical members; overlap={overlap!r}"
            )
        occupied_members.update(members)
        selected_groups.append((declaration.group, members))
        member_set = set(members)
        terminals_by_net: Dict[str, list[Tuple[str, str]]] = {}
        for member in members:
            for terminal, net in devices[member].pins.items():
                if (member, terminal) in implicit_substrate_bodies:
                    continue
                terminals_by_net.setdefault(net, []).append((member, terminal))
        group_port_count = 0
        for net in sorted(terminals_by_net):
            external = (
                net in top_port_set
                or bool(net_devices.get(net, set()) - member_set)
            )
            if not external:
                continue
            derived_ports.append(
                GroupRoutePort(
                    group=declaration.group,
                    net=net,
                    terminals=tuple(sorted(terminals_by_net[net])),
                )
            )
            group_port_count += 1
        if group_port_count == 0:
            raise GroupRoutePortError(
                f"route_group {declaration.group!r} has an empty external-port obligation"
            )

    cell = _canonical_text(
            getattr(cell_def, "local_name", None),
            "group-route cell name",
        )
    canonical_ports = tuple(sorted(derived_ports, key=lambda item: (item.group, item.net)))
    fact_projection = None
    if fact_set is not None and source_inventory is not None:
        from clay.layout.physical_macro_graph import (
            physical_macro_graph_from_intent,
        )

        macro_graph = physical_macro_graph_from_intent(
            cell_def,
            intent,
            device_records={},
        )
        module_partitions = tuple(
            sorted((macro.macro_id, macro.members) for macro in macro_graph.macros)
        )
        fact_projection = _topology_fact_projection(
            cell=cell,
            place_groups=place_groups,
            declarations=declarations,
            devices=devices,
            top_port_set=top_port_set,
            net_devices=net_devices,
            ports=canonical_ports,
            module_partitions=module_partitions,
            fact_set=fact_set,
            source_inventory=source_inventory,
        )
    return GroupRoutePortInventory(
        cell=cell,
        declarations=declarations,
        place_group_members=tuple(sorted(selected_groups)),
        ports=canonical_ports,
        fact_projection=fact_projection,
        source_welded_body_terminals=_source_welded_pmos_body_terminals(
            cell_def, intent, devices, canonical_ports
        ),
    )


def build_group_route_intent_fact_fragment(
    cell_def: Any,
    intent: Any,
) -> GroupRouteIntentFactFragment:
    """Compile declared route groups into a mergeable, identity-free fragment.

    Nets and membership are rebuilt from current topology by
    :func:`build_group_route_port_inventory`; callers cannot inject either.
    This function intentionally does not create an ``IntentFactSet`` or a
    ``SourceInventory`` because those identities belong to the single combined
    exact-candidate authority assembled from canonical source replay.
    """

    inventory = build_group_route_port_inventory(cell_def, intent)
    members_by_group = dict(inventory.place_group_members)
    facts: list[IntentFact] = []
    entailments: list[IntentEntailment] = []
    entities: set[EntityRef] = set()

    def fact(
        *,
        fact_id: str,
        kind: IntentFactKind,
        subjects: tuple[EntityRef, ...],
        payload: dict[str, object],
    ) -> IntentFact:
        return IntentFact.from_payload(
            fact_id=fact_id,
            kind=kind,
            subjects=tuple(sorted(subjects)),
            payload=payload,
            authority=FactAuthority.DESIGN,
            required_consumers=tuple(sorted(MINIMUM_REQUIRED_CONSUMERS[kind])),
        )

    for declaration in inventory.declarations:
        route_ref = EntityRef(EntityKind.ROUTE_GROUP, declaration.group)
        source_groups = declaration.place_groups or (declaration.group,)
        place_refs = tuple(
            EntityRef(EntityKind.PLACE_GROUP, name) for name in source_groups
        )
        port_nets = tuple(
            port.net for port in inventory.ports
            if port.group == declaration.group
        )
        net_refs = tuple(EntityRef(EntityKind.NET, net) for net in port_nets)
        if declaration.group not in members_by_group or not port_nets:
            raise GroupRoutePortError(
                f"route fact fragment lacks replayed group {declaration.group!r}"
            )
        entities.update((route_ref, *place_refs, *net_refs))
        group_facts = (
            fact(
                fact_id=f"partition:route_group:{declaration.group}",
                kind=IntentFactKind.MEMBER_PARTITION,
                subjects=(route_ref, *place_refs),
                payload={"complete": True},
            ),
            fact(
                fact_id=f"conservation:route_group:{declaration.group}:members",
                kind=IntentFactKind.MEMBER_CONSERVATION,
                subjects=(route_ref, *place_refs),
                payload={"complete": True},
            ),
            fact(
                fact_id=f"ports:route_group:{declaration.group}:external",
                kind=IntentFactKind.EXTERNAL_PORT_DENOMINATOR,
                subjects=(route_ref, *place_refs, *net_refs),
                payload={"complete": True},
            ),
            fact(
                fact_id=(
                    f"topology:route_group:{declaration.group}:external_nets"
                ),
                kind=IntentFactKind.TOPOLOGY_EXTERNAL_NET_INVENTORY,
                subjects=(route_ref, *place_refs, *net_refs),
                payload={"complete": True},
            ),
            fact(
                fact_id=f"containment:route_group:{declaration.group}:final",
                kind=IntentFactKind.FINAL_CONTAINMENT,
                subjects=(route_ref,),
                payload={"scope": "final_layout"},
            ),
        )
        canonical_group_facts = tuple(
            sorted(group_facts, key=lambda row: row.fact_id)
        )
        facts.extend(canonical_group_facts)
        entailments.append(
            IntentEntailment(
                atom_id=f"route_group:{declaration.group}:external",
                fact_ids=tuple(row.fact_id for row in canonical_group_facts),
            )
        )

    return GroupRouteIntentFactFragment(
        facts=tuple(sorted(facts, key=lambda row: row.fact_id)),
        entailments=tuple(sorted(entailments)),
        source_entities=tuple(sorted(entities)),
    )


def verify_group_route_port_inventory(
    cell_def: Any,
    intent: Any,
    inventory: object,
    *,
    fact_set: IntentFactSet | None = None,
    source_inventory: SourceInventory | None = None,
) -> None:
    """Rebuild the denominator and reject a foreign or stale inventory."""

    if type(inventory) is not GroupRoutePortInventory:
        raise TypeError("group-route verification requires an exact inventory")
    fresh = build_group_route_port_inventory(
        cell_def,
        intent,
        fact_set=fact_set,
        source_inventory=source_inventory,
    )
    if (
        fresh.canonical_data() != inventory.canonical_data()
        or fresh.sha256 != inventory.sha256
        or fresh.source_welded_body_terminals != inventory.source_welded_body_terminals
    ):
        raise GroupRoutePortError("group-route port inventory does not replay")


def _materialized_member_bboxes(
    final_placement_certificate: Any,
) -> Tuple[
    Dict[str, _BBoxNm],
    Dict[str, Tuple[str, ...]],
    frozenset[str],
]:
    """Replay physical-block to canonical-source ownership from placement.

    ``materialized_block_members`` partitions final solver masters into emitted
    physical blocks.  ``materialized_block_source_members`` independently binds
    each emitted block to the source devices consumed by its canonical stamp.
    Route groups are expressed in source-device names, so neither block-name
    conventions nor PCell bbox guessing are valid substitutes for this join.
    """

    raw_members = getattr(
        final_placement_certificate, "materialized_block_members", None
    )
    if type(raw_members) is not tuple or not raw_members:
        raise TypeError("materialized block members must be a non-empty exact tuple")
    owned_blocks = []
    physical_blocks_by_master: Dict[str, Tuple[str, ...]] = {}
    for row in raw_members:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError("materialized block member row must be an exact pair")
        master = _canonical_text(row[0], "materialized block master")
        members = row[1]
        if type(members) is not tuple or not members:
            raise TypeError("materialized block member inventory must be an exact tuple")
        canonical_members = tuple(
            _canonical_text(member, "materialized block member")
            for member in members
        )
        if (
            canonical_members != tuple(sorted(canonical_members))
            or len(set(canonical_members)) != len(canonical_members)
        ):
            raise GroupRoutePortError(
                "materialized block member inventory must be canonical and unique"
            )
        if master in physical_blocks_by_master:
            raise GroupRoutePortError("materialized block ownership repeats a master")
        physical_blocks_by_master[master] = canonical_members
        owned_blocks.extend(canonical_members)
    if (
        raw_members != tuple(sorted(raw_members))
        or len(set(owned_blocks)) != len(owned_blocks)
    ):
        raise GroupRoutePortError(
            "materialized block ownership must be canonical and disjoint"
        )

    raw_bboxes = getattr(
        final_placement_certificate, "materialized_block_bboxes", None
    )
    if type(raw_bboxes) is not tuple or not raw_bboxes:
        raise TypeError("materialized block bboxes must be a non-empty exact tuple")
    bboxes: Dict[str, _BBoxNm] = {}
    for row in raw_bboxes:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError("materialized block bbox row must be an exact pair")
        member = _canonical_text(row[0], "materialized block bbox member")
        if member in bboxes:
            raise GroupRoutePortError("materialized block bboxes repeat a member")
        bboxes[member] = _canonical_bbox_nm(row[1], "materialized block bbox")
    expected_physical_blocks = set(owned_blocks)
    if set(bboxes) != expected_physical_blocks:
        raise GroupRoutePortError(
            "materialized block bboxes do not cover exact physical blocks"
        )

    raw_source_members = getattr(
        final_placement_certificate, "materialized_block_source_members", None
    )
    if type(raw_source_members) is not tuple or not raw_source_members:
        raise TypeError(
            "materialized source member ownership must be a non-empty exact tuple"
        )
    source_by_physical_block: Dict[str, Tuple[str, ...]] = {}
    all_source_members = []
    for row in raw_source_members:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError(
                "materialized source member ownership row must be an exact pair"
            )
        physical_block = _canonical_text(
            row[0], "materialized source member physical block"
        )
        raw_sources = row[1]
        if type(raw_sources) is not tuple or not raw_sources:
            raise TypeError(
                "materialized source member inventory must be a non-empty exact tuple"
            )
        sources = tuple(
            _canonical_text(source, "materialized source member")
            for source in raw_sources
        )
        if sources != tuple(sorted(sources)) or len(set(sources)) != len(sources):
            raise GroupRoutePortError(
                "materialized source member inventory must be canonical and unique"
            )
        if physical_block in source_by_physical_block:
            raise GroupRoutePortError(
                "materialized source member ownership repeats a physical block"
            )
        source_by_physical_block[physical_block] = sources
        all_source_members.extend(sources)
    if raw_source_members != tuple(sorted(raw_source_members)):
        raise GroupRoutePortError(
            "materialized source member ownership must be canonical"
        )
    if set(source_by_physical_block) != expected_physical_blocks:
        raise GroupRoutePortError(
            "materialized source member ownership does not cover exact physical blocks"
        )
    if len(all_source_members) != len(set(all_source_members)):
        raise GroupRoutePortError(
            "materialized source member ownership repeats a source member"
        )

    physical_block_by_source = {
        source: physical_block
        for physical_block, sources in source_by_physical_block.items()
        for source in sources
    }
    raw_source_bboxes = getattr(
        final_placement_certificate, "materialized_source_member_bboxes", ()
    )
    if type(raw_source_bboxes) is not tuple:
        raise TypeError("materialized source member bboxes must be an exact tuple")
    exact_source_bboxes: Dict[str, _BBoxNm] = {}
    if raw_source_bboxes:
        for row in raw_source_bboxes:
            if type(row) is not tuple or len(row) != 2:
                raise TypeError(
                    "materialized source member bbox row must be an exact pair"
                )
            source = _canonical_text(row[0], "materialized source member bbox")
            if source in exact_source_bboxes:
                raise GroupRoutePortError(
                    "materialized source member bboxes repeat a source"
                )
            exact_source_bboxes[source] = _canonical_bbox_nm(
                row[1], "materialized source member bbox"
            )
        if raw_source_bboxes != tuple(sorted(raw_source_bboxes)):
            raise GroupRoutePortError(
                "materialized source member bboxes must be canonical"
            )
        if not set(exact_source_bboxes) <= set(all_source_members):
            raise GroupRoutePortError(
                "materialized source member bboxes name foreign source ownership"
            )
        for physical_block, sources in source_by_physical_block.items():
            covered = set(sources).intersection(exact_source_bboxes)
            if covered and covered != set(sources):
                raise GroupRoutePortError(
                    "materialized source member bboxes must cover exact source "
                    f"ownership per physical owner {physical_block!r}"
                )
        for source, bbox in exact_source_bboxes.items():
            owner_bbox = bboxes[physical_block_by_source[source]]
            if not (
                owner_bbox[0] <= bbox[0] < bbox[2] <= owner_bbox[2]
                and owner_bbox[1] <= bbox[1] < bbox[3] <= owner_bbox[3]
            ):
                raise GroupRoutePortError(
                    "materialized source member bbox must stay inside its physical owner"
                )

    source_bboxes = (
        exact_source_bboxes
        if exact_source_bboxes
        else {
            source: bboxes[physical_block]
            for physical_block, sources in source_by_physical_block.items()
            for source in sources
        }
    )
    source_owners = {
        master: tuple(sorted(
            source
            for physical_block in physical_blocks
            for source in source_by_physical_block[physical_block]
        ))
        for master, physical_blocks in physical_blocks_by_master.items()
    }
    refinable_masters = frozenset(
        master
        for master, owned in source_owners.items()
        if set(owned) <= set(exact_source_bboxes)
    )
    return source_bboxes, source_owners, refinable_masters


def _bbox_has_positive_overlap(left: _BBoxNm, right: _BBoxNm) -> bool:
    return (
        min(left[2], right[2]) > max(left[0], right[0])
        and min(left[3], right[3]) > max(left[1], right[1])
    )


def bind_group_route_placement(
    inventory: GroupRoutePortInventory,
    final_placement_certificate: Any,
) -> GroupRoutePlacementCertificate:
    """Bind topology-defined route groups to exact final member geometry."""

    if type(inventory) is not GroupRoutePortInventory:
        raise TypeError("group-route placement requires an exact topology inventory")
    member_bboxes, owners, refinable_masters = _materialized_member_bboxes(
        final_placement_certificate
    )
    groups = []
    for group, members in inventory.place_group_members:
        member_set = set(members)
        for master, owned in owners.items():
            overlap = member_set.intersection(owned)
            if overlap and not set(owned) <= member_set:
                if master not in refinable_masters:
                    raise GroupRoutePortError(
                        f"group-route placement splits rigid master {master!r}"
                    )
                foreign = set(owned) - member_set
                if any(
                    _bbox_has_positive_overlap(
                        member_bboxes[selected], member_bboxes[unselected]
                    )
                    for selected in overlap
                    for unselected in foreign
                ):
                    raise GroupRoutePortError(
                        "group-route placement lacks a disjoint common refinement "
                        f"inside master {master!r}"
                    )
        missing = tuple(member for member in members if member not in member_bboxes)
        if missing:
            raise GroupRoutePortError(
                f"group-route placement missing exact member bbox {missing!r}"
            )
        rows = tuple((member, member_bboxes[member]) for member in members)
        envelope = (
            min(bbox[0] for _member, bbox in rows),
            min(bbox[1] for _member, bbox in rows),
            max(bbox[2] for _member, bbox in rows),
            max(bbox[3] for _member, bbox in rows),
        )
        foreign_intrusions = tuple(sorted(
            source
            for source, bbox in member_bboxes.items()
            if source not in member_set
            and _bbox_has_positive_overlap(envelope, bbox)
        ))
        if foreign_intrusions:
            raise GroupRoutePortError(
                "foreign source members "
                f"{foreign_intrusions!r} intrude group-route envelope {group!r}"
            )
        groups.append(
            GroupRoutePlacement(
                group=group,
                members=members,
                member_bboxes_nm=rows,
                envelope_nm=envelope,
            )
        )
    return GroupRoutePlacementCertificate(
        topology_inventory_sha256=inventory.sha256,
        groups=tuple(groups),
    )


def verify_group_route_placement(
    inventory: GroupRoutePortInventory,
    final_placement_certificate: Any,
    placed: object,
) -> None:
    """Rebind exact placement authority and reject stale group geometry."""

    if type(placed) is not GroupRoutePlacementCertificate:
        raise TypeError("group-route placement verification requires an exact certificate")
    fresh = bind_group_route_placement(inventory, final_placement_certificate)
    if fresh.canonical_data() != placed.canonical_data() or fresh.sha256 != placed.sha256:
        raise GroupRoutePortError("group-route placement certificate does not replay")


def build_group_route_routing_contract(
    inventory: GroupRoutePortInventory,
    placement: GroupRoutePlacementCertificate,
    pins_by_net: object,
) -> GroupRouteRoutingContract:
    """Freeze every topology terminal's exact final physical Pin alternatives."""

    if type(inventory) is not GroupRoutePortInventory:
        raise TypeError("group-route routing requires an exact topology inventory")
    if type(placement) is not GroupRoutePlacementCertificate:
        raise TypeError("group-route routing requires an exact placement certificate")
    if placement.topology_inventory_sha256 != inventory.sha256:
        raise GroupRoutePortError(
            "group-route routing placement has stale topology authority"
        )
    expected_groups = inventory.place_group_members
    observed_groups = tuple((group.group, group.members) for group in placement.groups)
    if observed_groups != expected_groups:
        raise GroupRoutePortError(
            "group-route routing placement does not replay group membership"
        )
    if type(pins_by_net) is not dict:
        raise TypeError("group-route physical pins must be an exact dictionary")
    from clay.layout.route.engine import Pin
    from clay.layout.route.terminal_identity import pin_source_aliases, pin_provenance

    placement_by_group = {group.group: group for group in placement.groups}
    eligible_welded = set(inventory.source_welded_body_terminals)
    all_pin_aliases = {
        alias
        for net_pins in pins_by_net.values()
        if isinstance(raw_list := net_pins, (list, tuple))
        for pin in raw_list
        if type(pin) is Pin
        for alias in pin_source_aliases(pin)
    }
    physical_ports = []
    for topology_port in inventory.ports:
        raw_pins = pins_by_net.get(topology_port.net)
        if type(raw_pins) is not list:
            raise TypeError(
                f"group-route net {topology_port.net!r} pins must be an exact list"
            )
        accesses = []
        member_bboxes = dict(
            placement_by_group[topology_port.group].member_bboxes_nm
        )
        required_aliases = set(topology_port.terminals)
        covered_aliases = set()
        seen_components = set()
        for pin in raw_pins:
            if type(pin) is not Pin:
                continue
            aliases = set(pin_source_aliases(pin))
            if not aliases.intersection(required_aliases):
                continue
            if not aliases <= required_aliases:
                raise GroupRoutePortError("group-route cannot partially consume a component with foreign source aliases")
            if pin.net != topology_port.net:
                raise GroupRoutePortError("group-route physical access carries a foreign net")
            binding = getattr(pin, "terminal_binding", None)
            if binding is not None:
                component = pin_provenance(pin)
                if component in seen_components:
                    raise GroupRoutePortError("group-route physical component has multiple accesses")
                seen_components.add(component)
            access = GroupRouteTerminalAccess(device=pin.device, terminal=pin.terminal,
                xy_nm=pin.xy_nm, layer_key=pin.layer, terminal_binding=binding)
            # This is the canonical representative source's access. Shared
            # aliases are logical associations, not additional seed locations.
            bbox = member_bboxes[pin.device]
            x, y = access.xy_nm
            if not (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]):
                raise GroupRoutePortError(
                    "group-route physical access escapes exact member bbox: "
                    f"{topology_port.group}:{pin.device}:{pin.terminal}:{access.xy_nm!r}"
                )
            accesses.append(access)
            covered_aliases.update(aliases)
        welded_for_port: Tuple[Tuple[str, str], ...] = ()
        if covered_aliases != required_aliases:
            missing = required_aliases - covered_aliases
            welded_candidates = tuple(
                sorted(
                    pair
                    for pair in missing
                    if pair[1] == "b"
                    and pair in eligible_welded
                    and (pair[0], "s") in required_aliases
                    and (pair[0], "s") in covered_aliases
                    and pair not in all_pin_aliases
                )
            )
            if set(welded_candidates) != missing:
                raise GroupRoutePortError(
                    "group-route terminal lacks exact physical access: "
                    f"{topology_port.group}:{topology_port.net}:{tuple(sorted(missing - set(welded_candidates)))!r}"
                )
            welded_for_port = welded_candidates
        physical_ports.append(
            GroupRoutePhysicalPort(
                group=topology_port.group,
                net=topology_port.net,
                envelope_nm=placement_by_group[topology_port.group].envelope_nm,
                terminals=topology_port.terminals,
                accesses=tuple(sorted(accesses)),
                welded_source_body_terminals=welded_for_port,
            )
        )
    return GroupRouteRoutingContract(
        topology_inventory_sha256=inventory.sha256,
        placement_certificate_sha256=placement.sha256,
        ports=tuple(sorted(physical_ports)),
    )


def verify_group_route_routing_contract(
    inventory: GroupRoutePortInventory,
    placement: GroupRoutePlacementCertificate,
    pins_by_net: object,
    contract: object,
) -> None:
    """Rebuild the route input and reject missing, foreign or stale Pins."""

    if type(contract) is not GroupRouteRoutingContract:
        raise TypeError("group-route routing verification requires an exact contract")
    fresh = build_group_route_routing_contract(inventory, placement, pins_by_net)
    if fresh.canonical_data() != contract.canonical_data() or fresh.sha256 != contract.sha256:
        raise GroupRoutePortError("group-route routing contract does not replay")


__all__ = [
    "GroupRoutePort",
    "GroupRoutePortError",
    "GroupRouteIntentFactFragment",
    "GroupRoutePortInventory",
    "GroupRoutePlacement",
    "GroupRoutePlacementCertificate",
    "GroupRoutePhysicalPort",
    "GroupRouteLanding",
    "GroupRouteLandingCertificate",
    "GdsGroupRoutePortObligation",
    "bind_group_route_placement",
    "bind_group_route_routing_fact_projection",
    "GroupRouteRoutingContract",
    "GroupRouteTerminalAccess",
    "build_group_route_routing_contract",
    "build_group_route_intent_fact_fragment",
    "build_group_route_port_inventory",
    "build_topology_fact_projection",
    "measure_group_route_port_manifest",
    "verify_group_route_routing_contract",
    "verify_group_route_landing_certificate",
    "verify_group_route_ports_on_gds",
    "verify_group_route_placement",
    "verify_group_route_port_inventory",
]
