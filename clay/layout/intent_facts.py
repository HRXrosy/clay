"""Closed, coordinate-free facts shared by physical-intent consumers.

The values in this module are authority records, not placement hints.  Their
canonical form deliberately admits only registered relational payload fields;
candidate geometry and artifact locators belong in diagnostic receipts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any

from clay.layout.intent_basis import ClaimScope


_ENTITY_SCHEMA = "clay.intent-entity-ref/v1"
_FACT_SCHEMA = "clay.intent-fact/v1"
_FACT_SET_SCHEMA = "clay.intent-fact-set/v1"
_SOURCE_INVENTORY_SCHEMA = "clay.source-inventory/v1"
_AUTHORITY_RECEIPT_SCHEMA = "clay.intent-fact-authority-receipt/v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:%+!\-\[\]]*\Z")
REGISTERED_CONSUMER_ORDER = (
    "topology",
    "placement_initial",
    "placement_replay",
    "routing",
    "final_gds",
)
REGISTERED_CONSUMERS = frozenset(REGISTERED_CONSUMER_ORDER)


class IntentFactSchemaError(ValueError):
    """A fact row is structurally invalid or outside the closed schema."""


class IntentFactTypeError(TypeError):
    """A fact boundary received a mutable or inexact Python value."""


class IntentFactAuthorityError(ValueError):
    """Fact authority or inventory binding failed at a typed blocker."""

    def __init__(self, message: str, *, blocker: str) -> None:
        if blocker not in {"AUTHORITY_IDENTITY", "FACT_SCHEMA"}:
            raise ValueError(f"unknown intent fact authority blocker {blocker!r}")
        super().__init__(message)
        self.blocker = blocker


class EntityKind(str, Enum):
    DEVICE = "device"
    NET = "net"
    PLACE_GROUP = "place_group"
    ROUTE_GROUP = "route_group"
    WELL_DOMAIN = "well_domain"
    MODULE = "module"


class FactAuthority(str, Enum):
    DESIGN = "design"
    PDK = "pdk"
    TARGET_REFERENCE_RELATION = "target_reference_relation"


class IntentFactKind(str, Enum):
    MEMBER_PARTITION = "member_partition"
    MEMBER_CONSERVATION = "member_conservation"
    ORDINAL_CELL = "ordinal_cell"
    ORDER = "order"
    ADJACENCY = "adjacency"
    NO_FOREIGN_ENVELOPE = "no_foreign_envelope"
    WELL_DOMAIN = "well_domain"
    EXTERNAL_PORT_DENOMINATOR = "external_port_denominator"
    TOPOLOGY_EXTERNAL_NET_INVENTORY = "topology_external_net_inventory"
    MATCHED_ROUTE_CLASS = "matched_route_class"
    NET_WIDTH_CLASS = "net_width_class"
    VIA_CAPACITY_CLASS = "via_capacity_class"
    FINAL_CONTAINMENT = "final_containment"


MINIMUM_REQUIRED_CONSUMERS: Mapping[IntentFactKind, frozenset[str]] = MappingProxyType(
    {
        IntentFactKind.MEMBER_PARTITION: frozenset(
            {"final_gds", "placement_initial", "placement_replay", "topology"}
        ),
        IntentFactKind.MEMBER_CONSERVATION: frozenset(
            {"final_gds", "placement_initial", "placement_replay"}
        ),
        IntentFactKind.ORDINAL_CELL: frozenset(
            {"final_gds", "placement_initial", "placement_replay"}
        ),
        IntentFactKind.ORDER: frozenset(
            {"final_gds", "placement_initial", "placement_replay"}
        ),
        IntentFactKind.ADJACENCY: frozenset(
            {"final_gds", "placement_initial", "placement_replay"}
        ),
        IntentFactKind.NO_FOREIGN_ENVELOPE: frozenset(
            {"final_gds", "placement_initial", "placement_replay"}
        ),
        IntentFactKind.WELL_DOMAIN: frozenset(
            {"final_gds", "placement_initial", "placement_replay"}
        ),
        IntentFactKind.EXTERNAL_PORT_DENOMINATOR: frozenset(
            {"final_gds", "routing", "topology"}
        ),
        IntentFactKind.TOPOLOGY_EXTERNAL_NET_INVENTORY: frozenset(
            {"final_gds", "routing", "topology"}
        ),
        IntentFactKind.MATCHED_ROUTE_CLASS: frozenset(
            {"final_gds", "routing"}
        ),
        IntentFactKind.NET_WIDTH_CLASS: frozenset({"final_gds", "routing"}),
        IntentFactKind.VIA_CAPACITY_CLASS: frozenset({"final_gds", "routing"}),
        IntentFactKind.FINAL_CONTAINMENT: frozenset({"final_gds"}),
    }
)


@dataclass(frozen=True)
class _PayloadSchema:
    allowed: frozenset[str]
    required: frozenset[str] = frozenset()


_PAYLOAD_SCHEMAS: dict[IntentFactKind, _PayloadSchema] = {
    IntentFactKind.MEMBER_PARTITION: _PayloadSchema(
        frozenset({"complete"}),
        frozenset({"complete"}),
    ),
    IntentFactKind.MEMBER_CONSERVATION: _PayloadSchema(
        frozenset({"complete"}),
        frozenset({"complete"}),
    ),
    IntentFactKind.ORDINAL_CELL: _PayloadSchema(
        frozenset({"column", "row", "tier"}),
        frozenset({"column", "tier"}),
    ),
    IntentFactKind.ORDER: _PayloadSchema(
        frozenset({"axis", "relation", "semantic"}),
        frozenset({"axis", "relation"}),
    ),
    IntentFactKind.ADJACENCY: _PayloadSchema(
        frozenset({"axis", "relation"}),
        frozenset({"axis", "relation"}),
    ),
    IntentFactKind.NO_FOREIGN_ENVELOPE: _PayloadSchema(
        frozenset({"scope"}),
        frozenset({"scope"}),
    ),
    IntentFactKind.WELL_DOMAIN: _PayloadSchema(
        frozenset({"body_relation", "sharing"}),
        frozenset({"body_relation", "sharing"}),
    ),
    IntentFactKind.EXTERNAL_PORT_DENOMINATOR: _PayloadSchema(
        frozenset({"complete"}),
        frozenset({"complete"}),
    ),
    IntentFactKind.TOPOLOGY_EXTERNAL_NET_INVENTORY: _PayloadSchema(
        frozenset({"complete"}),
        frozenset({"complete"}),
    ),
    IntentFactKind.MATCHED_ROUTE_CLASS: _PayloadSchema(
        frozenset({"route_class"}),
        frozenset({"route_class"}),
    ),
    IntentFactKind.NET_WIDTH_CLASS: _PayloadSchema(
        frozenset({"width_class"}),
        frozenset({"width_class"}),
    ),
    IntentFactKind.VIA_CAPACITY_CLASS: _PayloadSchema(
        frozenset({"via_class"}),
        frozenset({"via_class"}),
    ),
    IntentFactKind.FINAL_CONTAINMENT: _PayloadSchema(
        frozenset({"scope"}),
        frozenset({"scope"}),
    ),
}

_ALLOWED_AUTHORITIES: dict[IntentFactKind, frozenset[FactAuthority]] = {
    IntentFactKind.MEMBER_PARTITION: frozenset(
        {FactAuthority.DESIGN, FactAuthority.TARGET_REFERENCE_RELATION}
    ),
    IntentFactKind.MEMBER_CONSERVATION: frozenset(
        {FactAuthority.DESIGN, FactAuthority.TARGET_REFERENCE_RELATION}
    ),
    IntentFactKind.ORDINAL_CELL: frozenset(
        {FactAuthority.DESIGN, FactAuthority.TARGET_REFERENCE_RELATION}
    ),
    IntentFactKind.ORDER: frozenset(
        {FactAuthority.DESIGN, FactAuthority.TARGET_REFERENCE_RELATION}
    ),
    IntentFactKind.ADJACENCY: frozenset(
        {FactAuthority.DESIGN, FactAuthority.TARGET_REFERENCE_RELATION}
    ),
    IntentFactKind.NO_FOREIGN_ENVELOPE: frozenset(
        {FactAuthority.DESIGN, FactAuthority.TARGET_REFERENCE_RELATION}
    ),
    IntentFactKind.WELL_DOMAIN: frozenset(
        {
            FactAuthority.DESIGN,
            FactAuthority.PDK,
            FactAuthority.TARGET_REFERENCE_RELATION,
        }
    ),
    IntentFactKind.EXTERNAL_PORT_DENOMINATOR: frozenset(
        {FactAuthority.DESIGN, FactAuthority.TARGET_REFERENCE_RELATION}
    ),
    IntentFactKind.TOPOLOGY_EXTERNAL_NET_INVENTORY: frozenset(
        {FactAuthority.DESIGN, FactAuthority.TARGET_REFERENCE_RELATION}
    ),
    IntentFactKind.MATCHED_ROUTE_CLASS: frozenset(
        {FactAuthority.DESIGN, FactAuthority.TARGET_REFERENCE_RELATION}
    ),
    IntentFactKind.NET_WIDTH_CLASS: frozenset(
        {FactAuthority.DESIGN, FactAuthority.PDK}
    ),
    IntentFactKind.VIA_CAPACITY_CLASS: frozenset(
        {FactAuthority.DESIGN, FactAuthority.PDK}
    ),
    IntentFactKind.FINAL_CONTAINMENT: frozenset(
        {FactAuthority.DESIGN, FactAuthority.PDK}
    ),
}

_CLOSED_PAYLOAD_VALUES: dict[str, frozenset[str]] = {
    "axis": frozenset({"column", "horizontal", "tier", "vertical", "x", "y"}),
    "body_relation": frozenset({"isolated", "same_body", "separate_body"}),
    "relation": frozenset(
        {
            "abut",
            "above",
            "adjacent",
            "after",
            "before",
            "below",
            "left_of",
            "right_of",
        }
    ),
    "semantic": frozenset({"centroid_order_v1", "envelope_separation_v1"}),
    "route_class": frozenset(
        {"differential", "matched", "matched_sensitive"}
    ),
    "scope": frozenset(
        {
            "all_members",
            "final_layout",
            "members",
            "module",
            "place_group",
            "route_group",
        }
    ),
    "sharing": frozenset({"isolated", "private", "shared"}),
    "via_class": frozenset(
        {"default", "high_current", "power", "redundant", "signal"}
    ),
    "width_class": frozenset(
        {"default", "high_current", "power", "sensitive", "signal"}
    ),
}

_IDENTIFIER_PAYLOAD_FIELDS = frozenset(
    {
        "axis",
        "body_relation",
        "relation",
        "semantic",
        "route_class",
        "scope",
        "sharing",
        "via_class",
        "width_class",
    }
)
_ORDINAL_PAYLOAD_FIELDS = frozenset({"column", "row", "tier"})
_BOOLEAN_PAYLOAD_FIELDS = frozenset({"complete"})
_REFERENCE_DENIED_KEYWORDS = frozenset(
    {
        "area",
        "bbox",
        "centroid",
        "coordinate",
        "coordinates",
        "distance",
        "gds",
        "height",
        "image",
        "locator",
        "offset",
        "origin",
        "path",
        "placement",
        "png",
        "seed",
        "serialized",
        "track",
        "tracks",
        "width",
        "x",
        "xy",
        "y",
    }
)


def _canonical_json(data: object) -> str:
    try:
        return json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise IntentFactTypeError("payload must contain exact canonical JSON values") from exc


def _sha256(data: object) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _require_exact_dict(value: object, *, field: str) -> dict[str, object]:
    if type(value) is not dict:
        raise IntentFactTypeError(f"{field} must be an exact dict")
    return value


def _require_exact_tuple(value: object, *, field: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise IntentFactTypeError(f"{field} must be an exact tuple")
    return value


def _require_identifier(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise IntentFactTypeError(f"{field} must be an exact string")
    if _IDENTIFIER_RE.fullmatch(value) is None:
        raise IntentFactSchemaError(f"{field} must be a canonical entity identifier")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise IntentFactTypeError(f"{field} must be an exact string")
    if _SHA256_RE.fullmatch(value) is None:
        raise IntentFactSchemaError(
            f"{field} must be a lowercase 64-hex SHA-256 digest"
        )
    return value


def _require_exact_fields(
    data: object,
    *,
    expected: frozenset[str],
    schema: str,
) -> dict[str, object]:
    row = _require_exact_dict(data, field="canonical replay data")
    if any(type(key) is not str for key in row):
        raise IntentFactTypeError(
            "canonical replay data keys must be exact strings"
        )
    keys = frozenset(row)
    if keys != expected:
        raise IntentFactSchemaError(
            "canonical schema fields disagree: "
            f"missing={sorted(expected - keys)}, unknown={sorted(keys - expected)}"
        )
    raw_schema = row.get("schema")
    if type(raw_schema) is not str:
        raise IntentFactTypeError("schema discriminator must be an exact string")
    if raw_schema != schema:
        raise IntentFactSchemaError(f"canonical schema must be {schema!r}")
    return row


def _json_value(value: object, *, path: str = "payload") -> object:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        raise IntentFactTypeError(f"{path} contains a forbidden float JSON value")
    if type(value) is list:
        return [_json_value(item, path=f"{path}[]") for item in value]
    if type(value) is dict:
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise IntentFactTypeError(f"{path} keys must be exact strings")
            normalized[key] = _json_value(item, path=f"{path}.{key}")
        return normalized
    raise IntentFactTypeError(
        f"{path} must contain only exact JSON values; got {type(value).__name__}"
    )


def _loads_canonical_json(value: object, *, field: str) -> dict[str, object]:
    if type(value) is not str:
        raise IntentFactTypeError(f"{field} must be an exact string")

    def reject_constant(token: str) -> object:
        raise ValueError(f"non-finite JSON constant {token}")

    try:
        payload = json.loads(value, parse_constant=reject_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise IntentFactSchemaError(f"{field} must contain canonical JSON") from exc
    payload = _json_value(payload, path=field)
    if type(payload) is not dict:
        raise IntentFactTypeError(f"{field} must encode a JSON object")
    if _canonical_json(payload) != value:
        raise IntentFactSchemaError(f"{field} must contain canonical JSON")
    return payload


def _keyword_tokens(key: str) -> frozenset[str]:
    return frozenset(token for token in re.split(r"[^a-z0-9]+", key.casefold()) if token)


def _find_reference_leak(value: object) -> tuple[str, str] | None:
    if type(value) is dict:
        for key, child in value.items():
            normalized = key.casefold()
            if (
                normalized == "semantic"
                and type(child) is str
                and child in _CLOSED_PAYLOAD_VALUES["semantic"]
            ):
                # Registered semantic identifiers describe how a discrete fact is
                # interpreted; they are not measured coordinates.  Their closed
                # enum is validated again by ``_validate_payload`` below.
                continue
            tokens = _keyword_tokens(normalized)
            denied = tokens & _REFERENCE_DENIED_KEYWORDS
            if denied:
                word = sorted(denied)[0]
                if word in {"placement", "serialized"}:
                    return ("placement", key)
                if word in {"path", "gds", "locator"}:
                    return ("opaque locator", key)
                if word in {"png", "image"}:
                    return ("PNG locator", key)
                return ("coordinate", key)
            nested = _find_reference_leak(child)
            if nested is not None:
                return nested
    elif type(value) is list:
        for child in value:
            nested = _find_reference_leak(child)
            if nested is not None:
                return nested
    elif type(value) is str:
        folded = value.casefold()
        if ".png" in folded:
            return ("PNG locator", value)
        denied = (
            _keyword_tokens(folded) & _REFERENCE_DENIED_KEYWORDS
        ) - {"x", "xy", "y"}
        if denied:
            word = sorted(denied)[0]
            if word in {"placement", "serialized"}:
                return ("placement", value)
            if word in {"path", "gds", "locator"}:
                return ("opaque locator", value)
            if word in {"png", "image"}:
                return ("PNG locator", value)
            return ("coordinate", value)
        if (
            folded.startswith(("$.", "file:", "/", "\\"))
            or "../" in folded
            or "..\\" in folded
        ):
            return ("opaque locator", value)
    return None


def _find_png_locator(value: object) -> str | None:
    if type(value) is dict:
        for key, child in value.items():
            if "png" in key.casefold() or "locator" in key.casefold():
                return key
            nested = _find_png_locator(child)
            if nested is not None:
                return nested
    elif type(value) is list:
        for child in value:
            nested = _find_png_locator(child)
            if nested is not None:
                return nested
    elif type(value) is str and ".png" in value.casefold():
        return value
    return None


def _validate_payload(
    *,
    kind: IntentFactKind,
    payload: dict[str, object],
    authority: FactAuthority,
) -> None:
    if authority not in _ALLOWED_AUTHORITIES[kind]:
        raise IntentFactSchemaError(
            f"{authority.value} authority is not allowed for {kind.value} facts"
        )

    schema = _PAYLOAD_SCHEMAS[kind]
    keys = frozenset(payload)
    unknown = keys - schema.allowed
    missing = schema.required - keys
    png_locator = _find_png_locator(payload)
    if png_locator is not None:
        raise IntentFactSchemaError(
            f"intent fact contains forbidden PNG locator {png_locator!r}"
        )
    if unknown or missing:
        if authority is FactAuthority.TARGET_REFERENCE_RELATION:
            leak = _find_reference_leak(payload)
            if leak is not None:
                category, key = leak
                raise IntentFactSchemaError(
                    "target-reference fact contains forbidden "
                    f"{category} field {key!r}"
                )
        raise IntentFactSchemaError(
            f"{kind.value} payload schema disagrees: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )

    for field in keys & _ORDINAL_PAYLOAD_FIELDS:
        value = payload[field]
        if type(value) is not int or value < 0:
            raise IntentFactTypeError(
                f"{kind.value} payload field {field!r} must be an exact "
                "non-negative integer"
            )
    for field in keys & _BOOLEAN_PAYLOAD_FIELDS:
        if type(payload[field]) is not bool:
            raise IntentFactTypeError(
                f"{kind.value} payload field {field!r} must be an exact boolean"
            )
        if payload[field] is not True:
            raise IntentFactSchemaError(
                f"{kind.value} payload field {field!r} must be true"
            )
    for field in keys & _IDENTIFIER_PAYLOAD_FIELDS:
        value = _require_identifier(
            payload[field],
            field=f"{kind.value} payload field {field!r}",
        )
        if value not in _CLOSED_PAYLOAD_VALUES[field]:
            raise IntentFactSchemaError(
                f"{kind.value} payload field {field!r} has no closed value "
                f"registration for {value!r}"
            )
    if authority is FactAuthority.TARGET_REFERENCE_RELATION:
        leak = _find_reference_leak(payload)
        if leak is not None:
            category, key = leak
            raise IntentFactSchemaError(
                f"target-reference fact contains forbidden {category} field {key!r}"
            )


def _require_fact_id_template(
    fact_id: str,
    *,
    expected: str | frozenset[str],
) -> None:
    allowed = frozenset({expected}) if type(expected) is str else expected
    if fact_id not in allowed:
        raise IntentFactSchemaError(
            "fact_id does not match its kind/subjects canonical template: "
            f"expected={sorted(allowed)}, actual={fact_id!r}"
        )


def _escape_fact_id_component(value: str) -> str:
    return value.replace("%", "%25").replace(":", "%3A")


def _subject_signature_error(kind: IntentFactKind) -> IntentFactSchemaError:
    return IntentFactSchemaError(
        f"{kind.value} subjects violate the closed subject signature"
    )


def _validate_subjects_and_fact_id(
    *,
    fact_id: str,
    kind: IntentFactKind,
    subjects: tuple[EntityRef, ...],
    payload: dict[str, object],
) -> None:
    if kind is IntentFactKind.ORDINAL_CELL:
        if len(subjects) != 1 or subjects[0].kind not in {
            EntityKind.PLACE_GROUP,
            EntityKind.MODULE,
        }:
            raise _subject_signature_error(kind)
        _require_fact_id_template(
            fact_id,
            expected=(
                f"floorplan:{subjects[0].kind.value}:"
                f"{_escape_fact_id_component(subjects[0].name)}:cell"
            ),
        )
        return

    if kind in {IntentFactKind.ORDER, IntentFactKind.ADJACENCY}:
        allowed = {
            EntityKind.DEVICE,
            EntityKind.PLACE_GROUP,
            EntityKind.MODULE,
        }
        if kind is IntentFactKind.ORDER:
            allowed.add(EntityKind.ROUTE_GROUP)
        if (
            len(subjects) != 2
            or subjects[0].kind is not subjects[1].kind
            or subjects[0].kind not in allowed
        ):
            raise _subject_signature_error(kind)
        prefix = "order" if kind is IntentFactKind.ORDER else "adjacency"
        _require_fact_id_template(
            fact_id,
            expected=(
                f"{prefix}:{subjects[0].kind.value}:"
                f"{_escape_fact_id_component(subjects[0].name)}:"
                f"{_escape_fact_id_component(subjects[1].name)}:"
                f"{payload['axis']}"
            ),
        )
        return

    if kind in {
        IntentFactKind.MEMBER_PARTITION,
        IntentFactKind.MEMBER_CONSERVATION,
    }:
        compatibility = {
            EntityKind.PLACE_GROUP: {EntityKind.DEVICE},
            EntityKind.ROUTE_GROUP: {EntityKind.PLACE_GROUP},
            EntityKind.WELL_DOMAIN: {
                EntityKind.DEVICE,
                EntityKind.PLACE_GROUP,
            },
            EntityKind.MODULE: {
                EntityKind.DEVICE,
                EntityKind.PLACE_GROUP,
            },
        }
        containers: list[EntityRef] = []
        for index, container in enumerate(subjects):
            members = subjects[:index] + subjects[index + 1 :]
            if (
                members
                and members[0].kind in compatibility.get(container.kind, set())
                and all(member.kind is members[0].kind for member in members)
            ):
                containers.append(container)
        if len(containers) != 1:
            raise IntentFactSchemaError(
                "member_partition subjects violate the closed "
                "container/member compatibility matrix"
            )
        container = containers[0]
        if kind is IntentFactKind.MEMBER_PARTITION:
            expected = (
                f"partition:{container.kind.value}:"
                f"{_escape_fact_id_component(container.name)}"
            )
        else:
            expected = (
                f"conservation:{container.kind.value}:"
                f"{_escape_fact_id_component(container.name)}:members"
            )
        _require_fact_id_template(fact_id, expected=expected)
        return

    if kind in {
        IntentFactKind.NO_FOREIGN_ENVELOPE,
        IntentFactKind.FINAL_CONTAINMENT,
    }:
        if len(subjects) != 1 or subjects[0].kind not in {
            EntityKind.MODULE,
            EntityKind.PLACE_GROUP,
            EntityKind.ROUTE_GROUP,
            EntityKind.WELL_DOMAIN,
        }:
            raise _subject_signature_error(kind)
        if kind is IntentFactKind.NO_FOREIGN_ENVELOPE:
            expected = (
                f"envelope:{subjects[0].kind.value}:"
                f"{_escape_fact_id_component(subjects[0].name)}"
            )
        else:
            expected = (
                f"containment:{subjects[0].kind.value}:"
                f"{_escape_fact_id_component(subjects[0].name)}:final"
            )
        _require_fact_id_template(fact_id, expected=expected)
        return

    if kind is IntentFactKind.WELL_DOMAIN:
        domains = tuple(
            subject
            for subject in subjects
            if subject.kind is EntityKind.WELL_DOMAIN
        )
        members = tuple(
            subject
            for subject in subjects
            if subject.kind is not EntityKind.WELL_DOMAIN
        )
        if (
            len(domains) != 1
            or not members
            or members[0].kind
            not in {EntityKind.DEVICE, EntityKind.PLACE_GROUP}
            or any(member.kind is not members[0].kind for member in members)
        ):
            raise _subject_signature_error(kind)
        _require_fact_id_template(
            fact_id,
            expected=f"well:{_escape_fact_id_component(domains[0].name)}",
        )
        return

    if kind in {
        IntentFactKind.EXTERNAL_PORT_DENOMINATOR,
        IntentFactKind.TOPOLOGY_EXTERNAL_NET_INVENTORY,
    }:
        compatibility = {
            EntityKind.PLACE_GROUP: {EntityKind.DEVICE},
            EntityKind.ROUTE_GROUP: {EntityKind.PLACE_GROUP},
            EntityKind.MODULE: {
                EntityKind.DEVICE,
                EntityKind.PLACE_GROUP,
            },
        }
        nets = tuple(
            subject for subject in subjects if subject.kind is EntityKind.NET
        )
        non_nets = tuple(
            subject for subject in subjects if subject.kind is not EntityKind.NET
        )
        containers: list[EntityRef] = []
        for index, container in enumerate(non_nets):
            members = non_nets[:index] + non_nets[index + 1 :]
            if (
                members
                and members[0].kind in compatibility.get(container.kind, set())
                and all(member.kind is members[0].kind for member in members)
            ):
                containers.append(container)
        if len(containers) != 1 or not nets:
            raise _subject_signature_error(kind)
        if kind is IntentFactKind.EXTERNAL_PORT_DENOMINATOR:
            expected = (
                f"ports:{containers[0].kind.value}:"
                f"{_escape_fact_id_component(containers[0].name)}:external"
            )
        else:
            expected = (
                f"topology:{containers[0].kind.value}:"
                f"{_escape_fact_id_component(containers[0].name)}:external_nets"
            )
        _require_fact_id_template(fact_id, expected=expected)
        return

    if kind is IntentFactKind.MATCHED_ROUTE_CLASS:
        if len(subjects) != 2 or any(
            subject.kind is not EntityKind.NET for subject in subjects
        ):
            raise _subject_signature_error(kind)
        _require_fact_id_template(
            fact_id,
            expected=(
                f"route:net:{_escape_fact_id_component(subjects[0].name)}:"
                f"{_escape_fact_id_component(subjects[1].name)}:matched"
            ),
        )
        return

    if kind in {
        IntentFactKind.NET_WIDTH_CLASS,
        IntentFactKind.VIA_CAPACITY_CLASS,
    }:
        if len(subjects) != 1 or subjects[0].kind is not EntityKind.NET:
            raise _subject_signature_error(kind)
        if kind is IntentFactKind.NET_WIDTH_CLASS:
            expected = f"width:net:{_escape_fact_id_component(subjects[0].name)}"
        else:
            expected = (
                f"via:net:{_escape_fact_id_component(subjects[0].name)}:capacity"
            )
        _require_fact_id_template(fact_id, expected=expected)
        return

    raise IntentFactSchemaError(f"{kind.value} has no closed subject schema")


def _validate_payload_combination(
    *,
    kind: IntentFactKind,
    subjects: tuple[EntityRef, ...],
    payload: dict[str, object],
) -> None:
    if kind in {IntentFactKind.ORDER, IntentFactKind.ADJACENCY}:
        axis = payload["axis"]
        relation = payload["relation"]
        horizontal_axes = {"column", "horizontal", "x"}
        if kind is IntentFactKind.ORDER:
            allowed = (
                {"after", "before", "left_of", "right_of"}
                if axis in horizontal_axes
                else {"above", "after", "before", "below"}
            )
        else:
            allowed = (
                {"abut", "adjacent", "left_of", "right_of"}
                if axis in horizontal_axes
                else {"above", "abut", "adjacent", "below"}
            )
        if relation not in allowed:
            raise IntentFactSchemaError(
                f"{kind.value} axis/relation payload combination is invalid"
            )
    elif kind is IntentFactKind.NO_FOREIGN_ENVELOPE:
        scope = payload["scope"]
        if scope not in {"all_members", "members", subjects[0].kind.value}:
            raise IntentFactSchemaError(
                "no_foreign_envelope scope is incompatible with its container"
            )
    elif kind is IntentFactKind.WELL_DOMAIN:
        if (
            payload["sharing"] == "shared"
            and payload["body_relation"] != "same_body"
        ):
            raise IntentFactSchemaError(
                "shared well_domain facts require same_body relation"
            )
    elif kind is IntentFactKind.FINAL_CONTAINMENT:
        if payload["scope"] != "final_layout":
            raise IntentFactSchemaError(
                "final_containment scope must be final_layout"
            )


@dataclass(frozen=True, order=True)
class EntityRef:
    kind: EntityKind
    name: str

    def __post_init__(self) -> None:
        if type(self.kind) is not EntityKind:
            raise IntentFactTypeError("kind must be an exact EntityKind")
        _require_identifier(self.name, field="name")

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _ENTITY_SCHEMA,
            "kind": self.kind.value,
            "name": self.name,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_data())

    @classmethod
    def from_data(cls, data: object) -> "EntityRef":
        row = _require_exact_fields(
            data,
            expected=frozenset({"schema", "kind", "name"}),
            schema=_ENTITY_SCHEMA,
        )
        raw_kind = row["kind"]
        if type(raw_kind) is not str:
            raise IntentFactTypeError("entity kind canonical data must be a string")
        try:
            kind = EntityKind(raw_kind)
        except ValueError as exc:
            raise IntentFactSchemaError(f"unknown entity kind {raw_kind!r}") from exc
        return cls(kind=kind, name=row["name"])  # type: ignore[arg-type]


@dataclass(frozen=True)
class SourceInventory:
    circuit_id: str
    source_sha256: str
    topology_sha256: str
    entities: tuple[EntityRef, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.circuit_id, field="circuit_id")
        _require_sha256(self.source_sha256, field="source_sha256")
        _require_sha256(self.topology_sha256, field="topology_sha256")
        entities = _require_exact_tuple(self.entities, field="entities")
        if any(type(entity) is not EntityRef for entity in entities):
            raise IntentFactTypeError(
                "entities must contain exact EntityRef values"
            )
        if tuple(sorted(set(entities))) != entities:
            raise IntentFactSchemaError(
                "entities must contain sorted unique values"
            )

    @classmethod
    def from_inventory(
        cls,
        *,
        circuit_id: str,
        source_sha256: str,
        topology_sha256: str,
        inventory: dict[EntityKind, tuple[str, ...]],
    ) -> "SourceInventory":
        rows = _require_exact_dict(inventory, field="source inventory")
        entities: list[EntityRef] = []
        for kind, names in rows.items():
            if type(kind) is not EntityKind:
                raise IntentFactTypeError(
                    "source inventory keys must be exact EntityKind values"
                )
            typed_names = _require_exact_tuple(
                names,
                field=f"{kind.value} inventory",
            )
            canonical_names = tuple(
                _require_identifier(
                    name,
                    field=f"{kind.value} inventory entity",
                )
                for name in typed_names
            )
            if tuple(sorted(set(canonical_names))) != canonical_names:
                raise IntentFactSchemaError(
                    f"{kind.value} inventory must contain sorted unique identifiers"
                )
            entities.extend(EntityRef(kind, name) for name in canonical_names)
        return cls(
            circuit_id=circuit_id,
            source_sha256=source_sha256,
            topology_sha256=topology_sha256,
            entities=tuple(sorted(entities)),
        )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _SOURCE_INVENTORY_SCHEMA,
            "circuit_id": self.circuit_id,
            "source_sha256": self.source_sha256,
            "topology_sha256": self.topology_sha256,
            "entities": [entity.canonical_data() for entity in self.entities],
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_data())

    @classmethod
    def from_data(cls, data: object) -> "SourceInventory":
        row = _require_exact_fields(
            data,
            expected=frozenset(
                {
                    "schema",
                    "circuit_id",
                    "source_sha256",
                    "topology_sha256",
                    "entities",
                }
            ),
            schema=_SOURCE_INVENTORY_SCHEMA,
        )
        raw_entities = row["entities"]
        if type(raw_entities) is not list:
            raise IntentFactTypeError(
                "entities canonical data must be a list"
            )
        return cls(
            circuit_id=row["circuit_id"],  # type: ignore[arg-type]
            source_sha256=row["source_sha256"],  # type: ignore[arg-type]
            topology_sha256=row["topology_sha256"],  # type: ignore[arg-type]
            entities=tuple(EntityRef.from_data(entity) for entity in raw_entities),
        )


@dataclass(frozen=True)
class IntentFactAuthorityReceipt:
    circuit_id: str
    source_sha256: str
    topology_sha256: str
    fact_set_sha256: str
    source_inventory_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.circuit_id, field="circuit_id")
        _require_sha256(self.source_sha256, field="source_sha256")
        _require_sha256(self.topology_sha256, field="topology_sha256")
        _require_sha256(self.fact_set_sha256, field="fact_set_sha256")
        _require_sha256(
            self.source_inventory_sha256,
            field="source_inventory_sha256",
        )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _AUTHORITY_RECEIPT_SCHEMA,
            "circuit_id": self.circuit_id,
            "source_sha256": self.source_sha256,
            "topology_sha256": self.topology_sha256,
            "fact_set_sha256": self.fact_set_sha256,
            "source_inventory_sha256": self.source_inventory_sha256,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_data())

    @classmethod
    def from_data(cls, data: object) -> "IntentFactAuthorityReceipt":
        row = _require_exact_fields(
            data,
            expected=frozenset(
                {
                    "schema",
                    "circuit_id",
                    "source_sha256",
                    "topology_sha256",
                    "fact_set_sha256",
                    "source_inventory_sha256",
                }
            ),
            schema=_AUTHORITY_RECEIPT_SCHEMA,
        )
        return cls(
            circuit_id=row["circuit_id"],  # type: ignore[arg-type]
            source_sha256=row["source_sha256"],  # type: ignore[arg-type]
            topology_sha256=row["topology_sha256"],  # type: ignore[arg-type]
            fact_set_sha256=row["fact_set_sha256"],  # type: ignore[arg-type]
            source_inventory_sha256=row[
                "source_inventory_sha256"
            ],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, order=True)
class IntentFact:
    fact_id: str
    kind: IntentFactKind
    subjects: tuple[EntityRef, ...]
    payload_json: str
    authority: FactAuthority
    required_consumers: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.fact_id, field="fact_id")
        if type(self.kind) is not IntentFactKind:
            raise IntentFactTypeError("kind must be an exact IntentFactKind")
        subjects = _require_exact_tuple(self.subjects, field="subjects")
        if not subjects or any(type(subject) is not EntityRef for subject in subjects):
            raise IntentFactTypeError(
                "subjects must contain one or more exact EntityRef values"
            )
        if tuple(sorted(set(subjects))) != subjects:
            raise IntentFactSchemaError("subjects must contain sorted unique values")
        if type(self.authority) is not FactAuthority:
            raise IntentFactTypeError("authority must be an exact FactAuthority")
        consumers = _require_exact_tuple(
            self.required_consumers,
            field="required_consumers",
        )
        if not consumers:
            raise IntentFactSchemaError("required_consumers cannot be empty")
        for consumer in consumers:
            _require_identifier(consumer, field="required consumer")
        if tuple(sorted(set(consumers))) != consumers:
            raise IntentFactSchemaError(
                "required_consumers must contain sorted unique values"
            )
        unknown_consumers = frozenset(consumers) - REGISTERED_CONSUMERS
        if unknown_consumers:
            raise IntentFactSchemaError(
                "unknown required consumer values: "
                f"{sorted(unknown_consumers)}"
            )
        missing_consumers = MINIMUM_REQUIRED_CONSUMERS[self.kind] - frozenset(
            consumers
        )
        if missing_consumers:
            raise IntentFactSchemaError(
                f"{self.kind.value} is missing mandatory consumers: "
                f"{sorted(missing_consumers)}"
            )
        payload = _loads_canonical_json(self.payload_json, field="payload_json")
        _validate_payload(kind=self.kind, payload=payload, authority=self.authority)
        _validate_subjects_and_fact_id(
            fact_id=self.fact_id,
            kind=self.kind,
            subjects=subjects,
            payload=payload,
        )
        _validate_payload_combination(
            kind=self.kind,
            subjects=subjects,
            payload=payload,
        )

    @classmethod
    def from_payload(
        cls,
        *,
        fact_id: str,
        kind: IntentFactKind,
        subjects: tuple[EntityRef, ...],
        payload: dict[str, object],
        authority: FactAuthority,
        required_consumers: tuple[str, ...],
    ) -> "IntentFact":
        row = _require_exact_dict(payload, field="payload")
        normalized = _json_value(row)
        return cls(
            fact_id=fact_id,
            kind=kind,
            subjects=subjects,
            payload_json=_canonical_json(normalized),
            authority=authority,
            required_consumers=required_consumers,
        )

    @property
    def payload(self) -> dict[str, object]:
        return _loads_canonical_json(self.payload_json, field="payload_json")

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _FACT_SCHEMA,
            "fact_id": self.fact_id,
            "kind": self.kind.value,
            "subjects": [subject.canonical_data() for subject in self.subjects],
            "payload": self.payload,
            "authority": self.authority.value,
            "required_consumers": list(self.required_consumers),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_data())

    @classmethod
    def from_data(cls, data: object) -> "IntentFact":
        row = _require_exact_fields(
            data,
            expected=frozenset(
                {
                    "schema",
                    "fact_id",
                    "kind",
                    "subjects",
                    "payload",
                    "authority",
                    "required_consumers",
                }
            ),
            schema=_FACT_SCHEMA,
        )
        raw_kind = row["kind"]
        if type(raw_kind) is not str:
            raise IntentFactTypeError("kind canonical data must be a string")
        try:
            kind = IntentFactKind(raw_kind)
        except ValueError as exc:
            raise IntentFactSchemaError(
                f"unknown intent fact kind {raw_kind!r}"
            ) from exc
        raw_authority = row["authority"]
        if type(raw_authority) is not str:
            raise IntentFactTypeError("authority canonical data must be a string")
        try:
            authority = FactAuthority(raw_authority)
        except ValueError as exc:
            raise IntentFactSchemaError(
                f"unknown fact authority {raw_authority!r}"
            ) from exc
        raw_subjects = row["subjects"]
        if type(raw_subjects) is not list:
            raise IntentFactTypeError("subjects canonical data must be a list")
        raw_consumers = row["required_consumers"]
        if type(raw_consumers) is not list:
            raise IntentFactTypeError(
                "required_consumers canonical data must be a list"
            )
        if any(type(consumer) is not str for consumer in raw_consumers):
            raise IntentFactTypeError(
                "required_consumers canonical data must contain strings"
            )
        payload = row["payload"]
        if type(payload) is not dict:
            raise IntentFactTypeError("payload canonical data must be an exact dict")
        return cls.from_payload(
            fact_id=row["fact_id"],  # type: ignore[arg-type]
            kind=kind,
            subjects=tuple(EntityRef.from_data(subject) for subject in raw_subjects),
            payload=payload,
            authority=authority,
            required_consumers=tuple(raw_consumers),
        )


@dataclass(frozen=True)
class IntentFactSet:
    circuit_id: str
    claim_scope: ClaimScope
    source_sha256: str
    topology_sha256: str
    pdk_snapshot_sha256: str
    pdk_installed_authority_sha256: str
    facts: tuple[IntentFact, ...]
    target_reference_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.circuit_id, field="circuit_id")
        if type(self.claim_scope) is not ClaimScope:
            raise IntentFactTypeError("claim_scope must be an exact ClaimScope")
        _require_sha256(self.source_sha256, field="source_sha256")
        _require_sha256(self.topology_sha256, field="topology_sha256")
        _require_sha256(self.pdk_snapshot_sha256, field="pdk_snapshot_sha256")
        _require_sha256(
            self.pdk_installed_authority_sha256,
            field="pdk_installed_authority_sha256",
        )
        facts = _require_exact_tuple(self.facts, field="facts")
        if any(type(fact) is not IntentFact for fact in facts):
            raise IntentFactTypeError("facts must contain exact IntentFact values")
        fact_ids = tuple(fact.fact_id for fact in facts)
        if tuple(sorted(set(fact_ids))) != fact_ids:
            raise IntentFactSchemaError(
                "facts must contain sorted unique fact IDs"
            )

        has_target_reference = any(
            fact.authority is FactAuthority.TARGET_REFERENCE_RELATION
            for fact in facts
        )
        if has_target_reference:
            if self.claim_scope is ClaimScope.HELD_OUT_GENERALIZATION:
                raise IntentFactSchemaError(
                    "held-out fact set cannot contain a target-reference fact"
                )
            if self.claim_scope is ClaimScope.REFERENCE_FREE_SINGLE_CIRCUIT:
                raise IntentFactSchemaError(
                    "reference-free fact set cannot contain a target-reference fact"
                )

        receipt = self.target_reference_receipt_sha256
        if receipt is not None:
            _require_sha256(receipt, field="target_reference_receipt_sha256")
            if (
                self.claim_scope
                is not ClaimScope.REFERENCE_CONDITIONED_SINGLE_CIRCUIT
            ):
                raise IntentFactSchemaError(
                    "target-reference receipt is allowed only for a "
                    "reference-conditioned fact set"
                )
        if has_target_reference and receipt is None:
            raise IntentFactSchemaError(
                "target-reference facts require a bound reference receipt"
            )

    def validate_inventory(
        self,
        inventory: dict[EntityKind, tuple[str, ...]],
    ) -> None:
        rows = _require_exact_dict(inventory, field="source inventory")
        normalized: dict[EntityKind, frozenset[str]] = {}
        for kind, names in rows.items():
            if type(kind) is not EntityKind:
                raise IntentFactTypeError(
                    "source inventory keys must be exact EntityKind values"
                )
            typed_names = _require_exact_tuple(
                names,
                field=f"{kind.value} inventory",
            )
            canonical_names = tuple(
                _require_identifier(name, field=f"{kind.value} inventory entity")
                for name in typed_names
            )
            if tuple(sorted(set(canonical_names))) != canonical_names:
                raise IntentFactSchemaError(
                    f"{kind.value} inventory must contain sorted unique identifiers"
                )
            normalized[kind] = frozenset(canonical_names)

        for fact in self.facts:
            for subject in fact.subjects:
                if subject.name not in normalized.get(subject.kind, frozenset()):
                    raise IntentFactSchemaError(
                        "unknown source entity "
                        f"{subject.kind.value}:{subject.name!s} in fact {fact.fact_id!r}"
                    )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _FACT_SET_SCHEMA,
            "circuit_id": self.circuit_id,
            "claim_scope": self.claim_scope.value,
            "source_sha256": self.source_sha256,
            "topology_sha256": self.topology_sha256,
            "pdk_snapshot_sha256": self.pdk_snapshot_sha256,
            "pdk_installed_authority_sha256": self.pdk_installed_authority_sha256,
            "facts": [fact.canonical_data() for fact in self.facts],
            "target_reference_receipt_sha256": (
                self.target_reference_receipt_sha256
            ),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_data())

    @classmethod
    def from_data(cls, data: object) -> "IntentFactSet":
        row = _require_exact_fields(
            data,
            expected=frozenset(
                {
                    "schema",
                    "circuit_id",
                    "claim_scope",
                    "source_sha256",
                    "topology_sha256",
                    "pdk_snapshot_sha256",
                    "pdk_installed_authority_sha256",
                    "facts",
                    "target_reference_receipt_sha256",
                }
            ),
            schema=_FACT_SET_SCHEMA,
        )
        raw_scope = row["claim_scope"]
        if type(raw_scope) is not str:
            raise IntentFactTypeError("claim_scope canonical data must be a string")
        try:
            scope = ClaimScope(raw_scope)
        except ValueError as exc:
            raise IntentFactSchemaError(
                f"unknown claim scope {raw_scope!r}"
            ) from exc
        raw_facts = row["facts"]
        if type(raw_facts) is not list:
            raise IntentFactTypeError("facts canonical data must be a list")
        receipt = row["target_reference_receipt_sha256"]
        if receipt is not None and type(receipt) is not str:
            raise IntentFactTypeError(
                "target_reference_receipt_sha256 canonical data must be null or a string"
            )
        return cls(
            circuit_id=row["circuit_id"],  # type: ignore[arg-type]
            claim_scope=scope,
            source_sha256=row["source_sha256"],  # type: ignore[arg-type]
            topology_sha256=row["topology_sha256"],  # type: ignore[arg-type]
            pdk_snapshot_sha256=row["pdk_snapshot_sha256"],  # type: ignore[arg-type]
            pdk_installed_authority_sha256=row[
                "pdk_installed_authority_sha256"
            ],  # type: ignore[arg-type]
            facts=tuple(IntentFact.from_data(fact) for fact in raw_facts),
            target_reference_receipt_sha256=receipt,
        )


def verify_intent_fact_authority(
    *,
    fact_set: IntentFactSet,
    source_inventory: SourceInventory,
) -> IntentFactAuthorityReceipt:
    """Bind an exact fact set to its typed source inventory and identities."""

    if type(fact_set) is not IntentFactSet:
        raise IntentFactAuthorityError(
            "fact_set must be an exact IntentFactSet",
            blocker="AUTHORITY_IDENTITY",
        )
    if type(source_inventory) is not SourceInventory:
        raise IntentFactAuthorityError(
            "source_inventory must be an exact SourceInventory",
            blocker="AUTHORITY_IDENTITY",
        )
    for field in ("circuit_id", "source_sha256", "topology_sha256"):
        if getattr(fact_set, field) != getattr(source_inventory, field):
            raise IntentFactAuthorityError(
                f"{field} differs between fact set and source inventory",
                blocker="AUTHORITY_IDENTITY",
            )

    known_entities = frozenset(source_inventory.entities)
    for fact in fact_set.facts:
        for subject in fact.subjects:
            if subject not in known_entities:
                raise IntentFactAuthorityError(
                    "unknown source entity "
                    f"{subject.kind.value}:{subject.name!s} in fact "
                    f"{fact.fact_id!r}",
                    blocker="FACT_SCHEMA",
                )

    return IntentFactAuthorityReceipt(
        circuit_id=fact_set.circuit_id,
        source_sha256=fact_set.source_sha256,
        topology_sha256=fact_set.topology_sha256,
        fact_set_sha256=fact_set.sha256,
        source_inventory_sha256=source_inventory.sha256,
    )


__all__ = [
    "ClaimScope",
    "EntityKind",
    "EntityRef",
    "FactAuthority",
    "IntentFact",
    "IntentFactAuthorityError",
    "IntentFactAuthorityReceipt",
    "IntentFactKind",
    "IntentFactSchemaError",
    "IntentFactSet",
    "IntentFactTypeError",
    "MINIMUM_REQUIRED_CONSUMERS",
    "REGISTERED_CONSUMER_ORDER",
    "REGISTERED_CONSUMERS",
    "SourceInventory",
    "verify_intent_fact_authority",
]
