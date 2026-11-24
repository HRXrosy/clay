"""Replayable, PDK-bound area floors for structure-first physical search.

This module deliberately computes only mathematical lower bounds.  It never
uses a reference layout bbox, a placed candidate bbox, or a cheap-route score.
MOS arrays use origin-free canonical active artwork; standalone MOS leaves use
only mandatory channel rectangles.  Proved non-shareable atom projections sum
on each physical layer, while alternative layer and resource-class projections
take a maximum to avoid double counting.  No PCell bounding box is an area term.
Optional tap/ring increments stay zero until canonical
materialization proves they cannot be shared or PCell-integrated; undeclared
channels are also zero.  All PDK-dependent terms bind exact installed rule
bytes.  Consumers must replay the input receipt before using the result as a
search gap denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal


_MAX_COUNT = 4096
_MAX_NM = 1_000_000_000_000
_MAX_TEXT = 256
_FOOTPRINT_KINDS = {"device", "passive", "tap"}
_GROUP_KINDS = {"passive_bank", "well_island"}
_LAYER_RESOURCE_KINDS = {"device_active", "passive", "tap_active"}
_LAYER_RESOURCE_PROOFS = {
    "canonical_common_centroid_active",
    "canonical_device_active",
    "canonical_device_channel",
    "canonical_fixed_cell_layer",
    "canonical_passive_recognition",
    "canonical_shared_diffusion_active",
    "canonical_tap_active",
}
_NONSHAREABLE_DOMAIN_PROOFS = {
    "canonical_array_atom",
    "canonical_channel_atom",
    "canonical_rigid_macro",
    "canonical_rigid_atom",
    "canonical_tap_atom",
}
_RESOURCE_CLASSES = {
    "device_functional_area",
    "passive_banks",
    "taps",
    "well_islands",
    "well_domain_spacing",
    "guard_rings",
    "rails",
    "vias",
    "channel_capacity",
}


class PdkLegalLowerBoundError(ValueError):
    """One typed, fail-closed lower-bound compilation error."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = _text(code, "lower-bound error code")
        self.detail = _text(detail, "lower-bound error detail")
        super().__init__(f"{self.code}: {self.detail}")


def _text(value: object, label: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or len(value) > _MAX_TEXT:
        raise ValueError(f"{label} must be a bounded exact string")
    if not allow_empty and not value:
        raise ValueError(f"{label} must be non-empty")
    if value.casefold().endswith(".png"):
        raise ValueError(f"{label} cannot name PNG evidence")
    return value


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or not 0 < value <= _MAX_NM:
        raise ValueError(f"{label} must be a bounded positive exact integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_NM:
        raise ValueError(f"{label} must be a bounded non-negative exact integer")
    return value


def _exact_tuple(value: object, label: str, *, nonempty: bool = False) -> tuple:
    if type(value) is not tuple or len(value) > _MAX_COUNT:
        raise ValueError(f"{label} must be a bounded exact tuple")
    if nonempty and not value:
        raise ValueError(f"{label} must be non-empty")
    return value


def _ceil_integer_sqrt_product(coefficient: int, radicand: int) -> int:
    """Return ``ceil(coefficient * sqrt(radicand))`` using integer math."""

    squared = coefficient * coefficient * radicand
    root = math.isqrt(squared)
    return root if root * root == squared else root + 1


def _enclosed_area_floor(area_nm2: int, margin_nm: int) -> int:
    # For any inner rectangle w*h >= A, AM-GM gives
    # (w+2m)(h+2m) >= A + 4m*sqrt(A) + 4m^2.
    return (
        area_nm2
        + _ceil_integer_sqrt_product(4 * margin_nm, area_nm2)
        + 4 * margin_nm * margin_nm
    )


@dataclass(frozen=True, order=True)
class LegalFootprint:
    member_id: str
    kind: Literal["device", "passive", "tap"]
    width_nm: int
    height_nm: int
    geometry_sha256: str
    body_net: str = ""

    def __post_init__(self) -> None:
        _text(self.member_id, "footprint member_id")
        if type(self.kind) is not str or self.kind not in _FOOTPRINT_KINDS:
            raise ValueError("footprint kind is unsupported")
        _positive_int(self.width_nm, "footprint width_nm")
        _positive_int(self.height_nm, "footprint height_nm")
        _sha256(self.geometry_sha256, "footprint geometry_sha256")
        _text(self.body_net, "footprint body_net", allow_empty=True)
        if self.kind == "passive" and self.body_net:
            raise ValueError("passive footprint cannot declare a body net")

    @property
    def area_nm2(self) -> int:
        self.__post_init__()
        return self.width_nm * self.height_nm

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "body_net": self.body_net,
            "geometry_sha256": self.geometry_sha256,
            "height_nm": self.height_nm,
            "kind": self.kind,
            "member_id": self.member_id,
            "width_nm": self.width_nm,
        }


@dataclass(frozen=True, order=True)
class LegalLayerResource:
    resource_id: str
    kind: Literal["device_active", "passive", "tap_active"]
    members: tuple[str, ...]
    layer_key: tuple[int, int]
    area_nm2: int
    width_nm: int
    height_nm: int
    geometry_sha256: str
    proof_kind: str
    nonshareable_domain_sha256: str
    nonshareable_proof_kind: str

    def __post_init__(self) -> None:
        _text(self.resource_id, "layer resource_id")
        if type(self.kind) is not str or self.kind not in _LAYER_RESOURCE_KINDS:
            raise ValueError("layer resource kind is unsupported")
        members = _exact_tuple(
            self.members, "layer resource members", nonempty=True
        )
        if any(type(member) is not str for member in members):
            raise ValueError("layer resource members must be exact strings")
        canonical_members = tuple(
            sorted(_text(member, "layer resource member") for member in members)
        )
        if len(set(canonical_members)) != len(canonical_members):
            raise ValueError("layer resource members contain a duplicate")
        if (
            type(self.layer_key) is not tuple
            or len(self.layer_key) != 2
            or any(type(value) is not int or value < 0 for value in self.layer_key)
        ):
            raise ValueError(
                "layer resource layer_key must be an exact non-negative pair"
            )
        _positive_int(self.area_nm2, "layer resource area_nm2")
        _positive_int(self.width_nm, "layer resource width_nm")
        _positive_int(self.height_nm, "layer resource height_nm")
        if self.area_nm2 > self.width_nm * self.height_nm:
            raise ValueError("layer resource area exceeds its projection envelope")
        _sha256(self.geometry_sha256, "layer resource geometry_sha256")
        if (
            type(self.proof_kind) is not str
            or self.proof_kind not in _LAYER_RESOURCE_PROOFS
        ):
            raise ValueError("layer resource proof kind is unsupported")
        _sha256(
            self.nonshareable_domain_sha256,
            "layer resource nonshareable_domain_sha256",
        )
        if (
            type(self.nonshareable_proof_kind) is not str
            or self.nonshareable_proof_kind not in _NONSHAREABLE_DOMAIN_PROOFS
        ):
            raise ValueError("layer resource nonshareable proof kind is unsupported")
        object.__setattr__(self, "members", canonical_members)

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "area_nm2": self.area_nm2,
            "geometry_sha256": self.geometry_sha256,
            "height_nm": self.height_nm,
            "kind": self.kind,
            "layer_key": list(self.layer_key),
            "members": list(self.members),
            "nonshareable_domain_sha256": self.nonshareable_domain_sha256,
            "nonshareable_proof_kind": self.nonshareable_proof_kind,
            "proof_kind": self.proof_kind,
            "resource_id": self.resource_id,
            "width_nm": self.width_nm,
        }


@dataclass(frozen=True, order=True)
class LegalPlacementGroup:
    group_id: str
    kind: Literal["passive_bank", "well_island"]
    members: tuple[str, ...]
    tie_net: str = ""
    well_enclosure_nm: int = 0
    well_spacing_nm: int = 0
    guard_ring_setback_nm: int = 0
    guard_ring_width_nm: int = 0
    tap_integrated: bool = False

    def __post_init__(self) -> None:
        _text(self.group_id, "placement group_id")
        if type(self.kind) is not str or self.kind not in _GROUP_KINDS:
            raise ValueError("placement group kind is unsupported")
        members = _exact_tuple(self.members, "placement group members", nonempty=True)
        if any(type(member) is not str for member in members):
            raise ValueError("placement group members must be exact strings")
        canonical = tuple(sorted(_text(member, "placement group member") for member in members))
        if len(set(canonical)) != len(canonical):
            raise ValueError("placement group members contain a duplicate")
        _text(self.tie_net, "placement group tie_net", allow_empty=True)
        for name in (
            "well_enclosure_nm",
            "well_spacing_nm",
            "guard_ring_setback_nm",
            "guard_ring_width_nm",
        ):
            _nonnegative_int(getattr(self, name), f"placement group {name}")
        if type(self.tap_integrated) is not bool:
            raise ValueError("placement group tap_integrated must be an exact bool")
        if self.kind == "passive_bank" and (
            self.tie_net
            or self.well_enclosure_nm
            or self.well_spacing_nm
            or self.guard_ring_setback_nm
            or self.guard_ring_width_nm
            or self.tap_integrated
        ):
            raise ValueError("passive bank cannot carry well/ring rules")
        if self.kind == "well_island" and not self.tie_net:
            raise ValueError("well island requires a tie net")
        if self.tap_integrated and not self.guard_ring_width_nm:
            raise ValueError("an integrated tap requires a materialized guard ring")
        object.__setattr__(self, "members", canonical)

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "group_id": self.group_id,
            "guard_ring_setback_nm": self.guard_ring_setback_nm,
            "guard_ring_width_nm": self.guard_ring_width_nm,
            "kind": self.kind,
            "members": list(self.members),
            "tie_net": self.tie_net,
            "tap_integrated": self.tap_integrated,
            "well_enclosure_nm": self.well_enclosure_nm,
            "well_spacing_nm": self.well_spacing_nm,
        }


@dataclass(frozen=True, order=True)
class RailRequirement:
    net: str
    layer_n: int
    required_count: int
    min_width_nm: int
    min_length_nm: int

    def __post_init__(self) -> None:
        _text(self.net, "rail net")
        _positive_int(self.layer_n, "rail layer_n")
        _positive_int(self.required_count, "rail required_count")
        _positive_int(self.min_width_nm, "rail min_width_nm")
        _positive_int(self.min_length_nm, "rail min_length_nm")

    @property
    def area_nm2(self) -> int:
        self.__post_init__()
        return self.required_count * self.min_width_nm * self.min_length_nm

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "layer_n": self.layer_n,
            "min_length_nm": self.min_length_nm,
            "min_width_nm": self.min_width_nm,
            "net": self.net,
            "required_count": self.required_count,
        }


@dataclass(frozen=True, order=True)
class ViaRequirement:
    via_name: str
    lower_layer_n: int
    required_count: int
    cut_size_nm: int

    def __post_init__(self) -> None:
        _text(self.via_name, "via name")
        _positive_int(self.lower_layer_n, "via lower_layer_n")
        _positive_int(self.required_count, "via required_count")
        _positive_int(self.cut_size_nm, "via cut_size_nm")

    @property
    def area_nm2(self) -> int:
        self.__post_init__()
        return self.required_count * self.cut_size_nm * self.cut_size_nm

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "cut_size_nm": self.cut_size_nm,
            "lower_layer_n": self.lower_layer_n,
            "required_count": self.required_count,
            "via_name": self.via_name,
        }


@dataclass(frozen=True, order=True)
class ChannelRequirement:
    channel_id: str
    layer_n: int
    required_tracks: int
    min_width_nm: int
    min_space_nm: int
    min_run_nm: int

    def __post_init__(self) -> None:
        _text(self.channel_id, "channel id")
        _positive_int(self.layer_n, "channel layer_n")
        _positive_int(self.required_tracks, "channel required_tracks")
        _positive_int(self.min_width_nm, "channel min_width_nm")
        _nonnegative_int(self.min_space_nm, "channel min_space_nm")
        _positive_int(self.min_run_nm, "channel min_run_nm")

    @property
    def span_nm(self) -> int:
        self.__post_init__()
        return (
            self.required_tracks * self.min_width_nm
            + (self.required_tracks - 1) * self.min_space_nm
        )

    @property
    def area_nm2(self) -> int:
        return self.span_nm * self.min_run_nm

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "channel_id": self.channel_id,
            "layer_n": self.layer_n,
            "min_run_nm": self.min_run_nm,
            "min_space_nm": self.min_space_nm,
            "min_width_nm": self.min_width_nm,
            "required_tracks": self.required_tracks,
        }


@dataclass(frozen=True)
class PdkLegalLowerBoundInput:
    circuit_id: str
    topology_sha256: str
    source_sha256: str
    pdk_id: str
    pdk_source_snapshot_sha256: str
    pdk_installed_authority_sha256: str
    deck_scope: Literal["cited_partial", "foundry"]
    footprints: tuple[LegalFootprint, ...]
    layer_resources: tuple[LegalLayerResource, ...]
    groups: tuple[LegalPlacementGroup, ...]
    matching_components: tuple[tuple[str, ...], ...]
    rails: tuple[RailRequirement, ...]
    vias: tuple[ViaRequirement, ...]
    channels: tuple[ChannelRequirement, ...]
    schema: str = "clay.pdk-legal-lower-bound-input/v5"

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != "clay.pdk-legal-lower-bound-input/v5":
            raise ValueError("lower-bound input schema is unsupported")
        _text(self.circuit_id, "lower-bound circuit_id")
        _text(self.pdk_id, "lower-bound pdk_id")
        _sha256(self.topology_sha256, "lower-bound topology_sha256")
        _sha256(self.source_sha256, "lower-bound source_sha256")
        _sha256(self.pdk_source_snapshot_sha256, "lower-bound PDK snapshot")
        _sha256(
            self.pdk_installed_authority_sha256,
            "lower-bound installed PDK authority",
        )
        if type(self.deck_scope) is not str or self.deck_scope not in {
            "cited_partial",
            "foundry",
        }:
            raise ValueError("lower-bound deck_scope is unsupported")

        typed_fields = (
            ("footprints", self.footprints, LegalFootprint),
            ("layer_resources", self.layer_resources, LegalLayerResource),
            ("groups", self.groups, LegalPlacementGroup),
            ("rails", self.rails, RailRequirement),
            ("vias", self.vias, ViaRequirement),
            ("channels", self.channels, ChannelRequirement),
        )
        for label, values, expected in typed_fields:
            exact = _exact_tuple(values, f"lower-bound {label}")
            if any(type(item) is not expected for item in exact):
                raise ValueError(f"lower-bound {label} contain an invalid item")
            for item in exact:
                item.__post_init__()
            object.__setattr__(self, label, tuple(sorted(exact)))
        identity_keys = {
            "footprints": tuple(item.member_id for item in self.footprints),
            "layer_resources": tuple(
                item.resource_id for item in self.layer_resources
            ),
            "groups": tuple(item.group_id for item in self.groups),
            "rails": tuple((item.net, item.layer_n) for item in self.rails),
            "vias": tuple(
                (item.via_name, item.lower_layer_n) for item in self.vias
            ),
            "channels": tuple(item.channel_id for item in self.channels),
        }
        for label, keys in identity_keys.items():
            if len(keys) != len(set(keys)):
                raise ValueError(f"lower-bound {label} contain a duplicate identity")

        raw_components = _exact_tuple(
            self.matching_components, "lower-bound matching_components"
        )
        components: list[tuple[str, ...]] = []
        for component in raw_components:
            members = _exact_tuple(component, "matching component", nonempty=True)
            if any(type(member) is not str for member in members):
                raise ValueError("matching component members must be exact strings")
            canonical = tuple(sorted(_text(member, "matching member") for member in members))
            if len(set(canonical)) != len(canonical):
                raise ValueError("matching component contains a duplicate")
            components.append(canonical)
        canonical_components = tuple(sorted(components))
        if len(set(canonical_components)) != len(canonical_components):
            raise ValueError("matching components contain a duplicate")
        object.__setattr__(self, "matching_components", canonical_components)

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "channels": [item.canonical_data() for item in self.channels],
            "circuit_id": self.circuit_id,
            "deck_scope": self.deck_scope,
            "footprints": [item.canonical_data() for item in self.footprints],
            "groups": [item.canonical_data() for item in self.groups],
            "layer_resources": [
                item.canonical_data() for item in self.layer_resources
            ],
            "matching_components": [list(item) for item in self.matching_components],
            "pdk_id": self.pdk_id,
            "pdk_installed_authority_sha256": (
                self.pdk_installed_authority_sha256
            ),
            "pdk_source_snapshot_sha256": self.pdk_source_snapshot_sha256,
            "rails": [item.canonical_data() for item in self.rails],
            "schema": self.schema,
            "source_sha256": self.source_sha256,
            "topology_sha256": self.topology_sha256,
            "vias": [item.canonical_data() for item in self.vias],
        }

    @property
    def sha256(self) -> str:
        raw = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return hashlib.sha256(raw.encode("ascii")).hexdigest()


@dataclass(frozen=True, order=True)
class LowerBoundComponent:
    resource_class: str
    area_floor_nm2: int
    evidence_count: int

    def __post_init__(self) -> None:
        if type(self.resource_class) is not str or self.resource_class not in _RESOURCE_CLASSES:
            raise ValueError("lower-bound component resource class is unsupported")
        _nonnegative_int(self.area_floor_nm2, "component area_floor_nm2")
        _positive_int(self.evidence_count, "component evidence_count")

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "area_floor_nm2": self.area_floor_nm2,
            "evidence_count": self.evidence_count,
            "resource_class": self.resource_class,
        }


@dataclass(frozen=True)
class PdkLegalLowerBoundReceipt:
    input_sha256: str
    status: Literal["supported", "unsupported"]
    lower_bound_nm2: int | None
    components: tuple[LowerBoundComponent, ...]
    blockers: tuple[str, ...]
    binding_truth: bool = False
    signoff_authority: bool = False
    schema: str = "clay.pdk-legal-lower-bound-receipt/v5"

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != "clay.pdk-legal-lower-bound-receipt/v5":
            raise ValueError("lower-bound receipt schema is unsupported")
        _sha256(self.input_sha256, "lower-bound receipt input_sha256")
        if type(self.status) is not str or self.status not in {"supported", "unsupported"}:
            raise ValueError("lower-bound receipt status is unsupported")
        if self.binding_truth is not False or self.signoff_authority is not False:
            raise ValueError(
                "unbound lower-bound computation cannot carry promotion authority"
            )
        components = _exact_tuple(self.components, "lower-bound receipt components")
        if any(type(item) is not LowerBoundComponent for item in components):
            raise ValueError("lower-bound receipt components are invalid")
        for item in components:
            item.__post_init__()
        canonical_components = tuple(sorted(components))
        if len({item.resource_class for item in canonical_components}) != len(canonical_components):
            raise ValueError("lower-bound receipt duplicates a resource class")
        blockers = _exact_tuple(self.blockers, "lower-bound receipt blockers")
        if any(type(item) is not str for item in blockers):
            raise ValueError("lower-bound receipt blockers must be exact strings")
        canonical_blockers = tuple(sorted(_text(item, "lower-bound blocker") for item in blockers))
        if len(set(canonical_blockers)) != len(canonical_blockers):
            raise ValueError("lower-bound receipt blockers contain a duplicate")
        if self.status == "supported":
            _positive_int(self.lower_bound_nm2, "lower-bound receipt value")
            if canonical_blockers or {item.resource_class for item in canonical_components} != _RESOURCE_CLASSES:
                raise ValueError("supported lower-bound receipt is incomplete")
        elif self.lower_bound_nm2 is not None or not canonical_blockers:
            raise ValueError("unsupported lower-bound receipt requires blockers and no value")
        object.__setattr__(self, "components", canonical_components)
        object.__setattr__(self, "blockers", canonical_blockers)

    @property
    def lower_bound_um2(self) -> float | None:
        self.__post_init__()
        return None if self.lower_bound_nm2 is None else self.lower_bound_nm2 / 1_000_000.0

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "authority": {
                "binding_truth": self.binding_truth,
                "signoff_authority": self.signoff_authority,
            },
            "blockers": list(self.blockers),
            "components": [item.canonical_data() for item in self.components],
            "input_sha256": self.input_sha256,
            "lower_bound_nm2": self.lower_bound_nm2,
            "schema": self.schema,
            "status": self.status,
        }

    @property
    def sha256(self) -> str:
        raw = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _component(resource_class: str, area_nm2: int, count: int) -> LowerBoundComponent:
    return LowerBoundComponent(resource_class, area_nm2, count)


def _independent_layer_projection_floor(
    resources: tuple[LegalLayerResource, ...],
) -> int:
    """Sum proved non-shareable atoms per layer, then maximize across layers."""

    area_by_layer: dict[tuple[int, int], int] = {}
    for resource in resources:
        area_by_layer[resource.layer_key] = (
            area_by_layer.get(resource.layer_key, 0) + resource.area_nm2
        )
    return max(area_by_layer.values(), default=0)


def compute_pdk_legal_lower_bound(
    inputs: PdkLegalLowerBoundInput,
) -> PdkLegalLowerBoundReceipt:
    """Compute one deterministic lower-bound receipt from canonical inputs."""

    if type(inputs) is not PdkLegalLowerBoundInput:
        raise TypeError("lower-bound compiler requires exact typed inputs")
    inputs.__post_init__()
    footprints = {item.member_id: item for item in inputs.footprints}
    if len(footprints) != len(inputs.footprints):
        raise PdkLegalLowerBoundError("MEMBER_CONSERVATION", "duplicate footprint member")
    groups = {item.group_id: item for item in inputs.groups}
    if len(groups) != len(inputs.groups):
        raise PdkLegalLowerBoundError("MEMBER_CONSERVATION", "duplicate placement group")
    flattened = tuple(member for group in inputs.groups for member in group.members)
    expected_members = set(footprints)
    if set(flattened) != expected_members or len(flattened) != len(expected_members):
        raise PdkLegalLowerBoundError(
            "MEMBER_CONSERVATION",
            "placement groups must own every footprint exactly once",
        )
    member_group = {
        member: group.group_id for group in inputs.groups for member in group.members
    }
    resources = {item.resource_id: item for item in inputs.layer_resources}
    if len(resources) != len(inputs.layer_resources):
        raise PdkLegalLowerBoundError(
            "LAYER_RESOURCE_CONSERVATION", "duplicate layer resource"
        )
    resource_by_member: dict[str, LegalLayerResource] = {}
    for resource in inputs.layer_resources:
        unknown = tuple(member for member in resource.members if member not in footprints)
        if unknown:
            raise PdkLegalLowerBoundError(
                "LAYER_RESOURCE_CONSERVATION",
                f"layer resource {resource.resource_id!r} owns unknown members {unknown!r}",
            )
        owned_groups = {member_group[member] for member in resource.members}
        if len(owned_groups) != 1:
            raise PdkLegalLowerBoundError(
                "LAYER_RESOURCE_GROUP_SPLIT",
                f"layer resource {resource.resource_id!r} crosses placement groups",
            )
        expected_kind = {
            "device_active": "device",
            "passive": "passive",
            "tap_active": "tap",
        }[resource.kind]
        if any(footprints[member].kind != expected_kind for member in resource.members):
            raise PdkLegalLowerBoundError(
                "LAYER_RESOURCE_KIND_MISMATCH",
                f"layer resource {resource.resource_id!r} disagrees with its footprints",
            )
        duplicate = tuple(
            member for member in resource.members if member in resource_by_member
        )
        if duplicate:
            raise PdkLegalLowerBoundError(
                "LAYER_RESOURCE_CONSERVATION",
                f"members {duplicate!r} occur in more than one layer resource",
            )
        for member in resource.members:
            resource_by_member[member] = resource

    integrated_tap_ids = {
        member
        for group in inputs.groups
        if group.kind == "well_island" and group.tap_integrated
        for member in group.members
        if footprints[member].kind == "tap"
    }
    required_resource_members = set(footprints) - integrated_tap_ids
    missing_resource_members = tuple(
        sorted(required_resource_members - set(resource_by_member))
    )
    if missing_resource_members:
        raise PdkLegalLowerBoundError(
            "LAYER_RESOURCE_CONSERVATION",
            f"mandatory members lack exact layer resources: {missing_resource_members!r}",
        )
    for component in inputs.matching_components:
        if not any(set(component) <= set(group.members) for group in inputs.groups):
            raise PdkLegalLowerBoundError(
                "MATCHING_COMPONENT_SPLIT",
                f"matching component {component!r} crosses placement groups",
            )

    well_outer_floors: list[int] = []
    well_spacing_terms: list[int] = []
    ring_outer_floors: list[int] = []
    standalone_tap_ids: set[str] = set()
    for group in inputs.groups:
        owned = tuple(footprints[member] for member in group.members)
        owned_resources = tuple(
            sorted(
                {
                    resource_by_member[member].resource_id: resource_by_member[member]
                    for member in group.members
                    if member in resource_by_member
                    and not (
                        group.tap_integrated and footprints[member].kind == "tap"
                    )
                }.values(),
                key=lambda item: item.resource_id,
            )
        )
        if group.kind == "passive_bank":
            if any(item.kind != "passive" for item in owned):
                raise PdkLegalLowerBoundError(
                    "PASSIVE_BANK_KIND_MISMATCH",
                    f"passive bank {group.group_id!r} owns a non-passive member",
                )
            continue
        if any(item.kind not in {"device", "passive", "tap"} for item in owned):
            raise PdkLegalLowerBoundError(
                "WELL_MEMBER_KIND_MISMATCH",
                f"well island {group.group_id!r} owns an unsupported member",
            )
        taps = tuple(item for item in owned if item.kind == "tap")
        if len(taps) > 1 or (group.tap_integrated and len(taps) != 1):
            raise PdkLegalLowerBoundError(
                "TAP_DOMAIN_MISMATCH",
                f"well island {group.group_id!r} has incompatible tap evidence",
            )
        if taps and not group.tap_integrated:
            standalone_tap_ids.add(taps[0].member_id)
        if any(
            item.kind != "passive" and item.body_net != group.tie_net
            for item in owned
        ):
            raise PdkLegalLowerBoundError(
                "BODY_TIE_MISMATCH",
                f"well island {group.group_id!r} disagrees with body/tap tie",
            )
        base_area = _independent_layer_projection_floor(owned_resources)
        well_floor = _enclosed_area_floor(base_area, group.well_enclosure_nm)
        total_margin = (
            group.well_enclosure_nm
            + group.guard_ring_setback_nm
            + group.guard_ring_width_nm
        )
        outer_floor = _enclosed_area_floor(base_area, total_margin)
        well_outer_floors.append(well_floor)
        minimum_cross_section = min(
            min(item.width_nm, item.height_nm) for item in owned_resources
        )
        well_spacing_terms.append(
            group.well_spacing_nm * minimum_cross_section
        )
        if group.guard_ring_width_nm:
            ring_outer_floors.append(outer_floor)

    blockers: list[str] = []
    if inputs.deck_scope != "foundry":
        blockers.append("foundry PDK rule-byte authority is unavailable")
    kinds = {item.kind for item in inputs.footprints}
    if "device" not in kinds:
        blockers.append("device envelope evidence is missing")
    if "passive" not in kinds:
        blockers.append("passive bank evidence is missing")
    if not well_outer_floors:
        blockers.append("well island evidence is missing")
    if not inputs.rails:
        blockers.append("rail evidence is missing")
    if not inputs.vias:
        blockers.append("via evidence is missing")

    components: list[LowerBoundComponent] = []
    if devices := tuple(
        item for item in inputs.layer_resources if item.kind == "device_active"
    ):
        components.append(
            _component(
                "device_functional_area",
                _independent_layer_projection_floor(devices),
                len(devices),
            )
        )
    if passives := tuple(
        item for item in inputs.layer_resources if item.kind == "passive"
    ):
        components.append(
            _component(
                "passive_banks",
                _independent_layer_projection_floor(passives),
                len(passives),
            )
        )
    taps = tuple(item for item in inputs.footprints if item.kind == "tap")
    components.append(
        _component(
            "taps",
            sum(
                resource_by_member[item.member_id].area_nm2
                for item in taps
                if item.member_id in standalone_tap_ids
            ),
            max(1, len(taps)),
        )
    )
    if well_outer_floors:
        components.append(_component("well_islands", sum(well_outer_floors), len(well_outer_floors)))
    components.append(
        _component(
            "well_domain_spacing",
            min(well_spacing_terms) if len(well_outer_floors) > 1 else 0,
            max(1, len(well_outer_floors) - 1),
        )
    )
    components.append(
        _component(
            "guard_rings",
            sum(ring_outer_floors),
            max(1, len(ring_outer_floors)),
        )
    )

    rail_by_layer: dict[int, int] = {}
    for item in inputs.rails:
        rail_by_layer[item.layer_n] = rail_by_layer.get(item.layer_n, 0) + item.area_nm2
    if rail_by_layer:
        components.append(_component("rails", max(rail_by_layer.values()), len(inputs.rails)))
    via_by_layer: dict[int, int] = {}
    for item in inputs.vias:
        via_by_layer[item.lower_layer_n] = via_by_layer.get(item.lower_layer_n, 0) + item.area_nm2
    if via_by_layer:
        components.append(_component("vias", max(via_by_layer.values()), len(inputs.vias)))
    components.append(
        _component(
            "channel_capacity",
            max((item.area_nm2 for item in inputs.channels), default=0),
            max(1, len(inputs.channels)),
        )
    )

    if blockers:
        return PdkLegalLowerBoundReceipt(
            input_sha256=inputs.sha256,
            status="unsupported",
            lower_bound_nm2=None,
            components=tuple(components),
            blockers=tuple(blockers),
        )
    lower_bound = max(item.area_floor_nm2 for item in components)
    return PdkLegalLowerBoundReceipt(
        input_sha256=inputs.sha256,
        status="supported",
        lower_bound_nm2=lower_bound,
        components=tuple(components),
        blockers=(),
    )


def verify_pdk_legal_lower_bound(
    inputs: PdkLegalLowerBoundInput,
    receipt: PdkLegalLowerBoundReceipt,
) -> bool:
    """Replay ``inputs`` and require the claimed receipt byte-for-byte.

    A receipt is intentionally not a self-signing authority.  Publication and
    search promotion must retain the canonical inputs and call this verifier;
    replacing a value, component, blocker, status, or input binding is rejected.
    """

    if type(receipt) is not PdkLegalLowerBoundReceipt:
        raise TypeError("lower-bound verifier requires an exact typed receipt")
    receipt.__post_init__()
    expected = compute_pdk_legal_lower_bound(inputs)
    if receipt.canonical_data() != expected.canonical_data():
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_RECEIPT_MISMATCH",
            "receipt differs from canonical lower-bound replay",
        )
    return True


def verify_pdk_legal_lower_bound_topology(
    source_cell: object,
    graph: object,
    inputs: PdkLegalLowerBoundInput,
    receipt: PdkLegalLowerBoundReceipt,
) -> bool:
    """Replay source-owned graph members, macro groups, and matching components.

    This closes the structure denominator independently from the numerical PDK
    rule replay.  The graph is rebuilt from the canonical source bytes rather
    than trusted as a caller-owned denominator.  Synthesized tap footprints may
    appear only inside one graph-declared well island; every source
    device/passive must match the graph inventory exactly.
    """

    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.physical_macro_graph import (
        PhysicalMacroGraph,
        physical_macro_graph_from_intent,
    )
    from clay.layout.route.exact_artifact_evidence import (
        _replayed_source_cell_authority,
    )

    if type(graph) is not PhysicalMacroGraph:
        raise TypeError("lower-bound topology replay requires PhysicalMacroGraph")
    graph.__post_init__()
    verify_pdk_legal_lower_bound(inputs, receipt)
    try:
        source_sha256, _inventory_sha256, _classes, canonical_cell = (
            _replayed_source_cell_authority(source_cell)
        )
        replayed_graph = physical_macro_graph_from_intent(
            canonical_cell,
            layout_intent_from_cell_def(canonical_cell),
            device_records={},
        )
    except (OSError, ValueError) as exc:
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_TOPOLOGY_MISMATCH",
            f"canonical source replay failed: {exc}",
        ) from exc
    if source_sha256 != inputs.source_sha256:
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_TOPOLOGY_MISMATCH",
            "input source digest differs from canonical source bytes",
        )
    if graph.canonical_data() != replayed_graph.canonical_data():
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_TOPOLOGY_MISMATCH",
            "supplied macro graph differs from canonical source lowering",
        )
    if inputs.topology_sha256 != graph.sha256:
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_TOPOLOGY_MISMATCH",
            "input topology digest differs from the canonical macro graph",
        )
    source_members = tuple(
        sorted(item.member_id for item in inputs.footprints if item.kind != "tap")
    )
    if source_members != graph.devices:
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_TOPOLOGY_MISMATCH",
            "footprint inventory differs from canonical graph devices",
        )
    expected_matching = tuple(sorted(set(
        macro.members for macro in graph.macros if macro.kind == "matched_array"
    )))
    if inputs.matching_components != expected_matching:
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_TOPOLOGY_MISMATCH",
            "matching components differ from canonical graph macros",
        )

    structural_macros = tuple(
        macro
        for macro in graph.macros
        if macro.kind in {"passive_bank", "well_island"}
    )
    expected_groups = {
        macro.macro_id: (macro.kind, macro.members, dict(macro.attributes))
        for macro in structural_macros
    }
    if set(expected_groups) != {group.group_id for group in inputs.groups}:
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_TOPOLOGY_MISMATCH",
            "placement groups differ from canonical graph macro inventory",
        )
    footprint_kind = {item.member_id: item.kind for item in inputs.footprints}
    for group in inputs.groups:
        expected_kind, expected_members, attributes = expected_groups[group.group_id]
        source_group_members = tuple(
            member for member in group.members if footprint_kind[member] != "tap"
        )
        if group.kind != expected_kind or source_group_members != expected_members:
            raise PdkLegalLowerBoundError(
                "LOWER_BOUND_TOPOLOGY_MISMATCH",
                f"group {group.group_id!r} differs from canonical macro ownership",
            )
        if group.kind == "well_island" and group.tie_net != attributes.get("body", ""):
            raise PdkLegalLowerBoundError(
                "LOWER_BOUND_TOPOLOGY_MISMATCH",
                f"well group {group.group_id!r} differs from canonical body tie",
            )
    return True


def _materialized_footprint(
    name: str,
    kind: str,
    geometry: Any,
    *,
    width_nm: int,
    height_nm: int,
    body_net: str = "",
) -> LegalFootprint:
    """Bind a safe functional-area leaf to canonical materializer artwork."""

    if hasattr(geometry, "polys_by_layer"):
        from clay.layout.route.exact_artifact_evidence import (
            _emitted_device_snapshot_sha256,
        )

        digest = _emitted_device_snapshot_sha256(
            {name: geometry}, {name: (0.0, 0.0)}
        )
    else:
        from clay.layout.physical_macro_contracts import _artwork_sha256

        x0, y0, x1, y1 = geometry.extent()
        digest = _artwork_sha256(
            geometry.polygons, origin_nm=(int(x0), int(y0))
        )
    return LegalFootprint(
        name, kind, width_nm, height_nm, digest, body_net=body_net
    )


def _materialized_layer_resource(
    resource_id: str,
    kind: Literal["device_active", "passive", "tap_active"],
    members: tuple[str, ...],
    layer_key: tuple[int, int],
    polygons: object,
    proof_kind: str,
    nonshareable_domain_sha256: str,
    nonshareable_proof_kind: str,
) -> LegalLayerResource:
    """Measure one exact, origin-free union on a proved functional PDK layer."""

    try:
        import klayout.db as kdb
    except ImportError as exc:  # pragma: no cover - environment contract
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_LAYER_RESOURCE_UNMEASURABLE",
            "KLayout is unavailable for exact layer-resource union",
        ) from exc
    selected = tuple(
        polygon
        for polygon in tuple(polygons)
        if (int(polygon.layer), int(polygon.datatype)) == layer_key
    )
    if not selected:
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_LAYER_RESOURCE_UNMEASURABLE",
            f"resource {resource_id!r} has no artwork on proved layer {layer_key!r}",
        )
    region = kdb.Region()
    for item in selected:
        points = [
            kdb.Point(int(vertex.x), int(vertex.y))
            for vertex in item.polygon.vertices
        ]
        if len(points) < 3:
            raise PdkLegalLowerBoundError(
                "LOWER_BOUND_LAYER_RESOURCE_UNMEASURABLE",
                f"resource {resource_id!r} contains a degenerate polygon",
            )
        region.insert(kdb.Polygon(points))
    merged = region.merged()
    area_nm2 = int(merged.area())
    bounds = merged.bbox()
    width_nm = int(bounds.right) - int(bounds.left)
    height_nm = int(bounds.top) - int(bounds.bottom)
    if area_nm2 <= 0 or width_nm <= 0 or height_nm <= 0:
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_LAYER_RESOURCE_UNMEASURABLE",
            f"resource {resource_id!r} has no positive exact projection",
        )
    from clay.layout.physical_macro_contracts import _artwork_sha256

    geometry_sha256 = _artwork_sha256(
        selected, origin_nm=(int(bounds.left), int(bounds.bottom))
    )
    return LegalLayerResource(
        resource_id,
        kind,
        members,
        layer_key,
        area_nm2,
        width_nm,
        height_nm,
        geometry_sha256,
        proof_kind,
        nonshareable_domain_sha256,
        nonshareable_proof_kind,
    )


def _canonical_rule_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _minimum_legal_metal_segment(
    tech: Any,
    rules: Any,
    layer_n: int,
    *,
    declared_width_nm: int = 0,
) -> tuple[int, int]:
    """Return one rule-derived width/run rectangle for a mandatory route."""

    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

    try:
        tech.metal_key(layer_n)
    except ValueError as exc:
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_PDK_RULE_MISSING",
            f"PDK does not expose metal{layer_n}",
        ) from exc
    minimum_width = tech.min_width(f"metal{layer_n}")
    if minimum_width is None and layer_n == 1:
        minimum_width = rules.metal1_min_width
    if type(minimum_width) is not int or minimum_width <= 0:
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_PDK_RULE_MISSING",
            f"PDK does not expose a positive metal{layer_n} minimum width",
        )
    if type(declared_width_nm) is not int or declared_width_nm < 0:
        raise ValueError("declared metal width must be an exact non-negative integer")
    area = int(tech.min_area(f"metal{layer_n}") or 0)
    grid = max(1, int(pdk_manufacturing_grid_nm(tech)))
    width = max(minimum_width, declared_width_nm)
    width = ((width + grid - 1) // grid) * grid
    run = max(grid, minimum_width, (area + width - 1) // width)
    run = ((run + grid - 1) // grid) * grid
    return width, run


def _declared_metal_layer_n(layer: object) -> int:
    if type(layer) is not str:
        raise TypeError("trunk layer must be an exact string")
    folded = layer.casefold()
    suffix = folded.removeprefix("metal").removeprefix("m")
    if not suffix.isdigit() or int(suffix) <= 0:
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_PDK_RULE_MISSING",
            f"trunk declares a non-canonical metal layer {layer!r}",
        )
    return int(suffix)


def _mandatory_via_cut_size(tech: Any, via_name: str) -> int:
    """Return one exact PDK cut size or fail; an enclosure is never a cut."""

    try:
        value = tech.via_size(via_name)
    except ValueError as exc:
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_PDK_RULE_MISSING",
            f"PDK does not expose exact {via_name!r} cut size",
        ) from exc
    if type(value) is not int or value <= 0:
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_PDK_RULE_MISSING",
            f"PDK does not expose exact {via_name!r} cut size",
        )
    return value


def compile_pdk_legal_lower_bound_source(
    source_path: str | Path,
    *,
    circuit_id: str | None = None,
    pdk_id: str | None = None,
) -> tuple[PdkLegalLowerBoundInput, PdkLegalLowerBoundReceipt]:
    """Compile one numeric floor from exact source, PCells, and PDK rule bytes."""

    from types import SimpleNamespace

    from clay.layout.device_cell import _resolve_layer_keys, _rules_from_tech
    from clay.layout.extract.realize_render import (
        _group_ring_base_clearance_nm,
        _group_ring_nwell_clearance_nm,
        _norm_kind,
        _ring_extension_nm,
        build_device_geometry,
    )
    from clay.layout.generic_placement import (
        _build_passive_geom,
        _collect_bjts,
        _collect_passives,
        compile_rigid_nonmos_macro_plans,
    )
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.physical_macro_graph import physical_macro_graph_from_intent
    from clay.layout.route.pins import _mim_spec_for_passive
    from clay.layout.route.route_cell import (
        bridge_devices_from_cell_def,
        resolve_common_centroids,
        resolve_shared_diffusion_arrays,
    )
    from clay.layout.tech import LayoutTech
    from clay.parser import parse_clay_file
    from clay.pdk import get_pdk, load_all

    path = Path(source_path).resolve()
    source_bytes = path.read_bytes()
    parsed = parse_clay_file(str(path))
    if path.read_bytes() != source_bytes:
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_SOURCE_REPLAY_MISMATCH", "source changed while replaying"
        )
    if len(parsed.cell_defs) != 1:
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_SOURCE_REPLAY_MISMATCH",
            "lower-bound source must select exactly one cell",
        )
    cell = parsed.cell_defs[0]
    selected_pdk = str(pdk_id or parsed.pdk_id)
    intent = layout_intent_from_cell_def(cell)
    graph = physical_macro_graph_from_intent(cell, intent, device_records={})

    load_all()
    adapter = get_pdk(selected_pdk)
    snapshot = adapter.source_snapshot()
    if (
        snapshot is None
        or snapshot.installed_authority_sha256 is None
        or not snapshot.rules()
    ):
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_PDK_AUTHORITY_MISSING",
            f"{selected_pdk} has no exact installed rule-byte snapshot",
        )
    tech = LayoutTech.resolve(selected_pdk)
    rules = _rules_from_tech(tech)
    contact_cut_size = _mandatory_via_cut_size(tech, "contact")
    implant_well = tech.implant_well()
    footprints: list[LegalFootprint] = []
    layer_resources: list[LegalLayerResource] = []
    device_domains: dict[str, tuple[str, str]] = {}

    mos, _passive_views = bridge_devices_from_cell_def(cell)
    mos_by_name = {device.name: device for device in mos}
    mos_resource_by_member: dict[str, LegalLayerResource] = {}
    for device in mos:
        body = str(device.nets.get("body", "") or tech.substrate_net())
        geometry = build_device_geometry(
            selected_pdk,
            device.kind,
            device.W,
            device.L,
            device.M,
            nets=dict(device.nets),
            oxide=device.oxide,
        )
        footprint = _materialized_footprint(
                device.name,
                "device",
                geometry,
                width_nm=max(
                    int(rules.channel_min_width),
                    math.floor(float(device.W) * 1000.0),
                )
                * int(device.M),
                height_nm=max(
                    int(rules.gate_min_length),
                    math.floor(float(device.L) * 1000.0),
                ),
                body_net=body,
            )
        footprints.append(footprint)
        kind = _norm_kind(device.kind)
        active_key = tuple(_resolve_layer_keys(tech, kind).active)
        channel_domain_sha256 = _canonical_rule_digest(
            {
                "geometry_sha256": footprint.geometry_sha256,
                "member": device.name,
                "pdk_source_snapshot_sha256": snapshot.snapshot_sha256,
                "schema": "clay.nonshareable-channel-atom/v1",
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "topology_sha256": graph.sha256,
            }
        )
        mos_resource_by_member[device.name] = LegalLayerResource(
            f"channel:{device.name}",
            "device_active",
            (device.name,),
            active_key,
            footprint.area_nm2,
            footprint.width_nm,
            footprint.height_nm,
            footprint.geometry_sha256,
            "canonical_device_channel",
            channel_domain_sha256,
            "canonical_channel_atom",
        )
        well = "n" if kind == "pmos" else "p"
        device_domains[device.name] = (well, body)

    passive_records = _collect_passives(cell, None)
    bjt_records = _collect_bjts(cell, tech, intent)
    nonmos_records = {**passive_records, **bjt_records}
    try:
        rigid_plans, rigid_inputs, rigid_interfaces = (
            compile_rigid_nonmos_macro_plans(
                graph, nonmos_records, tech, intent
            )
        )
    except ValueError as exc:
        if str(exc).startswith("@resistor_segments PDK "):
            raise PdkLegalLowerBoundError(
                "LOWER_BOUND_INTENT_PDK_MISMATCH", str(exc)
            ) from exc
        raise
    rigid_domain_by_member: dict[str, str] = {}
    for plan in rigid_plans.plans:
        domain_sha256 = _canonical_rule_digest(
            {
                "compile_inputs_sha256": rigid_inputs.sha256,
                "interfaces_sha256": rigid_interfaces.sha256,
                "macro_plan_sha256": plan.sha256,
                "pdk_source_snapshot_sha256": snapshot.snapshot_sha256,
                "schema": "clay.nonshareable-rigid-macro/v1",
            }
        )
        for member in plan.members:
            rigid_domain_by_member[member] = domain_sha256
    extra_rigid_members = tuple(
        sorted(set(rigid_domain_by_member) - set(nonmos_records))
    )
    if extra_rigid_members:
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_NONSHAREABLE_PROOF_MISSING",
            f"rigid non-MOS plan owns unknown members {extra_rigid_members!r}",
        )

    def nonmos_domain(name: str, geometry: Any) -> tuple[str, str]:
        macro_domain = rigid_domain_by_member.get(name)
        if macro_domain is not None:
            return macro_domain, "canonical_rigid_macro"
        from clay.layout.physical_macro_contracts import _artwork_sha256

        x0, y0, _x1, _y1 = geometry.extent()
        artwork_sha256 = _artwork_sha256(
            geometry.polygons, origin_nm=(int(x0), int(y0))
        )
        return (
            _canonical_rule_digest(
                {
                    "artwork_sha256": artwork_sha256,
                    "member": name,
                    "pdk_source_snapshot_sha256": snapshot.snapshot_sha256,
                    "schema": "clay.nonshareable-rigid-atom/v1",
                    "topology_sha256": graph.sha256,
                }
            ),
            "canonical_rigid_atom",
        )

    for name, record in sorted(passive_records.items()):
        geometry = _build_passive_geom(
            record, tech, str(record.get("orient", "R0"))
        )
        passive_domain_sha256, passive_domain_kind = nonmos_domain(name, geometry)
        footprints.append(
            _materialized_footprint(
                name, "passive", geometry, width_nm=1, height_nm=1
            )
        )
        drawn_layers = {
            (int(item.layer), int(item.datatype)) for item in geometry.polygons
        }
        passive_kind = str(record.get("kind", ""))
        candidates: set[tuple[int, int]] = set()
        if passive_kind == "res":
            marker = tech.marker_key("res_mk")
            if marker is not None:
                candidates.add(tuple(marker))
        elif passive_kind == "cap":
            marker = tech.marker_key("fusetop")
            if marker is not None:
                candidates.add(tuple(marker))
            spec = _mim_spec_for_passive(
                tech, SimpleNamespace(model=str(record.get("model", "") or ""))
            )
            if spec is not None:
                routing_layers = set(tuple(item) for item in spec.metal_order)
                candidates.update(
                    tuple(item) for item in spec.metals if tuple(item) not in routing_layers
                )
        proved_layers = tuple(sorted(candidates & drawn_layers))
        if not proved_layers:
            raise PdkLegalLowerBoundError(
                "LOWER_BOUND_PASSIVE_RECOGNITION_MISSING",
                f"passive {name!r} has no drawn PDK recognition-layer proof",
            )
        passive_candidates = tuple(
            _materialized_layer_resource(
                f"passive:{name}",
                "passive",
                (name,),
                layer_key,
                geometry.polygons,
                "canonical_passive_recognition",
                passive_domain_sha256,
                passive_domain_kind,
            )
            for layer_key in proved_layers
        )
        layer_resources.append(
            max(
                passive_candidates,
                key=lambda item: (item.area_nm2, item.layer_key),
            )
        )

    for name, record in sorted(bjt_records.items()):
        geometry = _build_passive_geom(record, tech, "R0")
        bjt_domain_sha256, bjt_domain_kind = nonmos_domain(name, geometry)
        body = str(record["pins"].get("c", "") or tech.substrate_net())
        footprints.append(
            _materialized_footprint(
                name,
                "device",
                geometry,
                width_nm=1,
                height_nm=1,
                body_net=body,
            )
        )
        bjt_layers = tuple(
            sorted({(int(item.layer), int(item.datatype)) for item in geometry.polygons})
        )
        bjt_candidates = tuple(
            _materialized_layer_resource(
                f"fixed-cell:{name}",
                "device_active",
                (name,),
                layer_key,
                geometry.polygons,
                "canonical_fixed_cell_layer",
                bjt_domain_sha256,
                bjt_domain_kind,
            )
            for layer_key in bjt_layers
        )
        if not bjt_candidates:
            raise PdkLegalLowerBoundError(
                "LOWER_BOUND_LAYER_RESOURCE_UNMEASURABLE",
                f"fixed-cell device {name!r} contains no measurable layer",
            )
        layer_resources.append(
            max(bjt_candidates, key=lambda item: (item.area_nm2, item.layer_key))
        )
        device_domains[name] = ("p", body)

    array_members: set[str] = set()
    array_specs = (
        (
            resolve_shared_diffusion_arrays(cell, intent, tech)[0],
            "canonical_shared_diffusion_active",
        ),
        (
            resolve_common_centroids(cell, intent, tech)[0],
            "canonical_common_centroid_active",
        ),
    )
    for consumptions, proof_kind in array_specs:
        for consumption in consumptions:
            if consumption.array is None:
                continue
            members = tuple(
                sorted(
                    set(str(item) for item in consumption.members)
                    | set(str(item) for item in consumption.flank_dummies)
                )
            )
            if not members or any(member not in mos_by_name for member in members):
                raise PdkLegalLowerBoundError(
                    "LOWER_BOUND_ARRAY_RESOURCE_MISMATCH",
                    f"array {consumption.group!r} has an unknown or empty member set",
                )
            overlap = tuple(sorted(array_members & set(members)))
            if overlap:
                raise PdkLegalLowerBoundError(
                    "LOWER_BOUND_ARRAY_RESOURCE_MISMATCH",
                    f"array {consumption.group!r} overlaps members {overlap!r}",
                )
            normalized_kinds = {_norm_kind(mos_by_name[name].kind) for name in members}
            if len(normalized_kinds) != 1:
                raise PdkLegalLowerBoundError(
                    "LOWER_BOUND_ARRAY_RESOURCE_MISMATCH",
                    f"array {consumption.group!r} mixes active-layer families",
                )
            active_key = tuple(
                _resolve_layer_keys(tech, next(iter(normalized_kinds))).active
            )
            from clay.layout.physical_macro_contracts import _artwork_sha256

            x0, y0, _x1, _y1 = consumption.array.extent()
            array_artwork_sha256 = _artwork_sha256(
                consumption.array.polygons, origin_nm=(int(x0), int(y0))
            )
            array_domain_sha256 = _canonical_rule_digest(
                {
                    "array_artwork_sha256": array_artwork_sha256,
                    "group": consumption.group,
                    "members": list(members),
                    "pdk_source_snapshot_sha256": snapshot.snapshot_sha256,
                    "proof_kind": proof_kind,
                    "schema": "clay.nonshareable-array-atom/v1",
                    "topology_sha256": graph.sha256,
                }
            )
            layer_resources.append(
                _materialized_layer_resource(
                    f"array:{proof_kind}:{consumption.group}",
                    "device_active",
                    members,
                    active_key,
                    consumption.array.polygons,
                    proof_kind,
                    array_domain_sha256,
                    "canonical_array_atom",
                )
            )
            array_members.update(members)
    layer_resources.extend(
        resource
        for member, resource in sorted(mos_resource_by_member.items())
        if member not in array_members
    )

    original_footprints = {item.member_id: item for item in footprints}
    tap_side = max(
        int(rules.channel_min_width),
        contact_cut_size + 2 * int(rules.comp_contact_enc),
    )
    base_ring_clearance = int(_group_ring_base_clearance_nm(tech))
    nwell_clearance = int(_group_ring_nwell_clearance_nm(tech))
    assigned_group: dict[str, str] = {}
    groups: list[LegalPlacementGroup] = []

    def add_well_group(
        label: str,
        members: tuple[str, ...],
        *,
        well: str,
        tie_net: str,
        ring_required: bool,
        declaration_sha256: str = "",
    ) -> None:
        canonical_members = tuple(sorted(members))
        if not canonical_members or any(name not in original_footprints for name in canonical_members):
            raise PdkLegalLowerBoundError(
                "LOWER_BOUND_TOPOLOGY_MISMATCH",
                f"well-domain {label!r} has an unknown or empty member inventory",
            )
        overlap = sorted(set(canonical_members) & set(assigned_group))
        if overlap:
            raise PdkLegalLowerBoundError(
                "LOWER_BOUND_TOPOLOGY_MISMATCH",
                f"well-domain {label!r} overlaps {overlap!r}",
            )
        if well not in {"p", "n"} or not tie_net:
            raise PdkLegalLowerBoundError(
                "LOWER_BOUND_TOPOLOGY_MISMATCH",
                f"well-domain {label!r} lacks one well polarity/tie net",
            )
        domain_digest = _canonical_rule_digest(
            {
                "label": label,
                "members": list(canonical_members),
                "pdk_source_snapshot_sha256": snapshot.snapshot_sha256,
                "source_declaration_sha256": declaration_sha256,
                "schema": "clay.lower-bound-well-domain/v1",
                "tie_net": tie_net,
                "well": well,
            }
        )
        group_id = f"well_{domain_digest[:20]}"
        tap_members: tuple[str, ...] = ()
        if ring_required:
            tap_name = f"__lb_tap_{domain_digest[:20]}"
            tap_digest = _canonical_rule_digest(
                {
                    "active_side_nm": tap_side,
                    "comp_contact_enclosure_nm": int(rules.comp_contact_enc),
                    "contact_size_nm": contact_cut_size,
                    "domain_sha256": domain_digest,
                    "pdk_source_snapshot_sha256": snapshot.snapshot_sha256,
                    "schema": "clay.mandatory-tap-floor/v1",
                }
            )
            footprints.append(
                LegalFootprint(
                    tap_name,
                    "tap",
                    tap_side,
                    tap_side,
                    tap_digest,
                    body_net=tie_net,
                )
            )
            tap_members = (tap_name,)
        ring_width = 0
        ring_setback = 0
        well_enclosure = (
            int(implant_well.nwell_pcomp_enclosure) if well == "n" else 0
        )
        well_spacing = max(
            base_ring_clearance,
            nwell_clearance if well == "n" else 0,
        )
        if ring_required:
            diff_width = tap_side
            ring_width = diff_width
            if well == "n":
                ring_width = max(
                    diff_width + 2 * int(rules.nwell_ncomp_overlap),
                    int(rules.nwell_min_width),
                )
            extension = int(_ring_extension_nm(tech, well))
            ring_setback = max(0, extension - ring_width)
            well_enclosure = 0
        groups.append(
            LegalPlacementGroup(
                group_id,
                "well_island",
                (*canonical_members, *tap_members),
                tie_net=tie_net,
                well_enclosure_nm=well_enclosure,
                well_spacing_nm=well_spacing,
                guard_ring_setback_nm=ring_setback,
                guard_ring_width_nm=ring_width,
                tap_integrated=ring_required,
            )
        )
        for name in canonical_members:
            assigned_group[name] = group_id

    declared_rings = tuple(
        sorted(intent.guard_rings, key=lambda item: item.sha256)
    )
    for ordinal, ring in enumerate(declared_rings):
        members = tuple(str(name) for name in ring.members)
        member_ties = {
            device_domains[name][1] for name in members if name in device_domains
        }
        tie_net = str(ring.tie_net or (next(iter(member_ties)) if len(member_ties) == 1 else ""))
        add_well_group(
            f"declared_guard_ring_{ordinal}",
            members,
            well=str(ring.well),
            tie_net=tie_net,
            ring_required=True,
            declaration_sha256=ring.sha256,
        )

    for name in sorted(str(item) for item in intent.well_isolated_devices):
        if name in assigned_group:
            continue
        well, body = device_domains.get(name, ("", ""))
        add_well_group(
            f"isolated_{name}",
            (name,),
            well=well,
            tie_net=body,
            ring_required=False,
        )

    from clay.layout.well_island_sharing import compile_well_island_share_plans

    for plan in compile_well_island_share_plans(
        cell, intent.well_island_shares, tech
    ):
        owners = {assigned_group.get(name) for name in plan.members}
        if owners != {None}:
            if len(owners) == 1 and None not in owners:
                continue
            raise PdkLegalLowerBoundError(
                "LOWER_BOUND_TOPOLOGY_MISMATCH",
                f"shared well {plan.group!r} crosses mandatory domains",
            )
        add_well_group(
            f"shared_{plan.group}",
            plan.members,
            well="n" if plan.well == "nwell" else "p",
            tie_net=plan.body,
            ring_required=False,
            declaration_sha256=plan.declaration_sha256,
        )

    matched_components = tuple(sorted(set(
        macro.members for macro in graph.macros if macro.kind == "matched_array"
    )))
    for macro in sorted(
        (item for item in graph.macros if item.kind == "well_island"),
        key=lambda item: item.macro_id,
    ):
        members = tuple(str(name) for name in macro.members)
        owners = {assigned_group.get(name) for name in members}
        if owners != {None}:
            if len(owners) == 1 and None not in owners:
                continue
            raise PdkLegalLowerBoundError(
                "LOWER_BOUND_TOPOLOGY_MISMATCH",
                f"well island {macro.macro_id!r} crosses mandatory domains",
            )
        attributes = dict(macro.attributes)
        add_well_group(
            macro.macro_id,
            members,
            well=str(attributes.get("well", "")),
            tie_net=str(attributes.get("body", "")),
            ring_required=False,
        )

    remaining_domains: dict[tuple[str, str], list[str]] = {}
    for name, domain in sorted(device_domains.items()):
        if name not in assigned_group:
            remaining_domains.setdefault(domain, []).append(name)
    for ordinal, ((well, body), members) in enumerate(sorted(remaining_domains.items())):
        add_well_group(
            f"body_domain_{ordinal}",
            tuple(members),
            well=well,
            tie_net=body,
            ring_required=False,
        )

    passive_names = tuple(sorted(
        item.member_id
        for item in footprints
        if item.kind == "passive" and item.member_id not in assigned_group
    ))
    if passive_names:
        groups.append(
            LegalPlacementGroup("passive_domain", "passive_bank", passive_names)
        )

    rails_by_key: dict[tuple[str, int], tuple[int, int]] = {}
    wire_width_by_net = dict(intent.wire_widths)
    if any(
        type(net) is not str
        or type(width) is not int
        or width <= 0
        for net, width in wire_width_by_net.items()
    ):
        raise TypeError("wire-width intent must map exact net names to positive nm")
    for net in sorted(str(port) for port in cell.ports):
        external_width, external_run = _minimum_legal_metal_segment(
            tech,
            rules,
            1,
            declared_width_nm=wire_width_by_net.get(net, 0),
        )
        rails_by_key[(net, 1)] = (external_width, external_run)
    for trunk in intent.trunks:
        layer_n = _declared_metal_layer_n(trunk.layer)
        declared_width_nm = max(
            int(round(float(trunk.width_um) * 1000.0)),
            wire_width_by_net.get(str(trunk.net), 0),
        )
        width, run = _minimum_legal_metal_segment(
            tech,
            rules,
            layer_n,
            declared_width_nm=declared_width_nm,
        )
        key = (str(trunk.net), layer_n)
        prior = rails_by_key.get(key)
        if prior is not None:
            width = max(width, prior[0])
            run = max(run, prior[1])
        rails_by_key[key] = (width, run)
    rails = tuple(
        RailRequirement(net, layer, 1, width, run)
        for (net, layer), (width, run) in sorted(rails_by_key.items())
    )

    via_counts: dict[tuple[str, int], int] = {
        ("contact", 1): max(
            1,
            sum(
                group.kind == "well_island" and group.tap_integrated
                for group in groups
            ),
        )
    }
    vias = tuple(
        ViaRequirement(name, lower, count, _mandatory_via_cut_size(tech, name))
        for (name, lower), count in sorted(via_counts.items())
    )

    declared_channels = tuple(intent.channels)
    channels = []
    for ordinal, channel in enumerate(declared_channels):
        channel_layer = 2
        try:
            tech.metal_key(channel_layer)
        except ValueError as exc:
            raise PdkLegalLowerBoundError(
                "LOWER_BOUND_PDK_RULE_MISSING",
                "PDK does not expose declared channel layer metal2",
            ) from exc
        channel_width = tech.min_width(f"metal{channel_layer}")
        channel_space = tech.min_space(f"metal{channel_layer}")
        if (
            type(channel_width) is not int
            or channel_width <= 0
            or type(channel_space) is not int
            or channel_space < 0
        ):
            raise PdkLegalLowerBoundError(
                "LOWER_BOUND_PDK_RULE_MISSING",
                "PDK does not expose exact metal2 channel width/spacing",
            )
        declared_nm = int(round(float(channel.width_um) * 1000.0))
        tracks = max(
            1,
            math.ceil((declared_nm + channel_space) / (channel_width + channel_space)),
        )
        channels.append(
            ChannelRequirement(
                f"channel_{ordinal}",
                channel_layer,
                tracks,
                channel_width,
                channel_space,
                1,
            )
        )

    confirmed = adapter.source_snapshot()
    if (
        confirmed is None
        or confirmed.snapshot_sha256 != snapshot.snapshot_sha256
        or confirmed.installed_authority_sha256
        != snapshot.installed_authority_sha256
    ):
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_PDK_AUTHORITY_MISSING",
            "PDK source snapshot changed during PCell/rule replay",
        )
    if path.read_bytes() != source_bytes:
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_SOURCE_REPLAY_MISMATCH",
            "source changed during PCell/rule replay",
        )
    inputs = PdkLegalLowerBoundInput(
        circuit_id=str(circuit_id or cell.local_name),
        topology_sha256=graph.sha256,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        pdk_id=selected_pdk,
        pdk_source_snapshot_sha256=snapshot.snapshot_sha256,
        pdk_installed_authority_sha256=str(snapshot.installed_authority_sha256),
        deck_scope="foundry",
        footprints=tuple(footprints),
        layer_resources=tuple(layer_resources),
        groups=tuple(groups),
        matching_components=matched_components,
        rails=rails,
        vias=vias,
        channels=tuple(channels),
    )
    return inputs, compute_pdk_legal_lower_bound(inputs)


def verify_pdk_legal_lower_bound_source(
    source_path: str | Path,
    inputs: PdkLegalLowerBoundInput,
    receipt: PdkLegalLowerBoundReceipt,
) -> bool:
    """Replay source, PDK bytes, PCells, and rules before accepting one floor."""

    expected_inputs, expected_receipt = compile_pdk_legal_lower_bound_source(
        source_path, circuit_id=inputs.circuit_id, pdk_id=inputs.pdk_id
    )
    if (
        inputs.canonical_data() != expected_inputs.canonical_data()
        or receipt.canonical_data() != expected_receipt.canonical_data()
    ):
        raise PdkLegalLowerBoundError(
            "LOWER_BOUND_SOURCE_REPLAY_MISMATCH",
            "source/PDK replay differs from supplied lower-bound evidence",
        )
    return True


__all__ = [
    "ChannelRequirement",
    "LegalFootprint",
    "LegalLayerResource",
    "LegalPlacementGroup",
    "LowerBoundComponent",
    "PdkLegalLowerBoundError",
    "PdkLegalLowerBoundInput",
    "PdkLegalLowerBoundReceipt",
    "RailRequirement",
    "ViaRequirement",
    "compile_pdk_legal_lower_bound_source",
    "compute_pdk_legal_lower_bound",
    "verify_pdk_legal_lower_bound",
    "verify_pdk_legal_lower_bound_source",
    "verify_pdk_legal_lower_bound_topology",
]
