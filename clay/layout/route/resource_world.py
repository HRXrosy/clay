"""Deterministic provenance for the exact world a resource plan may route.

The candidate-pool solver can prove that a pool is internally self-consistent,
but that proof says nothing about whether the pool belongs to this final routing
world.  This module binds a pool to final pins, projected obstacles, materialized
rails, placement evidence, topology declarations, and PDK routing facts.  It
enumerates explicit legal track segments and either an eager landing-site list
or an exact lazy landing domain; it neither creates complete-net candidates nor
invokes a router.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from clay.layout.minimum_area import minimum_area_square_side_nm
from clay.layout.route.resource_meter import bind_budget, resource_boundary
from clay.layout.route.resource_plan import (
    CandidatePool,
    CorridorLayerKey,
    LandingSite,
    LandingSiteCatalog,
    LandingSitePage,
    LandingSiteConflictDomain,
    LandingSiteConflictMetalRule,
    LandingSitePairConflict,
    LandingSiteConflictViaRule,
    RouteResourcePlanVerificationError,
    MAX_TOTAL_ASSIGNMENT_ARCS,
    TrackResource,
    TopologyPlan,
    canonical_explicit_routing_resource_ledger_json,
    explicit_routing_resource_ledger_sha256,
    landing_site_collection_sha256,
    paged_routing_resource_ledger_sha256,
)
from clay.layout.route.resource_conductors import TERMINAL_CONDUCTOR_KINDS


LayerKey = tuple[int, int]
BBoxNm = tuple[int, int, int, int]

MAX_ENUMERATED_TRACK_CENTERS = 8_192
MAX_ENUMERATED_TRACK_SEGMENTS = 32_768
MAX_ENUMERATED_TRACK_RESOURCES = 8_192
MAX_ENUMERATED_VIA_SITES = 32_768
MAX_ENUMERATED_VIA_RESOURCE_PAIRS = 500_000
MAX_ENUMERATED_LANDING_SITE_PAIR_CONFLICTS = 100_000
# Exact-query work is capped at the production budget.  Callers may lower the
# budget for probes/tests, but may not buy their way around a typed stop with a
# larger override; a stop must be removed by a sound index or compiler change.
DEFAULT_ENUMERATED_GEOMETRY_EVALUATIONS = 5_000_000
MAX_ENUMERATED_GEOMETRY_EVALUATIONS = 5_000_000
MAX_ROUTING_WORLD_REQUIRED_NETS = 50
MAX_ROUTING_WORLD_PINS = 100_000
MAX_ROUTING_WORLD_OBSTACLES = 200_000
MAX_ROUTING_WORLD_RAILS = 100_000
_PACKED_GEOMETRY_NODE_CAPACITY = 16


class CandidatePoolApplicabilityError(RouteResourcePlanVerificationError):
    """A replay-valid pool was compiled for a different final routing world."""


class RoutingResourceEnumerationError(ValueError):
    """Final-world facts cannot be lowered into explicit legal resources."""


class RoutingResourceEnumerationLimitError(RoutingResourceEnumerationError):
    """Explicit enumeration exceeded a declared deterministic work bound."""


class LandingSitePairConflictLimitError(
    RoutingResourceEnumerationLimitError
):
    """Exact pair materialization exceeded its finite tuple representation."""


class RoutingResourceLedgerVerificationError(RoutingResourceEnumerationError):
    """An explicit resource ledger does not replay against its claimed world."""


def _bounded_tuple(values: Any, *, name: str, limit: int) -> tuple[Any, ...]:
    """Materialize at most ``limit`` facts, including from a generator."""

    result: list[Any] = []
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise RoutingResourceEnumerationError(f"{name} must be iterable") from exc
    for value in iterator:
        result.append(value)
        if len(result) > limit:
            raise RoutingResourceEnumerationLimitError(
                f"{name} count exceeds hard maximum {limit}"
            )
    return tuple(result)


def _is_device_projection_fact(item: Any) -> bool:
    return (
        isinstance(item, (tuple, list))
        and len(item) >= 8
        and item[7] == "device_projection"
    )


@dataclass
class _GeometryEvaluationBudget:
    limit: int
    used: int = 0

    def consume(self, reason: str) -> None:
        self.used += 1
        if self.used > self.limit:
            raise RoutingResourceEnumerationLimitError(
                f"geometry evaluation budget {self.limit} exhausted during {reason}"
            )


@dataclass(frozen=True)
class _PackedGeometryNode:
    """One immutable node in a deterministic STR-packed obstacle R-tree."""

    bbox: BBoxNm
    minimum_item_index: int
    item_indices: tuple[int, ...] = ()
    children: tuple["_PackedGeometryNode", ...] = ()


def _bbox_union(bboxes: Sequence[BBoxNm]) -> BBoxNm:
    if not bboxes:
        raise RoutingResourceEnumerationError(
            "packed geometry node cannot be built without a bounding box"
        )
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _bboxes_touch_or_overlap(left: BBoxNm, right: BBoxNm) -> bool:
    return (
        max(left[0], right[0]) <= min(left[2], right[2])
        and max(left[1], right[1]) <= min(left[3], right[3])
    )


def _str_groups(
    entries: Sequence[Any],
    *,
    bbox_of: Any,
    stable_index_of: Any,
) -> tuple[tuple[Any, ...], ...]:
    """Pack entries with deterministic sort-tile-recursive (STR) grouping."""

    if not entries:
        return ()
    group_count = (
        len(entries) + _PACKED_GEOMETRY_NODE_CAPACITY - 1
    ) // _PACKED_GEOMETRY_NODE_CAPACITY
    slice_count = math.isqrt(group_count)
    if slice_count * slice_count < group_count:
        slice_count += 1
    groups_per_slice = (group_count + slice_count - 1) // slice_count
    slice_size = _PACKED_GEOMETRY_NODE_CAPACITY * groups_per_slice

    def x_key(entry: Any) -> tuple[Any, ...]:
        bbox = bbox_of(entry)
        return (
            bbox[0] + bbox[2],
            bbox[1] + bbox[3],
            bbox,
            stable_index_of(entry),
        )

    def y_key(entry: Any) -> tuple[Any, ...]:
        bbox = bbox_of(entry)
        return (
            bbox[1] + bbox[3],
            bbox[0] + bbox[2],
            bbox,
            stable_index_of(entry),
        )

    by_x = tuple(sorted(entries, key=x_key))
    groups: list[tuple[Any, ...]] = []
    for slice_start in range(0, len(by_x), slice_size):
        spatial_slice = tuple(sorted(
            by_x[slice_start:slice_start + slice_size],
            key=y_key,
        ))
        for group_start in range(
            0, len(spatial_slice), _PACKED_GEOMETRY_NODE_CAPACITY
        ):
            groups.append(tuple(
                spatial_slice[
                    group_start:group_start + _PACKED_GEOMETRY_NODE_CAPACITY
                ]
            ))
    return tuple(groups)


def _build_packed_obstacle_root(
    obstacles: Sequence[tuple[Any, ...]],
) -> _PackedGeometryNode | None:
    if not obstacles:
        return None
    item_bboxes = tuple(
        (int(item[0]), int(item[1]), int(item[2]), int(item[3]))
        for item in obstacles
    )
    leaf_groups = _str_groups(
        tuple(range(len(obstacles))),
        bbox_of=lambda index: item_bboxes[index],
        stable_index_of=lambda index: index,
    )
    level = tuple(
        _PackedGeometryNode(
            bbox=_bbox_union(tuple(item_bboxes[index] for index in group)),
            minimum_item_index=min(group),
            item_indices=tuple(sorted(group)),
        )
        for group in leaf_groups
    )
    while len(level) > 1:
        parent_groups = _str_groups(
            level,
            bbox_of=lambda node: node.bbox,
            stable_index_of=lambda node: node.minimum_item_index,
        )
        level = tuple(
            _PackedGeometryNode(
                bbox=_bbox_union(tuple(node.bbox for node in group)),
                minimum_item_index=min(
                    node.minimum_item_index for node in group
                ),
                children=tuple(group),
            )
            for group in parent_groups
        )
    return level[0]


def _query_packed_bbox_indices(
    root: _PackedGeometryNode | None,
    item_bboxes: Sequence[BBoxNm],
    bbox: BBoxNm,
    budget: _GeometryEvaluationBudget,
    reason: str,
    *,
    max_index_exclusive: int | None = None,
) -> tuple[int, ...]:
    """Return exact intersecting item indices from one immutable packed tree."""

    if root is None:
        return ()
    matched: list[int] = []
    pending = [root]
    while pending:
        node = pending.pop()
        if (
            max_index_exclusive is not None
            and node.minimum_item_index >= max_index_exclusive
        ):
            continue
        budget.consume(reason)
        if not _bboxes_touch_or_overlap(node.bbox, bbox):
            continue
        if node.item_indices:
            for index in node.item_indices:
                if (
                    max_index_exclusive is not None
                    and index >= max_index_exclusive
                ):
                    break
                budget.consume(reason)
                if _bboxes_touch_or_overlap(item_bboxes[index], bbox):
                    matched.append(index)
            continue
        pending.extend(reversed(node.children))
    return tuple(sorted(matched))


@dataclass(frozen=True)
class _LayerGeometryIndex:
    obstacles_by_layer: Mapping[LayerKey, tuple[tuple[Any, ...], ...]]
    obstacle_roots_by_layer: Mapping[LayerKey, _PackedGeometryNode]
    pins_by_layer: Mapping[LayerKey, tuple[tuple[Any, ...], ...]]
    pin_x_by_layer: Mapping[LayerKey, tuple[int, ...]]

    @classmethod
    def build(
        cls,
        obstacles: Sequence[tuple[Any, ...]],
        pins: Sequence[tuple[Any, ...]],
    ) -> "_LayerGeometryIndex":
        obstacle_groups: dict[LayerKey, list[tuple[Any, ...]]] = {}
        for obstacle in obstacles:
            key = int(obstacle[4]), int(obstacle[5])
            obstacle_groups.setdefault(key, []).append(obstacle)
        pin_groups: dict[LayerKey, list[tuple[Any, ...]]] = {}
        for pin in pins:
            key = int(pin[5]), int(pin[6])
            pin_groups.setdefault(key, []).append(pin)
        obstacles_by_layer = {
            key: tuple(sorted(
                values,
                key=lambda item: (
                    int(item[0]),
                    int(item[1]),
                    int(item[2]),
                    int(item[3]),
                    json.dumps(item, separators=(",", ":")),
                ),
            ))
            for key, values in obstacle_groups.items()
        }
        pins_by_layer = {
            key: tuple(sorted(
                values,
                key=lambda item: (
                    int(item[3]),
                    int(item[4]),
                    str(item[0]),
                    str(item[1]),
                    str(item[2]),
                ),
            ))
            for key, values in pin_groups.items()
        }
        return cls(
            obstacles_by_layer=obstacles_by_layer,
            obstacle_roots_by_layer={
                key: root
                for key, values in obstacles_by_layer.items()
                for root in (_build_packed_obstacle_root(values),)
                if root is not None
            },
            pins_by_layer=pins_by_layer,
            pin_x_by_layer={
                key: tuple(int(item[3]) for item in values)
                for key, values in pins_by_layer.items()
            },
        )

    def query_obstacles(
        self,
        layer_key: LayerKey,
        bbox: BBoxNm,
        budget: _GeometryEvaluationBudget,
        reason: str,
    ):
        values = self.obstacles_by_layer.get(layer_key, ())
        root = self.obstacle_roots_by_layer.get(layer_key)
        if root is None:
            return
        matched_indices: list[int] = []
        pending = [root]
        while pending:
            node = pending.pop()
            budget.consume(reason)
            if not _bboxes_touch_or_overlap(node.bbox, bbox):
                continue
            if node.item_indices:
                for index in node.item_indices:
                    budget.consume(reason)
                    obstacle = values[index]
                    obstacle_bbox = (
                        int(obstacle[0]),
                        int(obstacle[1]),
                        int(obstacle[2]),
                        int(obstacle[3]),
                    )
                    if _bboxes_touch_or_overlap(obstacle_bbox, bbox):
                        matched_indices.append(index)
                continue
            pending.extend(reversed(node.children))
        for index in sorted(matched_indices):
            yield values[index]

    def query_pins(
        self,
        layer_key: LayerKey,
        bbox: BBoxNm,
        budget: _GeometryEvaluationBudget,
    ):
        values = self.pins_by_layer.get(layer_key, ())
        xs = self.pin_x_by_layer.get(layer_key, ())
        start = bisect_left(xs, bbox[0])
        stop = bisect_right(xs, bbox[2])
        for index in range(start, stop):
            pin = values[index]
            budget.consume("landing-pin spatial check")
            if bbox[1] <= int(pin[4]) <= bbox[3]:
                yield pin


@dataclass(frozen=True)
class ExplicitRoutingResourceLedger:
    """Geometry-complete resources derived only from one final routing world."""

    routing_world_provenance_digest: str
    track_resources: tuple[TrackResource, ...]
    landing_sites: tuple[LandingSite, ...]
    landing_site_pair_conflicts: tuple[LandingSitePairConflict, ...] = ()
    owned_conductor_index_digest: str = ""
    landing_site_domain_digest: str = ""
    landing_sites_complete: bool = True
    landing_site_conflict_domain: LandingSiteConflictDomain | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.routing_world_provenance_digest, str)
            or not self.routing_world_provenance_digest
        ):
            raise RoutingResourceEnumerationError(
                "ledger routing-world digest must be non-empty"
            )
        if not isinstance(self.owned_conductor_index_digest, str):
            raise RoutingResourceEnumerationError(
                "owned-conductor index digest must be a string"
            )
        if not isinstance(self.landing_site_domain_digest, str):
            raise RoutingResourceEnumerationError(
                "landing-site domain digest must be a string"
            )
        if type(self.landing_sites_complete) is not bool:
            raise RoutingResourceEnumerationError(
                "landing_sites_complete must be an exact boolean"
            )
        if not self.landing_sites_complete and not self.landing_site_domain_digest:
            raise RoutingResourceEnumerationError(
                "an incomplete landing-site listing requires a domain digest"
            )
        resources = _bounded_tuple(
            self.track_resources,
            name="ledger track resources",
            limit=MAX_ENUMERATED_TRACK_RESOURCES,
        )
        sites = _bounded_tuple(
            self.landing_sites,
            name="ledger landing sites",
            limit=MAX_ENUMERATED_VIA_SITES,
        )
        site_conflicts = _bounded_tuple(
            self.landing_site_pair_conflicts,
            name="ledger landing-site conflicts",
            limit=MAX_ENUMERATED_LANDING_SITE_PAIR_CONFLICTS,
        )
        conflict_domain = self.landing_site_conflict_domain
        if conflict_domain is not None and (
            not isinstance(conflict_domain, LandingSiteConflictDomain)
            or conflict_domain.routing_world_provenance_digest
            != self.routing_world_provenance_digest
            or conflict_domain.landing_site_collection_digest
            != landing_site_collection_sha256(sites)
        ):
            raise RoutingResourceEnumerationError(
                "ledger landing conflict domain has stale world/site binding"
            )
        if not all(isinstance(resource, TrackResource) for resource in resources):
            raise RoutingResourceEnumerationError(
                "ledger track resources must be TrackResource values"
            )
        if not all(isinstance(site, LandingSite) for site in sites):
            raise RoutingResourceEnumerationError(
                "ledger landing sites must be LandingSite values"
            )
        if not all(
            isinstance(conflict, LandingSitePairConflict)
            for conflict in site_conflicts
        ):
            raise RoutingResourceEnumerationError(
                "ledger site conflicts must be LandingSitePairConflict values"
            )
        for resource in resources:
            if (
                resource.through_interval_nm is None
                or resource.corridor_bbox_nm is None
                or resource.gds_layer_key is None
                or not resource.routing_world_provenance_digest
            ):
                raise RoutingResourceEnumerationError(
                    "ledger track resources must be geometry-complete"
                )
            if (
                resource.routing_world_provenance_digest
                != self.routing_world_provenance_digest
            ):
                raise RoutingResourceEnumerationError(
                    "ledger track resource belongs to a foreign routing world"
                )
        for site in sites:
            if (
                site.cut_bbox_nm is None
                or site.lower_pad_bbox_nm is None
                or site.upper_pad_bbox_nm is None
                or not site.routing_world_provenance_digest
            ):
                raise RoutingResourceEnumerationError(
                    "ledger landing sites must be geometry-complete"
                )
            if (
                site.routing_world_provenance_digest
                != self.routing_world_provenance_digest
            ):
                raise RoutingResourceEnumerationError(
                    "ledger landing site belongs to a foreign routing world"
                )
        resource_keys = tuple(resource.key for resource in resources)
        if len(resource_keys) != len(set(resource_keys)):
            raise RoutingResourceEnumerationError(
                "ledger contains duplicate track resource keys"
            )
        site_ids = tuple(site.site_id for site in sites)
        if len(site_ids) != len(set(site_ids)):
            raise RoutingResourceEnumerationError(
                "ledger contains duplicate landing site ids"
            )
        site_id_set = set(site_ids)
        conflict_pairs = tuple(
            (conflict.left_site_id, conflict.right_site_id)
            for conflict in site_conflicts
        )
        if len(conflict_pairs) != len(set(conflict_pairs)):
            raise RoutingResourceEnumerationError(
                "ledger contains duplicate landing site-pair conflicts"
            )
        for conflict in site_conflicts:
            if (
                conflict.left_site_id not in site_id_set
                or conflict.right_site_id not in site_id_set
            ):
                raise RoutingResourceEnumerationError(
                    "ledger site conflict references a missing landing site"
                )
            if (
                conflict.routing_world_provenance_digest
                != self.routing_world_provenance_digest
            ):
                raise RoutingResourceEnumerationError(
                    "ledger site conflict belongs to a foreign routing world"
                )
        object.__setattr__(
            self,
            "track_resources",
            tuple(sorted(resources, key=lambda resource: resource.key)),
        )
        object.__setattr__(
            self,
            "landing_sites",
            tuple(sorted(sites, key=lambda site: site.site_id)),
        )
        object.__setattr__(
            self,
            "landing_site_pair_conflicts",
            tuple(sorted(
                site_conflicts,
                key=lambda conflict: (
                    conflict.left_site_id,
                    conflict.right_site_id,
                ),
            )),
        )
        object.__setattr__(
            self, "landing_site_conflict_domain", conflict_domain
        )

    def canonical_json(self) -> str:
        return canonical_explicit_routing_resource_ledger_json(
            self.routing_world_provenance_digest,
            self.track_resources,
            self.landing_sites,
            self.landing_site_pair_conflicts,
            self.owned_conductor_index_digest,
            self.landing_site_domain_digest,
            self.landing_sites_complete,
            self.landing_site_conflict_domain,
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _RoutingLayerFacts:
    layer_n: int
    key: LayerKey
    preferred_axis: str
    min_width_nm: int
    routing_width_nm: int
    min_space_nm: int
    track_step_nm: int
    manufacturing_grid_nm: int


def _first_rule_value(tech: Any, method_name: str, names: Sequence[str]) -> Any:
    method = getattr(tech, method_name)
    for name in names:
        try:
            value = method(name)
        except (KeyError, TypeError, ValueError):
            continue
        if value is not None:
            return int(value)
    return None


def _via_rule_payload(tech: Any, name: str) -> dict[str, Any]:
    try:
        rule = tech.via_rule(name)
    except (AttributeError, KeyError, TypeError, ValueError):
        rule = None
    payload: dict[str, Any] = {}
    for field in ("size", "spacing", "enclosure_below", "enclosure_above"):
        raw = None if rule is None else getattr(rule, field, None)
        if raw is None:
            payload[field] = None
            continue
        value = getattr(raw, "value", raw)
        source = getattr(raw, "source", None)
        payload[field] = {
            "rule_id": str(getattr(raw, "rule_id", "")),
            "source": str(getattr(source, "value", source or "")),
            "value": int(value),
        }
    return payload


def _first_via_rule_payload(tech: Any, names: Sequence[str]) -> dict[str, Any]:
    empty = {
        field: None
        for field in ("size", "spacing", "enclosure_below", "enclosure_above")
    }
    for name in names:
        payload = _via_rule_payload(tech, name)
        if any(value is not None for value in payload.values()):
            return payload
    return empty


def _parsed_rule_payload(tech: Any) -> dict[str, dict[str, Any]]:
    """Numeric parsed-deck facts used ahead of consolidated router fallbacks."""

    try:
        rules = tech.drc_rules() or {}
    except (AttributeError, KeyError, TypeError, ValueError):
        return {}
    payload: dict[str, dict[str, Any]] = {}
    for name, raw in sorted(rules.items()):
        if not isinstance(raw, Mapping):
            continue
        values = {
            str(key): value
            for key, value in sorted(raw.items())
            if str(key).startswith("value_")
            and isinstance(value, (int, float, str, bool))
        }
        if values:
            payload[str(name)] = values
    return payload


def canonical_pdk_routing_json(tech: Any) -> str:
    """Canonical PDK identity plus every routing-layer rule consumed here."""

    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    from clay.layout.route.via_model import via_cut_size_nm, via_enclosure_nm

    metals: list[dict[str, Any]] = []
    for layer_n in range(1, 33):
        try:
            key = tuple(int(value) for value in tech.metal_key(layer_n))
        except (KeyError, TypeError, ValueError):
            break
        names = (f"metal{layer_n}", f"met{layer_n}", f"m{layer_n}")
        axis_fn = getattr(tech, "metal_preferred_axis_declared", None)
        try:
            declared_axis = axis_fn(layer_n) if callable(axis_fn) else None
        except (AttributeError, KeyError, TypeError, ValueError):
            declared_axis = None
        declared_axis = (
            declared_axis.upper()
            if isinstance(declared_axis, str)
            and declared_axis.upper() in ("H", "V")
            else None
        )
        metals.append({
            "declared_axis": declared_axis,
            "gds_key": list(key),
            "layer_n": layer_n,
            "min_area_nm2": _first_rule_value(tech, "min_area", names),
            "min_space_nm": _first_rule_value(tech, "min_space", names),
            "min_width_nm": _first_rule_value(tech, "min_width", names),
            "width_spacing_entries": [
                {"wide_ge_nm": int(band.wide_ge),
                 "run_length_ge_nm": int(band.run_length_ge),
                 "spacing_nm": int(band.spacing.value),
                 "rule_id": str(band.spacing.rule_id),
                 "source": str(getattr(band.spacing.source, "value", band.spacing.source)),
                 "opening_nm": band.opening_nm,
                 "attached_halo_nm": band.attached_halo_nm}
                for band in (tech.width_spacing_entries(names[0])
                             if callable(getattr(tech, "width_spacing_entries", None)) else ())
            ],
        })

    vias: list[dict[str, Any]] = []
    for lower_layer_n in range(1, len(metals)):
        names = (
            f"via{lower_layer_n}",
            f"vi{lower_layer_n}",
            f"v{lower_layer_n}",
        )
        try:
            key = list(int(value) for value in tech.via_key(lower_layer_n))
        except (KeyError, TypeError, ValueError):
            key = None
        rule_payload = _first_via_rule_payload(tech, names)
        effective_name = f"via{lower_layer_n}"
        vias.append({
            "cut_gds_key": key,
            "effective_cut_size_nm": via_cut_size_nm(tech, effective_name),
            "effective_enclosure_nm": list(
                via_enclosure_nm(tech, effective_name)
            ),
            "effective_spacing_nm": _via_spacing_nm(tech, effective_name),
            "lower_layer_n": lower_layer_n,
            "rule": rule_payload,
        })

    contact: dict[str, Any]
    try:
        contact_key = list(int(value) for value in tech.contact_key())
    except (AttributeError, KeyError, TypeError, ValueError):
        contact_key = None
    contact = {
        "cut_gds_key": contact_key,
        "rule": _via_rule_payload(tech, "contact"),
    }
    payload = {
        "contact": contact,
        "metals": metals,
        "manufacturing_grid_nm": pdk_manufacturing_grid_nm(tech),
        "parsed_rules": _parsed_rule_payload(tech),
        "pdk_id": str(getattr(tech, "pdk_id", "")),
        "schema": "clay.pdk-routing-provenance.v3",
        "mim_routing_clearance_rules": [
            {"rule_id": rule.rule_id, "metal_layer": list(rule.metal_layer),
             "marker_layer": list(rule.marker_layer), "predicate": rule.predicate,
             "grow_nm": rule.grow_nm, "spacing_nm": rule.spacing_nm,
             "metric": rule.metric}
            for rule in (tech.mim_routing_clearance_rules()
                         if callable(getattr(tech, "mim_routing_clearance_rules", None)) else ())
        ],
        "vias": vias,
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def pdk_provenance_digest(tech: Any) -> str:
    return hashlib.sha256(
        canonical_pdk_routing_json(tech).encode("utf-8")
    ).hexdigest()


def _nm(value_um: Any) -> int:
    value = float(value_um)
    if not math.isfinite(value):
        raise ValueError(f"routing-world coordinate must be finite, got {value_um!r}")
    return int(round(value * 1000.0))


def _layer_key(layer: Any) -> LayerKey:
    if not isinstance(layer, (tuple, list)) or len(layer) != 2:
        raise ValueError(f"routing-world layer must be a GDS key, got {layer!r}")
    return int(layer[0]), int(layer[1])


@dataclass(frozen=True)
class RoutingWorldProvenance:
    """Canonical final routing inputs and their component provenance digests."""

    required_nets: tuple[str, ...]
    placement_provenance_digest: str
    topology_provenance_digest: str
    pdk_provenance_digest: str
    pins: tuple[tuple[Any, ...], ...]
    obstacles: tuple[tuple[Any, ...], ...]
    rails: tuple[tuple[Any, ...], ...]
    # Parallel canonical binding keeps the established eight-field obstacle
    # currency stable while making device-terminal seam identity hash material.
    # Each record is ``(*exact_obstacle_fact, terminal_conductor_ref)``.
    terminal_conductor_refs: tuple[tuple[Any, ...], ...] = ()

    def __post_init__(self) -> None:
        required_nets = _bounded_tuple(
            self.required_nets,
            name="routing-world required_nets",
            limit=MAX_ROUTING_WORLD_REQUIRED_NETS,
        )
        if (
            any(not isinstance(net, str) or not net for net in required_nets)
            or len(required_nets) != len(set(required_nets))
            or required_nets != tuple(sorted(required_nets))
        ):
            raise ValueError(
                "required_nets must be sorted, unique, non-empty strings"
            )
        for name in (
            "placement_provenance_digest",
            "topology_provenance_digest",
            "pdk_provenance_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        pins = _bounded_tuple(
            self.pins,
            name="routing-world pins",
            limit=MAX_ROUTING_WORLD_PINS,
        )
        from clay.layout.route.terminal_identity import unpack_pin_fact
        for pin in pins:
            if isinstance(pin, (tuple, list)) and len(pin) in (7, 8) and any(
                not isinstance(identity, str) or not identity for identity in pin[:3]
            ):
                raise ValueError(
                    "routing-world pin net/device/terminal identity must be "
                    "non-empty strings"
                )
            unpack_pin_fact(pin)
        pins = tuple(tuple(pin) for pin in pins)
        obstacles = _bounded_tuple(
            self.obstacles,
            name="routing-world obstacles",
            limit=MAX_ROUTING_WORLD_OBSTACLES,
        )
        obstacles = tuple(
            obstacle
            for obstacle in obstacles
            if not _is_device_projection_fact(obstacle)
        )
        rails = _bounded_tuple(
            self.rails,
            name="routing-world rails",
            limit=MAX_ROUTING_WORLD_RAILS,
        )
        terminal_bindings = _bounded_tuple(
            self.terminal_conductor_refs,
            name="routing-world terminal_conductor_refs",
            limit=MAX_ROUTING_WORLD_OBSTACLES,
        )
        obstacle_set = {tuple(obstacle) for obstacle in obstacles}
        binding_by_fact: dict[tuple[Any, ...], str] = {}
        for raw_binding in terminal_bindings:
            if not isinstance(raw_binding, (tuple, list)) or len(raw_binding) != 9:
                raise ValueError(
                    "terminal_conductor_ref binding must contain one exact "
                    "eight-field obstacle fact plus one ref"
                )
            fact = tuple(raw_binding[:8])
            ref = raw_binding[8]
            if fact not in obstacle_set:
                raise ValueError(
                    "terminal_conductor_ref binding names no exact obstacle fact"
                )
            if fact[6] is None or fact[7] not in TERMINAL_CONDUCTOR_KINDS:
                raise ValueError(
                    "terminal_conductor_ref may bind only owned literal terminal "
                    "artwork"
                )
            if not isinstance(ref, str) or not ref:
                raise ValueError("terminal_conductor_ref must be a non-empty string")
            previous = binding_by_fact.setdefault(fact, ref)
            if previous != ref:
                raise ValueError(
                    "one exact obstacle fact has ambiguous terminal_conductor_refs"
                )
        terminal_bindings = tuple(sorted(
            ((*fact, ref) for fact, ref in binding_by_fact.items()),
            key=lambda item: json.dumps(item, separators=(",", ":")),
        ))
        object.__setattr__(self, "required_nets", required_nets)
        object.__setattr__(self, "pins", tuple(sorted(pins)))
        object.__setattr__(
            self,
            "obstacles",
            tuple(sorted(
                obstacles,
                key=lambda item: json.dumps(item, separators=(",", ":")),
            )),
        )
        object.__setattr__(self, "rails", tuple(sorted(rails)))
        object.__setattr__(self, "terminal_conductor_refs", terminal_bindings)

    def canonical_json(self) -> str:
        payload = {
            "obstacles": [list(item) for item in self.obstacles],
            "pdk_provenance_digest": self.pdk_provenance_digest,
            "pins": [list(item) for item in self.pins],
            "placement_provenance_digest": self.placement_provenance_digest,
            "rails": [list(item) for item in self.rails],
            "required_nets": list(self.required_nets),
            "schema": ("clay.final-routing-world.v3" if any(len(pin) == 8 for pin in self.pins)
                       else "clay.final-routing-world.v2"),
            "terminal_conductor_refs": [
                list(item) for item in self.terminal_conductor_refs
            ],
            "topology_provenance_digest": self.topology_provenance_digest,
        }
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def routing_world_provenance_digest(self) -> str:
        return self.sha256()


def build_routing_world_provenance(
    *,
    pins_by_net: Mapping[str, Sequence[Any]],
    obstacles: Sequence[Any],
    rails_by_net: Mapping[str, Sequence[Any]],
    placement_provenance_digest: str,
    topology_plan: TopologyPlan,
    tech: Any,
) -> RoutingWorldProvenance:
    """Freeze the final canonical route inputs after all materialization/projection."""

    if not isinstance(placement_provenance_digest, str) or not placement_provenance_digest:
        raise ValueError("placement_provenance_digest must be non-empty")
    if not isinstance(topology_plan, TopologyPlan):
        raise TypeError("topology_plan must be TopologyPlan")
    required_nets = tuple(sorted(_bounded_tuple(
        (demand.net for demand in topology_plan.demands),
        name="routing-world required_nets",
        limit=MAX_ROUTING_WORLD_REQUIRED_NETS,
    )))

    pin_items = _bounded_tuple(
        pins_by_net.items(),
        name="pins_by_net entries",
        limit=MAX_ROUTING_WORLD_REQUIRED_NETS,
    )
    pin_facts: list[tuple[Any, ...]] = []
    from clay.layout.route.terminal_identity import pin_fact
    for net, net_pins in pin_items:
        for pin in net_pins:
            fact_input = pin
            if not hasattr(pin, "net"):
                # Historical duck pins inherit their net from this actual
                # containing map. Keep that adapter without mutating the input;
                # an explicit Pin net still has to agree with the same map.
                from types import SimpleNamespace
                fact_input = SimpleNamespace(
                    net=str(net), device=getattr(pin, "device", ""),
                    terminal=getattr(pin, "terminal", ""),
                    xy_um=getattr(pin, "xy_um", None), xy_nm=getattr(pin, "xy_nm", None),
                    layer=getattr(pin, "layer", None),
                    terminal_binding=getattr(pin, "terminal_binding", None))
            fact = pin_fact(fact_input, layer_key=_layer_key(getattr(pin, "layer")))
            if fact[0] != str(net):
                raise ValueError("routing-world pin differs from its containing net")
            pin_facts.append(fact)
            if len(pin_facts) > MAX_ROUTING_WORLD_PINS:
                raise RoutingResourceEnumerationLimitError(
                    "routing-world pins count exceeds hard maximum "
                    f"{MAX_ROUTING_WORLD_PINS}"
                )
    pins = tuple(sorted(pin_facts))
    bounded_obstacles = _bounded_tuple(
        obstacles,
        name="routing-world obstacles",
        limit=MAX_ROUTING_WORLD_OBSTACLES,
    )
    obstacle_facts_list: list[tuple[Any, ...]] = []
    terminal_conductor_refs: list[tuple[Any, ...]] = []
    for obstacle in bounded_obstacles:
        if getattr(obstacle, "kind", None) == "device_projection":
            continue
        fact = (
                min(_nm(obstacle.x0), _nm(obstacle.x1)),
                min(_nm(obstacle.y0), _nm(obstacle.y1)),
                max(_nm(obstacle.x0), _nm(obstacle.x1)),
                max(_nm(obstacle.y0), _nm(obstacle.y1)),
                *_layer_key(obstacle.layer),
                None if obstacle.owner_net is None else str(obstacle.owner_net),
                None if obstacle.kind is None else str(obstacle.kind),
        )
        obstacle_facts_list.append(fact)
        terminal_ref = getattr(obstacle, "terminal_conductor_ref", None)
        if terminal_ref is not None:
            terminal_conductor_refs.append((*fact, terminal_ref))
    obstacle_facts = tuple(sorted(
        obstacle_facts_list,
        key=lambda item: json.dumps(item, separators=(",", ":")),
    ))
    rail_items = _bounded_tuple(
        rails_by_net.items(),
        name="rails_by_net entries",
        limit=MAX_ROUTING_WORLD_REQUIRED_NETS,
    )
    rail_facts: list[tuple[Any, ...]] = []
    for net, net_rails in rail_items:
        for layer, bbox in net_rails:
            rail_facts.append((
                str(net),
                *_layer_key(layer),
                min(_nm(bbox[0]), _nm(bbox[2])),
                min(_nm(bbox[1]), _nm(bbox[3])),
                max(_nm(bbox[0]), _nm(bbox[2])),
                max(_nm(bbox[1]), _nm(bbox[3])),
            ))
            if len(rail_facts) > MAX_ROUTING_WORLD_RAILS:
                raise RoutingResourceEnumerationLimitError(
                    "routing-world rails count exceeds hard maximum "
                    f"{MAX_ROUTING_WORLD_RAILS}"
                )
    rails = tuple(sorted(rail_facts))
    topology_digest = topology_plan.sha256()
    pdk_digest = pdk_provenance_digest(tech)
    return RoutingWorldProvenance(
        required_nets=required_nets,
        placement_provenance_digest=placement_provenance_digest,
        topology_provenance_digest=topology_digest,
        pdk_provenance_digest=pdk_digest,
        pins=pins,
        obstacles=obstacle_facts,
        rails=rails,
        terminal_conductor_refs=tuple(terminal_conductor_refs),
    )


def _ceil_to_multiple(value: int, step: int) -> int:
    return -((-int(value)) // int(step)) * int(step)


def _floor_to_multiple(value: int, step: int) -> int:
    return (int(value) // int(step)) * int(step)


def _routing_layer_facts(
    tech: Any,
    manufacturing_grid_nm: int | None = None,
) -> tuple[_RoutingLayerFacts, ...]:
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

    if manufacturing_grid_nm is None:
        manufacturing_grid_nm = pdk_manufacturing_grid_nm(tech)
    declared: list[
        tuple[int, LayerKey, str | None, int | None, int | None]
    ] = []
    for layer_n in range(1, 33):
        try:
            key = _layer_key(tech.metal_key(layer_n))
        except (AttributeError, KeyError, TypeError, ValueError):
            break
        names = (f"metal{layer_n}", f"met{layer_n}", f"m{layer_n}")
        width = _first_rule_value(tech, "min_width", names)
        space = _first_rule_value(tech, "min_space", names)
        try:
            axis = tech.metal_preferred_axis_declared(layer_n)
        except (AttributeError, KeyError, TypeError, ValueError):
            axis = None
        axis = axis.upper() if isinstance(axis, str) else ""
        declared.append((
            layer_n,
            key,
            axis if axis in ("H", "V") else None,
            width,
            space,
        ))

    routing_indexes = tuple(
        index
        for index, (_layer_n, _key, axis, width, space) in enumerate(declared)
        if (
            axis is not None
            and width is not None
            and width > 0
            and space is not None
            and space >= 0
        )
    )
    if not routing_indexes:
        if declared:
            first_axis_layer = next(
                (
                    layer_n
                    for layer_n, _key, axis, _width, _space in declared
                    if axis is not None
                ),
                None,
            )
            if first_axis_layer is not None:
                raise RoutingResourceEnumerationError(
                    f"metal{first_axis_layer} has no valid routing "
                    "width/spacing declaration"
                )
            raise RoutingResourceEnumerationError(
                "metal1 has no valid declared preferred routing axis"
            )
        raise RoutingResourceEnumerationError("PDK declares no routing metal layers")
    highest_routing_index = routing_indexes[-1]
    higher_layer_n = declared[highest_routing_index][0]
    for index, (layer_n, _key, axis, width, space) in enumerate(declared):
        if index <= highest_routing_index and axis is None:
            raise RoutingResourceEnumerationError(
                f"metal{layer_n} declared routing axis is missing below "
                f"routable metal{higher_layer_n}"
            )
        if index <= highest_routing_index and (
            width is None or width <= 0 or space is None or space < 0
        ):
            raise RoutingResourceEnumerationError(
                f"metal{layer_n} routing width/spacing declaration is missing "
                f"below routable metal{higher_layer_n}"
            )

    layers: list[_RoutingLayerFacts] = []
    for layer_n, key, axis, width, space in declared[: highest_routing_index + 1]:
        if axis is None:  # Proven impossible by the contiguous-prefix check.
            raise RoutingResourceEnumerationError(
                f"metal{layer_n} has no valid declared preferred routing axis"
            )
        if width is None or space is None or width <= 0 or space < 0:
            raise RoutingResourceEnumerationError(
                f"metal{layer_n} has no valid routing width/spacing declaration"
            )
        routing_width_nm = _ceil_to_multiple(
            int(width), 2 * manufacturing_grid_nm
        )
        layers.append(_RoutingLayerFacts(
            layer_n=layer_n,
            key=key,
            preferred_axis=axis,
            min_width_nm=int(width),
            routing_width_nm=routing_width_nm,
            min_space_nm=int(space),
            track_step_nm=_ceil_to_multiple(
                routing_width_nm + int(space), manufacturing_grid_nm
            ),
            manufacturing_grid_nm=manufacturing_grid_nm,
        ))
    if not layers:
        raise RoutingResourceEnumerationError(
            "PDK exposes no metal with key, width, spacing, and preferred axis"
        )
    return tuple(layers)


def _world_envelope(world: RoutingWorldProvenance) -> BBoxNm:
    xs: list[int] = []
    ys: list[int] = []
    for pin in world.pins:
        xs.append(int(pin[3]))
        ys.append(int(pin[4]))
    for obstacle in world.obstacles:
        if _is_device_projection_fact(obstacle):
            continue
        xs.extend((int(obstacle[0]), int(obstacle[2])))
        ys.extend((int(obstacle[1]), int(obstacle[3])))
    for rail in world.rails:
        xs.extend((int(rail[3]), int(rail[5])))
        ys.extend((int(rail[4]), int(rail[6])))
    if not xs or not ys:
        raise RoutingResourceEnumerationError(
            "final routing world has no geometric envelope"
        )
    bbox = (min(xs), min(ys), max(xs), max(ys))
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise RoutingResourceEnumerationError(
            "final routing-world envelope must have positive area"
        )
    return bbox


def _world_obstacle_facts(
    world: RoutingWorldProvenance,
) -> tuple[tuple[Any, ...], ...]:
    rails = tuple(
        (
            int(rail[3]),
            int(rail[4]),
            int(rail[5]),
            int(rail[6]),
            int(rail[1]),
            int(rail[2]),
            str(rail[0]),
            "rail",
        )
        for rail in world.rails
    )
    hard_obstacles = tuple(
        obstacle
        for obstacle in world.obstacles
        if not _is_device_projection_fact(obstacle)
    )
    return tuple(sorted(
        hard_obstacles + rails,
        key=lambda item: json.dumps(item, separators=(",", ":")),
    ))


def _clear_intervals(
    run_bounds: tuple[int, int],
    blocked: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    start, end = run_bounds
    clipped = tuple(sorted(
        (max(start, left), min(end, right))
        for left, right in blocked
        if max(start, left) < min(end, right)
    ))
    merged: list[tuple[int, int]] = []
    for left, right in clipped:
        if not merged or left > merged[-1][1]:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
    clear: list[tuple[int, int]] = []
    cursor = start
    for left, right in merged:
        if cursor < left:
            clear.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        clear.append((cursor, end))
    return tuple(clear)


def _enumerate_track_resources(
    world: RoutingWorldProvenance,
    layers: Sequence[_RoutingLayerFacts],
    owned_conductor_index: Any,
    *,
    max_track_centers: int,
    max_track_segments: int,
    max_track_resources: int,
    geometry_budget: _GeometryEvaluationBudget,
    geometry_index: _LayerGeometryIndex,
) -> tuple[TrackResource, ...]:
    envelope = _world_envelope(world)
    world_digest = world.sha256()
    track_center_count = 0
    track_segment_count = 0
    resources: list[TrackResource] = []
    layer_n_by_key = {layer.key: layer.layer_n for layer in layers}

    for layer in layers:
        half_width = layer.routing_width_nm // 2
        clearance = half_width + layer.min_space_nm
        if layer.preferred_axis == "H":
            run_limits = (envelope[0], envelope[2])
            cross_limits = (envelope[1], envelope[3])
        else:
            run_limits = (envelope[1], envelope[3])
            cross_limits = (envelope[0], envelope[2])
        run_bounds = (run_limits[0] + half_width, run_limits[1] - half_width)
        if run_bounds[1] <= run_bounds[0]:
            continue
        first_center = _ceil_to_multiple(
            cross_limits[0] + half_width, layer.track_step_nm
        )
        last_center = _floor_to_multiple(
            cross_limits[1] - half_width, layer.track_step_nm
        )
        if last_center < first_center:
            continue

        regular_center_count = (
            (last_center - first_center) // layer.track_step_nm + 1
        )
        if track_center_count + regular_center_count > max_track_centers:
            raise RoutingResourceEnumerationLimitError(
                f"track center count exceeds limit {max_track_centers}"
            )
        regular_centers = set(range(
            first_center,
            last_center + 1,
            layer.track_step_nm,
        ))
        centers = set(regular_centers)
        terminal_access_centers: set[int] = set()
        # A global pitch lattice is not by itself a complete access model: a
        # legal device terminal can fall between two tracks even though an
        # adjacent-layer local runner at the terminal's manufacturing-grid
        # phase is available.  Seed that phase from declared pin access facts;
        # all obstacle, spacing, landing, and conflict checks below remain
        # unchanged.  Only upward transport is seeded, so this cannot create a
        # lower-metal route through a higher-metal pin.
        for pin in world.pins:
            source_layer_n = layer_n_by_key.get((int(pin[5]), int(pin[6])))
            if source_layer_n != layer.layer_n - 1:
                continue
            cross_coordinate = (
                int(pin[4])
                if layer.preferred_axis == "H"
                else int(pin[3])
            )
            for access_center in (
                _floor_to_multiple(
                    cross_coordinate, layer.manufacturing_grid_nm
                ),
                _ceil_to_multiple(
                    cross_coordinate, layer.manufacturing_grid_nm
                ),
            ):
                if (
                    cross_limits[0] + half_width
                    <= access_center
                    <= cross_limits[1] - half_width
                ):
                    centers.add(access_center)
                    terminal_access_centers.add(access_center)
        for component in owned_conductor_index.components:
            if not component.pin_accesses or not component.atom_facts:
                continue
            source_layer_n = layer_n_by_key.get(
                component.layer_key
            )
            if source_layer_n != layer.layer_n - 1:
                continue
            for atom in component.atom_facts:
                cross_lo, cross_hi = (
                    (int(atom[1]), int(atom[3]))
                    if layer.preferred_axis == "H"
                    else (int(atom[0]), int(atom[2]))
                )
                for coordinate in (
                    cross_lo,
                    (cross_lo + cross_hi) // 2,
                    cross_hi,
                ):
                    for access_center in (
                        _floor_to_multiple(
                            coordinate, layer.manufacturing_grid_nm
                        ),
                        _ceil_to_multiple(
                            coordinate, layer.manufacturing_grid_nm
                        ),
                    ):
                        if (
                            cross_limits[0] + half_width
                            <= access_center
                            <= cross_limits[1] - half_width
                        ):
                            centers.add(access_center)
                            terminal_access_centers.add(access_center)

        segments_by_interval: dict[tuple[int, int], list[int]] = {}
        for center in sorted(centers):
            track_center_count += 1
            if track_center_count > max_track_centers:
                raise RoutingResourceEnumerationLimitError(
                    f"track center count exceeds limit {max_track_centers}"
                )
            blocked: list[tuple[int, int]] = []
            query_bbox = (
                (
                    run_bounds[0],
                    center - clearance,
                    run_bounds[1],
                    center + clearance,
                )
                if layer.preferred_axis == "H"
                else (
                    center - clearance,
                    run_bounds[0],
                    center + clearance,
                    run_bounds[1],
                )
            )
            for obstacle in geometry_index.query_obstacles(
                layer.key,
                query_bbox,
                geometry_budget,
                "track-obstacle spatial check",
            ):
                if layer.preferred_axis == "H":
                    run_rect = (int(obstacle[0]), int(obstacle[2]))
                    cross_rect = (int(obstacle[1]), int(obstacle[3]))
                else:
                    run_rect = (int(obstacle[1]), int(obstacle[3]))
                    cross_rect = (int(obstacle[0]), int(obstacle[2]))
                if not (
                    cross_rect[0] - clearance
                    < center
                    < cross_rect[1] + clearance
                ):
                    continue
                blocked.append((
                    run_rect[0] - clearance,
                    run_rect[1] + clearance,
                ))
            clear = tuple(
                (snapped_start, snapped_end)
                for left, right in _clear_intervals(run_bounds, blocked)
                for snapped_start, snapped_end in ((
                    _ceil_to_multiple(left, layer.manufacturing_grid_nm),
                    _floor_to_multiple(right, layer.manufacturing_grid_nm),
                ),)
                if snapped_start < snapped_end
            )
            track_segment_count += len(clear)
            if track_segment_count > max_track_segments:
                raise RoutingResourceEnumerationLimitError(
                    f"track segment count exceeds limit {max_track_segments}"
                )
            for interval in clear:
                segments_by_interval.setdefault(interval, []).append(center)
        for interval in sorted(segments_by_interval):
            centers = sorted(segments_by_interval[interval])
            # A TrackResource is one exact through interval plus a finite set
            # of legal starts.  Track centres need not form one arithmetic
            # run: CP choices use explicit indices and every physical-spacing
            # check consumes the actual centre.  Splitting equal-interval
            # terminal phases into one resource per pitch island inflated one
            # local two-pin net into 3,685 resource-triple candidates.  Keep
            # only the semantic distinction (global lattice vs access-only)
            # and expose all same-interval starts in that resource.
            groups = []
            for access_only in (False, True):
                group = [
                    center
                    for center in centers
                    if (
                        center in terminal_access_centers
                        and center not in regular_centers
                    )
                    is access_only
                ]
                if group:
                    groups.append((access_only, group))
            for access_only, group in groups:
                if access_only != all(
                    center in terminal_access_centers
                    and center not in regular_centers
                    for center in group
                ):
                    raise RoutingResourceEnumerationError(
                        "track access-phase partition is inconsistent"
                    )
                if layer.preferred_axis == "H":
                    corridor_bbox = (
                        interval[0] - half_width,
                        group[0] - half_width,
                        interval[1] + half_width,
                        group[-1] + half_width,
                    )
                else:
                    corridor_bbox = (
                        group[0] - half_width,
                        interval[0] - half_width,
                        group[-1] + half_width,
                        interval[1] + half_width,
                    )
                corridor_id = (
                    f"world:{world_digest[:16]}:m{layer.layer_n}:"
                    f"{'terminal-access' if access_only else 'global'}:"
                    f"{layer.preferred_axis}:{interval[0]}:{interval[1]}:"
                    f"{group[0]}:{group[-1]}"
                )
                resources.append(TrackResource(
                    key=CorridorLayerKey(corridor_id, layer.layer_n),
                    preferred_axis=layer.preferred_axis,
                    track_centers_nm=tuple(group),
                    gds_layer_key=layer.key,
                    provenance=(
                        f"final-routing-world:{world_digest}:"
                        f"metal{layer.layer_n}:pitch={layer.track_step_nm}:"
                        f"width={layer.routing_width_nm}:"
                        "terminal-access-phase="
                        f"{'only' if access_only else 'global'}"
                    ),
                    through_interval_nm=interval,
                    corridor_bbox_nm=corridor_bbox,
                    routing_world_provenance_digest=world_digest,
                    min_space_nm=layer.min_space_nm,
                ))
                if len(resources) > max_track_resources:
                    raise RoutingResourceEnumerationLimitError(
                        f"track resource count exceeds limit {max_track_resources}"
                    )

        # A terminal escape is a net-owned access obligation, not neutral
        # routing capacity.  Give the terminal's own layer and its adjacent
        # upper layer a manufacturing-grid-phase resource owned by that net.
        # Same-layer access is required when a PCell has already consumed the
        # cut below its published terminal (for example an internal MIM
        # down-riser): the route must first leave on that real pad, then place a
        # new cut at a free point.  Foreign/unknown conductors remain hard
        # obstacles, and later exact joins still prove every new via.  The owner
        # is carried into the allocation IR, so no foreign net can use this
        # less-conservative terminal throat.
        owned_centers_by_net: dict[str, set[int]] = {}
        for pin in world.pins:
            pin_layer_key = (int(pin[5]), int(pin[6]))
            pin_layer_n = layer_n_by_key.get(pin_layer_key)
            is_adjacent_upper_escape = pin_layer_n == layer.layer_n - 1
            is_same_layer_escape = pin_layer_key == layer.key
            if not (
                is_adjacent_upper_escape or is_same_layer_escape
            ):
                continue
            owner_net = str(pin[0])
            cross_coordinate = (
                int(pin[4]) if layer.preferred_axis == "H" else int(pin[3])
            )
            for access_center in (
                _floor_to_multiple(
                    cross_coordinate, layer.manufacturing_grid_nm
                ),
                _ceil_to_multiple(
                    cross_coordinate, layer.manufacturing_grid_nm
                ),
            ):
                if (
                    cross_limits[0] + half_width
                    <= access_center
                    <= cross_limits[1] - half_width
                ):
                    owned_centers_by_net.setdefault(owner_net, set()).add(
                        access_center
                    )
        # A pin point can sit on a frame edge even though a contained VIA pad
        # exists elsewhere on the same exact owned component.  Mirror the
        # neutral component-atom phase enumeration above into the net-owned
        # adjacent-layer domain.  The later component/landing join, not this
        # broad-phase seed, decides which centers have a legal PDK landing.
        for component in owned_conductor_index.components:
            atom_facts = tuple(component.atom_facts)
            if not atom_facts:
                continue
            source_layer_n = layer_n_by_key.get(component.layer_key)
            if source_layer_n != layer.layer_n - 1:
                continue
            # A declared power rail is itself a structural routing root.  It
            # need not also carry a synthetic pin access before the topology
            # compiler may lift from it.  Keep that extra domain scoped to the
            # exact rail atoms: unrelated atom-only conductors in the same
            # electrical component do not become escape obligations.
            access_atom_facts = (
                atom_facts
                if component.pin_accesses
                else tuple(
                    atom
                    for atom in atom_facts
                    if str(atom[7]).strip().lower() == "rail"
                )
            )
            if not access_atom_facts:
                continue
            owner_net = str(component.net)
            if not owner_net:
                continue
            for atom in access_atom_facts:
                cross_lo, cross_hi = (
                    (int(atom[1]), int(atom[3]))
                    if layer.preferred_axis == "H"
                    else (int(atom[0]), int(atom[2]))
                )
                for coordinate in (
                    cross_lo,
                    (cross_lo + cross_hi) // 2,
                    cross_hi,
                ):
                    for access_center in (
                        _floor_to_multiple(
                            coordinate, layer.manufacturing_grid_nm
                        ),
                        _ceil_to_multiple(
                            coordinate, layer.manufacturing_grid_nm
                        ),
                    ):
                        if (
                            cross_limits[0] + half_width
                            <= access_center
                            <= cross_limits[1] - half_width
                        ):
                            owned_centers_by_net.setdefault(
                                owner_net, set()
                            ).add(access_center)

        for owner_net in sorted(owned_centers_by_net):
            owned_segments: dict[tuple[int, int], list[int]] = {}
            for center in sorted(owned_centers_by_net[owner_net]):
                track_center_count += 1
                if track_center_count > max_track_centers:
                    raise RoutingResourceEnumerationLimitError(
                        f"track center count exceeds limit {max_track_centers}"
                    )
                query_bbox = (
                    (
                        run_bounds[0],
                        center - clearance,
                        run_bounds[1],
                        center + clearance,
                    )
                    if layer.preferred_axis == "H"
                    else (
                        center - clearance,
                        run_bounds[0],
                        center + clearance,
                        run_bounds[1],
                    )
                )
                blocked: list[tuple[int, int]] = []
                for obstacle in geometry_index.query_obstacles(
                    layer.key,
                    query_bbox,
                    geometry_budget,
                    "owned terminal-track obstacle spatial check",
                ):
                    if obstacle[6] is not None and str(obstacle[6]) == owner_net:
                        continue
                    if layer.preferred_axis == "H":
                        run_rect = (int(obstacle[0]), int(obstacle[2]))
                        cross_rect = (int(obstacle[1]), int(obstacle[3]))
                    else:
                        run_rect = (int(obstacle[1]), int(obstacle[3]))
                        cross_rect = (int(obstacle[0]), int(obstacle[2]))
                    if not (
                        cross_rect[0] - clearance
                        < center
                        < cross_rect[1] + clearance
                    ):
                        continue
                    blocked.append((
                        run_rect[0] - clearance,
                        run_rect[1] + clearance,
                    ))
                clear = tuple(
                    (snapped_start, snapped_end)
                    for left, right in _clear_intervals(run_bounds, blocked)
                    for snapped_start, snapped_end in ((
                        _ceil_to_multiple(
                            left, layer.manufacturing_grid_nm
                        ),
                        _floor_to_multiple(
                            right, layer.manufacturing_grid_nm
                        ),
                    ),)
                    if snapped_start < snapped_end
                )
                track_segment_count += len(clear)
                if track_segment_count > max_track_segments:
                    raise RoutingResourceEnumerationLimitError(
                        f"track segment count exceeds limit {max_track_segments}"
                    )
                for interval in clear:
                    owned_segments.setdefault(interval, []).append(center)

            for interval in sorted(owned_segments):
                group = sorted(set(owned_segments[interval]))
                if layer.preferred_axis == "H":
                    corridor_bbox = (
                        interval[0] - half_width,
                        group[0] - half_width,
                        interval[1] + half_width,
                        group[-1] + half_width,
                    )
                else:
                    corridor_bbox = (
                        group[0] - half_width,
                        interval[0] - half_width,
                        group[-1] + half_width,
                        interval[1] + half_width,
                    )
                owner_ref = hashlib.sha256(
                    owner_net.encode("utf-8")
                ).hexdigest()[:16]
                corridor_id = (
                    f"world:{world_digest[:16]}:m{layer.layer_n}:"
                    f"owned-terminal:{owner_ref}:{layer.preferred_axis}:"
                    f"{interval[0]}:{interval[1]}:{group[0]}:{group[-1]}"
                )
                resources.append(TrackResource(
                    key=CorridorLayerKey(corridor_id, layer.layer_n),
                    preferred_axis=layer.preferred_axis,
                    track_centers_nm=tuple(group),
                    gds_layer_key=layer.key,
                    provenance=(
                        f"final-routing-world:{world_digest}:"
                        f"metal{layer.layer_n}:owned-terminal-net={owner_net}:"
                        f"width={layer.routing_width_nm}"
                    ),
                    through_interval_nm=interval,
                    corridor_bbox_nm=corridor_bbox,
                    routing_world_provenance_digest=world_digest,
                    min_space_nm=layer.min_space_nm,
                    owner_net=owner_net,
                ))
                if len(resources) > max_track_resources:
                    raise RoutingResourceEnumerationLimitError(
                        f"track resource count exceeds limit {max_track_resources}"
                    )
    return tuple(sorted(resources, key=lambda resource: resource.key))


def _rects_overlap(left: BBoxNm, right: BBoxNm) -> bool:
    return (
        max(left[0], right[0]) < min(left[2], right[2])
        and max(left[1], right[1]) < min(left[3], right[3])
    )


def _rects_touch_or_overlap(left: BBoxNm, right: BBoxNm) -> bool:
    return (
        max(left[0], right[0]) <= min(left[2], right[2])
        and max(left[1], right[1]) <= min(left[3], right[3])
    )


def _grow_bbox(bbox: BBoxNm, amount_nm: int) -> BBoxNm:
    amount = max(0, int(amount_nm))
    return (
        bbox[0] - amount,
        bbox[1] - amount,
        bbox[2] + amount,
        bbox[3] + amount,
    )


def _bbox_inside(inner: BBoxNm, outer: BBoxNm) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _translate_bbox(bbox: Sequence[int], x_nm: int, y_nm: int) -> BBoxNm:
    return (
        int(bbox[0]) + x_nm,
        int(bbox[1]) + y_nm,
        int(bbox[2]) + x_nm,
        int(bbox[3]) + y_nm,
    )


def _pad_bbox_with_width_floor(
    bbox: Sequence[int],
    *,
    width_floor_nm: int,
    manufacturing_grid_nm: int,
    minimum_area_nm2: int | None = None,
) -> BBoxNm:
    """Expand a via landing to its adjacent metal's width and area floors.

    Via cut/enclosure rules establish a minimum *coverage* rectangle, but that
    rectangle is still drawn metal and independently owes the layer minimum
    width and minimum area. Keep the enclosure rectangle whole and distribute any shortfall on
    its two sides in manufacturing-grid quanta.  The routing-layer width floor
    is already rounded to an even grid multiple, so the common symmetric case
    remains centred on the via; an odd-grid enclosure span gets the sole extra
    quantum deterministically on the high side.
    """

    grid = max(1, int(manufacturing_grid_nm))
    floor = _ceil_to_multiple(max(
        1, int(width_floor_nm), minimum_area_square_side_nm(minimum_area_nm2)
    ), grid)
    left, bottom, right, top = (int(value) for value in bbox)

    def expand(low: int, high: int) -> tuple[int, int]:
        shortfall = max(0, floor - (high - low))
        low_grow = (shortfall // (2 * grid)) * grid
        high_grow = shortfall - low_grow
        return low - low_grow, high + high_grow

    left, right = expand(left, right)
    bottom, top = expand(bottom, top)
    return left, bottom, right, top


def _landing_metal_area(tech: Any, layer_n: int) -> int | None:
    """Use the same canonical alias resolution as the routing PDK digest."""
    return _first_rule_value(
        tech, "min_area", (f"metal{layer_n}", f"met{layer_n}", f"m{layer_n}")
    )


def _point_on_track_resource(point: tuple[int, int], resource: TrackResource) -> bool:
    interval = resource.through_interval_nm
    if interval is None:
        return False
    x, y = point
    centers = resource.track_centers_nm
    coordinate = y if resource.preferred_axis == "H" else x
    center_index = bisect_left(centers, coordinate)
    on_center = (
        center_index < len(centers) and centers[center_index] == coordinate
    )
    if resource.preferred_axis == "H":
        return on_center and interval[0] <= x <= interval[1]
    return on_center and interval[0] <= y <= interval[1]


def _via_spacing_nm(tech: Any, via_name: str) -> int:
    try:
        parsed = tech.drc_rules() or {}
    except Exception:
        parsed = {}
    raw = parsed.get(f"{via_name}_space") if isinstance(parsed, Mapping) else None
    if isinstance(raw, Mapping):
        if raw.get("value_nm") is not None:
            return max(0, int(raw["value_nm"]))
        if raw.get("value_um") is not None:
            return max(0, int(round(float(raw["value_um"]) * 1000.0)))
    try:
        value = tech.via_spacing(via_name)
    except Exception:
        value = None
    if value is not None:
        return max(0, int(value))
    try:
        raw_value = tech.via_rule(via_name).spacing
    except Exception:
        raw_value = None
    return max(0, int(getattr(raw_value, "value", raw_value) or 0))


def _landing_pad_contact_owners(
    pad_bboxes: Sequence[BBoxNm],
    layer_key: LayerKey,
    geometry_budget: _GeometryEvaluationBudget,
    owned_conductor_index: Any,
    *,
    cache: dict[
        tuple[LayerKey, BBoxNm],
        tuple[bool, frozenset[str], frozenset[str]],
    ] | None = None,
) -> tuple[tuple[bool, frozenset[str], frozenset[str]], ...]:
    """Batch exact pad/conductor incidence and publish only complete results."""

    requested = tuple(
        tuple(int(value) for value in bbox) for bbox in pad_bboxes
    )
    contact_cache = {} if cache is None else cache
    missing = tuple(sorted({
        bbox
        for bbox in requested
        if (layer_key, bbox) not in contact_cache
    }))
    contacts = (
        owned_conductor_index.landing_pad_contacts(
            layer_key,
            missing,
            evaluation_consumer=geometry_budget.consume,
        )
        if missing
        else ()
    )
    staged = {}
    for bbox, contact in zip(missing, contacts, strict=True):
        owners = frozenset(contact.nets)
        component_refs = frozenset(
            component_ref
            for net in contact.nets
            for component_ref in contact.refs_for_net(net)
        )
        staged[(layer_key, bbox)] = True, owners, component_refs
    # A typed budget failure above cannot leak a partial proof into the shared
    # query context.  Publish the complete batch atomically.
    contact_cache.update(staged)
    return tuple(contact_cache[(layer_key, bbox)] for bbox in requested)


def _landing_pad_contact_owner(
    pad_bbox: BBoxNm,
    layer_key: LayerKey,
    geometry_budget: _GeometryEvaluationBudget,
    owned_conductor_index: Any,
    *,
    cache: dict[
        tuple[LayerKey, BBoxNm],
        tuple[bool, frozenset[str], frozenset[str]],
    ] | None = None,
) -> tuple[bool, frozenset[str], frozenset[str]]:
    return _landing_pad_contact_owners(
        (pad_bbox,),
        layer_key,
        geometry_budget,
        owned_conductor_index,
        cache=cache,
    )[0]


def _device_interface_pad_bbox(
    pad_bbox: BBoxNm,
    components: Sequence[Any],
    *,
    min_space_nm: int,
    ambiguity_diagnostics: dict[str, int] | None = None,
) -> BBoxNm | None:
    """Compile an immutable device/conductor seam into one landing-pad bbox.

    A via enclosure that only overlaps the end of an off-grid device terminal
    can form a sub-min-width staircase after the two legal rectangles merge.
    The landing option owns that interface: span the pad and the contacted
    device component, then carry the collar one PDK minimum-space beyond the
    component along its transport (long) axis.  Non-device conductors are
    already route-owned and remain byte-identical.
    """

    def is_device_kind(kind: Any) -> bool:
        normalized = str(kind).strip().lower().replace("_", "-")
        return normalized == "device"

    def ambiguous(reason: str) -> None:
        if ambiguity_diagnostics is not None:
            key = f"collar_detail_{reason}"
            ambiguity_diagnostics[key] = ambiguity_diagnostics.get(key, 0) + 1
        return None

    # Component identity is electrical connectivity, not physical seam
    # identity.  A guard-ring atom can touch a remote device atom of the same
    # net and merge both into one component_ref.  Growing to component.bbox_nm
    # would then turn a local via pad into a cell-scale plate.  Seed from device
    # atoms that the raw pad physically touches and stay inside their
    # device-only geometric component; AtomFact[7] preserves the originating
    # conductor kind through component union-find.
    device_atoms: list[tuple[BBoxNm, str | None]] = []
    atom_free_device_touched = False
    for component in components:
        atom_facts = tuple(getattr(component, "atom_facts", ()) or ())
        if atom_facts:
            terminal_refs = tuple(
                getattr(component, "atom_terminal_conductor_refs", ()) or ()
            )
            if len(terminal_refs) != len(atom_facts):
                if any(
                    is_device_kind(fact[7])
                    and _rects_touch_or_overlap(pad_bbox, fact[:4])
                    for fact in atom_facts
                ):
                    return ambiguous("atom_terminal_ref_alignment")
                continue
            device_atoms.extend(
                (
                    tuple(int(value) for value in fact[:4]),
                    terminal_ref,
                )
                for fact, terminal_ref in zip(
                    atom_facts, terminal_refs, strict=True
                )
                if is_device_kind(fact[7])
            )
            continue
        # An atom-free device component has no compiler seam identity.  It may
        # still exist as legacy evidence, but it cannot authorize a collar.
        if any(
            is_device_kind(kind)
            for kind in tuple(getattr(component, "kinds", ()) or ())
        ):
            bbox = tuple(int(value) for value in component.bbox_nm)
            if _rects_touch_or_overlap(pad_bbox, bbox):
                atom_free_device_touched = True

    touched = tuple(
        (bbox, terminal_ref)
        for bbox, terminal_ref in device_atoms
        if _rects_touch_or_overlap(pad_bbox, bbox)
    )
    if not touched:
        if atom_free_device_touched:
            return ambiguous("atom_free_device")
        return tuple(int(value) for value in pad_bbox)  # type: ignore[return-value]
    touched_refs = {terminal_ref for _bbox, terminal_ref in touched}
    if touched_refs == {None}:
        return ambiguous("missing_terminal_ref")
    if None in touched_refs:
        return ambiguous("mixed_terminal_refs")
    if len(touched_refs) != 1:
        return ambiguous("multiple_terminal_refs")
    terminal_ref = next(iter(touched_refs))

    # Follow geometric connectivity only inside the one compiler-proven seam.
    # Same-net touching atoms from another device instance have a different ref
    # and are therefore a hard closure boundary.
    remaining = set(sorted(
        bbox
        for bbox, atom_terminal_ref in device_atoms
        if atom_terminal_ref == terminal_ref
    ))
    frontier = sorted(bbox for bbox, _ref in touched)
    for bbox in frontier:
        remaining.discard(bbox)
    device_bboxes = list(frontier)
    while frontier:
        next_frontier = sorted(
            bbox
            for bbox in remaining
            if any(
                _rects_touch_or_overlap(bbox, reached)
                for reached in frontier
            )
        )
        if not next_frontier:
            break
        for bbox in next_frontier:
            remaining.remove(bbox)
        device_bboxes.extend(next_frontier)
        frontier = next_frontier

    spacing = max(0, int(min_space_nm))
    left, bottom, right, top = (int(value) for value in pad_bbox)
    grow_x = False
    grow_y = False
    for bbox in device_bboxes:
        left = min(left, bbox[0])
        bottom = min(bottom, bbox[1])
        right = max(right, bbox[2])
        top = max(top, bbox[3])
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width > height:
            grow_x = True
        elif height > width:
            grow_y = True
    if grow_x:
        left -= spacing
        right += spacing
    if grow_y:
        bottom -= spacing
        top += spacing
    return left, bottom, right, top


def _device_interface_local_escape_pad_bbox(
    pad_bbox: BBoxNm,
    components: Sequence[Any],
    *,
    min_space_nm: int,
) -> BBoxNm | None:
    """Compile a bounded escape throat from a directly touched long terminal.

    The full interface collar above follows a proven terminal's complete
    touch-connected atom chain.  That remains the primary construction.  A long
    array strap can, however, cross unrelated terminals far from the selected via;
    spanning the whole strap would then consume foreign metal even though a local
    PDK-space-deep throat at the exact contact is legal.  This fallback uses only
    directly touched atoms of one certified terminal, includes their complete
    cross-axis width, and grows solely along their declared long axis.  Square-only
    contacts have no structural transport axis and therefore receive no fallback.
    """

    def is_device_kind(kind: Any) -> bool:
        return str(kind).strip().lower().replace("_", "-") == "device"

    device_atoms: list[tuple[BBoxNm, str | None]] = []
    for component in components:
        atom_facts = tuple(getattr(component, "atom_facts", ()) or ())
        terminal_refs = tuple(
            getattr(component, "atom_terminal_conductor_refs", ()) or ()
        )
        if len(atom_facts) != len(terminal_refs):
            return None
        device_atoms.extend(
            (tuple(int(value) for value in fact[:4]), terminal_ref)
            for fact, terminal_ref in zip(atom_facts, terminal_refs, strict=True)
            if is_device_kind(fact[7])
        )
    touched = tuple(
        (bbox, terminal_ref)
        for bbox, terminal_ref in device_atoms
        if _rects_touch_or_overlap(pad_bbox, bbox)
    )
    touched_refs = {terminal_ref for _bbox, terminal_ref in touched}
    if not touched or None in touched_refs or len(touched_refs) != 1:
        return None

    spacing = max(0, int(min_space_nm))
    left, bottom, right, top = (int(value) for value in pad_bbox)
    grow_x = False
    grow_y = False
    for bbox, _terminal_ref in touched:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width > height:
            bottom = min(bottom, bbox[1])
            top = max(top, bbox[3])
            grow_x = True
        elif height > width:
            left = min(left, bbox[0])
            right = max(right, bbox[2])
            grow_y = True
    if not grow_x and not grow_y:
        return None
    if grow_x:
        left -= spacing
        right += spacing
    if grow_y:
        bottom -= spacing
        top += spacing
    return left, bottom, right, top


def _same_net_component_pair_at_point(
    point: tuple[int, int],
    *,
    pin_net: str | None,
    lower_layer_key: LayerKey,
    upper_layer_key: LayerKey,
    landing_template: Any,
    geometry_budget: _GeometryEvaluationBudget,
    owned_conductor_index: Any,
    require_pad_containment: bool = False,
    landing_contact_cache: dict[
        tuple[LayerKey, BBoxNm],
        tuple[bool, frozenset[str], frozenset[str]],
    ] | None = None,
) -> bool:
    """Prove that one via joins adjacent owned components of the same net."""

    x_nm, y_nm = point
    lower_pad = _translate_bbox(
        tuple(int(value) for value in landing_template.below_metal_rect),
        x_nm,
        y_nm,
    )
    upper_pad = _translate_bbox(
        tuple(int(value) for value in landing_template.above_metal_rect),
        x_nm,
        y_nm,
    )
    _, lower_nets, lower_refs = _landing_pad_contact_owner(
        lower_pad,
        lower_layer_key,
        geometry_budget,
        owned_conductor_index,
        cache=landing_contact_cache,
    )
    _, upper_nets, upper_refs = _landing_pad_contact_owner(
        upper_pad,
        upper_layer_key,
        geometry_budget,
        owned_conductor_index,
        cache=landing_contact_cache,
    )
    common_nets = lower_nets & upper_nets
    if pin_net is not None:
        common_nets &= {pin_net}
    if not lower_refs or not upper_refs or not common_nets:
        return False
    if not require_pad_containment:
        return True

    def enclosing_refs(refs: frozenset[str], pad_bbox: BBoxNm) -> set[str]:
        result = set()
        for component_ref in refs:
            component = owned_conductor_index.component(component_ref)
            if component.net not in common_nets:
                continue
            if any(
                int(atom[0]) <= pad_bbox[0]
                and int(atom[1]) <= pad_bbox[1]
                and pad_bbox[2] <= int(atom[2])
                and pad_bbox[3] <= int(atom[3])
                for atom in component.atom_facts
            ):
                result.add(component_ref)
        return result

    lower_enclosing = enclosing_refs(lower_refs, lower_pad)
    upper_enclosing = enclosing_refs(upper_refs, upper_pad)
    return any(
        owned_conductor_index.component(lower_ref).net
        == owned_conductor_index.component(upper_ref).net
        for lower_ref in lower_enclosing
        for upper_ref in upper_enclosing
    )


def _strict_overlap_center_rect(
    relative_bbox: BBoxNm,
    obstacle_bbox: BBoxNm,
    growth_nm: int,
) -> BBoxNm | None:
    """Integer centers whose translated/grown box strictly overlaps obstacle."""

    growth = max(0, int(growth_nm))
    if (
        int(relative_bbox[0]) - growth
        >= int(relative_bbox[2]) + growth
        or int(relative_bbox[1]) - growth
        >= int(relative_bbox[3]) + growth
        or int(obstacle_bbox[0]) >= int(obstacle_bbox[2])
        or int(obstacle_bbox[1]) >= int(obstacle_bbox[3])
    ):
        return None
    result = (
        int(obstacle_bbox[0]) - int(relative_bbox[2]) - growth + 1,
        int(obstacle_bbox[1]) - int(relative_bbox[3]) - growth + 1,
        int(obstacle_bbox[2]) - int(relative_bbox[0]) + growth - 1,
        int(obstacle_bbox[3]) - int(relative_bbox[1]) + growth - 1,
    )
    return result if result[0] <= result[2] and result[1] <= result[3] else None


def _closed_touch_center_rect(
    relative_bbox: BBoxNm,
    obstacle_bbox: BBoxNm,
) -> BBoxNm | None:
    """Integer centers whose translated box closed-touches the obstacle."""

    result = (
        int(obstacle_bbox[0]) - int(relative_bbox[2]),
        int(obstacle_bbox[1]) - int(relative_bbox[3]),
        int(obstacle_bbox[2]) - int(relative_bbox[0]),
        int(obstacle_bbox[3]) - int(relative_bbox[1]),
    )
    return result if result[0] <= result[2] and result[1] <= result[3] else None


def _inclusive_rect_difference(
    outer: BBoxNm,
    excluded: BBoxNm,
) -> tuple[BBoxNm, ...]:
    """Return a disjoint decomposition of ``outer - excluded`` on integer points."""

    intersection = (
        max(outer[0], excluded[0]),
        max(outer[1], excluded[1]),
        min(outer[2], excluded[2]),
        min(outer[3], excluded[3]),
    )
    if intersection[0] > intersection[2] or intersection[1] > intersection[3]:
        return (outer,)
    candidates = (
        (outer[0], outer[1], intersection[0] - 1, outer[3]),
        (intersection[2] + 1, outer[1], outer[2], outer[3]),
        (
            intersection[0],
            outer[1],
            intersection[2],
            intersection[1] - 1,
        ),
        (
            intersection[0],
            intersection[3] + 1,
            intersection[2],
            outer[3],
        ),
    )
    return tuple(
        rect
        for rect in candidates
        if rect[0] <= rect[2] and rect[1] <= rect[3]
    )


def _points_covered_by_inclusive_rects(
    points: Sequence[tuple[int, int]],
    rectangles: Sequence[BBoxNm],
) -> frozenset[tuple[int, int]]:
    """Classify point/rectangle incidence by one offline x sweep.

    Rectangle bounds are inclusive integer coordinates.  Range updates over the
    compressed candidate y coordinates and point queries make the sweep
    ``O((rectangles + points) log points)`` without a geometry query per point.
    """

    canonical_points = tuple(sorted({
        (int(point[0]), int(point[1])) for point in points
    }))
    if not canonical_points or not rectangles:
        return frozenset()
    ys = tuple(sorted({point[1] for point in canonical_points}))
    min_x = canonical_points[0][0]
    max_x = canonical_points[-1][0]
    events: list[tuple[int, int, int, int]] = []
    for rectangle in rectangles:
        x0, y0, x1, y1 = (int(value) for value in rectangle)
        if x1 < min_x or max_x < x0:
            continue
        first_y = bisect_left(ys, y0)
        after_y = bisect_right(ys, y1)
        if first_y >= after_y:
            continue
        start_x = max(x0, min_x)
        events.append((start_x, 1, first_y, after_y))
        if x1 < max_x:
            events.append((x1 + 1, -1, first_y, after_y))
    if not events:
        return frozenset()
    events.sort()

    # Fenwick tree over the difference array supports range-add/point-query.
    fenwick = [0] * (len(ys) + 2)

    def add(index: int, delta: int) -> None:
        fenwick_index = index + 1
        while fenwick_index < len(fenwick):
            fenwick[fenwick_index] += delta
            fenwick_index += fenwick_index & -fenwick_index

    def count(index: int) -> int:
        result = 0
        fenwick_index = index + 1
        while fenwick_index > 0:
            result += fenwick[fenwick_index]
            fenwick_index -= fenwick_index & -fenwick_index
        return result

    blocked: set[tuple[int, int]] = set()
    event_index = 0
    y_index = {value: index for index, value in enumerate(ys)}
    for point in canonical_points:
        while event_index < len(events) and events[event_index][0] <= point[0]:
            _, delta, first_y, after_y = events[event_index]
            add(first_y, delta)
            add(after_y, -delta)
            event_index += 1
        if count(y_index[point[1]]) > 0:
            blocked.add(point)
    return frozenset(blocked)


def _landing_obstacle_invalid_center_rects(
    *,
    cut_relative_bbox: BBoxNm,
    cut_layer_key: LayerKey,
    cut_spacing_nm: int,
    pad_specs: Sequence[tuple[BBoxNm, LayerKey, int]],
    geometry_index: _LayerGeometryIndex,
    geometry_budget: _GeometryEvaluationBudget,
    diagnostic_facts: list[tuple[BBoxNm, str]] | None = None,
    classification_cache: dict[
        tuple[str, BBoxNm, LayerKey, int],
        tuple[tuple[BBoxNm, str], ...],
    ] | None = None,
) -> tuple[BBoxNm, ...]:
    """Compile immutable obstacle geometry into forbidden via-centre rectangles."""

    invalid_center_rects: list[BBoxNm] = []

    def obstacle_label(rule: str, layer_key: LayerKey, obstacle: Any) -> str:
        owner = obstacle[6]
        owner_text = (
            str(owner) if owner is not None and str(owner) else "unowned"
        )
        kind = obstacle[7] if len(obstacle) > 7 else None
        kind_text = str(kind) if kind is not None and str(kind) else "unknown"
        return (
            f"{rule}:L{int(layer_key[0])}.{int(layer_key[1])}:"
            f"owner={owner_text}:kind={kind_text}"
        )

    def append(rectangle: BBoxNm, label: str) -> None:
        invalid_center_rects.append(rectangle)
        if diagnostic_facts is not None:
            diagnostic_facts.append((rectangle, label))

    cut_key = (
        "cut",
        tuple(int(value) for value in cut_relative_bbox),
        tuple(int(value) for value in cut_layer_key),
        max(0, int(cut_spacing_nm)),
    )
    cut_facts = (
        None if classification_cache is None else classification_cache.get(cut_key)
    )
    if cut_facts is None:
        staged_cut_facts: list[tuple[BBoxNm, str]] = []
        for obstacle in geometry_index.obstacles_by_layer.get(cut_layer_key, ()):
            geometry_budget.consume("landing batch cut-obstacle classification")
            obstacle_bbox = tuple(int(value) for value in obstacle[:4])
            invalid = _strict_overlap_center_rect(
                cut_relative_bbox,
                obstacle_bbox,
                cut_spacing_nm,
            )
            if invalid is not None:
                staged_cut_facts.append((
                    invalid,
                    obstacle_label("cut_spacing", cut_layer_key, obstacle),
                ))
        cut_facts = tuple(staged_cut_facts)
        if classification_cache is not None:
            # A typed budget stop above cannot publish a partial layer scan.
            classification_cache[cut_key] = cut_facts
    for rectangle, label in cut_facts:
        append(rectangle, label)

    for pad_relative_bbox, layer_key, min_space_nm in pad_specs:
        spacing = max(0, int(min_space_nm))
        pad_key = (
            "pad",
            tuple(int(value) for value in pad_relative_bbox),
            tuple(int(value) for value in layer_key),
            spacing,
        )
        pad_facts = (
            None
            if classification_cache is None
            else classification_cache.get(pad_key)
        )
        if pad_facts is None:
            staged_pad_facts: list[tuple[BBoxNm, str]] = []
            for obstacle in geometry_index.obstacles_by_layer.get(layer_key, ()):
                geometry_budget.consume(
                    "landing batch pad-obstacle classification"
                )
                obstacle_bbox = tuple(int(value) for value in obstacle[:4])
                touch = _closed_touch_center_rect(
                    pad_relative_bbox,
                    obstacle_bbox,
                )
                owner = obstacle[6]
                is_owned = owner is not None and bool(str(owner))
                if not is_owned and touch is not None:
                    staged_pad_facts.append((
                        touch,
                        obstacle_label("pad_touch", layer_key, obstacle),
                    ))
                if spacing <= 0:
                    continue
                keepout_overlap = _strict_overlap_center_rect(
                    pad_relative_bbox,
                    obstacle_bbox,
                    spacing,
                )
                if keepout_overlap is None:
                    continue
                if is_owned and touch is not None:
                    # The same obstacle's owned physical contact waives spacing.
                    # Any different obstacle contributes its own rectangles and
                    # can therefore still reject this point.
                    for rectangle in _inclusive_rect_difference(
                        keepout_overlap, touch
                    ):
                        staged_pad_facts.append((
                            rectangle,
                            obstacle_label(
                                "pad_spacing", layer_key, obstacle
                            ),
                        ))
                else:
                    staged_pad_facts.append((
                        keepout_overlap,
                        obstacle_label("pad_spacing", layer_key, obstacle),
                    ))
            pad_facts = tuple(staged_pad_facts)
            if classification_cache is not None:
                # Publish only after the complete signed layer scan succeeds.
                classification_cache[pad_key] = pad_facts
        for rectangle, label in pad_facts:
            append(rectangle, label)

    return tuple(invalid_center_rects)


def _landing_batch_prefers_spatial_queries(
    point_count: int,
    *,
    cut_layer_key: LayerKey,
    pad_specs: Sequence[tuple[BBoxNm, LayerKey, int]],
    geometry_index: _LayerGeometryIndex,
) -> bool:
    """Choose between two exact classifiers from their finite input sizes."""

    global_scan_count = len(
        geometry_index.obstacles_by_layer.get(cut_layer_key, ())
    ) + sum(
        len(geometry_index.obstacles_by_layer.get(layer_key, ()))
        for _pad_bbox, layer_key, _spacing in pad_specs
    )
    spatial_probe_scale = (
        max(0, int(point_count))
        * (1 + len(tuple(pad_specs)))
        * _PACKED_GEOMETRY_NODE_CAPACITY
    )
    return global_scan_count > spatial_probe_scale


def _point_in_inclusive_bbox(point: tuple[int, int], bbox: BBoxNm) -> bool:
    return (
        bbox[0] <= point[0] <= bbox[2]
        and bbox[1] <= point[1] <= bbox[3]
    )


def _batch_landing_obstacle_blocked_points_spatial(
    points: Sequence[tuple[int, int]],
    *,
    cut_relative_bbox: BBoxNm,
    cut_layer_key: LayerKey,
    cut_spacing_nm: int,
    pad_specs: Sequence[tuple[BBoxNm, LayerKey, int]],
    geometry_index: _LayerGeometryIndex,
    geometry_budget: _GeometryEvaluationBudget,
) -> frozenset[tuple[int, int]]:
    """Classify a sparse exact point set through the immutable obstacle STR."""

    blocked: set[tuple[int, int]] = set()
    for point in tuple(sorted({
        (int(raw[0]), int(raw[1])) for raw in points
    })):
        cut_bbox = _translate_bbox(cut_relative_bbox, *point)
        cut_space = max(0, int(cut_spacing_nm))
        cut_query_bbox = (
            cut_bbox[0] - cut_space,
            cut_bbox[1] - cut_space,
            cut_bbox[2] + cut_space,
            cut_bbox[3] + cut_space,
        )
        for obstacle in geometry_index.query_obstacles(
            cut_layer_key,
            cut_query_bbox,
            geometry_budget,
            "landing sparse cut-obstacle spatial classification",
        ):
            invalid = _strict_overlap_center_rect(
                cut_relative_bbox,
                tuple(int(value) for value in obstacle[:4]),
                cut_space,
            )
            if invalid is not None and _point_in_inclusive_bbox(point, invalid):
                blocked.add(point)
                break
        if point in blocked:
            continue

        for pad_relative_bbox, layer_key, min_space_nm in pad_specs:
            spacing = max(0, int(min_space_nm))
            pad_bbox = _translate_bbox(pad_relative_bbox, *point)
            pad_query_bbox = (
                pad_bbox[0] - spacing,
                pad_bbox[1] - spacing,
                pad_bbox[2] + spacing,
                pad_bbox[3] + spacing,
            )
            for obstacle in geometry_index.query_obstacles(
                layer_key,
                pad_query_bbox,
                geometry_budget,
                "landing sparse pad-obstacle spatial classification",
            ):
                obstacle_bbox = tuple(int(value) for value in obstacle[:4])
                touch_bbox = _closed_touch_center_rect(
                    pad_relative_bbox, obstacle_bbox
                )
                touches = (
                    touch_bbox is not None
                    and _point_in_inclusive_bbox(point, touch_bbox)
                )
                owner = obstacle[6]
                is_owned = owner is not None and bool(str(owner))
                if touches and not is_owned:
                    blocked.add(point)
                    break
                keepout_bbox = _strict_overlap_center_rect(
                    pad_relative_bbox, obstacle_bbox, spacing
                )
                violates_spacing = (
                    keepout_bbox is not None
                    and _point_in_inclusive_bbox(point, keepout_bbox)
                )
                if violates_spacing and not (is_owned and touches):
                    blocked.add(point)
                    break
            if point in blocked:
                break
    return frozenset(blocked)


def _batch_landing_obstacle_blocked_points(
    points: Sequence[tuple[int, int]],
    *,
    cut_relative_bbox: BBoxNm,
    cut_layer_key: LayerKey,
    cut_spacing_nm: int,
    pad_specs: Sequence[tuple[BBoxNm, LayerKey, int]],
    geometry_index: _LayerGeometryIndex,
    geometry_budget: _GeometryEvaluationBudget,
) -> frozenset[tuple[int, int]]:
    """Return points rejected by cut/pad obstacle rules in one batch sweep."""

    canonical_points = tuple(sorted({
        (int(point[0]), int(point[1])) for point in points
    }))
    if not canonical_points:
        return frozenset()
    if _landing_batch_prefers_spatial_queries(
        len(canonical_points),
        cut_layer_key=cut_layer_key,
        pad_specs=pad_specs,
        geometry_index=geometry_index,
    ):
        return _batch_landing_obstacle_blocked_points_spatial(
            canonical_points,
            cut_relative_bbox=cut_relative_bbox,
            cut_layer_key=cut_layer_key,
            cut_spacing_nm=cut_spacing_nm,
            pad_specs=pad_specs,
            geometry_index=geometry_index,
            geometry_budget=geometry_budget,
        )
    invalid_center_rects = _landing_obstacle_invalid_center_rects(
        cut_relative_bbox=cut_relative_bbox,
        cut_layer_key=cut_layer_key,
        cut_spacing_nm=cut_spacing_nm,
        pad_specs=pad_specs,
        geometry_index=geometry_index,
        geometry_budget=geometry_budget,
    )

    for _ in canonical_points:
        geometry_budget.consume("landing batch candidate classification")
    return _points_covered_by_inclusive_rects(
        canonical_points,
        invalid_center_rects,
    )


def _iter_track_intersection_points(
    lower: TrackResource,
    upper: TrackResource,
    lower_step_nm: int,
    upper_step_nm: int,
):
    lower_interval = lower.through_interval_nm
    upper_interval = upper.through_interval_nm
    if lower_interval is None or upper_interval is None:
        return
    if lower.preferred_axis != upper.preferred_axis:
        horizontal = lower if lower.preferred_axis == "H" else upper
        vertical = upper if upper.preferred_axis == "V" else lower
        horizontal_interval = horizontal.through_interval_nm
        vertical_interval = vertical.through_interval_nm
        if horizontal_interval is None or vertical_interval is None:
            return
        for x in vertical.track_centers_nm:
            if not horizontal_interval[0] <= x <= horizontal_interval[1]:
                continue
            for y in horizontal.track_centers_nm:
                if vertical_interval[0] <= y <= vertical_interval[1]:
                    yield x, y
        return

    common_centers = sorted(
        set(lower.track_centers_nm) & set(upper.track_centers_nm)
    )
    overlap = (
        max(lower_interval[0], upper_interval[0]),
        min(lower_interval[1], upper_interval[1]),
    )
    if overlap[1] < overlap[0]:
        return
    common_step = math.lcm(lower_step_nm, upper_step_nm)
    first = _ceil_to_multiple(overlap[0], common_step)
    last = _floor_to_multiple(overlap[1], common_step)
    for center in common_centers:
        coordinate = first
        while coordinate <= last:
            yield (
                (coordinate, center)
                if lower.preferred_axis == "H"
                else (center, coordinate)
            )
            coordinate += common_step


def _iter_owned_conductor_track_points(
    conductor_bbox: BBoxNm,
    conductor_is_lower: bool,
    opposite_tracks: Sequence[TrackResource],
    landing_template: Any,
    via_pitch_nm: int,
    manufacturing_grid_nm: int,
    min_contact_width_nm: int,
    geometry_budget: _GeometryEvaluationBudget,
):
    """Enumerate lane vias whose emitted pad contacts an owned conductor.

    The via pad is newly emitted metal and supplies its own cut enclosure.  It
    therefore need not be contained by the pre-existing conductor; a positive
    overlap with a PDK-min-width access throat is sufficient electrical
    contact.  The full landing materializer still rejects every foreign/unowned
    pad contact and all cut/pad spacing violations.
    """

    pad = (
        landing_template.below_metal_rect
        if conductor_is_lower
        else landing_template.above_metal_rect
    )
    feasible_regions = _owned_access_feasible_regions(
        conductor_bbox,
        pad,
        manufacturing_grid_nm,
    )
    if not feasible_regions:
        return
    contained_region = _owned_contained_access_region(conductor_bbox, pad)
    emitted: set[tuple[int, int]] = set()
    for resource in opposite_tracks:
        geometry_budget.consume("owned-conductor track intersection")
        interval = resource.through_interval_nm
        if interval is None:
            continue
        for feasible in feasible_regions:
            if resource.preferred_axis == "H":
                base_run = (
                    max(interval[0], feasible[0]),
                    min(interval[1], feasible[2]),
                )
                if base_run[1] < base_run[0]:
                    continue
                for y in resource.track_centers_nm:
                    if feasible[1] <= y <= feasible[3]:
                        run = base_run
                        if feasible != contained_region:
                            run = _owned_extension_throat_run(
                                preferred_axis="H",
                                cross_coordinate=y,
                                run=run,
                                conductor_bbox=conductor_bbox,
                                relative_pad_bbox=pad,
                                min_contact_width_nm=min_contact_width_nm,
                                manufacturing_grid_nm=manufacturing_grid_nm,
                            )
                            if run is None:
                                continue
                        first = _ceil_to_multiple(
                            run[0], manufacturing_grid_nm
                        )
                        last = _floor_to_multiple(
                            run[1], manufacturing_grid_nm
                        )
                        x = first
                        while x <= last:
                            point = (x, y)
                            if (
                                point not in emitted
                                and _pad_contact_has_minimum_throat(
                                    point,
                                    conductor_bbox,
                                    pad,
                                    min_contact_width_nm,
                                    manufacturing_grid_nm,
                                )
                            ):
                                emitted.add(point)
                                yield point
                            x += via_pitch_nm
            else:
                base_run = (
                    max(interval[0], feasible[1]),
                    min(interval[1], feasible[3]),
                )
                if base_run[1] < base_run[0]:
                    continue
                for x in resource.track_centers_nm:
                    if feasible[0] <= x <= feasible[2]:
                        run = base_run
                        if feasible != contained_region:
                            run = _owned_extension_throat_run(
                                preferred_axis="V",
                                cross_coordinate=x,
                                run=run,
                                conductor_bbox=conductor_bbox,
                                relative_pad_bbox=pad,
                                min_contact_width_nm=min_contact_width_nm,
                                manufacturing_grid_nm=manufacturing_grid_nm,
                            )
                            if run is None:
                                continue
                        first = _ceil_to_multiple(
                            run[0], manufacturing_grid_nm
                        )
                        last = _floor_to_multiple(
                            run[1], manufacturing_grid_nm
                        )
                        y = first
                        while y <= last:
                            point = (x, y)
                            if (
                                point not in emitted
                                and _pad_contact_has_minimum_throat(
                                    point,
                                    conductor_bbox,
                                    pad,
                                    min_contact_width_nm,
                                    manufacturing_grid_nm,
                                )
                            ):
                                emitted.add(point)
                                yield point
                            y += via_pitch_nm


def _owned_access_feasible_regions(
    conductor_bbox: BBoxNm,
    relative_pad_bbox: Sequence[int],
    manufacturing_grid_nm: int,
) -> tuple[BBoxNm, ...]:
    """Return legacy-contained then extended-contact center domains."""

    pad = tuple(int(value) for value in relative_pad_bbox)
    contained = _owned_contained_access_region(conductor_bbox, pad)
    contact = (
        conductor_bbox[0] - pad[2] + manufacturing_grid_nm,
        conductor_bbox[1] - pad[3] + manufacturing_grid_nm,
        conductor_bbox[2] - pad[0] - manufacturing_grid_nm,
        conductor_bbox[3] - pad[1] - manufacturing_grid_nm,
    )
    return tuple(
        region
        for region in dict.fromkeys((contained, contact))
        if region[2] >= region[0] and region[3] >= region[1]
    )


def _owned_contained_access_region(
    conductor_bbox: BBoxNm,
    relative_pad_bbox: Sequence[int],
) -> BBoxNm:
    pad = tuple(int(value) for value in relative_pad_bbox)
    return (
        conductor_bbox[0] - pad[0],
        conductor_bbox[1] - pad[1],
        conductor_bbox[2] - pad[2],
        conductor_bbox[3] - pad[3],
    )


def _owned_extension_throat_run(
    *,
    preferred_axis: str,
    cross_coordinate: int,
    run: tuple[int, int],
    conductor_bbox: BBoxNm,
    relative_pad_bbox: Sequence[int],
    min_contact_width_nm: int,
    manufacturing_grid_nm: int,
) -> tuple[int, int] | None:
    """Clip an extension phase to its first PDK-width-valid run interval."""

    pad = tuple(int(value) for value in relative_pad_bbox)
    if preferred_axis == "H":
        cross_overlap = min(
            conductor_bbox[3], cross_coordinate + pad[3]
        ) - max(conductor_bbox[1], cross_coordinate + pad[1])
        if cross_overlap < manufacturing_grid_nm:
            return None
        required_run_overlap = (
            manufacturing_grid_nm
            if cross_overlap >= min_contact_width_nm
            else min_contact_width_nm
        )
        throat_run = (
            conductor_bbox[0] + required_run_overlap - pad[2],
            conductor_bbox[2] - required_run_overlap - pad[0],
        )
    elif preferred_axis == "V":
        cross_overlap = min(
            conductor_bbox[2], cross_coordinate + pad[2]
        ) - max(conductor_bbox[0], cross_coordinate + pad[0])
        if cross_overlap < manufacturing_grid_nm:
            return None
        required_run_overlap = (
            manufacturing_grid_nm
            if cross_overlap >= min_contact_width_nm
            else min_contact_width_nm
        )
        throat_run = (
            conductor_bbox[1] + required_run_overlap - pad[3],
            conductor_bbox[3] - required_run_overlap - pad[1],
        )
    else:
        raise RoutingResourceEnumerationError(
            f"owned access track axis {preferred_axis!r} is invalid"
        )
    clipped = max(run[0], throat_run[0]), min(run[1], throat_run[1])
    return clipped if clipped[1] >= clipped[0] else None


def _pad_contact_has_minimum_throat(
    point: tuple[int, int],
    conductor_bbox: BBoxNm,
    relative_pad_bbox: Sequence[int],
    min_contact_width_nm: int,
    manufacturing_grid_nm: int,
) -> bool:
    """Prove contained access or a PDK-min-width path through the union."""

    pad_bbox = _translate_bbox(relative_pad_bbox, point[0], point[1])
    if _bbox_inside(pad_bbox, conductor_bbox):
        return True
    overlap_x = min(conductor_bbox[2], pad_bbox[2]) - max(
        conductor_bbox[0], pad_bbox[0]
    )
    overlap_y = min(conductor_bbox[3], pad_bbox[3]) - max(
        conductor_bbox[1], pad_bbox[1]
    )
    return (
        overlap_x >= manufacturing_grid_nm
        and overlap_y >= manufacturing_grid_nm
        and max(overlap_x, overlap_y) >= min_contact_width_nm
    )


def _iter_owned_component_pair_points(
    lower_component: Any,
    upper_component: Any,
    landing_template: Any,
    via_pitch_nm: int,
    manufacturing_grid_nm: int,
    geometry_budget: _GeometryEvaluationBudget,
):
    """Enumerate rule-pitched via centers enclosed by both owned components."""

    if lower_component.net != upper_component.net:
        return
    lower_pad = tuple(
        int(value) for value in landing_template.below_metal_rect
    )
    upper_pad = tuple(
        int(value) for value in landing_template.above_metal_rect
    )
    for lower_atom in lower_component.atom_facts:
        lower_feasible = (
            int(lower_atom[0]) - lower_pad[0],
            int(lower_atom[1]) - lower_pad[1],
            int(lower_atom[2]) - lower_pad[2],
            int(lower_atom[3]) - lower_pad[3],
        )
        if (
            lower_feasible[2] < lower_feasible[0]
            or lower_feasible[3] < lower_feasible[1]
        ):
            continue
        for upper_atom in upper_component.atom_facts:
            geometry_budget.consume("owned component-pair landing overlap")
            upper_feasible = (
                int(upper_atom[0]) - upper_pad[0],
                int(upper_atom[1]) - upper_pad[1],
                int(upper_atom[2]) - upper_pad[2],
                int(upper_atom[3]) - upper_pad[3],
            )
            feasible = (
                max(lower_feasible[0], upper_feasible[0]),
                max(lower_feasible[1], upper_feasible[1]),
                min(lower_feasible[2], upper_feasible[2]),
                min(lower_feasible[3], upper_feasible[3]),
            )
            if feasible[2] < feasible[0] or feasible[3] < feasible[1]:
                continue
            # A global pitch anchor makes overlapping atom-pair enumeration
            # identical and exactly replayable independent of iteration order.
            first_x = _ceil_to_multiple(feasible[0], via_pitch_nm)
            first_y = _ceil_to_multiple(feasible[1], via_pitch_nm)
            last_x = _floor_to_multiple(feasible[2], via_pitch_nm)
            last_y = _floor_to_multiple(feasible[3], via_pitch_nm)
            x_nm = first_x
            while x_nm <= last_x:
                y_nm = first_y
                while y_nm <= last_y:
                    if (
                        x_nm % manufacturing_grid_nm == 0
                        and y_nm % manufacturing_grid_nm == 0
                    ):
                        yield x_nm, y_nm
                    y_nm += via_pitch_nm
                x_nm += via_pitch_nm


def _owned_conductor_track_contains_point(
    point: tuple[int, int],
    conductor_bbox: BBoxNm,
    conductor_is_lower: bool,
    opposite_tracks: Sequence[TrackResource],
    landing_template: Any,
    via_pitch_nm: int,
    manufacturing_grid_nm: int,
    min_contact_width_nm: int,
    geometry_budget: _GeometryEvaluationBudget,
) -> bool:
    """Check exact owned pad-contact lattice membership without enumerating it."""

    pad = (
        landing_template.below_metal_rect
        if conductor_is_lower
        else landing_template.above_metal_rect
    )
    feasible_regions = _owned_access_feasible_regions(
        conductor_bbox,
        pad,
        manufacturing_grid_nm,
    )
    if not feasible_regions:
        return False
    contained_region = _owned_contained_access_region(conductor_bbox, pad)
    x, y = point
    if not _pad_contact_has_minimum_throat(
        point,
        conductor_bbox,
        pad,
        min_contact_width_nm,
        manufacturing_grid_nm,
    ):
        return False
    for resource in opposite_tracks:
        geometry_budget.consume("owned-conductor track membership")
        interval = resource.through_interval_nm
        if interval is None:
            continue
        centers = resource.track_centers_nm
        cross_coordinate = y if resource.preferred_axis == "H" else x
        center_index = bisect_left(centers, cross_coordinate)
        if (
            center_index >= len(centers)
            or centers[center_index] != cross_coordinate
        ):
            continue
        for feasible in feasible_regions:
            if resource.preferred_axis == "H":
                if not feasible[1] <= y <= feasible[3]:
                    continue
                run = (
                    max(interval[0], feasible[0]),
                    min(interval[1], feasible[2]),
                )
                run_coordinate = x
                if feasible != contained_region:
                    run = _owned_extension_throat_run(
                        preferred_axis="H",
                        cross_coordinate=y,
                        run=run,
                        conductor_bbox=conductor_bbox,
                        relative_pad_bbox=pad,
                        min_contact_width_nm=min_contact_width_nm,
                        manufacturing_grid_nm=manufacturing_grid_nm,
                    )
            else:
                if not feasible[0] <= x <= feasible[2]:
                    continue
                run = (
                    max(interval[0], feasible[1]),
                    min(interval[1], feasible[3]),
                )
                run_coordinate = y
                if feasible != contained_region:
                    run = _owned_extension_throat_run(
                        preferred_axis="V",
                        cross_coordinate=x,
                        run=run,
                        conductor_bbox=conductor_bbox,
                        relative_pad_bbox=pad,
                        min_contact_width_nm=min_contact_width_nm,
                        manufacturing_grid_nm=manufacturing_grid_nm,
                    )
            if run is None or run[1] < run[0]:
                continue
            first = _ceil_to_multiple(run[0], manufacturing_grid_nm)
            last = _floor_to_multiple(run[1], manufacturing_grid_nm)
            if (
                first <= run_coordinate <= last
                and (run_coordinate - first) % via_pitch_nm == 0
            ):
                return True
    return False


def _landing_site_domain_digest(
    world_digest: str,
    tracks: Sequence[TrackResource],
    pdk_digest: str,
    owned_conductor_index_digest: str,
) -> str:
    """Bind the exact lazy landing query to its immutable geometric basis."""

    payload = {
        "owned_conductor_index_digest": owned_conductor_index_digest,
        "pdk_provenance_digest": pdk_digest,
        "routing_world_provenance_digest": world_digest,
        "schema": "clay.exact-landing-site-domain.v3",
        "track_resources": [
            {
                "corridor_id": resource.key.corridor_id,
                "gds_layer_key": list(resource.gds_layer_key or ()),
                "layer_n": resource.key.layer_n,
                "preferred_axis": resource.preferred_axis,
                "through_interval_nm": list(resource.through_interval_nm or ()),
                "track_centers_nm": list(resource.track_centers_nm),
            }
            for resource in sorted(tracks, key=lambda item: item.key)
        ],
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _materialize_landing_points(
    points: Sequence[tuple[int, int]],
    *,
    world: RoutingWorldProvenance,
    tech: Any,
    lower_layer: _RoutingLayerFacts,
    upper_layer: _RoutingLayerFacts,
    geometry_index: _LayerGeometryIndex,
    owned_conductor_index: Any,
    geometry_budget: _GeometryEvaluationBudget,
    manufacturing_grid_nm: int | None = None,
    landing_template_override: Any | None = None,
    via_key_override: LayerKey | None = None,
    via_space_nm_override: int | None = None,
    blocker_center_rects: Sequence[BBoxNm] | None = None,
    diagnostics: dict[str, int] | None = None,
    rejection_by_point: dict[tuple[int, int], str] | None = None,
    landing_contact_cache: dict[
        tuple[LayerKey, BBoxNm],
        tuple[bool, frozenset[str], frozenset[str]],
    ] | None = None,
    routing_world_digest_override: str | None = None,
    world_envelope_override: BBoxNm | None = None,
) -> tuple[LandingSite, ...]:
    """Apply the full PDK/obstacle/ownership contract to requested coordinates."""

    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    from clay.layout.route.via_model import via_landing

    lower_layer_n = lower_layer.layer_n
    via_name = f"via{lower_layer_n}"
    via_key = (
        _layer_key(tech.via_key(lower_layer_n))
        if via_key_override is None
        else _layer_key(via_key_override)
    )
    via_space_nm = (
        _via_spacing_nm(tech, via_name)
        if via_space_nm_override is None
        else max(0, int(via_space_nm_override))
    )
    landing_template = (
        via_landing(tech, via_name, 0, 0)
        if landing_template_override is None
        else landing_template_override
    )
    if landing_template is None:
        return ()
    if manufacturing_grid_nm is None:
        manufacturing_grid_nm = pdk_manufacturing_grid_nm(tech)
    lower_pad_template = _pad_bbox_with_width_floor(
        landing_template.below_metal_rect,
        width_floor_nm=lower_layer.routing_width_nm,
        manufacturing_grid_nm=manufacturing_grid_nm,
        minimum_area_nm2=_landing_metal_area(tech, lower_layer.layer_n),
    )
    upper_pad_template = _pad_bbox_with_width_floor(
        landing_template.above_metal_rect,
        width_floor_nm=upper_layer.routing_width_nm,
        manufacturing_grid_nm=manufacturing_grid_nm,
        minimum_area_nm2=_landing_metal_area(tech, upper_layer.layer_n),
    )
    if world_envelope_override is None:
        envelope = _world_envelope(world)
    else:
        envelope = tuple(world_envelope_override)
        if (
            len(envelope) != 4
            or any(type(value) is not int for value in envelope)
            or envelope[2] < envelope[0]
            or envelope[3] < envelope[1]
        ):
            raise RoutingResourceEnumerationError(
                "routing-world envelope override must be a valid integer bbox"
            )
    prepared_points: list[
        tuple[tuple[int, int], BBoxNm, BBoxNm, BBoxNm]
    ] = []

    def reject(point: tuple[int, int], reason: str) -> None:
        if diagnostics is not None:
            diagnostics[reason] = diagnostics.get(reason, 0) + 1
        if rejection_by_point is not None:
            prior = rejection_by_point.setdefault(point, reason)
            if prior != reason:
                raise RoutingResourceEnumerationError(
                    "one exact landing point received conflicting rejection reasons"
                )

    for x, y in sorted({(int(x), int(y)) for x, y in points}):
        point = (x, y)
        if x % manufacturing_grid_nm or y % manufacturing_grid_nm:
            reject(point, "off_grid")
            continue
        cut_bbox = _translate_bbox(landing_template.cut, x, y)
        lower_pad = _translate_bbox(lower_pad_template, x, y)
        upper_pad = _translate_bbox(upper_pad_template, x, y)
        if any(
            coordinate % manufacturing_grid_nm
            for bbox in (cut_bbox, lower_pad, upper_pad)
            for coordinate in bbox
        ):
            reject(point, "off_grid")
            continue
        if not all(
            _bbox_inside(bbox, envelope)
            for bbox in (cut_bbox, lower_pad, upper_pad)
        ):
            reject(point, "outside_envelope")
            continue
        prepared_points.append((point, cut_bbox, lower_pad, upper_pad))
    prepared_xy = tuple(point for point, _, _, _ in prepared_points)
    if blocker_center_rects is None:
        blocked_points = _batch_landing_obstacle_blocked_points(
            prepared_xy,
            cut_relative_bbox=tuple(int(value) for value in landing_template.cut),
            cut_layer_key=via_key,
            cut_spacing_nm=via_space_nm,
            pad_specs=(
                (
                    lower_pad_template,
                    lower_layer.key,
                    lower_layer.min_space_nm,
                ),
                (
                    upper_pad_template,
                    upper_layer.key,
                    upper_layer.min_space_nm,
                ),
            ),
            geometry_index=geometry_index,
            geometry_budget=geometry_budget,
        )
    else:
        for _ in prepared_xy:
            geometry_budget.consume("landing cached candidate classification")
        blocked_points = _points_covered_by_inclusive_rects(
            prepared_xy, blocker_center_rects
        )
    if routing_world_digest_override is not None and (
        not isinstance(routing_world_digest_override, str)
        or not routing_world_digest_override
    ):
        raise RoutingResourceEnumerationError(
            "routing-world digest override must be a non-empty string"
        )
    world_digest = (
        world.sha256()
        if routing_world_digest_override is None
        else routing_world_digest_override
    )
    sites: list[LandingSite] = []
    exact_candidates = []
    for prepared in prepared_points:
        point = prepared[0]
        if point in blocked_points:
            reject(point, "obstacle_blocked")
            continue
        exact_candidates.append(prepared)
    lower_contacts = _landing_pad_contact_owners(
        tuple(prepared[2] for prepared in exact_candidates),
        lower_layer.key,
        geometry_budget,
        owned_conductor_index,
        cache=landing_contact_cache,
    )
    upper_contacts = _landing_pad_contact_owners(
        tuple(prepared[3] for prepared in exact_candidates),
        upper_layer.key,
        geometry_budget,
        owned_conductor_index,
        cache=landing_contact_cache,
    )
    for prepared, lower_contact, upper_contact in zip(
        exact_candidates, lower_contacts, upper_contacts, strict=True
    ):
        point, cut_bbox, lower_pad, upper_pad = prepared
        x, y = point
        lower_ok, lower_owners, lower_components = lower_contact
        upper_ok, upper_owners, upper_components = upper_contact
        owners = lower_owners | upper_owners
        if not lower_ok or not upper_ok or len(owners) > 1:
            reject(point, "owner_rejected")
            continue
        owner_net = next(iter(owners)) if owners else None

        # Compile the immutable device/M1 interface into the exact landing
        # option before candidate selection.  Every full or local-escape collar
        # is re-adjudicated against the same PDK obstacle/ownership facts as the
        # raw via template; it may consume neither foreign clearance nor a new net.
        raw_lower_components = frozenset(lower_components)
        raw_upper_components = frozenset(upper_components)
        raw_lower_pad = lower_pad
        raw_upper_pad = upper_pad
        lower_component_objects = tuple(
            owned_conductor_index.component(ref)
            for ref in sorted(raw_lower_components)
        )
        upper_component_objects = tuple(
            owned_conductor_index.component(ref)
            for ref in sorted(raw_upper_components)
        )
        lower_pad = _device_interface_pad_bbox(
            lower_pad,
            lower_component_objects,
            min_space_nm=lower_layer.min_space_nm,
            ambiguity_diagnostics=diagnostics,
        )
        upper_pad = _device_interface_pad_bbox(
            upper_pad,
            upper_component_objects,
            min_space_nm=upper_layer.min_space_nm,
            ambiguity_diagnostics=diagnostics,
        )
        if lower_pad is None or upper_pad is None:
            reject(point, "collar_terminal_ambiguous")
            continue
        collar_variant = "full"
        collar_changed = (
            lower_pad != raw_lower_pad or upper_pad != raw_upper_pad
        )
        if collar_changed:
            def adjudicate_collar(
                candidate_lower_pad: BBoxNm,
                candidate_upper_pad: BBoxNm,
            ) -> tuple[
                str | None,
                tuple[
                    frozenset[str],
                    frozenset[str],
                    frozenset[str],
                    frozenset[str],
                ] | None,
            ]:
                if not all(
                    _bbox_inside(bbox, envelope)
                    for bbox in (cut_bbox, candidate_lower_pad, candidate_upper_pad)
                ):
                    return "collar_outside_envelope", None
                relative_lower_pad = (
                    candidate_lower_pad[0] - x,
                    candidate_lower_pad[1] - y,
                    candidate_lower_pad[2] - x,
                    candidate_lower_pad[3] - y,
                )
                relative_upper_pad = (
                    candidate_upper_pad[0] - x,
                    candidate_upper_pad[1] - y,
                    candidate_upper_pad[2] - x,
                    candidate_upper_pad[3] - y,
                )
                if point in _batch_landing_obstacle_blocked_points(
                    (point,),
                    cut_relative_bbox=tuple(
                        int(value) for value in landing_template.cut
                    ),
                    cut_layer_key=via_key,
                    cut_spacing_nm=via_space_nm,
                    pad_specs=(
                        (
                            relative_lower_pad,
                            lower_layer.key,
                            lower_layer.min_space_nm,
                        ),
                        (
                            relative_upper_pad,
                            upper_layer.key,
                            upper_layer.min_space_nm,
                        ),
                    ),
                    geometry_index=geometry_index,
                    geometry_budget=geometry_budget,
                ):
                    return "collar_obstacle_blocked", None
                _, widened_lower_owners, widened_lower_components = (
                    _landing_pad_contact_owner(
                        candidate_lower_pad,
                        lower_layer.key,
                        geometry_budget,
                        owned_conductor_index,
                        cache=landing_contact_cache,
                    )
                )
                _, widened_upper_owners, widened_upper_components = (
                    _landing_pad_contact_owner(
                        candidate_upper_pad,
                        upper_layer.key,
                        geometry_budget,
                        owned_conductor_index,
                        cache=landing_contact_cache,
                    )
                )
                widened_owners = widened_lower_owners | widened_upper_owners
                if widened_owners != owners or len(widened_owners) > 1:
                    return "collar_owner_changed", None
                return None, (
                    widened_lower_owners,
                    widened_upper_owners,
                    frozenset(widened_lower_components),
                    frozenset(widened_upper_components),
                )

            collar_reason, collar_contact = adjudicate_collar(
                lower_pad, upper_pad
            )
            if collar_reason is not None:
                local_lower_pad = _device_interface_local_escape_pad_bbox(
                    raw_lower_pad,
                    lower_component_objects,
                    min_space_nm=lower_layer.min_space_nm,
                )
                local_upper_pad = _device_interface_local_escape_pad_bbox(
                    raw_upper_pad,
                    upper_component_objects,
                    min_space_nm=upper_layer.min_space_nm,
                )
                fallback_lower_pad = local_lower_pad or lower_pad
                fallback_upper_pad = local_upper_pad or upper_pad
                fallback_changed = (
                    fallback_lower_pad != lower_pad
                    or fallback_upper_pad != upper_pad
                )
                if fallback_changed:
                    if diagnostics is not None:
                        detail_key = f"collar_detail_primary_{collar_reason}"
                        diagnostics[detail_key] = diagnostics.get(detail_key, 0) + 1
                    collar_reason, collar_contact = adjudicate_collar(
                        fallback_lower_pad, fallback_upper_pad
                    )
                    if collar_reason is None:
                        lower_pad = fallback_lower_pad
                        upper_pad = fallback_upper_pad
                        collar_variant = "local_escape"
                if collar_reason is not None:
                    reject(point, collar_reason)
                    continue
            assert collar_contact is not None
            (
                widened_lower_owners,
                widened_upper_owners,
                widened_lower_components,
                widened_upper_components,
            ) = collar_contact
            # The collar is emitted route metal.  It may intentionally bridge
            # additional disconnected components only when the exact widened
            # ownership replay proves they all belong to the already-selected
            # net.  Preserve every newly contacted component ref in the landing
            # certificate so the plan/runtime replay sees the resulting
            # cross-component route-owned union; no contact is silently dropped.
            lower_owners = widened_lower_owners
            upper_owners = widened_upper_owners
            lower_components = frozenset(widened_lower_components)
            upper_components = frozenset(widened_upper_components)
        sites.append(LandingSite(
            site_id=f"world:{world_digest[:16]}:via{lower_layer_n}:{x}:{y}",
            xy_nm=(x, y),
            lower_layer_n=lower_layer_n,
            upper_layer_n=upper_layer.layer_n,
            capacity=1,
            provenance=(
                f"final-routing-world:{world_digest}:via{lower_layer_n}:"
                f"cut={landing_template.cut_size_nm}:"
                f"enc={landing_template.enclosure_below_nm},"
                f"{landing_template.enclosure_above_nm}:"
                f"metal-width-floor={lower_layer.routing_width_nm},"
                f"{upper_layer.routing_width_nm}:interface={collar_variant}"
            ),
            cut_bbox_nm=cut_bbox,
            lower_pad_bbox_nm=lower_pad,
            upper_pad_bbox_nm=upper_pad,
            owner_net=owner_net,
            owner_layer_ns=tuple(
                layer_n
                for layer_n, side_owners in (
                    (lower_layer_n, lower_owners),
                    (upper_layer.layer_n, upper_owners),
                )
                if side_owners
            ),
            lower_owner_component_refs=tuple(sorted(lower_components)),
            upper_owner_component_refs=tuple(sorted(upper_components)),
            routing_world_provenance_digest=world_digest,
        ))
    if diagnostics is not None:
        diagnostics["accepted"] = diagnostics.get("accepted", 0) + len(sites)
    return tuple(sorted(sites, key=lambda site: site.site_id))


def _enumerate_landing_sites(
    world: RoutingWorldProvenance,
    tech: Any,
    layers: Sequence[_RoutingLayerFacts],
    tracks: Sequence[TrackResource],
    obstacles: Sequence[tuple[Any, ...]],
    geometry_index: _LayerGeometryIndex,
    owned_conductor_index: Any,
    *,
    max_via_sites: int,
    max_via_resource_pairs: int,
    geometry_budget: _GeometryEvaluationBudget,
) -> tuple[tuple[LandingSite, ...], bool]:
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    from clay.layout.route.via_model import via_landing

    manufacturing_grid_nm = pdk_manufacturing_grid_nm(tech)
    layer_by_n = {layer.layer_n: layer for layer in layers}
    tracks_by_layer: dict[int, list[TrackResource]] = {}
    for resource in tracks:
        tracks_by_layer.setdefault(resource.key.layer_n, []).append(resource)
    mandatory_sites: list[LandingSite] = []
    eager_owned_sites: list[LandingSite] = []
    eager_free_sites: list[LandingSite] = []
    complete = True
    resource_pair_count = 0

    for lower_layer_n in sorted(layer_by_n):
        upper_layer_n = lower_layer_n + 1
        if upper_layer_n not in layer_by_n:
            continue
        lower_layer = layer_by_n[lower_layer_n]
        upper_layer = layer_by_n[upper_layer_n]
        lower_tracks = tuple(tracks_by_layer.get(lower_layer_n, ()))
        upper_tracks = tuple(tracks_by_layer.get(upper_layer_n, ()))
        via_name = f"via{lower_layer_n}"
        try:
            _layer_key(tech.via_key(lower_layer_n))
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        via_space_nm = _via_spacing_nm(tech, via_name)
        landing_template = via_landing(tech, via_name, 0, 0)
        if landing_template is None:
            continue
        via_pitch_nm = _ceil_to_multiple(
            landing_template.cut_size_nm + via_space_nm,
            manufacturing_grid_nm,
        )
        free_points: set[tuple[int, int]] = set()
        for lower_track in lower_tracks:
            for upper_track in upper_tracks:
                resource_pair_count += 1
                if resource_pair_count > max_via_resource_pairs:
                    raise RoutingResourceEnumerationLimitError(
                        "via resource-pair count exceeds limit "
                        f"{max_via_resource_pairs}"
                    )
                if complete:
                    for point in _iter_track_intersection_points(
                        lower_track,
                        upper_track,
                        lower_layer.track_step_nm,
                        upper_layer.track_step_nm,
                    ):
                        free_points.add(point)
                        if (
                            len(free_points)
                            + len(eager_free_sites)
                            + len(eager_owned_sites)
                            + len(mandatory_sites)
                            > max_via_sites
                        ):
                            # The complete lattice remains represented by the
                            # exact query-domain digest.  It is not truncated
                            # into a misleading prefix.
                            complete = False
                            eager_free_sites.clear()
                            eager_owned_sites.clear()
                            free_points.clear()
                            break
        mandatory_points: set[tuple[int, int]] = set()
        lower_components = tuple(
            component
            for component in owned_conductor_index.components
            if component.layer_key == lower_layer.key and component.atom_facts
        )
        upper_components_by_net: dict[str, list[Any]] = {}
        for component in owned_conductor_index.components:
            if component.layer_key == upper_layer.key and component.atom_facts:
                upper_components_by_net.setdefault(component.net, []).append(
                    component
                )
        for lower_component in lower_components:
            for upper_component in upper_components_by_net.get(
                lower_component.net, ()
            ):
                geometry_budget.consume("owned component-pair landing expansion")
                for point in _iter_owned_component_pair_points(
                    lower_component,
                    upper_component,
                    landing_template,
                    via_pitch_nm,
                    manufacturing_grid_nm,
                    geometry_budget,
                ):
                    mandatory_points.add(point)
                    if len(mandatory_points) + len(mandatory_sites) > max_via_sites:
                        raise RoutingResourceEnumerationLimitError(
                            "mandatory owned component-pair via site count exceeds "
                            f"limit {max_via_sites}"
                        )
        for pin in world.pins:
            point = (int(pin[3]), int(pin[4]))
            pin_layer = (int(pin[5]), int(pin[6]))
            reaches_opposite_track = (
                pin_layer == lower_layer.key
                and any(
                    _point_on_track_resource(point, resource)
                    for resource in upper_tracks
                )
            ) or (
                pin_layer == upper_layer.key
                and any(
                    _point_on_track_resource(point, resource)
                    for resource in lower_tracks
                )
            )
            reaches_same_net_component = (
                pin_layer in (lower_layer.key, upper_layer.key)
                and _same_net_component_pair_at_point(
                    point,
                    pin_net=str(pin[0]),
                    lower_layer_key=lower_layer.key,
                    upper_layer_key=upper_layer.key,
                    landing_template=landing_template,
                    geometry_budget=geometry_budget,
                    owned_conductor_index=owned_conductor_index,
                )
            )
            if reaches_opposite_track or reaches_same_net_component:
                mandatory_points.add(point)
        if len(mandatory_points) + len(mandatory_sites) > max_via_sites:
            raise RoutingResourceEnumerationLimitError(
                "mandatory component-pair/pin via site count exceeds limit "
                f"{max_via_sites}"
            )
        seed_points = set(mandatory_points)
        owned_access_points: set[tuple[int, int]] = set()
        if complete:
            owned_access_overflow = False
            for obstacle in obstacles:
                owner = obstacle[6]
                if owner is None or not str(owner):
                    continue
                layer_key = (int(obstacle[4]), int(obstacle[5]))
                if layer_key == lower_layer.key:
                    opposite_tracks = upper_tracks
                    conductor_is_lower = True
                elif layer_key == upper_layer.key:
                    opposite_tracks = lower_tracks
                    conductor_is_lower = False
                else:
                    continue
                conductor_bbox = tuple(int(value) for value in obstacle[:4])
                for point in _iter_owned_conductor_track_points(
                    conductor_bbox,
                    conductor_is_lower,
                    opposite_tracks,
                    landing_template,
                    via_pitch_nm,
                    manufacturing_grid_nm,
                    (
                        lower_layer.min_width_nm
                        if conductor_is_lower
                        else upper_layer.min_width_nm
                    ),
                    geometry_budget,
                ):
                    owned_access_points.add(point)
                    if (
                        len(seed_points)
                        + len(owned_access_points)
                        + len(mandatory_sites)
                        + len(eager_owned_sites)
                        + len(eager_free_sites)
                        + len(free_points)
                        > max_via_sites
                    ):
                        # Owned conductor/lane access has an exact finite query
                        # path.  Once the eager projection is too large, keep
                        # only topology-critical component-pair/pin seeds and
                        # represent every owned access through the signed lazy
                        # domain instead of retaining an order-biased prefix.
                        complete = False
                        eager_free_sites.clear()
                        eager_owned_sites.clear()
                        free_points.clear()
                        owned_access_points.clear()
                        owned_access_overflow = True
                        break
                if owned_access_overflow:
                    break
        mandatory_sites.extend(_materialize_landing_points(
            tuple(seed_points),
            world=world,
            tech=tech,
            lower_layer=lower_layer,
            upper_layer=upper_layer,
            geometry_index=geometry_index,
            owned_conductor_index=owned_conductor_index,
            geometry_budget=geometry_budget,
        ))
        if complete:
            eager_owned_sites.extend(_materialize_landing_points(
                tuple(owned_access_points - seed_points),
                world=world,
                tech=tech,
                lower_layer=lower_layer,
                upper_layer=upper_layer,
                geometry_index=geometry_index,
                owned_conductor_index=owned_conductor_index,
                geometry_budget=geometry_budget,
            ))
            eager_free_sites.extend(_materialize_landing_points(
                tuple(free_points - seed_points - owned_access_points),
                world=world,
                tech=tech,
                lower_layer=lower_layer,
                upper_layer=upper_layer,
                geometry_index=geometry_index,
                owned_conductor_index=owned_conductor_index,
                geometry_budget=geometry_budget,
            ))
            if (
                len(mandatory_sites)
                + len(eager_owned_sites)
                + len(eager_free_sites)
                > max_via_sites
            ):
                complete = False
                eager_free_sites.clear()
                eager_owned_sites.clear()
    sites = mandatory_sites + (
        eager_owned_sites + eager_free_sites if complete else []
    )
    unique = {site.site_id: site for site in sites}
    if len(unique) > max_via_sites:
        raise RoutingResourceEnumerationLimitError(
            f"materialized via site count exceeds limit {max_via_sites}"
        )
    return tuple(unique[key] for key in sorted(unique)), complete


def _point_is_track_crossing(
    point: tuple[int, int],
    lower: TrackResource,
    upper: TrackResource,
    lower_step_nm: int,
    upper_step_nm: int,
) -> bool:
    if not (
        _point_on_track_resource(point, lower)
        and _point_on_track_resource(point, upper)
    ):
        return False
    if lower.preferred_axis != upper.preferred_axis:
        return True
    run_coordinate = point[0] if lower.preferred_axis == "H" else point[1]
    return run_coordinate % math.lcm(lower_step_nm, upper_step_nm) == 0


@dataclass(frozen=True)
class _TrackPointMembershipIndex:
    """Exact track resources keyed by layer, axis, and explicit center."""

    resources_by_layer_axis_center: Mapping[
        tuple[int, str, int], tuple[TrackResource, ...]
    ]

    @classmethod
    def build(
        cls,
        resources: Sequence[TrackResource],
    ) -> "_TrackPointMembershipIndex":
        rows: dict[
            tuple[int, str, int], list[TrackResource]
        ] = {}
        for resource in resources:
            for center_nm in resource.track_centers_nm:
                rows.setdefault(
                    (
                        resource.key.layer_n,
                        resource.preferred_axis,
                        center_nm,
                    ),
                    [],
                ).append(resource)
        return cls(resources_by_layer_axis_center={
            key: tuple(sorted(values, key=lambda resource: resource.key))
            for key, values in sorted(rows.items())
        })

    def resources_at_point(
        self,
        layer_n: int,
        point: tuple[int, int],
        geometry_budget: _GeometryEvaluationBudget,
        reason: str,
    ) -> tuple[TrackResource, ...]:
        candidates = (
            *self.resources_by_layer_axis_center.get(
                (layer_n, "H", point[1]), ()
            ),
            *self.resources_by_layer_axis_center.get(
                (layer_n, "V", point[0]), ()
            ),
        )
        matches = []
        for resource in candidates:
            geometry_budget.consume(reason)
            if _point_on_track_resource(point, resource):
                matches.append(resource)
        return tuple(matches)


def _point_is_any_track_crossing(
    point: tuple[int, int],
    lower_tracks: Sequence[TrackResource],
    upper_tracks: Sequence[TrackResource],
    lower_step_nm: int,
    upper_step_nm: int,
    geometry_budget: _GeometryEvaluationBudget,
    *,
    membership_index: _TrackPointMembershipIndex | None = None,
    lower_layer_n: int | None = None,
    upper_layer_n: int | None = None,
) -> bool:
    """Test exact crossing membership without a quadratic all-pairs scan.

    A point belongs to only the track resources whose finite interval and
    center contain it.  Filtering each layer first is equivalent to testing
    every lower/upper pair, while keeping lazy-domain replay proportional to
    the ledger size rather than its Cartesian square.
    """

    if membership_index is not None:
        if type(lower_layer_n) is not int or type(upper_layer_n) is not int:
            raise RoutingResourceEnumerationError(
                "indexed track membership requires exact layer numbers"
            )
        lower_at_point = list(membership_index.resources_at_point(
            lower_layer_n,
            point,
            geometry_budget,
            "landing-domain lower-track membership",
        ))
    else:
        lower_at_point = []
        for resource in lower_tracks:
            geometry_budget.consume("landing-domain lower-track membership")
            if _point_on_track_resource(point, resource):
                lower_at_point.append(resource)
    if not lower_at_point:
        return False

    if membership_index is not None:
        upper_at_point = list(membership_index.resources_at_point(
            upper_layer_n,
            point,
            geometry_budget,
            "landing-domain upper-track membership",
        ))
    else:
        upper_at_point = []
        for resource in upper_tracks:
            geometry_budget.consume("landing-domain upper-track membership")
            if _point_on_track_resource(point, resource):
                upper_at_point.append(resource)
    if not upper_at_point:
        return False

    for lower in lower_at_point:
        for upper in upper_at_point:
            geometry_budget.consume("landing-domain crossing-axis membership")
            if lower.preferred_axis != upper.preferred_axis:
                return True
            run_coordinate = point[0] if lower.preferred_axis == "H" else point[1]
            if run_coordinate % math.lcm(lower_step_nm, upper_step_nm) == 0:
                return True
    return False


@dataclass(frozen=True)
class _LandingViaQueryFacts:
    """PDK-resolved immutable via facts reused by one validated query session."""

    landing_template: Any
    via_key: LayerKey
    via_spacing_nm: int
    via_pitch_nm: int


@dataclass(frozen=True)
class _ExactLandingSiteQueryContext:
    """One validated, indexed, cumulatively budgeted lazy-domain session."""

    ledger_sha256: str
    world_sha256: str
    pdk_provenance_digest: str
    owned_conductor_index_digest: str
    layers: tuple[_RoutingLayerFacts, ...]
    obstacles: tuple[tuple[Any, ...], ...]
    world_envelope: BBoxNm
    geometry_index: _LayerGeometryIndex
    geometry_budget: _GeometryEvaluationBudget
    landing_blocker_rects_by_lower_layer: dict[int, tuple[BBoxNm, ...]]
    landing_obstacle_classification_by_rule: dict[
        tuple[str, BBoxNm, LayerKey, int],
        tuple[tuple[BBoxNm, str], ...],
    ]
    landing_site_by_layer_point: dict[
        tuple[int, tuple[int, int]], LandingSite | None
    ]
    landing_rejection_by_layer_point: dict[
        tuple[int, tuple[int, int]], str
    ]
    landing_contact_by_layer_bbox: dict[
        tuple[LayerKey, BBoxNm],
        tuple[bool, frozenset[str], frozenset[str]],
    ]
    via_query_facts_by_lower_layer: Mapping[int, _LandingViaQueryFacts]
    manufacturing_grid_nm: int
    resources_by_key: Mapping[CorridorLayerKey, TrackResource]
    resources_by_layer: Mapping[int, tuple[TrackResource, ...]]
    track_point_membership_index: _TrackPointMembershipIndex
    ledger: ExplicitRoutingResourceLedger
    world: RoutingWorldProvenance
    tech: Any
    owned_conductor_index: Any


@resource_boundary(world_argument="world", tech_argument="tech")
def _prepare_exact_landing_site_query_context(
    ledger: ExplicitRoutingResourceLedger,
    world: RoutingWorldProvenance,
    tech: Any,
    owned_conductor_index: Any,
    *,
    max_geometry_evaluations: int = DEFAULT_ENUMERATED_GEOMETRY_EVALUATIONS,
    resource_meter: Any = None,
) -> _ExactLandingSiteQueryContext:
    """Validate immutable domain inputs once and build reusable spatial indexes."""

    if type(ledger) is not ExplicitRoutingResourceLedger:
        raise RoutingResourceEnumerationError(
            "ledger must be exact ExplicitRoutingResourceLedger"
        )
    if type(world) is not RoutingWorldProvenance:
        raise RoutingResourceEnumerationError(
            "world must be exact RoutingWorldProvenance"
        )
    if (
        type(max_geometry_evaluations) is not int
        or max_geometry_evaluations <= 0
        or max_geometry_evaluations > MAX_ENUMERATED_GEOMETRY_EVALUATIONS
    ):
        raise RoutingResourceEnumerationLimitError(
            "max_geometry_evaluations is outside the exact query bound"
        )
    world_digest = world.sha256()
    if ledger.routing_world_provenance_digest != world_digest:
        raise RoutingResourceEnumerationError(
            "landing-site query ledger belongs to a foreign world"
        )
    if owned_conductor_index.world_digest != world_digest:
        raise RoutingResourceEnumerationError(
            "landing-site query conductor index belongs to a foreign world"
        )
    conductor_digest = owned_conductor_index.sha256()
    if conductor_digest != ledger.owned_conductor_index_digest:
        raise RoutingResourceEnumerationError(
            "landing-site query conductor index digest is stale"
        )
    tech_digest = pdk_provenance_digest(tech)
    if tech_digest != world.pdk_provenance_digest:
        raise RoutingResourceEnumerationError(
            "landing-site query PDK digest does not match the world"
        )
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    from clay.layout.route.via_model import via_landing

    manufacturing_grid_nm = pdk_manufacturing_grid_nm(tech)
    layers = _routing_layer_facts(tech, manufacturing_grid_nm)
    layer_ns = {layer.layer_n for layer in layers}
    via_query_facts_by_lower_layer: dict[int, _LandingViaQueryFacts] = {}
    for lower_layer_n in sorted(layer_ns):
        if lower_layer_n + 1 not in layer_ns:
            continue
        via_name = f"via{lower_layer_n}"
        landing_template = via_landing(tech, via_name, 0, 0)
        if landing_template is None:
            continue
        via_spacing_nm = _via_spacing_nm(tech, via_name)
        via_query_facts_by_lower_layer[lower_layer_n] = _LandingViaQueryFacts(
            landing_template=landing_template,
            via_key=_layer_key(tech.via_key(lower_layer_n)),
            via_spacing_nm=via_spacing_nm,
            via_pitch_nm=_ceil_to_multiple(
                landing_template.cut_size_nm + via_spacing_nm,
                manufacturing_grid_nm,
            ),
        )
    obstacles = _world_obstacle_facts(world)
    geometry_index = _LayerGeometryIndex.build(obstacles, world.pins)
    track_replay_budget = bind_budget(
        _GeometryEvaluationBudget(max_geometry_evaluations), resource_meter, "geometry"
    )
    canonical_tracks = _enumerate_track_resources(
        world,
        layers,
        owned_conductor_index,
        max_track_centers=MAX_ENUMERATED_TRACK_CENTERS,
        max_track_segments=MAX_ENUMERATED_TRACK_SEGMENTS,
        max_track_resources=MAX_ENUMERATED_TRACK_RESOURCES,
        geometry_budget=track_replay_budget,
        geometry_index=geometry_index,
    )
    if ledger.track_resources != canonical_tracks:
        raise RoutingResourceEnumerationError(
            "landing-site ledger does not match canonical track replay"
        )
    expected_domain_digest = _landing_site_domain_digest(
        world_digest,
        canonical_tracks,
        world.pdk_provenance_digest,
        conductor_digest,
    )
    if (
        ledger.landing_site_domain_digest != expected_domain_digest
        and not (
            ledger.landing_sites_complete
            and not ledger.landing_site_domain_digest
        )
    ):
        raise RoutingResourceEnumerationError(
            "landing-site query domain digest does not replay"
        )
    # Canonical track replay and exact lazy landing queries are separate proof
    # phases.  The ledger's track domain must first replay within its own hard
    # bound; a successful replay must not silently debit the independent
    # landing-site query allowance.  Both phases remain explicitly bounded by
    # the same caller-supplied ceiling (total work <= two phase ceilings).
    landing_query_budget = bind_budget(
        _GeometryEvaluationBudget(max_geometry_evaluations), resource_meter, "geometry"
    )
    blocker_rects_by_lower_layer: dict[int, tuple[BBoxNm, ...]] = {}
    resources_by_key = {
        resource.key: resource for resource in ledger.track_resources
    }
    resources_by_layer: dict[int, list[TrackResource]] = {}
    for resource in ledger.track_resources:
        resources_by_layer.setdefault(resource.key.layer_n, []).append(resource)
    return _ExactLandingSiteQueryContext(
        ledger_sha256=ledger.sha256(),
        world_sha256=world_digest,
        pdk_provenance_digest=tech_digest,
        owned_conductor_index_digest=conductor_digest,
        layers=layers,
        obstacles=obstacles,
        world_envelope=_world_envelope(world),
        geometry_index=geometry_index,
        geometry_budget=landing_query_budget,
        landing_blocker_rects_by_lower_layer=blocker_rects_by_lower_layer,
        landing_obstacle_classification_by_rule={},
        landing_site_by_layer_point={},
        landing_rejection_by_layer_point={},
        landing_contact_by_layer_bbox={},
        via_query_facts_by_lower_layer=via_query_facts_by_lower_layer,
        manufacturing_grid_nm=manufacturing_grid_nm,
        resources_by_key=resources_by_key,
        resources_by_layer={
            layer_n: tuple(resources)
            for layer_n, resources in resources_by_layer.items()
        },
        track_point_membership_index=(
            _TrackPointMembershipIndex.build(ledger.track_resources)
        ),
        ledger=ledger,
        world=world,
        tech=tech,
        owned_conductor_index=owned_conductor_index,
    )


def _verify_exact_landing_site_query_context(
    context: _ExactLandingSiteQueryContext,
    ledger: ExplicitRoutingResourceLedger,
    world: RoutingWorldProvenance,
    tech: Any,
    owned_conductor_index: Any,
) -> None:
    """Accept exact session objects cheaply; replay equivalent foreign inputs."""

    exact_inputs = (
        type(context) is _ExactLandingSiteQueryContext
        and context.ledger is ledger
        and context.world is world
        and context.tech is tech
        and context.owned_conductor_index is owned_conductor_index
    )
    if not exact_inputs and (
        type(context) is not _ExactLandingSiteQueryContext
        or context.ledger_sha256 != ledger.sha256()
        or context.world_sha256 != world.sha256()
        or context.pdk_provenance_digest != pdk_provenance_digest(tech)
        or context.owned_conductor_index_digest
        != owned_conductor_index.sha256()
    ):
        raise RoutingResourceEnumerationError(
            "landing-site query context belongs to foreign inputs"
        )


@resource_boundary(world_argument="world", tech_argument="tech")
def _query_exact_landing_sites_at_points(
    ledger: ExplicitRoutingResourceLedger,
    world: RoutingWorldProvenance,
    tech: Any,
    owned_conductor_index: Any,
    *,
    lower_layer_n: int,
    points: Sequence[tuple[int, int]],
    max_geometry_evaluations: int = DEFAULT_ENUMERATED_GEOMETRY_EVALUATIONS,
    signed_points: bool,
    query_context: _ExactLandingSiteQueryContext | None = None,
    diagnostics: dict[str, int] | None = None,
    resource_meter: Any = None,
) -> tuple[LandingSite, ...]:
    """Materialize a finite exact subset of the signed landing-site domain.

    The caller supplies coordinates discovered by a topology/pattern query.  A
    coordinate is accepted only when it is in the full eager definition: an
    exact lower/upper lane crossing, a pin access crossing the opposite layer,
    or a rule-pitched owned-conductor access.  PDK, obstacle, ownership, and
    enclosure checks are then replayed unchanged.
    """

    if type(ledger) is not ExplicitRoutingResourceLedger:
        raise RoutingResourceEnumerationError(
            "ledger must be exact ExplicitRoutingResourceLedger"
        )
    if type(world) is not RoutingWorldProvenance:
        raise RoutingResourceEnumerationError(
            "world must be exact RoutingWorldProvenance"
        )
    if type(lower_layer_n) is not int or lower_layer_n <= 0:
        raise RoutingResourceEnumerationError(
            "lower_layer_n must be a positive exact integer"
        )
    requested = _bounded_tuple(
        points,
        name="landing-site query points",
        limit=MAX_ENUMERATED_VIA_SITES,
    )
    if any(
        not isinstance(point, (tuple, list))
        or len(point) != 2
        or any(type(value) is not int for value in point)
        for point in requested
    ):
        raise RoutingResourceEnumerationError(
            "landing-site query points must be integer-nm pairs"
        )
    context = query_context or _prepare_exact_landing_site_query_context(
        ledger,
        world,
        tech,
        owned_conductor_index,
        max_geometry_evaluations=max_geometry_evaluations,
        **({"resource_meter": resource_meter} if resource_meter is not None else {}),
    )
    _verify_exact_landing_site_query_context(
        context, ledger, world, tech, owned_conductor_index
    )
    layers = context.layers
    layer_by_n = {layer.layer_n: layer for layer in layers}
    upper_layer_n = lower_layer_n + 1
    if lower_layer_n not in layer_by_n or upper_layer_n not in layer_by_n:
        return ()
    lower_layer = layer_by_n[lower_layer_n]
    upper_layer = layer_by_n[upper_layer_n]
    lower_tracks = context.resources_by_layer.get(lower_layer_n, ())
    upper_tracks = context.resources_by_layer.get(upper_layer_n, ())
    geometry_budget = context.geometry_budget
    obstacles = context.obstacles
    geometry_index = context.geometry_index
    via_query_facts = context.via_query_facts_by_lower_layer.get(lower_layer_n)
    if via_query_facts is None:
        return ()
    landing_template = via_query_facts.landing_template
    manufacturing_grid_nm = context.manufacturing_grid_nm
    via_pitch_nm = via_query_facts.via_pitch_nm

    if type(signed_points) is not bool:
        raise RoutingResourceEnumerationError(
            "signed_points must be an exact boolean"
        )
    cached_sites: list[LandingSite] = []
    uncached_points: list[tuple[int, int]] = []
    for raw_point in requested:
        point = int(raw_point[0]), int(raw_point[1])
        cache_key = lower_layer_n, point
        if cache_key in context.landing_site_by_layer_point:
            cached = context.landing_site_by_layer_point[cache_key]
            if cached is not None:
                cached_sites.append(cached)
            elif diagnostics is not None:
                rejection = context.landing_rejection_by_layer_point.get(cache_key)
                if rejection is None:
                    raise RoutingResourceEnumerationError(
                        "cached exact landing rejection lacks its certified reason"
                    )
                diagnostics[rejection] = diagnostics.get(rejection, 0) + 1
        else:
            uncached_points.append(point)
    requested = tuple(sorted(set(uncached_points)))
    valid_points: set[tuple[int, int]] = set()
    rejection_by_point: dict[tuple[int, int], str] = {}
    for raw_point in requested:
        point = int(raw_point[0]), int(raw_point[1])
        is_candidate = signed_points
        if not is_candidate:
            is_candidate = _point_is_any_track_crossing(
                point,
                lower_tracks,
                upper_tracks,
                lower_layer.track_step_nm,
                upper_layer.track_step_nm,
                geometry_budget,
                membership_index=context.track_point_membership_index,
                lower_layer_n=lower_layer_n,
                upper_layer_n=upper_layer_n,
            )
        if not is_candidate:
            for pin in world.pins:
                geometry_budget.consume("landing-domain pin membership")
                if (int(pin[3]), int(pin[4])) != point:
                    continue
                pin_layer = int(pin[5]), int(pin[6])
                if pin_layer not in (lower_layer.key, upper_layer.key):
                    continue
                opposite_tracks = (
                    upper_tracks
                    if pin_layer == lower_layer.key
                    else lower_tracks
                )
                for resource in opposite_tracks:
                    geometry_budget.consume(
                        "landing-domain pin/track membership"
                    )
                    if _point_on_track_resource(point, resource):
                        is_candidate = True
                        break
                if is_candidate:
                    break
                if _same_net_component_pair_at_point(
                    point,
                    pin_net=str(pin[0]),
                    lower_layer_key=lower_layer.key,
                    upper_layer_key=upper_layer.key,
                    landing_template=landing_template,
                    geometry_budget=geometry_budget,
                    owned_conductor_index=owned_conductor_index,
                    landing_contact_cache=(
                        context.landing_contact_by_layer_bbox
                    ),
                ):
                    is_candidate = True
                    break
        if (
            not is_candidate
            and point[0] % via_pitch_nm == 0
            and point[1] % via_pitch_nm == 0
            and _same_net_component_pair_at_point(
                point,
                pin_net=None,
                lower_layer_key=lower_layer.key,
                upper_layer_key=upper_layer.key,
                landing_template=landing_template,
                geometry_budget=geometry_budget,
                owned_conductor_index=owned_conductor_index,
                require_pad_containment=True,
                landing_contact_cache=(
                    context.landing_contact_by_layer_bbox
                ),
            )
        ):
            is_candidate = True
        if not is_candidate:
            for obstacle in obstacles:
                geometry_budget.consume(
                    "landing-domain owned-obstacle membership"
                )
                owner = obstacle[6]
                if owner is None or not str(owner):
                    continue
                layer_key = (int(obstacle[4]), int(obstacle[5]))
                if layer_key == lower_layer.key:
                    opposite_tracks = upper_tracks
                    conductor_is_lower = True
                elif layer_key == upper_layer.key:
                    opposite_tracks = lower_tracks
                    conductor_is_lower = False
                else:
                    continue
                if _owned_conductor_track_contains_point(
                    point,
                    tuple(int(value) for value in obstacle[:4]),
                    conductor_is_lower,
                    opposite_tracks,
                    landing_template,
                    via_pitch_nm,
                    manufacturing_grid_nm,
                    (
                        lower_layer.min_width_nm
                        if conductor_is_lower
                        else upper_layer.min_width_nm
                    ),
                    geometry_budget,
                ):
                    is_candidate = True
                    break
        if is_candidate:
            valid_points.add(point)
        else:
            rejection_by_point[point] = "not_in_signed_domain"
            if diagnostics is not None:
                diagnostics["not_in_signed_domain"] = (
                    diagnostics.get("not_in_signed_domain", 0) + 1
                )
    blocker_center_rects = context.landing_blocker_rects_by_lower_layer.get(
        lower_layer_n
    )
    if valid_points and blocker_center_rects is None:
        lower_pad_template = _pad_bbox_with_width_floor(
            landing_template.below_metal_rect,
            width_floor_nm=lower_layer.routing_width_nm,
            manufacturing_grid_nm=manufacturing_grid_nm,
            minimum_area_nm2=_landing_metal_area(tech, lower_layer.layer_n),
        )
        upper_pad_template = _pad_bbox_with_width_floor(
            landing_template.above_metal_rect,
            width_floor_nm=upper_layer.routing_width_nm,
            manufacturing_grid_nm=manufacturing_grid_nm,
            minimum_area_nm2=_landing_metal_area(tech, upper_layer.layer_n),
        )
        pad_specs = (
            (
                lower_pad_template,
                lower_layer.key,
                lower_layer.min_space_nm,
            ),
            (
                upper_pad_template,
                upper_layer.key,
                upper_layer.min_space_nm,
            ),
        )
        cut_layer_key = via_query_facts.via_key
        if not _landing_batch_prefers_spatial_queries(
            len(valid_points),
            cut_layer_key=cut_layer_key,
            pad_specs=pad_specs,
            geometry_index=geometry_index,
        ):
            blocker_center_rects = _landing_obstacle_invalid_center_rects(
                cut_relative_bbox=tuple(
                    int(value) for value in landing_template.cut
                ),
                cut_layer_key=cut_layer_key,
                cut_spacing_nm=via_query_facts.via_spacing_nm,
                pad_specs=pad_specs,
                geometry_index=geometry_index,
                geometry_budget=geometry_budget,
                classification_cache=(
                    context.landing_obstacle_classification_by_rule
                ),
            )
            # Publish only a complete exact classification.  A budget/PDK
            # failure above is never memoized as an empty ruleset.
            context.landing_blocker_rects_by_lower_layer[
                lower_layer_n
            ] = blocker_center_rects
    elif not valid_points and blocker_center_rects is None:
        # An empty point query needs no obstacle-domain compilation and must not
        # publish a misleading empty cache entry for this via layer.
        blocker_center_rects = ()
    materialized = _materialize_landing_points(
        tuple(valid_points),
        world=world,
        tech=tech,
        lower_layer=lower_layer,
        upper_layer=upper_layer,
        geometry_index=geometry_index,
        owned_conductor_index=owned_conductor_index,
        geometry_budget=geometry_budget,
        manufacturing_grid_nm=manufacturing_grid_nm,
        landing_template_override=landing_template,
        via_key_override=via_query_facts.via_key,
        via_space_nm_override=via_query_facts.via_spacing_nm,
        blocker_center_rects=blocker_center_rects,
        diagnostics=diagnostics,
        rejection_by_point=rejection_by_point,
        landing_contact_cache=context.landing_contact_by_layer_bbox,
        routing_world_digest_override=context.world_sha256,
        world_envelope_override=context.world_envelope,
    )
    materialized_by_point = {site.xy_nm: site for site in materialized}
    for point in requested:
        cache_key = lower_layer_n, point
        site = materialized_by_point.get(point)
        context.landing_site_by_layer_point[cache_key] = site
        if site is None:
            rejection = rejection_by_point.get(point)
            if rejection is None:
                raise RoutingResourceEnumerationError(
                    "exact landing miss lacks a certified rejection reason"
                )
            context.landing_rejection_by_layer_point[cache_key] = rejection
    unique = {
        site.site_id: site for site in (*cached_sites, *materialized)
    }
    return tuple(unique[site_id] for site_id in sorted(unique))


@resource_boundary(world_argument="world", tech_argument="tech")
def query_exact_landing_sites_at_points(
    ledger: ExplicitRoutingResourceLedger,
    world: RoutingWorldProvenance,
    tech: Any,
    owned_conductor_index: Any,
    *,
    lower_layer_n: int,
    points: Sequence[tuple[int, int]],
    max_geometry_evaluations: int = DEFAULT_ENUMERATED_GEOMETRY_EVALUATIONS,
    resource_meter: Any = None,
) -> tuple[LandingSite, ...]:
    """Public point query; every coordinate must replay domain membership."""

    return _query_exact_landing_sites_at_points(
        ledger,
        world,
        tech,
        owned_conductor_index,
        lower_layer_n=lower_layer_n,
        points=points,
        max_geometry_evaluations=max_geometry_evaluations,
        signed_points=False,
        **({"resource_meter": resource_meter} if resource_meter is not None else {}),
    )


@resource_boundary(world_argument="world", tech_argument="tech")
def _query_exact_landing_sites_for_lane_pair(
    ledger: ExplicitRoutingResourceLedger,
    world: RoutingWorldProvenance,
    tech: Any,
    owned_conductor_index: Any,
    *,
    left_resource: TrackResource,
    left_track_start: int,
    right_resource: TrackResource,
    right_track_start: int,
    query_context: _ExactLandingSiteQueryContext | None = None,
    resource_meter: Any = None,
) -> tuple[LandingSite, ...]:
    """Return every exact legal site joining two selected adjacent lanes."""

    context = query_context or _prepare_exact_landing_site_query_context(
        ledger, world, tech, owned_conductor_index,
        **({"resource_meter": resource_meter} if resource_meter is not None else {}),
    )
    _verify_exact_landing_site_query_context(
        context, ledger, world, tech, owned_conductor_index
    )
    resources_by_key = context.resources_by_key
    for resource, track_start in (
        (left_resource, left_track_start),
        (right_resource, right_track_start),
    ):
        if resources_by_key.get(resource.key) != resource:
            raise RoutingResourceEnumerationError(
                "landing lane-pair query names a foreign track resource"
            )
        if (
            type(track_start) is not int
            or track_start < 0
            or track_start >= len(resource.track_centers_nm)
        ):
            raise RoutingResourceEnumerationError(
                "landing lane-pair query track start is outside the resource"
            )
    if abs(left_resource.key.layer_n - right_resource.key.layer_n) != 1:
        return ()
    lower_resource, lower_start, upper_resource, upper_start = (
        (left_resource, left_track_start, right_resource, right_track_start)
        if left_resource.key.layer_n < right_resource.key.layer_n
        else (right_resource, right_track_start, left_resource, left_track_start)
    )
    layer_by_n = {layer.layer_n: layer for layer in context.layers}
    lower_layer = layer_by_n[lower_resource.key.layer_n]
    upper_layer = layer_by_n[upper_resource.key.layer_n]
    selected_lower = replace(
        lower_resource,
        track_centers_nm=(lower_resource.track_centers_nm[lower_start],),
    )
    selected_upper = replace(
        upper_resource,
        track_centers_nm=(upper_resource.track_centers_nm[upper_start],),
    )
    points = _bounded_tuple(
        _iter_track_intersection_points(
            selected_lower,
            selected_upper,
            lower_layer.track_step_nm,
            upper_layer.track_step_nm,
        ),
        name="exact lane-pair landing points",
        limit=MAX_ENUMERATED_VIA_SITES,
    )
    return _query_exact_landing_sites_at_points(
        ledger,
        world,
        tech,
        owned_conductor_index,
        lower_layer_n=lower_resource.key.layer_n,
        points=points,
        signed_points=True,
        query_context=context,
    )


@resource_boundary(world_argument="world", tech_argument="tech")
def query_exact_landing_sites_for_lane_pair(
    ledger: ExplicitRoutingResourceLedger,
    world: RoutingWorldProvenance,
    tech: Any,
    owned_conductor_index: Any,
    *,
    left_resource: TrackResource,
    left_track_start: int,
    right_resource: TrackResource,
    right_track_start: int,
    resource_meter: Any = None,
) -> tuple[LandingSite, ...]:
    """Public exact lane-pair query with an internally validated context."""

    return _query_exact_landing_sites_for_lane_pair(
        ledger,
        world,
        tech,
        owned_conductor_index,
        left_resource=left_resource,
        left_track_start=left_track_start,
        right_resource=right_resource,
        right_track_start=right_track_start,
        **({"resource_meter": resource_meter} if resource_meter is not None else {}),
    )


@resource_boundary(world_argument="world", tech_argument="tech")
def _query_exact_landing_sites_for_component_lane(
    ledger: ExplicitRoutingResourceLedger,
    world: RoutingWorldProvenance,
    tech: Any,
    owned_conductor_index: Any,
    *,
    component_ref: str,
    lane_resource: TrackResource,
    lane_track_start: int,
    query_context: _ExactLandingSiteQueryContext | None = None,
    diagnostics: dict[str, int] | None = None,
    resource_meter: Any = None,
) -> tuple[LandingSite, ...]:
    """Return all rule-pitched accesses from one owned component to one lane."""

    context = query_context or _prepare_exact_landing_site_query_context(
        ledger, world, tech, owned_conductor_index,
        **({"resource_meter": resource_meter} if resource_meter is not None else {}),
    )
    _verify_exact_landing_site_query_context(
        context, ledger, world, tech, owned_conductor_index
    )
    resources_by_key = context.resources_by_key
    if resources_by_key.get(lane_resource.key) != lane_resource:
        raise RoutingResourceEnumerationError(
            "component/lane landing query names a foreign track resource"
        )
    if (
        type(lane_track_start) is not int
        or lane_track_start < 0
        or lane_track_start >= len(lane_resource.track_centers_nm)
    ):
        raise RoutingResourceEnumerationError(
            "component/lane query track start is outside the resource"
        )
    component = owned_conductor_index.component(component_ref)
    layer_n_by_key = {
        layer.key: layer.layer_n for layer in context.layers
    }
    layer_by_key = {layer.key: layer for layer in context.layers}
    component_layer_n = layer_n_by_key.get(component.layer_key)
    if (
        component_layer_n is None
        or abs(component_layer_n - lane_resource.key.layer_n) != 1
    ):
        return ()
    lower_layer_n = min(component_layer_n, lane_resource.key.layer_n)
    via_query_facts = context.via_query_facts_by_lower_layer.get(
        lower_layer_n
    )
    if via_query_facts is None:
        return ()
    landing_template = via_query_facts.landing_template
    manufacturing_grid_nm = context.manufacturing_grid_nm
    via_pitch_nm = via_query_facts.via_pitch_nm
    selected_lane = replace(
        lane_resource,
        track_centers_nm=(
            lane_resource.track_centers_nm[lane_track_start],
        ),
    )
    geometry_budget = context.geometry_budget
    component_is_lower = component_layer_n == lower_layer_n
    points: set[tuple[int, int]] = set()
    for atom in component.atom_facts:
        for point in _iter_owned_conductor_track_points(
            tuple(int(value) for value in atom[:4]),
            component_is_lower,
            (selected_lane,),
            landing_template,
            via_pitch_nm,
            manufacturing_grid_nm,
            layer_by_key[component.layer_key].min_width_nm,
            geometry_budget,
        ):
            points.add(point)
            if len(points) > MAX_ENUMERATED_VIA_SITES:
                raise RoutingResourceEnumerationLimitError(
                    "exact component/lane landing points exceed materialization "
                    f"limit {MAX_ENUMERATED_VIA_SITES}"
                )
    for pin in component.pin_accesses:
        geometry_budget.consume("component/lane pin membership")
        point = int(pin[3]), int(pin[4])
        if _point_on_track_resource(point, selected_lane):
            points.add(point)
        if len(points) > MAX_ENUMERATED_VIA_SITES:
            raise RoutingResourceEnumerationLimitError(
                "exact component/lane landing points exceed materialization "
                f"limit {MAX_ENUMERATED_VIA_SITES}"
            )
    sites = _query_exact_landing_sites_at_points(
        ledger,
        world,
        tech,
        owned_conductor_index,
        lower_layer_n=lower_layer_n,
        points=tuple(points),
        signed_points=True,
        query_context=context,
        diagnostics=diagnostics,
    )
    if diagnostics is not None:
        diagnostics["landing_points"] = (
            diagnostics.get("landing_points", 0) + len(points)
        )
        diagnostics["materialized_sites"] = (
            diagnostics.get("materialized_sites", 0) + len(sites)
        )
    return sites


@resource_boundary(world_argument="world", tech_argument="tech")
def query_exact_landing_sites_for_component_lane(
    ledger: ExplicitRoutingResourceLedger,
    world: RoutingWorldProvenance,
    tech: Any,
    owned_conductor_index: Any,
    *,
    component_ref: str,
    lane_resource: TrackResource,
    lane_track_start: int,
    resource_meter: Any = None,
) -> tuple[LandingSite, ...]:
    """Public component/lane query with an internally validated context."""

    return _query_exact_landing_sites_for_component_lane(
        ledger,
        world,
        tech,
        owned_conductor_index,
        component_ref=component_ref,
        lane_resource=lane_resource,
        lane_track_start=lane_track_start,
        **({"resource_meter": resource_meter} if resource_meter is not None else {}),
    )


def _spacing_conflict(
    left: BBoxNm,
    right: BBoxNm,
    spacing_nm: int,
) -> tuple[bool, bool]:
    """Return ``(hard_conflict, physical_touch)`` with exact spacing legal."""

    if _rects_touch_or_overlap(left, right):
        return False, True
    if spacing_nm > 0 and _rects_overlap(_grow_bbox(left, spacing_nm), right):
        return True, False
    return False, False


def _site_influence_bbox(
    site: LandingSite,
    layer_by_n: Mapping[int, _RoutingLayerFacts],
    via_spacing_by_lower_layer: Mapping[int, int],
) -> BBoxNm:
    cut = site.cut_bbox_nm
    lower = site.lower_pad_bbox_nm
    upper = site.upper_pad_bbox_nm
    if cut is None or lower is None or upper is None:
        raise RoutingResourceEnumerationError(
            "site-pair enumeration requires geometry-complete landing sites"
        )
    boxes = (
        _grow_bbox(cut, via_spacing_by_lower_layer[site.lower_layer_n]),
        _grow_bbox(lower, layer_by_n[site.lower_layer_n].min_space_nm),
        _grow_bbox(upper, layer_by_n[site.upper_layer_n].min_space_nm),
    )
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _derive_site_pair_conflict(
    left: LandingSite,
    right: LandingSite,
    layer_by_n: Mapping[int, _RoutingLayerFacts],
    via_spacing_by_lower_layer: Mapping[int, int],
    world_digest: str,
) -> LandingSitePairConflict | None:
    hard_refs: set[str] = set()
    touch_refs: set[str] = set()
    if left.lower_layer_n == right.lower_layer_n:
        left_cut = left.cut_bbox_nm
        right_cut = right.cut_bbox_nm
        if left_cut is None or right_cut is None:
            raise RoutingResourceEnumerationError(
                "site-pair cut check requires geometry-complete sites"
            )
        via_name = f"via{left.lower_layer_n}"
        hard, touch = _spacing_conflict(
            left_cut,
            right_cut,
            via_spacing_by_lower_layer[left.lower_layer_n],
        )
        if hard or touch:
            # Via cuts never merge electrically, so touching cuts are hard too.
            hard_refs.add(f"{via_name}:cut-spacing")

    left_pads = (
        (left.lower_layer_n, left.lower_pad_bbox_nm),
        (left.upper_layer_n, left.upper_pad_bbox_nm),
    )
    right_pads = (
        (right.lower_layer_n, right.lower_pad_bbox_nm),
        (right.upper_layer_n, right.upper_pad_bbox_nm),
    )
    for left_layer_n, left_pad in left_pads:
        for right_layer_n, right_pad in right_pads:
            if left_layer_n != right_layer_n:
                continue
            if left_pad is None or right_pad is None:
                raise RoutingResourceEnumerationError(
                    "site-pair pad check requires geometry-complete sites"
                )
            hard, touch = _spacing_conflict(
                left_pad,
                right_pad,
                layer_by_n[left_layer_n].min_space_nm,
            )
            if hard:
                hard_refs.add(f"metal{left_layer_n}:pad-spacing")
            elif touch:
                touch_refs.add(f"metal{left_layer_n}:pad-merge")

    if hard_refs:
        # Via cuts never become one legal cut by adding route metal, so any cut
        # rule remains unconditional.  A metal-only pad-spacing conflict is
        # different: an explicitly allocated same-net track may bridge both
        # pads and turn the three rectangles into one conductor.  Preserve that
        # conditional obligation in the ledger; selection and replay must both
        # prove the exact bridge before allowing the pair.
        scope = (
            "same_net_bridge"
            if all(ref.startswith("metal") for ref in hard_refs)
            else "all_nets"
        )
        return LandingSitePairConflict(
            left_site_id=left.site_id,
            right_site_id=right.site_id,
            conflict_scope=scope,
            rule_refs=tuple(sorted(hard_refs | touch_refs)),
            routing_world_provenance_digest=world_digest,
        )
    if touch_refs:
        return LandingSitePairConflict(
            left_site_id=left.site_id,
            right_site_id=right.site_id,
            conflict_scope="foreign_nets",
            rule_refs=tuple(sorted(touch_refs)),
            routing_world_provenance_digest=world_digest,
        )
    return None


def _enumerate_landing_site_pair_conflicts(
    sites: Any,
    layers: Sequence[_RoutingLayerFacts],
    tech: Any,
    world_digest: str,
    *,
    max_site_pair_conflicts: int,
    geometry_budget: _GeometryEvaluationBudget,
) -> tuple[LandingSitePairConflict, ...]:
    layer_by_n = {layer.layer_n: layer for layer in layers}
    via_spacing_by_lower_layer = {
        lower_layer_n: _via_spacing_nm(tech, f"via{lower_layer_n}")
        for lower_layer_n in sorted({site.lower_layer_n for site in sites})
    }
    entries = tuple(sorted(
        [
            (
                _site_influence_bbox(
                    site, layer_by_n, via_spacing_by_lower_layer
                ),
                site,
            )
            for site in sites
        ],
        key=lambda item: (item[0][0], item[0][1], item[1].site_id),
    ))
    influence_bboxes = tuple(item[0] for item in entries)
    influence_root = _build_packed_obstacle_root(influence_bboxes)
    conflicts: list[LandingSitePairConflict] = []
    for right_index, (influence, site) in enumerate(entries):
        # A one-axis active sweep still compares every same-X row before its
        # Y rejection.  Dense landing columns made the cap+1 proof quadratic
        # and delayed an already-certain typed stop until after minutes of
        # unrelated pairs.  Query the immutable two-dimensional influence
        # index and derive only exact bbox intersections.  Under the cap this
        # enumerates the identical pair set; over the cap it remains atomic at
        # the first cap+1 conflict.
        for left_index in _query_packed_bbox_indices(
            influence_root,
            influence_bboxes,
            influence,
            geometry_budget,
            "landing site-pair spatial index check",
            max_index_exclusive=right_index,
        ):
            left_site = entries[left_index][1]
            conflict = _derive_site_pair_conflict(
                left_site,
                site,
                layer_by_n,
                via_spacing_by_lower_layer,
                world_digest,
            )
            if conflict is not None:
                conflicts.append(conflict)
                if len(conflicts) > max_site_pair_conflicts:
                    raise LandingSitePairConflictLimitError(
                        "landing site-pair conflict count exceeds limit "
                        f"{max_site_pair_conflicts}"
                    )
    return tuple(sorted(
        conflicts,
        key=lambda conflict: (conflict.left_site_id, conflict.right_site_id),
    ))


def _landing_site_conflict_domain(
    sites: Sequence[LandingSite] | LandingSiteCatalog,
    layers: Sequence[_RoutingLayerFacts],
    tech: Any,
    world_digest: str,
    *,
    landing_site_collection_digest: str | None = None,
) -> LandingSiteConflictDomain:
    site_values = (
        sites.iter_sites
        if isinstance(sites, LandingSiteCatalog)
        else lambda: iter(sites)
    )
    metal_layer_ns = {
        layer_n
        for site in site_values()
        for layer_n in (site.lower_layer_n, site.upper_layer_n)
    }
    via_layer_ns = {site.lower_layer_n for site in site_values()}
    layer_by_n = {layer.layer_n: layer for layer in layers}
    return LandingSiteConflictDomain(
        routing_world_provenance_digest=world_digest,
        pdk_provenance_digest=pdk_provenance_digest(tech),
        landing_site_collection_digest=(
            (
                sites.collection_sha256
                if isinstance(sites, LandingSiteCatalog)
                else landing_site_collection_sha256(sites)
            )
            if landing_site_collection_digest is None
            else landing_site_collection_digest
        ),
        metal_rules=tuple(
            LandingSiteConflictMetalRule(
                layer_n=layer_n,
                min_space_nm=layer_by_n[layer_n].min_space_nm,
            )
            for layer_n in sorted(metal_layer_ns)
        ),
        via_rules=tuple(
            LandingSiteConflictViaRule(
                lower_layer_n=layer_n,
                cut_spacing_nm=_via_spacing_nm(tech, f"via{layer_n}"),
            )
            for layer_n in sorted(via_layer_ns)
        ),
    )


def _replay_exact_landing_site_membership(
    ledger: ExplicitRoutingResourceLedger,
    world: RoutingWorldProvenance,
    tech: Any,
    owned_conductor_index: Any,
    sites: Sequence[LandingSite],
    *,
    query_context: _ExactLandingSiteQueryContext | None = None,
) -> tuple[LandingSite, ...]:
    """Replay only finite site membership in the exact lazy domain."""

    requested = _bounded_tuple(
        sites,
        name="landing-site domain projection",
        limit=MAX_ENUMERATED_VIA_SITES,
    )
    if not all(isinstance(site, LandingSite) for site in requested):
        raise RoutingResourceEnumerationError(
            "landing-site domain projection contains invalid values"
        )
    replayed: list[LandingSite] = []
    query_context = query_context or _prepare_exact_landing_site_query_context(
        ledger, world, tech, owned_conductor_index
    )
    by_lower_layer: dict[int, list[LandingSite]] = {}
    for site in requested:
        by_lower_layer.setdefault(site.lower_layer_n, []).append(site)
    for lower_layer_n, layer_sites in sorted(by_lower_layer.items()):
        replayed.extend(_query_exact_landing_sites_at_points(
            ledger,
            world,
            tech,
            owned_conductor_index,
            lower_layer_n=lower_layer_n,
            points=tuple(site.xy_nm for site in layer_sites),
            signed_points=False,
            query_context=query_context,
        ))
    expected = tuple(sorted(replayed, key=lambda site: site.site_id))
    actual = tuple(sorted(requested, key=lambda site: site.site_id))
    if actual != expected:
        raise RoutingResourceEnumerationError(
            "landing-site projection does not exactly replay from its domain"
        )
    return actual


def replay_exact_landing_site_catalog_compressed(
    ledger: ExplicitRoutingResourceLedger,
    world: RoutingWorldProvenance,
    tech: Any,
    owned_conductor_index: Any,
    sites: Sequence[LandingSite],
) -> tuple[
    LandingSiteCatalog,
    tuple[LandingSitePairConflict, ...],
    LandingSiteConflictDomain,
]:
    """Replay an aggregate exact projection through bounded shared-budget pages."""

    query_context = _prepare_exact_landing_site_query_context(
        ledger, world, tech, owned_conductor_index
    )
    from clay.layout.route import resource_plan as resource_plan_module

    page_size = min(
        MAX_ENUMERATED_VIA_SITES,
        resource_plan_module.MAX_LANDING_SITES,
    )
    try:
        iterator = iter(sites)
    except TypeError as exc:
        raise RoutingResourceEnumerationError(
            "landing-site catalog projection must be iterable"
        ) from exc
    pages = []
    pending = []
    previous_site_id: str | None = None
    total_count = 0

    def replay_page(page_sites: list[LandingSite]) -> None:
        page_index = len(pages)
        requested_page = tuple(page_sites)
        replayed_page = _replay_exact_landing_site_membership(
            ledger,
            world,
            tech,
            owned_conductor_index,
            requested_page,
            query_context=query_context,
        )
        if replayed_page != requested_page:
            raise RoutingResourceEnumerationError(
                "landing-site catalog projection does not exactly replay from "
                "its domain"
            )
        pages.append(LandingSitePage.build(page_index, replayed_page))

    for site in iterator:
        if not isinstance(site, LandingSite):
            raise RoutingResourceEnumerationError(
                "landing-site catalog projection contains invalid values"
            )
        if previous_site_id is not None and site.site_id <= previous_site_id:
            raise RoutingResourceEnumerationError(
                "landing-site catalog projection ids must be globally unique "
                "and sorted"
            )
        previous_site_id = site.site_id
        pending.append(site)
        total_count += 1
        if total_count > MAX_TOTAL_ASSIGNMENT_ARCS:
            raise RoutingResourceEnumerationLimitError(
                "landing-site catalog projection count exceeds hard maximum "
                f"{MAX_TOTAL_ASSIGNMENT_ARCS}"
            )
        if len(pending) == page_size:
            replay_page(pending)
            pending = []
    if pending:
        replay_page(pending)
    if not pages:
        raise RoutingResourceEnumerationError(
            "landing-site catalog projection must not be empty"
        )
    catalog = LandingSiteCatalog.from_pages(
        tuple(pages),
        routing_world_provenance_digest=world.sha256(),
        landing_site_domain_digest=ledger.landing_site_domain_digest,
    )
    return (
        catalog,
        (),
        _landing_site_conflict_domain(
            catalog,
            _routing_layer_facts(tech),
            tech,
            world.sha256(),
            landing_site_collection_digest=catalog.collection_sha256,
        ),
    )


def replay_exact_landing_site_subset(
    ledger: ExplicitRoutingResourceLedger,
    world: RoutingWorldProvenance,
    tech: Any,
    owned_conductor_index: Any,
    sites: Sequence[LandingSite],
) -> tuple[tuple[LandingSite, ...], tuple[LandingSitePairConflict, ...]]:
    """Replay a finite candidate-pool projection of the exact lazy domain."""

    actual = _replay_exact_landing_site_membership(
        ledger,
        world,
        tech,
        owned_conductor_index,
        sites,
    )
    conflicts = _enumerate_landing_site_pair_conflicts(
        actual,
        _routing_layer_facts(tech),
        tech,
        world.sha256(),
        max_site_pair_conflicts=MAX_ENUMERATED_LANDING_SITE_PAIR_CONFLICTS,
        geometry_budget=_GeometryEvaluationBudget(
            DEFAULT_ENUMERATED_GEOMETRY_EVALUATIONS
        ),
    )
    return actual, conflicts


def replay_exact_landing_site_subset_compressed(
    ledger: ExplicitRoutingResourceLedger,
    world: RoutingWorldProvenance,
    tech: Any,
    owned_conductor_index: Any,
    sites: Sequence[LandingSite],
) -> tuple[
    tuple[LandingSite, ...],
    tuple[LandingSitePairConflict, ...],
    LandingSiteConflictDomain | None,
]:
    """Replay a finite subset, atomically compressing only pair overflow."""

    actual = _replay_exact_landing_site_membership(
        ledger, world, tech, owned_conductor_index, sites
    )
    layers = _routing_layer_facts(tech)
    try:
        conflicts = _enumerate_landing_site_pair_conflicts(
            actual,
            layers,
            tech,
            world.sha256(),
            max_site_pair_conflicts=(
                MAX_ENUMERATED_LANDING_SITE_PAIR_CONFLICTS
            ),
            geometry_budget=_GeometryEvaluationBudget(
                DEFAULT_ENUMERATED_GEOMETRY_EVALUATIONS
            ),
        )
    except LandingSitePairConflictLimitError:
        return (
            actual,
            (),
            _landing_site_conflict_domain(
                actual, layers, tech, world.sha256()
            ),
        )
    return actual, conflicts, None


@resource_boundary(world_argument="world", tech_argument="tech")
def enumerate_explicit_routing_resources(
    world: RoutingWorldProvenance,
    tech: Any,
    *,
    max_track_centers: int = MAX_ENUMERATED_TRACK_CENTERS,
    max_track_segments: int = MAX_ENUMERATED_TRACK_SEGMENTS,
    max_track_resources: int = MAX_ENUMERATED_TRACK_RESOURCES,
    max_via_sites: int = MAX_ENUMERATED_VIA_SITES,
    max_via_resource_pairs: int = MAX_ENUMERATED_VIA_RESOURCE_PAIRS,
    max_site_pair_conflicts: int = MAX_ENUMERATED_LANDING_SITE_PAIR_CONFLICTS,
    max_geometry_evaluations: int = DEFAULT_ENUMERATED_GEOMETRY_EVALUATIONS,
    resource_meter: Any = None,
) -> ExplicitRoutingResourceLedger:
    """Enumerate tracks and an eager-or-exact-lazy landing domain.

    The 32k materialization bound limits certificate payload, not the physical
    lattice.  A larger lattice is represented by its replayable domain digest;
    downstream pattern queries materialize only their exact finite claims.
    """

    if not isinstance(world, RoutingWorldProvenance):
        raise RoutingResourceEnumerationError(
            "world must be RoutingWorldProvenance"
        )
    try:
        world = RoutingWorldProvenance(
            required_nets=world.required_nets,
            placement_provenance_digest=world.placement_provenance_digest,
            topology_provenance_digest=world.topology_provenance_digest,
            pdk_provenance_digest=world.pdk_provenance_digest,
            pins=world.pins,
            obstacles=world.obstacles,
            rails=world.rails,
            terminal_conductor_refs=world.terminal_conductor_refs,
        )
    except (TypeError, ValueError) as exc:
        raise RoutingResourceEnumerationError(
            f"routing world failed canonical entry replay: {exc}"
        ) from exc
    limits = (
        ("max_track_centers", max_track_centers, MAX_ENUMERATED_TRACK_CENTERS),
        ("max_track_segments", max_track_segments, MAX_ENUMERATED_TRACK_SEGMENTS),
        (
            "max_track_resources",
            max_track_resources,
            MAX_ENUMERATED_TRACK_RESOURCES,
        ),
        ("max_via_sites", max_via_sites, MAX_ENUMERATED_VIA_SITES),
        (
            "max_via_resource_pairs",
            max_via_resource_pairs,
            MAX_ENUMERATED_VIA_RESOURCE_PAIRS,
        ),
        (
            "max_site_pair_conflicts",
            max_site_pair_conflicts,
            MAX_ENUMERATED_LANDING_SITE_PAIR_CONFLICTS,
        ),
        (
            "max_geometry_evaluations",
            max_geometry_evaluations,
            MAX_ENUMERATED_GEOMETRY_EVALUATIONS,
        ),
    )
    for name, limit, hard_maximum in limits:
        if type(limit) is not int or limit <= 0:
            raise RoutingResourceEnumerationLimitError(
                f"{name} must be a positive integer"
            )
        if limit > hard_maximum:
            raise RoutingResourceEnumerationLimitError(
                f"{name} exceeds hard maximum {hard_maximum}"
            )
    actual_pdk_digest = pdk_provenance_digest(tech)
    if world.pdk_provenance_digest != actual_pdk_digest:
        raise RoutingResourceEnumerationError(
            "routing world PDK digest does not match enumeration tech"
        )
    layers = _routing_layer_facts(tech)
    geometry_budget = bind_budget(
        _GeometryEvaluationBudget(max_geometry_evaluations), resource_meter, "geometry"
    )
    obstacles = _world_obstacle_facts(world)
    geometry_index = _LayerGeometryIndex.build(obstacles, world.pins)
    from clay.layout.route.resource_conductors import (
        OwnedConductorComponentIndex,
        OwnedConductorIndexError,
    )

    try:
        owned_conductor_index = OwnedConductorComponentIndex.build_planar(
            world, tech
        )
    except OwnedConductorIndexError as exc:
        raise RoutingResourceEnumerationError(
            f"owned-conductor component index failed: {exc}"
        ) from exc
    tracks = _enumerate_track_resources(
        world,
        layers,
        owned_conductor_index,
        max_track_centers=max_track_centers,
        max_track_segments=max_track_segments,
        max_track_resources=max_track_resources,
        geometry_budget=geometry_budget,
        geometry_index=geometry_index,
    )
    landing_sites, landing_sites_complete = _enumerate_landing_sites(
        world,
        tech,
        layers,
        tracks,
        obstacles,
        geometry_index,
        owned_conductor_index,
        max_via_sites=max_via_sites,
        max_via_resource_pairs=max_via_resource_pairs,
        geometry_budget=geometry_budget,
    )
    # An incomplete ledger is only the immutable query domain plus mandatory
    # access evidence; it is not a candidate selection.  Sweeping every pair in
    # that base listing is both unnecessary and potentially quadratic.  The
    # finite option-site union is replayed, including every pair conflict, when
    # a CandidatePool is compiled or verified.
    site_conflicts = (
        _enumerate_landing_site_pair_conflicts(
            landing_sites,
            layers,
            tech,
            world.sha256(),
            max_site_pair_conflicts=max_site_pair_conflicts,
            geometry_budget=geometry_budget,
        )
        if landing_sites_complete
        else ()
    )
    ledger = ExplicitRoutingResourceLedger(
        routing_world_provenance_digest=world.sha256(),
        track_resources=tracks,
        landing_sites=landing_sites,
        landing_site_pair_conflicts=site_conflicts,
        owned_conductor_index_digest=owned_conductor_index.sha256(),
        landing_site_domain_digest=(
            ""
            if landing_sites_complete
            else _landing_site_domain_digest(
                world.sha256(),
                tracks,
                world.pdk_provenance_digest,
                owned_conductor_index.sha256(),
            )
        ),
        landing_sites_complete=landing_sites_complete,
    )
    if resource_meter is not None:
        resource_meter.register_ledger(ledger, geometry_budget)
    return ledger


def verify_explicit_routing_resource_ledger(
    ledger: ExplicitRoutingResourceLedger,
    world: RoutingWorldProvenance,
    tech: Any,
    *,
    max_track_centers: int = MAX_ENUMERATED_TRACK_CENTERS,
    max_track_segments: int = MAX_ENUMERATED_TRACK_SEGMENTS,
    max_track_resources: int = MAX_ENUMERATED_TRACK_RESOURCES,
    max_via_sites: int = MAX_ENUMERATED_VIA_SITES,
    max_via_resource_pairs: int = MAX_ENUMERATED_VIA_RESOURCE_PAIRS,
    max_site_pair_conflicts: int = MAX_ENUMERATED_LANDING_SITE_PAIR_CONFLICTS,
    max_geometry_evaluations: int = DEFAULT_ENUMERATED_GEOMETRY_EVALUATIONS,
) -> bool:
    """Replay final-world enumeration and require an exact ledger certificate."""

    if not isinstance(ledger, ExplicitRoutingResourceLedger) or not isinstance(
        world, RoutingWorldProvenance
    ):
        raise RoutingResourceLedgerVerificationError(
            "ledger and routing world types must match"
        )
    if ledger.routing_world_provenance_digest != world.sha256():
        raise RoutingResourceLedgerVerificationError(
            "ledger world digest does not match the final routing world"
        )
    expected = enumerate_explicit_routing_resources(
        world,
        tech,
        max_track_centers=max_track_centers,
        max_track_segments=max_track_segments,
        max_track_resources=max_track_resources,
        max_via_sites=max_via_sites,
        max_via_resource_pairs=max_via_resource_pairs,
        max_site_pair_conflicts=max_site_pair_conflicts,
        max_geometry_evaluations=max_geometry_evaluations,
    )
    if ledger == expected:
        return True
    if not ledger.landing_sites_complete:
        from clay.layout.route.resource_conductors import (
            OwnedConductorComponentIndex,
        )

        if ledger.landing_site_pair_conflicts:
            raise RoutingResourceLedgerVerificationError(
                "an incomplete base ledger must defer landing site-pair conflicts"
            )

        canonical_domain_digest = _landing_site_domain_digest(
            world.sha256(),
            expected.track_resources,
            world.pdk_provenance_digest,
            expected.owned_conductor_index_digest,
        )
        lazy_domain_ledger = ExplicitRoutingResourceLedger(
            routing_world_provenance_digest=world.sha256(),
            track_resources=expected.track_resources,
            landing_sites=(),
            landing_site_pair_conflicts=(),
            owned_conductor_index_digest=expected.owned_conductor_index_digest,
            landing_site_domain_digest=canonical_domain_digest,
            landing_sites_complete=False,
        )
        sites = _replay_exact_landing_site_membership(
            lazy_domain_ledger,
            world,
            tech,
            OwnedConductorComponentIndex.build_planar(world, tech),
            ledger.landing_sites,
        )
        expected = ExplicitRoutingResourceLedger(
            routing_world_provenance_digest=world.sha256(),
            track_resources=expected.track_resources,
            landing_sites=sites,
            landing_site_pair_conflicts=(),
            owned_conductor_index_digest=expected.owned_conductor_index_digest,
            landing_site_domain_digest=canonical_domain_digest,
            landing_sites_complete=False,
        )
    if ledger != expected:
        raise RoutingResourceLedgerVerificationError(
            "ledger does not match exact replay of the final routing world"
        )
    return True


def verify_candidate_pool_applicability(
    pool: CandidatePool,
    world: RoutingWorldProvenance,
    *,
    tech: Any = None,
) -> bool:
    """Require exact pool/world identity before any selected plan is forwarded."""

    if not isinstance(pool, CandidatePool) or not isinstance(
        world, RoutingWorldProvenance
    ):
        raise CandidatePoolApplicabilityError(
            "candidate pool and routing world types must match"
        )
    checks = (
        ("required_nets", pool.required_nets, world.required_nets),
        (
            "placement_provenance_digest",
            pool.placement_provenance_digest,
            world.placement_provenance_digest,
        ),
        (
            "topology_provenance_digest",
            pool.topology_provenance_digest,
            world.topology_provenance_digest,
        ),
        (
            "pdk_provenance_digest",
            pool.pdk_provenance_digest,
            world.pdk_provenance_digest,
        ),
        (
            "routing_world_provenance_digest",
            pool.routing_world_provenance_digest,
            world.routing_world_provenance_digest,
        ),
    )
    for name, actual, expected in checks:
        if actual != expected:
            raise CandidatePoolApplicabilityError(
                f"candidate pool {name} mismatch for final routing world"
            )
    if tech is None:
        expected_empty_hash = explicit_routing_resource_ledger_sha256(
            world.sha256(),
            (),
            (),
            (),
            pool.owned_conductor_index_digest,
        )
        if (
            not pool.resources
            and not pool.landing_sites
            and not pool.landing_site_conflicts
            and pool.routing_resource_ledger_hash == expected_empty_hash
        ):
            # Backward-compatible provenance-only diagnostics may validate an
            # explicitly hashed empty ledger without a PDK object.  The canonical
            # route path always supplies ``tech`` and takes exact replay below.
            return True
        raise CandidatePoolApplicabilityError(
            "world-bound explicit resource ledger requires tech for exact replay"
        )
    if pool.routing_world_provenance_digest:
        ledger = enumerate_explicit_routing_resources(world, tech)
        if pool.landing_sites_complete:
            expected_ledger = ledger
            expected_catalog = None
            expected_ledger_hash = expected_ledger.sha256()
        else:
            from clay.layout.route.resource_conductors import (
                OwnedConductorComponentIndex,
            )

            conductor_index = OwnedConductorComponentIndex.build_planar(
                world, tech
            )
            canonical_domain_digest = _landing_site_domain_digest(
                world.sha256(),
                ledger.track_resources,
                world.pdk_provenance_digest,
                ledger.owned_conductor_index_digest,
            )
            lazy_domain_ledger = ExplicitRoutingResourceLedger(
                routing_world_provenance_digest=world.sha256(),
                track_resources=ledger.track_resources,
                landing_sites=(),
                landing_site_pair_conflicts=(),
                owned_conductor_index_digest=ledger.owned_conductor_index_digest,
                landing_site_domain_digest=canonical_domain_digest,
                landing_sites_complete=False,
            )
            try:
                if pool.landing_site_catalog is None:
                    sites, conflicts, conflict_domain = (
                        replay_exact_landing_site_subset_compressed(
                            lazy_domain_ledger,
                            world,
                            tech,
                            conductor_index,
                            pool.landing_sites,
                        )
                    )
                    expected_catalog = None
                else:
                    (
                        expected_catalog,
                        conflicts,
                        conflict_domain,
                    ) = replay_exact_landing_site_catalog_compressed(
                        lazy_domain_ledger,
                        world,
                        tech,
                        conductor_index,
                        pool.iter_landing_sites(),
                    )
                    sites = ()
            except RoutingResourceEnumerationError as exc:
                raise CandidatePoolApplicabilityError(
                    f"candidate pool landing projection failed exact replay: {exc}"
                ) from exc
            if expected_catalog is None:
                expected_ledger = ExplicitRoutingResourceLedger(
                    routing_world_provenance_digest=world.sha256(),
                    track_resources=ledger.track_resources,
                    landing_sites=sites,
                    landing_site_pair_conflicts=conflicts,
                    owned_conductor_index_digest=(
                        ledger.owned_conductor_index_digest
                    ),
                    landing_site_domain_digest=canonical_domain_digest,
                    landing_sites_complete=False,
                    landing_site_conflict_domain=conflict_domain,
                )
                expected_ledger_hash = expected_ledger.sha256()
            else:
                expected_ledger = None
                expected_ledger_hash = paged_routing_resource_ledger_sha256(
                    world.sha256(),
                    ledger.track_resources,
                    expected_catalog,
                    conflicts,
                    ledger.owned_conductor_index_digest,
                    canonical_domain_digest,
                    False,
                    conflict_domain,
                )
        expected_sites = (
            expected_ledger.landing_sites
            if expected_ledger is not None
            else ()
        )
        expected_conflicts = (
            expected_ledger.landing_site_pair_conflicts
            if expected_ledger is not None
            else conflicts
        )
        expected_conflict_domain = (
            expected_ledger.landing_site_conflict_domain
            if expected_ledger is not None
            else conflict_domain
        )
        expected_domain_digest = (
            expected_ledger.landing_site_domain_digest
            if expected_ledger is not None
            else canonical_domain_digest
        )
        expected_complete = (
            expected_ledger.landing_sites_complete
            if expected_ledger is not None
            else False
        )
        if (
            pool.routing_resource_ledger_hash != expected_ledger_hash
            or pool.owned_conductor_index_digest
            != ledger.owned_conductor_index_digest
            or pool.resources != ledger.track_resources
            or pool.landing_sites != expected_sites
            or pool.landing_site_catalog != expected_catalog
            or pool.landing_site_conflicts
            != expected_conflicts
            or pool.landing_site_domain_digest
            != expected_domain_digest
            or pool.landing_sites_complete
            != expected_complete
            or pool.landing_site_conflict_domain
            != expected_conflict_domain
        ):
            raise CandidatePoolApplicabilityError(
                "candidate pool explicit resource ledger does not match exact replay"
            )
    return True


__all__ = [
    "CandidatePoolApplicabilityError",
    "ExplicitRoutingResourceLedger",
    "MAX_ENUMERATED_GEOMETRY_EVALUATIONS",
    "DEFAULT_ENUMERATED_GEOMETRY_EVALUATIONS",
    "MAX_ENUMERATED_LANDING_SITE_PAIR_CONFLICTS",
    "MAX_ENUMERATED_TRACK_CENTERS",
    "MAX_ENUMERATED_TRACK_RESOURCES",
    "MAX_ENUMERATED_TRACK_SEGMENTS",
    "MAX_ENUMERATED_VIA_SITES",
    "MAX_ENUMERATED_VIA_RESOURCE_PAIRS",
    "MAX_ROUTING_WORLD_OBSTACLES",
    "MAX_ROUTING_WORLD_PINS",
    "MAX_ROUTING_WORLD_RAILS",
    "MAX_ROUTING_WORLD_REQUIRED_NETS",
    "RoutingWorldProvenance",
    "RoutingResourceEnumerationError",
    "RoutingResourceEnumerationLimitError",
    "LandingSitePairConflictLimitError",
    "RoutingResourceLedgerVerificationError",
    "build_routing_world_provenance",
    "canonical_pdk_routing_json",
    "enumerate_explicit_routing_resources",
    "pdk_provenance_digest",
    "query_exact_landing_sites_at_points",
    "query_exact_landing_sites_for_component_lane",
    "query_exact_landing_sites_for_lane_pair",
    "replay_exact_landing_site_subset",
    "replay_exact_landing_site_catalog_compressed",
    "replay_exact_landing_site_subset_compressed",
    "verify_candidate_pool_applicability",
    "verify_explicit_routing_resource_ledger",
]
