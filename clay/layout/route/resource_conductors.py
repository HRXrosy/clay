"""Deterministic connected-component index for owned routing conductors.

The routing resource compiler needs to know whether a via pad lands on an
existing conductor of the same net, bridges two disconnected conductors, or
touches foreign metal.  This module builds that fact from the final routing
world.  Connectivity is deliberately geometric and exact: closed rectangles
that touch are connected; spacing halos and pins never merge components.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isqrt
from typing import Any, Iterable, Mapping, Sequence


BBoxNm = tuple[int, int, int, int]
LayerKey = tuple[int, int]
PinAccessFact = tuple[Any, ...]
AtomFact = tuple[int, int, int, int, int, int, str, str]

_HARD_MAX_ATOMS = 200_000
_HARD_MAX_RAW_FACTS = 400_000
_HARD_MAX_RAW_PINS = 200_000
_HARD_MAX_PAIR_EVALUATIONS = 2_000_000
_HARD_MAX_COMPONENTS = 200_000
_HARD_MAX_PIN_MEMBERSHIPS = 2_000_000
_HARD_MAX_ROUTING_STACK_LAYERS = 64
_HARD_MAX_LANDING_PAD_QUERIES = 200_000
_INDEX_BUILD_TOKEN = object()
# Landing provenance is queried at every candidate via centre.  A wide leaf
# makes each exact point query re-test many unrelated conductor atoms; on a
# small analog cell that can dominate the bounded geometry proof even though
# the STR tree itself is tiny.  A production replay measured 867,308 primitive
# probes (plus 1,005,106 node probes) at fanout eight before exhausting the 2M
# exact-geometry budget.  Four trades one shallow level for at most half as many
# unrelated leaf primitives per landing query.  This is an index-shape
# constant only: component identity and the public canonical digest do not
# depend on private STR nodes.
_SPATIAL_NODE_CAPACITY = 4

# Literal, owned conductor kinds that may carry a compiler-issued terminal seam.
# ``device`` is the historical M1 terminal artwork. ``device_upper`` is the same
# stamp terminal continued onto real upper metal (for example a MIM plate/tab).
# ``guard-ring-m1`` is literal M1 artwork whose unique exact tap pin authorizes
# the already materialized frame as one terminal component. Routing abstractions
# and cuts are intentionally absent. Device-interface collar construction remains
# M1/device-only in resource_world; this set only preserves identity in the
# canonical world/component index.
TERMINAL_CONDUCTOR_KINDS = frozenset({
    "device",
    "device_upper",
    "guard-ring-m1",
})


class OwnedConductorIndexError(ValueError):
    """The owned-conductor world or a query is malformed."""


class OwnedConductorIndexLimitError(OwnedConductorIndexError):
    """A declared bounded-build limit was exceeded."""


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _digest(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _bounded_positive_int(name: str, value: Any, hard_max: int) -> int:
    if type(value) is not int or value <= 0:
        raise OwnedConductorIndexError(f"{name} must be a positive integer")
    if value > hard_max:
        raise OwnedConductorIndexLimitError(
            f"{name}={value} exceeds hard maximum {hard_max}"
        )
    return value


def _integer(name: str, value: Any) -> int:
    if type(value) is not int:
        raise OwnedConductorIndexError(f"{name} must be an integer")
    return value


def _layer_key(value: Any) -> LayerKey:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise OwnedConductorIndexError(
            f"layer_key must be a two-integer GDS key, got {value!r}"
        )
    return _integer("layer_key[0]", value[0]), _integer(
        "layer_key[1]", value[1]
    )


def _routing_stack_layer_keys(
    tech: Any,
) -> tuple[frozenset[LayerKey], frozenset[LayerKey]]:
    """Resolve PDK-declared planar metals and non-planar interconnect cuts."""

    metals: list[LayerKey] = []
    for layer_n in range(1, _HARD_MAX_ROUTING_STACK_LAYERS + 1):
        try:
            metals.append(_layer_key(tech.metal_key(layer_n)))
        except (AttributeError, KeyError, TypeError, ValueError):
            break
    if not metals:
        raise OwnedConductorIndexError(
            "tech exposes no PDK-declared planar routing metal"
        )
    if len(metals) != len(set(metals)):
        raise OwnedConductorIndexError(
            "tech reuses one GDS key for multiple routing metals"
        )

    cuts: set[LayerKey] = set()
    for lower_layer_n in range(1, len(metals)):
        try:
            cuts.add(_layer_key(tech.via_key(lower_layer_n)))
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    try:
        cuts.add(_layer_key(tech.contact_key()))
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    if set(metals) & cuts:
        raise OwnedConductorIndexError(
            "tech aliases a planar routing metal and interconnect cut GDS key"
        )
    return frozenset(metals), frozenset(cuts)


def _bbox(value: Any, *, name: str = "bbox_nm") -> BBoxNm:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise OwnedConductorIndexError(
            f"{name} must contain four integer coordinates"
        )
    result = tuple(_integer(f"{name}[{index}]", item) for index, item in enumerate(value))
    if result[2] < result[0] or result[3] < result[1]:
        raise OwnedConductorIndexError(f"{name} is inverted: {result!r}")
    return result  # type: ignore[return-value]


def _closed_touch(left: BBoxNm, right: BBoxNm) -> bool:
    return not (
        left[2] < right[0]
        or right[2] < left[0]
        or left[3] < right[1]
        or right[3] < left[1]
    )


def _point_inside(point: tuple[int, int], bbox: BBoxNm) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


@dataclass(frozen=True)
class OwnedConductorComponent:
    """One same-layer, same-net connected conductor component."""

    component_ref: str
    component_geometry_hash: str
    world_digest: str
    layer_key: LayerKey
    net: str
    bbox_nm: BBoxNm
    atom_facts: tuple[AtomFact, ...]
    atom_refs: tuple[str, ...]
    kinds: tuple[str, ...]
    atom_terminal_conductor_refs: tuple[str | None, ...] = ()
    pin_accesses: tuple[PinAccessFact, ...] = ()

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "atom_facts": [list(fact) for fact in self.atom_facts],
            "atom_refs": list(self.atom_refs),
            "atom_terminal_conductor_refs": list(
                self.atom_terminal_conductor_refs
            ),
            "bbox_nm": list(self.bbox_nm),
            "component_geometry_hash": self.component_geometry_hash,
            "component_ref": self.component_ref,
            "kinds": list(self.kinds),
            "layer_key": list(self.layer_key),
            "net": self.net,
            "pin_accesses": [list(fact) for fact in self.pin_accesses],
            "world_digest": self.world_digest,
        }


@dataclass(frozen=True)
class LandingPadContact:
    """Exact owned-conductor contacts made by one landing-pad rectangle."""

    status: str
    contacts_by_net: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        if self.status not in {
            "FREE",
            "OWNED",
            "MERGES_COMPONENTS",
            "FOREIGN_NET_CONTACT",
        }:
            raise OwnedConductorIndexError(
                f"invalid landing-pad contact status {self.status!r}"
            )

    @property
    def nets(self) -> tuple[str, ...]:
        return tuple(net for net, _refs in self.contacts_by_net)

    def refs_for_net(self, net: str) -> tuple[str, ...]:
        for item_net, refs in self.contacts_by_net:
            if item_net == net:
                return refs
        return ()


@dataclass(frozen=True)
class _Atom:
    bbox_nm: BBoxNm
    layer_key: LayerKey
    net: str
    kind: str
    terminal_conductor_ref: str | None
    fact: AtomFact
    atom_ref: str


@dataclass(frozen=True)
class _ContactPrimitive:
    """One exact conductor atom (or one atom-free pin access) in the STR tree."""

    bbox_nm: BBoxNm
    net: str
    component_ref: str
    atom_ref: str


@dataclass(frozen=True)
class _PadCandidate:
    """One exact, deduplicated landing pad in a private batch self-join."""

    bbox_nm: BBoxNm
    ordinal: int
    stable_ref: str


@dataclass(frozen=True)
class _AtomPairCandidate:
    """One indexed atom used by the exact component self-join."""

    bbox_nm: BBoxNm
    ordinal: int
    stable_ref: str


@dataclass(frozen=True)
class _SpatialNode:
    """One immutable node in the private, bulk-loaded 2-D STR tree."""

    bbox_nm: BBoxNm
    children: tuple[Any, ...]
    leaves: bool
    stable_ref: str


def _bbox_union(items: Sequence[Any]) -> BBoxNm:
    if not items:
        raise OwnedConductorIndexError("spatial-index node cannot be empty")
    return (
        min(item.bbox_nm[0] for item in items),
        min(item.bbox_nm[1] for item in items),
        max(item.bbox_nm[2] for item in items),
        max(item.bbox_nm[3] for item in items),
    )


def _ceil_sqrt(value: int) -> int:
    root = isqrt(value)
    return root if root * root == value else root + 1


def _spatial_stable_ref(item: Any) -> str:
    if isinstance(item, _ContactPrimitive):
        return item.atom_ref
    if isinstance(item, _PadCandidate):
        return item.stable_ref
    if isinstance(item, _AtomPairCandidate):
        return item.stable_ref
    if isinstance(item, _SpatialNode):
        return item.stable_ref
    raise OwnedConductorIndexError("malformed private spatial-index item")


def _str_groups(items: Sequence[Any], capacity: int) -> tuple[tuple[Any, ...], ...]:
    """Return deterministic Sort-Tile-Recursive groups of at most ``capacity``."""

    if not items:
        return ()
    group_count = (len(items) + capacity - 1) // capacity
    slice_count = _ceil_sqrt(group_count)
    groups_per_slice = (group_count + slice_count - 1) // slice_count
    slice_size = groups_per_slice * capacity
    by_x = sorted(
        items,
        key=lambda item: (
            item.bbox_nm[0] + item.bbox_nm[2],
            item.bbox_nm[1] + item.bbox_nm[3],
            item.bbox_nm,
            _spatial_stable_ref(item),
        ),
    )
    groups: list[tuple[Any, ...]] = []
    for start in range(0, len(by_x), slice_size):
        spatial_slice = sorted(
            by_x[start : start + slice_size],
            key=lambda item: (
                item.bbox_nm[1] + item.bbox_nm[3],
                item.bbox_nm[0] + item.bbox_nm[2],
                item.bbox_nm,
                _spatial_stable_ref(item),
            ),
        )
        for offset in range(0, len(spatial_slice), capacity):
            groups.append(tuple(spatial_slice[offset : offset + capacity]))
    return tuple(groups)


def _spatial_node(children: Sequence[Any], *, leaves: bool) -> _SpatialNode:
    ordered_children = tuple(children)
    bbox_nm = _bbox_union(ordered_children)
    stable_ref = "osn:" + _digest(
        {
            "bbox_nm": list(bbox_nm),
            "children": [
                _spatial_stable_ref(child) for child in ordered_children
            ],
            "leaves": leaves,
            "schema": "clay.owned-conductor-private-str-node.v1",
        }
    )
    return _SpatialNode(
        bbox_nm=bbox_nm,
        children=ordered_children,
        leaves=leaves,
        stable_ref=stable_ref,
    )


def _build_str_tree(items: Sequence[Any]) -> _SpatialNode | None:
    if not items:
        return None
    leaves = [
        _spatial_node(group, leaves=True)
        for group in _str_groups(items, _SPATIAL_NODE_CAPACITY)
    ]
    level = leaves
    while len(level) > _SPATIAL_NODE_CAPACITY:
        level = [
            _spatial_node(group, leaves=False)
            for group in _str_groups(level, _SPATIAL_NODE_CAPACITY)
        ]
    if len(level) == 1:
        return level[0]
    return _spatial_node(tuple(level), leaves=False)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, item: int) -> int:
        parent = self._parent[item]
        while parent != item:
            self._parent[item] = self._parent[parent]
            item = self._parent[item]
            parent = self._parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self._rank[left_root] < self._rank[right_root]:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        if self._rank[left_root] == self._rank[right_root]:
            self._rank[left_root] += 1


class OwnedConductorComponentIndex:
    """Immutable, world-bound index of owned conductor components."""

    def __init__(
        self,
        *,
        world_digest: str,
        components: Iterable[OwnedConductorComponent],
        pin_bindings: Iterable[tuple[PinAccessFact, str]],
        _build_token: Any = None,
    ) -> None:
        if _build_token is not _INDEX_BUILD_TOKEN:
            raise OwnedConductorIndexError(
                "OwnedConductorComponentIndex is build-only; use build(world, ...)"
            )
        if not isinstance(world_digest, str) or not world_digest:
            raise OwnedConductorIndexError("world_digest must be non-empty")
        self.world_digest = world_digest
        self._components = tuple(
            sorted(components, key=lambda component: component.component_ref)
        )
        self._by_ref = {
            component.component_ref: component for component in self._components
        }
        if len(self._by_ref) != len(self._components):
            raise OwnedConductorIndexError("duplicate owned-conductor component_ref")
        for component in self._components:
            self._validate_component(component)
        grouped: dict[tuple[LayerKey, str], list[OwnedConductorComponent]] = {}
        by_layer: dict[LayerKey, list[OwnedConductorComponent]] = {}
        for component in self._components:
            grouped.setdefault((component.layer_key, component.net), []).append(component)
            by_layer.setdefault(component.layer_key, []).append(component)
        self._by_layer_net = {
            key: tuple(sorted(value, key=lambda item: item.component_ref))
            for key, value in grouped.items()
        }
        self._by_layer = {
            key: tuple(sorted(value, key=lambda item: item.component_ref))
            for key, value in by_layer.items()
        }
        binding_map: dict[PinAccessFact, str] = {}
        for pin, component_ref in sorted(
            pin_bindings, key=lambda item: _canonical_json([list(item[0]), item[1]])
        ):
            if component_ref not in self._by_ref:
                raise OwnedConductorIndexError(
                    f"pin binding names missing component {component_ref!r}"
                )
            if pin in binding_map and binding_map[pin] != component_ref:
                raise OwnedConductorIndexError(
                    "one pin-access fact binds to multiple conductor components"
                )
            binding_map[pin] = component_ref
        for component in self._components:
            for pin in component.pin_accesses:
                if binding_map.get(pin) != component.component_ref:
                    raise OwnedConductorIndexError(
                        "component pin_accesses and pin bindings disagree"
                    )
        expected_bindings = {
            (pin, component.component_ref)
            for component in self._components
            for pin in component.pin_accesses
        }
        if set(binding_map.items()) != expected_bindings:
            raise OwnedConductorIndexError(
                "pin bindings must exactly equal component pin_accesses"
            )
        self._pin_bindings = binding_map
        primitives_by_layer: dict[LayerKey, list[_ContactPrimitive]] = {}
        for component in self._components:
            if component.atom_facts:
                component_primitives = tuple(
                    _ContactPrimitive(
                        bbox_nm=fact[:4],
                        net=component.net,
                        component_ref=component.component_ref,
                        atom_ref=atom_ref,
                    )
                    for fact, atom_ref in zip(
                        component.atom_facts, component.atom_refs, strict=True
                    )
                )
            else:
                pin = component.pin_accesses[0]
                point = (int(pin[3]), int(pin[4]))
                component_primitives = (
                    _ContactPrimitive(
                        bbox_nm=(point[0], point[1], point[0], point[1]),
                        net=component.net,
                        component_ref=component.component_ref,
                        atom_ref="op:" + _digest(
                            {
                                "component_ref": component.component_ref,
                                "pin_access": list(pin),
                                "schema": (
                                    "clay.owned-conductor-private-pin-primitive.v1"
                                ),
                            }
                        ),
                    ),
                )
            primitives_by_layer.setdefault(component.layer_key, []).extend(
                component_primitives
            )
        self._spatial_roots = {
            layer_key: _build_str_tree(
                tuple(
                    sorted(
                        primitives,
                        key=lambda primitive: (
                            primitive.bbox_nm,
                            primitive.net,
                            primitive.component_ref,
                            primitive.atom_ref,
                        ),
                    )
                )
            )
            for layer_key, primitives in sorted(primitives_by_layer.items())
        }

    def _validate_component(self, component: Any) -> None:
        if not isinstance(component, OwnedConductorComponent):
            raise OwnedConductorIndexError(
                "components must contain OwnedConductorComponent values"
            )
        if component.world_digest != self.world_digest:
            raise OwnedConductorIndexError(
                "component world_digest does not match index world_digest"
            )
        if component.component_ref != (
            f"oc:{self.world_digest}:{component.component_geometry_hash}"
        ):
            raise OwnedConductorIndexError(
                "component_ref is not bound to index world and geometry hash"
            )
        atom_facts = tuple(component.atom_facts)
        atom_terminal_refs = tuple(component.atom_terminal_conductor_refs)
        pin_accesses = tuple(component.pin_accesses)
        if (
            not isinstance(component.net, str)
            or not component.net
            or _layer_key(component.layer_key) != component.layer_key
            or _bbox(component.bbox_nm) != component.bbox_nm
        ):
            raise OwnedConductorIndexError(
                "component owner or bbox is malformed"
            )
        if atom_facts and len(atom_terminal_refs) != len(atom_facts):
            raise OwnedConductorIndexError(
                "component terminal refs must align exactly with atom facts"
            )
        if not atom_facts and atom_terminal_refs:
            raise OwnedConductorIndexError(
                "pin-only component cannot carry atom terminal refs"
            )
        atom_members = tuple(zip(
            atom_facts,
            atom_terminal_refs,
            strict=True,
        ))
        if (
            len(atom_facts) != len(set(atom_facts))
            or len(component.atom_refs) != len(set(component.atom_refs))
            or len(atom_members) != len(set(atom_members))
        ):
            raise OwnedConductorIndexError(
                "component atom facts, refs, and terminal-ref pairs must each "
                "be exact unique members"
            )
        for fact, terminal_ref in zip(
            atom_facts, atom_terminal_refs, strict=True
        ):
            if not isinstance(fact, tuple) or len(fact) != 8:
                raise OwnedConductorIndexError("component atom fact is malformed")
            _bbox(fact[:4], name="component atom bbox")
            _layer_key(fact[4:6])
            if (
                not isinstance(fact[6], str)
                or not fact[6]
                or not isinstance(fact[7], str)
                or not fact[7]
            ):
                raise OwnedConductorIndexError(
                    "component atom owner/kind is malformed"
                )
            if terminal_ref is not None and (
                not isinstance(terminal_ref, str)
                or not terminal_ref
                or fact[7] not in TERMINAL_CONDUCTOR_KINDS
            ):
                raise OwnedConductorIndexError(
                    "component terminal ref is malformed or bound to a "
                    "non-terminal conductor atom"
                )
        from clay.layout.route.terminal_identity import unpack_pin_fact
        for pin in pin_accesses:
            try:
                unpack_pin_fact(pin)
            except (TypeError, ValueError) as exc:
                raise OwnedConductorIndexError("component pin fact is malformed") from exc
            _integer("component pin x", pin[3])
            _integer("component pin y", pin[4])
            _layer_key(pin[5:7])
            if any(
                not isinstance(identity, str) or not identity
                for identity in pin[:3]
            ):
                raise OwnedConductorIndexError(
                    "component pin net/device/terminal identity is malformed"
                )
        if atom_facts:
            expected_atom_refs = tuple(
                "oa:" + _digest(
                    {
                        "fact": list(fact),
                        "schema": "clay.owned-conductor-atom.v2",
                        "terminal_conductor_ref": terminal_ref,
                    }
                )
                for fact, terminal_ref in zip(
                    atom_facts, atom_terminal_refs, strict=True
                )
            )
            if component.atom_refs != expected_atom_refs:
                raise OwnedConductorIndexError(
                    "component atom_refs do not match atom facts"
                )
            if tuple(sorted(component.atom_refs)) != component.atom_refs:
                raise OwnedConductorIndexError(
                    "component atoms must be in canonical atom-ref order"
                )
            if any(
                tuple(fact[4:6]) != component.layer_key
                or fact[6] != component.net
                for fact in atom_facts
            ):
                raise OwnedConductorIndexError(
                    "component atom layer/net does not match component owner"
                )
            expected_hash = _digest(
                {
                    "atoms": [list(fact) for fact in atom_facts],
                    "atom_terminal_conductor_refs": list(atom_terminal_refs),
                    "layer_key": list(component.layer_key),
                    "net": component.net,
                    "schema": "clay.owned-conductor-component-geometry.v2",
                }
            )
            expected_bbox = (
                min(fact[0] for fact in atom_facts),
                min(fact[1] for fact in atom_facts),
                max(fact[2] for fact in atom_facts),
                max(fact[3] for fact in atom_facts),
            )
            expected_kinds = tuple(sorted({fact[7] for fact in atom_facts}))
            if any(
                not any(
                    _point_inside((int(pin[3]), int(pin[4])), fact[:4])
                    for fact in atom_facts
                )
                for pin in pin_accesses
            ):
                raise OwnedConductorIndexError(
                    "bound pin lies outside its component atoms"
                )
        else:
            if len(pin_accesses) != 1:
                raise OwnedConductorIndexError(
                    "atom-free component must contain exactly one pin access"
                )
            pin = pin_accesses[0]
            expected_hash = _digest(
                {
                    "layer_key": list(component.layer_key),
                    "net": component.net,
                    "pin_access": list(pin),
                    "schema": "clay.owned-conductor-pin-access-geometry.v1",
                }
            )
            expected_bbox = (int(pin[3]), int(pin[4]), int(pin[3]), int(pin[4]))
            expected_kinds = ("pin-access",)
            if component.atom_refs:
                raise OwnedConductorIndexError(
                    "pin-only component cannot contain atom refs"
                )
        if component.component_geometry_hash != expected_hash:
            raise OwnedConductorIndexError(
                "component geometry hash does not replay"
            )
        if component.bbox_nm != expected_bbox or component.kinds != expected_kinds:
            raise OwnedConductorIndexError(
                "component bbox/kinds do not replay from its members"
            )
        for pin in pin_accesses:
            if pin[0] != component.net or tuple(pin[5:7]) != component.layer_key:
                raise OwnedConductorIndexError(
                    "component pin layer/net does not match component owner"
                )

    @classmethod
    def build(
        cls,
        world: Any,
        *,
        max_atoms: int = 100_000,
        max_pair_evaluations: int = 1_000_000,
        max_components: int = 100_000,
        max_pin_memberships: int = 1_000_000,
    ) -> "OwnedConductorComponentIndex":
        """Build an unclassified deterministic index for diagnostic consumers."""

        return cls._build_index(
            world,
            planar_metal_layer_keys=None,
            interconnect_cut_layer_keys=frozenset(),
            max_atoms=max_atoms,
            max_pair_evaluations=max_pair_evaluations,
            max_components=max_components,
            max_pin_memberships=max_pin_memberships,
        )

    @classmethod
    def build_planar(
        cls,
        world: Any,
        tech: Any,
        *,
        max_atoms: int = 100_000,
        max_pair_evaluations: int = 1_000_000,
        max_components: int = 100_000,
        max_pin_memberships: int = 1_000_000,
    ) -> "OwnedConductorComponentIndex":
        """Build the PDK-classified planar-metal index used by routing."""

        metal_keys, cut_keys = _routing_stack_layer_keys(tech)
        return cls._build_index(
            world,
            planar_metal_layer_keys=metal_keys,
            interconnect_cut_layer_keys=cut_keys,
            max_atoms=max_atoms,
            max_pair_evaluations=max_pair_evaluations,
            max_components=max_components,
            max_pin_memberships=max_pin_memberships,
        )

    @classmethod
    def _build_index(
        cls,
        world: Any,
        *,
        planar_metal_layer_keys: frozenset[LayerKey] | None,
        interconnect_cut_layer_keys: frozenset[LayerKey],
        max_atoms: int,
        max_pair_evaluations: int,
        max_components: int,
        max_pin_memberships: int,
    ) -> "OwnedConductorComponentIndex":
        """Build a deterministic index, failing atomically on any hard limit."""

        atom_limit = _bounded_positive_int(
            "max_atoms", max_atoms, _HARD_MAX_ATOMS
        )
        pair_limit = _bounded_positive_int(
            "max_pair_evaluations",
            max_pair_evaluations,
            _HARD_MAX_PAIR_EVALUATIONS,
        )
        component_limit = _bounded_positive_int(
            "max_components", max_components, _HARD_MAX_COMPONENTS
        )
        membership_limit = _bounded_positive_int(
            "max_pin_memberships",
            max_pin_memberships,
            _HARD_MAX_PIN_MEMBERSHIPS,
        )
        if not callable(getattr(world, "sha256", None)):
            raise OwnedConductorIndexError("world must provide sha256()")
        world_digest = world.sha256()
        if not isinstance(world_digest, str) or not world_digest:
            raise OwnedConductorIndexError("world sha256() must be non-empty")

        atoms = cls._normalize_atoms(
            world,
            atom_limit,
            planar_metal_layer_keys,
            interconnect_cut_layer_keys,
        )
        components, pair_evaluations = cls._build_atom_components(
            atoms,
            world_digest=world_digest,
            pair_limit=pair_limit,
            component_limit=component_limit,
        )
        del pair_evaluations  # The count is consumed only to enforce the proof bound.
        components, bindings = cls._bind_pins(
            world,
            world_digest=world_digest,
            components=components,
            planar_metal_layer_keys=planar_metal_layer_keys,
            component_limit=component_limit,
            membership_limit=membership_limit,
        )
        return cls(
            world_digest=world_digest,
            components=components,
            pin_bindings=bindings,
            _build_token=_INDEX_BUILD_TOKEN,
        )

    @staticmethod
    def _normalize_atoms(
        world: Any,
        atom_limit: int,
        planar_metal_layer_keys: frozenset[LayerKey] | None,
        interconnect_cut_layer_keys: frozenset[LayerKey],
    ) -> tuple[_Atom, ...]:
        facts: set[AtomFact] = set()
        seen = 0
        try:
            raw_obstacles = iter(world.obstacles)
            raw_rails = iter(world.rails)
        except (AttributeError, TypeError) as exc:
            raise OwnedConductorIndexError(
                "world must expose obstacles and rails"
            ) from exc
        obstacles: list[Any] = []
        rails: list[Any] = []
        raw_count = 0
        for raw in raw_obstacles:
            raw_count += 1
            if raw_count > _HARD_MAX_RAW_FACTS:
                raise OwnedConductorIndexLimitError(
                    "raw owned-conductor fact count exceeds hard inspection maximum "
                    f"{_HARD_MAX_RAW_FACTS}"
                )
            obstacles.append(raw)
        for raw in raw_rails:
            raw_count += 1
            if raw_count > _HARD_MAX_RAW_FACTS:
                raise OwnedConductorIndexLimitError(
                    "raw owned-conductor fact count exceeds hard inspection maximum "
                    f"{_HARD_MAX_RAW_FACTS}"
                )
            rails.append(raw)

        obstacle_set = {tuple(raw) for raw in obstacles}
        terminal_ref_by_fact: dict[AtomFact, str] = {}
        try:
            raw_terminal_bindings = iter(
                getattr(world, "terminal_conductor_refs", ())
            )
        except TypeError as exc:
            raise OwnedConductorIndexError(
                "world terminal_conductor_refs must be iterable"
            ) from exc
        for raw_binding in raw_terminal_bindings:
            if not isinstance(raw_binding, (tuple, list)) or len(raw_binding) != 9:
                raise OwnedConductorIndexError(
                    "malformed terminal_conductor_ref binding"
                )
            raw_fact = tuple(raw_binding[:8])
            terminal_ref = raw_binding[8]
            if raw_fact not in obstacle_set:
                raise OwnedConductorIndexError(
                    "terminal_conductor_ref binding names no obstacle atom"
                )
            if (
                raw_fact[6] is None
                or raw_fact[7] not in TERMINAL_CONDUCTOR_KINDS
                or not isinstance(terminal_ref, str)
                or not terminal_ref
            ):
                raise OwnedConductorIndexError(
                    "terminal_conductor_ref binding is malformed"
                )
            previous = terminal_ref_by_fact.setdefault(
                raw_fact, terminal_ref  # type: ignore[arg-type]
            )
            if previous != terminal_ref:
                raise OwnedConductorIndexError(
                    "one obstacle atom has ambiguous terminal_conductor_refs"
                )

        for raw in obstacles:
            if not isinstance(raw, (tuple, list)) or len(raw) != 8:
                raise OwnedConductorIndexError(
                    f"malformed routing-world obstacle fact {raw!r}"
                )
            owner = raw[6]
            if owner is None:
                continue
            net = str(owner)
            if not net:
                raise OwnedConductorIndexError("owned obstacle net must be non-empty")
            kind = "owned" if raw[7] is None else str(raw[7])
            layer_key = _layer_key(raw[4:6])
            if (
                planar_metal_layer_keys is not None
                and layer_key not in planar_metal_layer_keys
            ):
                if layer_key in interconnect_cut_layer_keys:
                    continue
                raise OwnedConductorIndexError(
                    "owned obstacle lies on neither a PDK routing metal nor "
                    f"an interconnect cut layer: {layer_key!r}"
                )
            fact: AtomFact = (
                *_bbox(raw[:4], name="obstacle bbox"),
                *layer_key,
                net,
                kind,
            )
            if fact not in facts:
                seen += 1
                if seen > atom_limit:
                    raise OwnedConductorIndexLimitError(
                        f"owned atom count exceeds max_atoms={atom_limit}"
                    )
                facts.add(fact)

        for raw in rails:
            if not isinstance(raw, (tuple, list)) or len(raw) != 7:
                raise OwnedConductorIndexError(
                    f"malformed routing-world rail fact {raw!r}"
                )
            net = str(raw[0])
            if not net:
                raise OwnedConductorIndexError("rail net must be non-empty")
            layer_key = _layer_key(raw[1:3])
            if (
                planar_metal_layer_keys is not None
                and layer_key not in planar_metal_layer_keys
            ):
                raise OwnedConductorIndexError(
                    f"rail must lie on a PDK routing metal, got {layer_key!r}"
                )
            fact = (
                *_bbox(raw[3:7], name="rail bbox"),
                *layer_key,
                net,
                "rail",
            )
            if fact not in facts:
                seen += 1
                if seen > atom_limit:
                    raise OwnedConductorIndexLimitError(
                        f"owned atom count exceeds max_atoms={atom_limit}"
                    )
                facts.add(fact)

        atoms: list[_Atom] = []
        for fact in sorted(facts, key=lambda item: _canonical_json(list(item))):
            atom_ref = "oa:" + _digest(
                {
                    "fact": list(fact),
                    "schema": "clay.owned-conductor-atom.v2",
                    "terminal_conductor_ref": terminal_ref_by_fact.get(fact),
                }
            )
            atoms.append(
                _Atom(
                    bbox_nm=fact[:4],
                    layer_key=fact[4:6],
                    net=fact[6],
                    kind=fact[7],
                    terminal_conductor_ref=terminal_ref_by_fact.get(fact),
                    fact=fact,
                    atom_ref=atom_ref,
                )
            )
        return tuple(atoms)

    @classmethod
    def _build_atom_components(
        cls,
        atoms: tuple[_Atom, ...],
        *,
        world_digest: str,
        pair_limit: int,
        component_limit: int,
    ) -> tuple[list[OwnedConductorComponent], int]:
        groups: dict[tuple[LayerKey, str], list[_Atom]] = {}
        for atom in atoms:
            groups.setdefault((atom.layer_key, atom.net), []).append(atom)

        pair_evaluations = 0
        components: list[OwnedConductorComponent] = []
        for (layer_key, net), group_atoms in sorted(groups.items()):
            ordered = sorted(
                group_atoms,
                key=lambda atom: (
                    atom.bbox_nm,
                    atom.kind,
                    atom.atom_ref,
                ),
            )
            union_find = _UnionFind(len(ordered))
            root = _build_str_tree(
                tuple(
                    _AtomPairCandidate(
                        bbox_nm=atom.bbox_nm,
                        ordinal=index,
                        stable_ref=atom.atom_ref,
                    )
                    for index, atom in enumerate(ordered)
                )
            )

            for current_index, current in enumerate(ordered):
                def visit(node: _SpatialNode) -> None:
                    nonlocal pair_evaluations
                    if not _closed_touch(node.bbox_nm, current.bbox_nm):
                        return
                    if node.leaves:
                        for primitive in node.children:
                            if not isinstance(primitive, _AtomPairCandidate):
                                raise OwnedConductorIndexError(
                                    "malformed atom-component spatial-index leaf"
                                )
                            if primitive.ordinal >= current_index:
                                continue
                            pair_evaluations += 1
                            if pair_evaluations > pair_limit:
                                raise OwnedConductorIndexLimitError(
                                    "owned-conductor overlap evaluation count "
                                    "exceeds "
                                    f"max_pair_evaluations={pair_limit}"
                                )
                            if _closed_touch(
                                primitive.bbox_nm, current.bbox_nm
                            ):
                                union_find.union(
                                    primitive.ordinal, current_index
                                )
                        return
                    for child in node.children:
                        if not isinstance(child, _SpatialNode):
                            raise OwnedConductorIndexError(
                                "malformed atom-component spatial-index node"
                            )
                        visit(child)

                if root is not None:
                    visit(root)

            members: dict[int, list[_Atom]] = {}
            for index, atom in enumerate(ordered):
                members.setdefault(union_find.find(index), []).append(atom)
            for component_atoms in members.values():
                components.append(
                    cls._atom_component(
                        world_digest=world_digest,
                        layer_key=layer_key,
                        net=net,
                        atoms=tuple(component_atoms),
                    )
                )
                if len(components) > component_limit:
                    raise OwnedConductorIndexLimitError(
                        f"component count exceeds max_components={component_limit}"
                    )
        return components, pair_evaluations

    @staticmethod
    def _atom_component(
        *,
        world_digest: str,
        layer_key: LayerKey,
        net: str,
        atoms: tuple[_Atom, ...],
    ) -> OwnedConductorComponent:
        ordered = tuple(sorted(atoms, key=lambda atom: atom.atom_ref))
        geometry_payload = {
            "atoms": [list(atom.fact) for atom in ordered],
            "atom_terminal_conductor_refs": [
                atom.terminal_conductor_ref for atom in ordered
            ],
            "layer_key": list(layer_key),
            "net": net,
            "schema": "clay.owned-conductor-component-geometry.v2",
        }
        geometry_hash = _digest(geometry_payload)
        return OwnedConductorComponent(
            component_ref=f"oc:{world_digest}:{geometry_hash}",
            component_geometry_hash=geometry_hash,
            world_digest=world_digest,
            layer_key=layer_key,
            net=net,
            bbox_nm=(
                min(atom.bbox_nm[0] for atom in ordered),
                min(atom.bbox_nm[1] for atom in ordered),
                max(atom.bbox_nm[2] for atom in ordered),
                max(atom.bbox_nm[3] for atom in ordered),
            ),
            atom_facts=tuple(atom.fact for atom in ordered),
            atom_refs=tuple(atom.atom_ref for atom in ordered),
            kinds=tuple(sorted({atom.kind for atom in ordered})),
            atom_terminal_conductor_refs=tuple(
                atom.terminal_conductor_ref for atom in ordered
            ),
        )

    @classmethod
    def _bind_pins(
        cls,
        world: Any,
        *,
        world_digest: str,
        components: list[OwnedConductorComponent],
        planar_metal_layer_keys: frozenset[LayerKey] | None,
        component_limit: int,
        membership_limit: int,
    ) -> tuple[list[OwnedConductorComponent], list[tuple[PinAccessFact, str]]]:
        try:
            raw_pins: list[Any] = []
            for raw_pin in world.pins:
                raw_pins.append(raw_pin)
                if len(raw_pins) > _HARD_MAX_RAW_PINS:
                    raise OwnedConductorIndexLimitError(
                        "raw pin fact count exceeds hard inspection maximum "
                        f"{_HARD_MAX_RAW_PINS}"
                    )
            pins = tuple(
                sorted(
                    {tuple(pin) for pin in raw_pins},
                    key=lambda pin: _canonical_json(list(pin)),
                )
            )
        except (AttributeError, TypeError) as exc:
            raise OwnedConductorIndexError("world must expose hashable pins") from exc

        grouped: dict[tuple[LayerKey, str], list[int]] = {}
        for index, component in enumerate(components):
            grouped.setdefault((component.layer_key, component.net), []).append(index)

        component_pins: dict[int, list[PinAccessFact]] = {}
        bindings: list[tuple[PinAccessFact, str]] = []
        membership_evaluations = 0

        def consume_membership() -> None:
            nonlocal membership_evaluations
            membership_evaluations += 1
            if membership_evaluations > membership_limit:
                raise OwnedConductorIndexLimitError(
                    "pin membership evaluation count exceeds "
                    f"max_pin_memberships={membership_limit}"
                )

        from clay.layout.route.terminal_identity import unpack_pin_fact
        for pin in pins:
            try:
                unpack_pin_fact(pin)
            except (TypeError, ValueError) as exc:
                raise OwnedConductorIndexError(
                    f"malformed routing-world pin fact {pin!r}"
                ) from exc
            if any(
                not isinstance(identity, str) or not identity
                for identity in pin[:3]
            ):
                raise OwnedConductorIndexError(
                    "pin net/device/terminal identity must be non-empty strings"
                )
            pin_layer_key = _layer_key(pin[5:7])
            if (
                planar_metal_layer_keys is not None
                and pin_layer_key not in planar_metal_layer_keys
            ):
                raise OwnedConductorIndexError(
                    "pin access must lie on a PDK routing metal, got "
                    f"{pin_layer_key!r}"
                )
            net = pin[0]
            point = (_integer("pin x", pin[3]), _integer("pin y", pin[4]))
            layer_key = _layer_key(pin[5:7])
            matches: list[int] = []
            for component_index in grouped.get((layer_key, net), ()):
                component = components[component_index]
                consume_membership()
                if not _point_inside(point, component.bbox_nm):
                    continue
                for fact in component.atom_facts:
                    consume_membership()
                    if _point_inside(point, fact[:4]):
                        matches.append(component_index)
                        break
            if len(matches) > 1:
                raise OwnedConductorIndexError(
                    "pin lies in multiple disconnected same-net components"
                )
            if matches:
                component_index = matches[0]
                component_pins.setdefault(component_index, []).append(pin)
                bindings.append((pin, components[component_index].component_ref))
                continue

            pin_component = cls._pin_only_component(
                world_digest=world_digest,
                pin=pin,
                net=net,
                layer_key=layer_key,
                point=point,
            )
            components.append(pin_component)
            grouped.setdefault((layer_key, net), []).append(len(components) - 1)
            bindings.append((pin, pin_component.component_ref))
            if len(components) > component_limit:
                raise OwnedConductorIndexLimitError(
                    f"component count exceeds max_components={component_limit}"
                )

        for component_index, bound_pins in component_pins.items():
            component = components[component_index]
            components[component_index] = OwnedConductorComponent(
                component_ref=component.component_ref,
                component_geometry_hash=component.component_geometry_hash,
                world_digest=component.world_digest,
                layer_key=component.layer_key,
                net=component.net,
                bbox_nm=component.bbox_nm,
                atom_facts=component.atom_facts,
                atom_refs=component.atom_refs,
                kinds=component.kinds,
                atom_terminal_conductor_refs=(
                    component.atom_terminal_conductor_refs
                ),
                pin_accesses=tuple(
                    sorted(bound_pins, key=lambda pin: _canonical_json(list(pin)))
                ),
            )
        return components, bindings

    @staticmethod
    def _pin_only_component(
        *,
        world_digest: str,
        pin: PinAccessFact,
        net: str,
        layer_key: LayerKey,
        point: tuple[int, int],
    ) -> OwnedConductorComponent:
        geometry_payload = {
            "layer_key": list(layer_key),
            "net": net,
            "pin_access": list(pin),
            "schema": "clay.owned-conductor-pin-access-geometry.v1",
        }
        geometry_hash = _digest(geometry_payload)
        return OwnedConductorComponent(
            component_ref=f"oc:{world_digest}:{geometry_hash}",
            component_geometry_hash=geometry_hash,
            world_digest=world_digest,
            layer_key=layer_key,
            net=net,
            bbox_nm=(point[0], point[1], point[0], point[1]),
            atom_facts=(),
            atom_refs=(),
            kinds=("pin-access",),
            pin_accesses=(pin,),
        )

    @property
    def components(self) -> tuple[OwnedConductorComponent, ...]:
        return self._components

    def components_for(
        self, layer_key: Sequence[int], net: str
    ) -> tuple[OwnedConductorComponent, ...]:
        key = _layer_key(layer_key)
        if not isinstance(net, str) or not net:
            raise OwnedConductorIndexError("net must be a non-empty string")
        return self._by_layer_net.get((key, net), ())

    def component(self, component_ref: str) -> OwnedConductorComponent:
        try:
            return self._by_ref[component_ref]
        except KeyError as exc:
            raise OwnedConductorIndexError(
                f"unknown owned-conductor component {component_ref!r}"
            ) from exc

    def refs_for_pin(self, pin_access_fact: Sequence[Any]) -> tuple[str, ...]:
        pin = tuple(pin_access_fact)
        component_ref = self._pin_bindings.get(pin)
        return () if component_ref is None else (component_ref,)

    @staticmethod
    def _component_touches_bbox(
        component: OwnedConductorComponent,
        bbox_nm: BBoxNm,
        evaluation_consumer: Any = None,
    ) -> bool:
        if evaluation_consumer is not None:
            evaluation_consumer(
                "owned-conductor component-bbox query"
            )
        if not _closed_touch(component.bbox_nm, bbox_nm):
            return False
        for fact in component.atom_facts:
            if evaluation_consumer is not None:
                evaluation_consumer(
                    "owned-conductor atom landing-pad query"
                )
            if _closed_touch(fact[:4], bbox_nm):
                return True
        for pin in component.pin_accesses:
            if evaluation_consumer is not None:
                evaluation_consumer(
                    "owned-conductor pin landing-pad query"
                )
            if _point_inside((int(pin[3]), int(pin[4])), bbox_nm):
                return True
        return False

    def refs_touching_bbox(
        self, layer_key: Sequence[int], net: str, bbox_nm: Sequence[int]
    ) -> tuple[str, ...]:
        bbox = _bbox(bbox_nm)
        return tuple(
            component.component_ref
            for component in self.components_for(layer_key, net)
            if self._component_touches_bbox(component, bbox)
        )

    def landing_pad_contact(
        self,
        layer_key: Sequence[int],
        pad_bbox_nm: Sequence[int],
        *,
        evaluation_consumer: Any = None,
    ) -> LandingPadContact:
        key = _layer_key(layer_key)
        bbox = _bbox(pad_bbox_nm, name="pad_bbox_nm")
        if evaluation_consumer is not None and not callable(evaluation_consumer):
            raise OwnedConductorIndexError(
                "evaluation_consumer must be callable when present"
            )
        contacts: dict[str, set[str]] = {}

        def consume(reason: str) -> None:
            if evaluation_consumer is not None:
                evaluation_consumer(reason)

        def visit(node: _SpatialNode) -> None:
            consume("owned-conductor spatial-index node component-bbox query")
            if not _closed_touch(node.bbox_nm, bbox):
                return
            if node.leaves:
                for primitive in node.children:
                    if not isinstance(primitive, _ContactPrimitive):
                        raise OwnedConductorIndexError(
                            "malformed private spatial-index primitive leaf"
                        )
                    consume(
                        "owned-conductor primitive leaf atom landing-pad query"
                    )
                    if _closed_touch(primitive.bbox_nm, bbox):
                        contacts.setdefault(primitive.net, set()).add(
                            primitive.component_ref
                        )
                return
            for child in node.children:
                if not isinstance(child, _SpatialNode):
                    raise OwnedConductorIndexError(
                        "malformed private spatial-index child node"
                    )
                visit(child)

        root = self._spatial_roots.get(key)
        if root is not None:
            visit(root)
        frozen = tuple(
            (net, tuple(sorted(refs))) for net, refs in sorted(contacts.items())
        )
        if not frozen:
            status = "FREE"
        elif len(frozen) > 1:
            status = "FOREIGN_NET_CONTACT"
        elif len(frozen[0][1]) > 1:
            status = "MERGES_COMPONENTS"
        else:
            status = "OWNED"
        return LandingPadContact(status=status, contacts_by_net=frozen)

    def landing_pad_contacts(
        self,
        layer_key: Sequence[int],
        pad_bboxes_nm: Sequence[Sequence[int]],
        *,
        evaluation_consumer: Any = None,
    ) -> tuple[LandingPadContact, ...]:
        """Classify a finite exact pad set with one deterministic STR self-join.

        Restarting :meth:`landing_pad_contact` at the conductor-tree root for
        every via candidate multiplies MBR probes by the full candidate count.
        This method bulk-loads the already-finite pad set into a second private
        STR tree and intersects the two trees.  Only leaf-pair candidates reach
        the exact closed-touch predicate.  Duplicate pad bboxes share one proof
        and one immutable result; caller order is preserved.
        """

        key = _layer_key(layer_key)
        if evaluation_consumer is not None and not callable(evaluation_consumer):
            raise OwnedConductorIndexError(
                "evaluation_consumer must be callable when present"
            )
        if len(pad_bboxes_nm) > _HARD_MAX_LANDING_PAD_QUERIES:
            raise OwnedConductorIndexLimitError(
                "landing-pad batch exceeds hard maximum "
                f"{_HARD_MAX_LANDING_PAD_QUERIES}"
            )
        requested = tuple(
            _bbox(value, name="pad_bboxes_nm[]") for value in pad_bboxes_nm
        )
        if not requested:
            return ()

        unique_bboxes = tuple(sorted(set(requested)))
        ordinal_by_bbox = {
            bbox: ordinal for ordinal, bbox in enumerate(unique_bboxes)
        }
        candidates = tuple(
            _PadCandidate(
                bbox_nm=bbox,
                ordinal=ordinal,
                stable_ref=f"pad:{ordinal:06d}",
            )
            for ordinal, bbox in enumerate(unique_bboxes)
        )
        candidate_root = _build_str_tree(candidates)
        conductor_root = self._spatial_roots.get(key)
        contacts: list[dict[str, set[str]]] = [
            {} for _ in unique_bboxes
        ]

        def consume(reason: str) -> None:
            if evaluation_consumer is not None:
                evaluation_consumer(reason)

        def visit_primitive_candidates(
            primitive: _ContactPrimitive,
            candidate_node: _SpatialNode,
        ) -> None:
            """Query one exact primitive, never its sparse leaf aggregate MBR."""

            consume(
                "owned-conductor batch spatial-index node-pair "
                "component-bbox query"
            )
            if not _closed_touch(primitive.bbox_nm, candidate_node.bbox_nm):
                return
            if candidate_node.leaves:
                for candidate in candidate_node.children:
                    if not isinstance(candidate, _PadCandidate):
                        raise OwnedConductorIndexError(
                            "malformed pad leaf in batch spatial index"
                        )
                    consume(
                        "owned-conductor batch exact candidate primitive "
                        "landing-pad query"
                    )
                    if _closed_touch(primitive.bbox_nm, candidate.bbox_nm):
                        contacts[candidate.ordinal].setdefault(
                            primitive.net, set()
                        ).add(primitive.component_ref)
                return
            for child in candidate_node.children:
                if not isinstance(child, _SpatialNode):
                    raise OwnedConductorIndexError(
                        "malformed pad child in batch spatial index"
                    )
                visit_primitive_candidates(primitive, child)

        def visit_pair(
            conductor_node: _SpatialNode,
            candidate_node: _SpatialNode,
        ) -> None:
            consume(
                "owned-conductor batch spatial-index node-pair "
                "component-bbox query"
            )
            if not _closed_touch(
                conductor_node.bbox_nm, candidate_node.bbox_nm
            ):
                return
            if conductor_node.leaves and candidate_node.leaves:
                for primitive in conductor_node.children:
                    if not isinstance(primitive, _ContactPrimitive):
                        raise OwnedConductorIndexError(
                            "malformed conductor leaf in batch spatial index"
                        )
                    for candidate in candidate_node.children:
                        if not isinstance(candidate, _PadCandidate):
                            raise OwnedConductorIndexError(
                                "malformed pad leaf in batch spatial index"
                            )
                        consume(
                            "owned-conductor batch exact candidate primitive "
                            "landing-pad query"
                        )
                        if _closed_touch(
                            primitive.bbox_nm, candidate.bbox_nm
                        ):
                            contacts[candidate.ordinal].setdefault(
                                primitive.net, set()
                            ).add(primitive.component_ref)
                return
            if conductor_node.leaves:
                for primitive in conductor_node.children:
                    if not isinstance(primitive, _ContactPrimitive):
                        raise OwnedConductorIndexError(
                            "malformed conductor leaf in batch spatial index"
                        )
                    visit_primitive_candidates(primitive, candidate_node)
                return
            if candidate_node.leaves:
                for child in conductor_node.children:
                    if not isinstance(child, _SpatialNode):
                        raise OwnedConductorIndexError(
                            "malformed conductor child in batch spatial index"
                        )
                    visit_pair(child, candidate_node)
                return
            for conductor_child in conductor_node.children:
                if not isinstance(conductor_child, _SpatialNode):
                    raise OwnedConductorIndexError(
                        "malformed conductor child in batch spatial index"
                    )
                for candidate_child in candidate_node.children:
                    if not isinstance(candidate_child, _SpatialNode):
                        raise OwnedConductorIndexError(
                            "malformed pad child in batch spatial index"
                        )
                    visit_pair(conductor_child, candidate_child)

        if conductor_root is not None and candidate_root is not None:
            visit_pair(conductor_root, candidate_root)

        results: list[LandingPadContact] = []
        for contact_by_net in contacts:
            frozen = tuple(
                (net, tuple(sorted(refs)))
                for net, refs in sorted(contact_by_net.items())
            )
            if not frozen:
                status = "FREE"
            elif len(frozen) > 1:
                status = "FOREIGN_NET_CONTACT"
            elif len(frozen[0][1]) > 1:
                status = "MERGES_COMPONENTS"
            else:
                status = "OWNED"
            results.append(
                LandingPadContact(status=status, contacts_by_net=frozen)
            )
        return tuple(
            results[ordinal_by_bbox[bbox]] for bbox in requested
        )

    def canonical_json(self) -> str:
        payload = {
            "components": [
                component.canonical_payload() for component in self._components
            ],
            "pin_bindings": [
                {"component_ref": component_ref, "pin_access": list(pin)}
                for pin, component_ref in sorted(
                    self._pin_bindings.items(),
                    key=lambda item: _canonical_json([list(item[0]), item[1]]),
                )
            ],
            "schema": "clay.owned-conductor-component-index.v2",
            "world_digest": self.world_digest,
        }
        return _canonical_json(payload)

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def replay_owned_conductor_component(
    component: OwnedConductorComponent,
    *,
    world_digest: str,
) -> None:
    """Replay one immutable v2 component record from its exact members.

    This is the public verification seam for carried component certificates.
    It deliberately delegates to the same validator used by the canonical
    index instead of maintaining a router-local approximation.
    """

    if not isinstance(world_digest, str) or not world_digest:
        raise OwnedConductorIndexError("world_digest must be non-empty")
    validator = object.__new__(OwnedConductorComponentIndex)
    validator.world_digest = world_digest
    validator._validate_component(component)


def validate_owned_conductor_component_index_certificate(
    *,
    world_digest: str,
    components: tuple[OwnedConductorComponent, ...],
    index_digest: str,
    max_pair_evaluations: int = 2_000_000,
) -> None:
    """Replay a complete v2 component table and its bound index digest.

    The table is not a used-component projection: every component participates
    in canonical ordering, member replay, partition replay, pin binding replay,
    and the final index digest.
    """

    if not isinstance(world_digest, str) or not world_digest:
        raise OwnedConductorIndexError("world_digest must be non-empty")
    if not isinstance(index_digest, str) or not index_digest:
        raise OwnedConductorIndexError("index_digest must be non-empty")
    if type(components) is not tuple:
        raise OwnedConductorIndexError("component certificate table must be a tuple")
    if len(components) > _HARD_MAX_COMPONENTS:
        raise OwnedConductorIndexLimitError(
            "component certificate table exceeds hard component maximum"
        )
    if (
        type(max_pair_evaluations) is not int
        or max_pair_evaluations < 0
        or max_pair_evaluations > _HARD_MAX_PAIR_EVALUATIONS
    ):
        raise OwnedConductorIndexLimitError(
            "max_pair_evaluations must be an exact non-negative integer no "
            f"greater than {_HARD_MAX_PAIR_EVALUATIONS}"
        )
    pair_limit = max_pair_evaluations
    if any(type(component) is not OwnedConductorComponent for component in components):
        raise OwnedConductorIndexError(
            "component certificate table contains a foreign record"
        )
    refs = tuple(component.component_ref for component in components)
    if refs != tuple(sorted(refs)) or len(refs) != len(set(refs)):
        raise OwnedConductorIndexError(
            "component certificate table must have canonical unique refs"
        )
    member_count = 0
    for component in components:
        if any(
            type(value) is not tuple
            for value in (
                component.layer_key,
                component.bbox_nm,
                component.atom_facts,
                component.atom_refs,
                component.kinds,
                component.atom_terminal_conductor_refs,
                component.pin_accesses,
            )
        ):
            raise OwnedConductorIndexError(
                "component certificate members must be canonical tuples"
            )
        member_count += len(component.atom_facts) + len(component.pin_accesses)
        if member_count > _HARD_MAX_PIN_MEMBERSHIPS + _HARD_MAX_ATOMS:
            raise OwnedConductorIndexLimitError(
                "component certificate member table exceeds hard maximum"
            )
        replay_owned_conductor_component(
            component,
            world_digest=world_digest,
        )
        pin_keys = tuple(
            _canonical_json(list(pin)) for pin in component.pin_accesses
        )
        if pin_keys != tuple(sorted(pin_keys)) or len(pin_keys) != len(set(pin_keys)):
            raise OwnedConductorIndexError(
                "component pin accesses must be canonical and unique"
            )

    pin_bindings = tuple(
        (pin, component.component_ref)
        for component in components
        for pin in component.pin_accesses
    )
    replayed = OwnedConductorComponentIndex(
        world_digest=world_digest,
        components=components,
        pin_bindings=pin_bindings,
        _build_token=_INDEX_BUILD_TOKEN,
    )

    pair_count = 0

    def consume_pair() -> None:
        nonlocal pair_count
        pair_count += 1
        if pair_count > pair_limit:
            raise OwnedConductorIndexLimitError(
                "component certificate replay exceeds pair evaluations budget"
            )

    atoms_by_owner: dict[
        tuple[LayerKey, str], list[tuple[AtomFact, str]]
    ] = {}
    pin_only_by_owner: dict[
        tuple[LayerKey, str], list[tuple[PinAccessFact, str]]
    ] = {}
    for component in components:
        atom_facts = component.atom_facts
        if atom_facts:
            ordered_indexes = tuple(sorted(
                range(len(atom_facts)),
                key=lambda index: (
                    atom_facts[index][0],
                    atom_facts[index][1],
                    atom_facts[index][2],
                    atom_facts[index][3],
                    component.atom_refs[index],
                ),
            ))
            parent = list(range(len(atom_facts)))

            def find(index: int) -> int:
                while parent[index] != index:
                    parent[index] = parent[parent[index]]
                    index = parent[index]
                return index

            active: list[int] = []
            for current_index in ordered_indexes:
                current = atom_facts[current_index]
                active = [
                    index for index in active
                    if atom_facts[index][2] >= current[0]
                ]
                for other_index in active:
                    consume_pair()
                    if _closed_touch(atom_facts[other_index][:4], current[:4]):
                        left = find(other_index)
                        right = find(current_index)
                        if left != right:
                            parent[right] = left
                active.append(current_index)
            if len({find(index) for index in range(len(atom_facts))}) != 1:
                raise OwnedConductorIndexError(
                    "component certificate contains disconnected atoms"
                )
        owner = (component.layer_key, component.net)
        if not atom_facts:
            pin_only_by_owner.setdefault(owner, []).append(
                (component.pin_accesses[0], component.component_ref)
            )
        atoms_by_owner.setdefault(owner, []).extend(
            (atom, component.component_ref) for atom in atom_facts
        )

    for atoms in atoms_by_owner.values():
        ordered_atoms = tuple(sorted(
            atoms,
            key=lambda item: (
                item[0][0], item[0][1], item[0][2], item[0][3], item[1]
            ),
        ))
        active_atoms: list[tuple[AtomFact, str]] = []
        for current_atom, current_ref in ordered_atoms:
            active_atoms = [
                item for item in active_atoms if item[0][2] >= current_atom[0]
            ]
            for other_atom, other_ref in active_atoms:
                if other_ref == current_ref:
                    continue
                consume_pair()
                if _closed_touch(other_atom[:4], current_atom[:4]):
                    raise OwnedConductorIndexError(
                        "component table must not split touching same-owner atoms"
                    )
            active_atoms.append((current_atom, current_ref))
    for owner, pin_components in pin_only_by_owner.items():
        for pin, _component_ref in pin_components:
            for atom, _atom_ref in atoms_by_owner.get(owner, ()):
                consume_pair()
                if _point_inside((int(pin[3]), int(pin[4])), atom[:4]):
                    raise OwnedConductorIndexError(
                        "pin-only component lies inside an atom component"
                    )

    if replayed.sha256() != index_digest:
        raise OwnedConductorIndexError(
            "component certificate table does not match index digest"
        )


__all__ = [
    "LandingPadContact",
    "OwnedConductorComponent",
    "OwnedConductorComponentIndex",
    "OwnedConductorIndexError",
    "OwnedConductorIndexLimitError",
    "TERMINAL_CONDUCTOR_KINDS",
    "replay_owned_conductor_component",
    "validate_owned_conductor_component_index_certificate",
]
