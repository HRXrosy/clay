"""Independent emitted-GDS check for unsafe router-owned via landings.

The electrical safety rule is deliberately small: upper routing metal may pass
over device artwork, but a router-owned VIA1 may land on device-terminal M1 only
when the route manifest's net equals the net labelled on that M1 island.

This module reads emitted GDS geometry.  Router ``tagged`` entries are used only
to identify which physical via cuts the router owns and what net it intended;
they are never used to infer the lower device net.  In addition, the checker
reconstructs the complete routing-metal graph and reports every connected
component carrying more than one M1 label.  Consequently an artifact without an
in-memory route manifest remains useful for detecting global conflicts, while
per-via attribution stays explicit instead of guessing ownership.

This is diagnostic defense-in-depth, not a sign-off replacement.  Only the real
KLayout LVS full-extraction path may certify absence of shorts.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

try:
    import klayout.db as kdb
except ImportError:  # pragma: no cover - exercised by monkeypatch in CI
    kdb = None  # type: ignore[assignment]

LayerKey = Tuple[int, int]
BBox = Tuple[int, int, int, int]
TaggedEntry = Tuple[str, LayerKey, BBox]
_MAX_ROUTER_MANIFEST_ENTRIES = 100_000
_MAX_TERMINAL_CLAIMS = 100_000
_MAX_TOPOLOGY_ISLANDS = 200_000
_MAX_TOPOLOGY_VIA_CUTS = 200_000
_MAX_TOPOLOGY_JOIN_EVALUATIONS = 2_000_000
_MAX_MANIFEST_JOIN_EVALUATIONS = 2_000_000
_MAX_PROVENANCE_GRAPH_VISITS = 2_000_000


@dataclass(frozen=True)
class ViaProvenanceObservation:
    """Verdict for one router-owned via cut on any declared routing level."""

    status: str
    expected_net: str
    lower_nets: Tuple[str, ...]
    upper_nets: Tuple[str, ...]
    via_layer: LayerKey
    via_bbox: BBox
    reason: str = ""

    @property
    def observed_nets(self) -> Tuple[str, ...]:
        """Backward-compatible name for the independently observed lower net."""
        return self.lower_nets


@dataclass(frozen=True)
class ViaProvenanceReport:
    """Five-state result from :func:`check_via_provenance`.

    ``status`` is one of ``pass``, ``fail``, ``unmeasurable``, ``unsupported``
    or ``advisory``.  A pass requires every router-owned via to be adjudicated
    (or an available manifest proving there are none), no unresolved landing,
    and no multi-label component.
    """

    status: str
    checked: int = 0
    observations: Tuple[ViaProvenanceObservation, ...] = ()
    violations: Tuple[ViaProvenanceObservation, ...] = ()
    unresolved: Tuple[ViaProvenanceObservation, ...] = ()
    conflicted_components: Tuple[Tuple[str, ...], ...] = ()
    manifest_available: bool = False
    router_owned_vias: int = 0
    artifact_sha256: str = ""
    route_manifest_sha256: str = ""
    terminal_claim_count: int = 0
    terminal_claims_sha256: str = ""
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "pass"


def _normalise_key(value: Any) -> LayerKey:
    return int(value[0]), int(value[1])


def _metal_stack(tech: Any, max_metal: int) -> Tuple[LayerKey, ...]:
    if type(max_metal) is not int or max_metal < 2:
        raise ValueError("max_metal must be an integer >= 2")
    probed: list[Optional[LayerKey]] = []
    for n in range(1, max_metal + 1):
        try:
            probed.append(_normalise_key(tech.metal_key(n)))
        except Exception:
            probed.append(None)
    declared = [index for index, key in enumerate(probed) if key is not None]
    if not declared:
        return ()
    last = declared[-1]
    holes = tuple(index + 1 for index, key in enumerate(probed[: last + 1]) if key is None)
    if holes:
        raise ValueError(
            "incomplete PDK metal stack; missing "
            + ", ".join(f"M{layer_n}" for layer_n in holes)
        )
    out = tuple(key for key in probed[: last + 1] if key is not None)
    if len(out) != len(set(out)):
        raise ValueError("PDK metal stack reuses a physical layer key")
    return out


def _via_stack(tech: Any, n_metals: int) -> Tuple[Tuple[LayerKey, int, int], ...]:
    out = []
    for n in range(1, n_metals):
        try:
            key = _normalise_key(tech.via_key(n))
        except Exception as exc:
            raise ValueError(
                f"incomplete PDK routing stack; VIA{n} is missing between M{n} and M{n + 1}"
            ) from exc
        out.append((key, n - 1, n))
    keys = tuple(item[0] for item in out)
    if len(keys) != len(set(keys)):
        raise ValueError("PDK routing stack reuses a via layer key")
    return tuple(out)


def _resolved_routing_stack(
    tech: Any, max_metal: int
) -> Tuple[
    Tuple[LayerKey, ...],
    Tuple[Tuple[LayerKey, int, int], ...],
    Tuple[Any, ...],
]:
    """Resolve the exact deck ladder when the technology has a PDK identity."""
    pdk_id = str(getattr(tech, "pdk_id", "") or "").strip()
    if not pdk_id:
        metals = _metal_stack(tech, max_metal)
        return metals, _via_stack(tech, len(metals)), ()

    from clay.layout.route.deck_isolation import DeckConnectStack

    stack = DeckConnectStack.from_tech(tech)
    metals = tuple(stack.conductors)
    if len(metals) > max_metal:
        raise ValueError(
            f"full-extraction deck declares {len(metals)} metals above "
            f"max_metal={max_metal}"
        )
    positions = {key: index for index, key in enumerate(metals)}
    expected_pairs = {
        (metals[index], metals[index + 1]) for index in range(len(metals) - 1)
    }
    actual_pairs = {(item.lower, item.upper) for item in stack.connections}
    if actual_pairs != expected_pairs or len(stack.connections) != len(expected_pairs):
        raise ValueError(
            "full-extraction deck routing connections are not one contiguous ladder"
        )
    vias = tuple(
        (
            item.cut,
            positions[item.lower],
            positions[item.upper],
        )
        for item in sorted(
            stack.connections, key=lambda item: positions[item.lower]
        )
    )
    return metals, vias, tuple(stack.mim_carves)


def _layer_region(layout: kdb.Layout, cell: kdb.Cell, key: LayerKey) -> kdb.Region:
    li = layout.find_layer(*key)
    if li is None:
        return kdb.Region()
    region = kdb.Region(cell.begin_shapes_rec(li))
    region.merge()
    return region


def _islands(region: kdb.Region) -> Tuple[kdb.Region, ...]:
    return tuple(kdb.Region(poly) for poly in region.each())


def _labels(
    layout: kdb.Layout,
    cell: kdb.Cell,
    label_key: LayerKey,
) -> Tuple[Tuple[str, int, int], ...]:
    li = layout.find_layer(*label_key)
    if li is None:
        return ()
    out = []
    it = cell.begin_shapes_rec(li)
    while not it.at_end():
        shape = it.shape()
        if shape.is_text():
            text = shape.text.transformed(it.trans())
            name = str(shape.text.string).strip().upper()
            if name:
                out.append((name, int(text.x), int(text.y)))
        it.next()
    return tuple(out)


def _device_body_keys(tech: Any) -> Tuple[LayerKey, ...]:
    out = []
    for kind in ("diff", "ndiff", "pdiff", "poly"):
        try:
            key = tech.build_layer(kind)
        except Exception:
            key = None
        if key is not None:
            normalised = _normalise_key(key)
            if normalised not in out:
                out.append(normalised)
    return tuple(out)


def _contact_key(tech: Any) -> Optional[LayerKey]:
    try:
        return _normalise_key(tech.contact_key())
    except Exception:
        try:
            key = tech.build_layer("contact")
        except Exception:
            return None
        return _normalise_key(key) if key is not None else None


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[a] = b


@dataclass(frozen=True)
class _Topology:
    islands: Tuple[Tuple[kdb.Region, ...], ...]
    offsets: Tuple[int, ...]
    labels_by_fragment: Tuple[Tuple[str, ...], ...]
    device_m1_fragments: frozenset[int]
    conflicted_components: Tuple[Tuple[str, ...], ...]
    via_cuts: dict[LayerKey, Tuple["_ViaCut", ...]]
    edges: Tuple["_ViaEdge", ...]
    adjacency: Tuple[Tuple[Tuple[int, Tuple[LayerKey, int, int]], ...], ...]


@dataclass(frozen=True)
class _ViaCut:
    via_layer: LayerKey
    bbox: BBox
    lower_fragments: Tuple[int, ...]
    upper_fragments: Tuple[int, ...]


@dataclass(frozen=True)
class _ViaEdge:
    via_layer: LayerKey
    lower_fragment: int
    upper_fragment: int


@dataclass
class _JoinBudget:
    limit: int
    used: int = 0

    def consume(self, what: str) -> None:
        self.used += 1
        if self.used > self.limit:
            raise ValueError(f"{what} exceeds bounded join limit {self.limit}")


class _BBoxIndex:
    """Deterministic x-sweep broad phase for bounded exact geometry joins."""

    def __init__(self, entries: Sequence[Tuple[BBox, Any]]) -> None:
        ordered = tuple(sorted(entries, key=lambda item: (item[0], repr(item[1]))))
        self._entries = ordered
        self._starts = tuple(item[0][0] for item in ordered)

    def overlapping(
        self,
        bbox: BBox,
        *,
        tolerance: int = 0,
        budget: Optional[_JoinBudget] = None,
        what: str = "geometry join",
    ) -> Tuple[Tuple[BBox, Any], ...]:
        stop = bisect_right(self._starts, bbox[2] + tolerance)
        out = []
        for index in range(stop):
            candidate_bbox, payload = self._entries[index]
            if budget is not None:
                # Charge the broad-phase scan itself, not merely hits surviving
                # the filters below.  Otherwise disjoint intervals ordered to
                # the left permit quadratic work without consuming the budget.
                budget.consume(what)
            if candidate_bbox[2] + tolerance < bbox[0]:
                continue
            if (
                candidate_bbox[3] + tolerance < bbox[1]
                or bbox[3] + tolerance < candidate_bbox[1]
            ):
                continue
            out.append((candidate_bbox, payload))
        return tuple(out)


def emitted_artifact_sha256(gds_path: str | Path) -> str:
    """Digest the exact emitted artifact consumed by both RG6 checkers."""
    digest = hashlib.sha256()
    with Path(gds_path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def route_manifest_sha256(tagged: Sequence[Any]) -> str:
    """Canonical multiset digest for the emitter-derived route manifest."""
    if not isinstance(tagged, (tuple, list)):
        raise TypeError("route manifest must be a bounded tuple/list")
    if len(tagged) > _MAX_ROUTER_MANIFEST_ENTRIES:
        raise ValueError(
            f"route manifest exceeds bounded limit {_MAX_ROUTER_MANIFEST_ENTRIES}"
        )
    payload = []
    for index, entry in enumerate(tagged):
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            raise TypeError(f"malformed tagged entry at index {index}")
        net = str(entry[0]).strip()
        if not net:
            raise ValueError(f"tagged entry {index} has unknown owner")
        layer = entry[1]
        bbox = entry[2]
        if (
            not isinstance(layer, (tuple, list))
            or len(layer) != 2
            or any(type(value) is not int for value in layer)
        ):
            raise TypeError(f"tagged entry {index} has invalid layer key")
        if (
            not isinstance(bbox, (tuple, list))
            or len(bbox) != 4
            or any(type(value) is not int for value in bbox)
            or bbox[2] <= bbox[0]
            or bbox[3] <= bbox[1]
        ):
            raise TypeError(f"tagged entry {index} has invalid half-open bbox")
        payload.append([
            net,
            int(layer[0]),
            int(layer[1]),
            [int(value) for value in bbox],
        ])
    payload.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _normalise_terminal_claims(
    claims: Sequence[Any],
) -> Tuple[Tuple[str, LayerKey, Tuple[int, int]], ...]:
    if not isinstance(claims, (tuple, list)):
        raise TypeError("terminal claims must be a bounded tuple/list")
    if len(claims) > _MAX_TERMINAL_CLAIMS:
        raise ValueError(
            f"terminal claims exceed bounded limit {_MAX_TERMINAL_CLAIMS}"
        )
    out = set()
    for index, entry in enumerate(claims):
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            raise TypeError(f"malformed terminal claim at index {index}")
        net = str(entry[0]).strip().upper()
        try:
            layer = entry[1]
            point = entry[2]
            key = (int(layer[0]), int(layer[1]))
            xy = (int(point[0]), int(point[1]))
        except Exception as exc:
            raise TypeError(f"malformed terminal claim at index {index}") from exc
        if not net or len(layer) != 2 or len(point) != 2:
            raise ValueError(f"malformed terminal claim at index {index}")
        out.add((net, key, xy))
    return tuple(sorted(out))


def terminal_claims_sha256(claims: Sequence[Any]) -> str:
    """Canonical digest of declared physical terminal ownership points."""
    canonical = _normalise_terminal_claims(claims)
    return hashlib.sha256(json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def terminal_claims_from_pins(
    pins_by_net: Mapping[str, Sequence[Any]], tech: Any
) -> Tuple[Tuple[str, LayerKey, Tuple[int, int]], ...]:
    """Project source-derived routed pins into the bounded checker currency."""
    if not isinstance(pins_by_net, Mapping):
        raise TypeError("pins_by_net must be a mapping")
    entries = []
    for raw_net in sorted(pins_by_net, key=lambda value: str(value)):
        net = str(raw_net).strip().upper()
        if not net:
            raise ValueError("terminal claim has an empty net")
        pins = pins_by_net[raw_net]
        if not isinstance(pins, (tuple, list)):
            raise TypeError(f"pins for net {net!r} must be a bounded tuple/list")
        for pin in pins:
            raw_layer = getattr(pin, "layer", None)
            if isinstance(raw_layer, tuple) and len(raw_layer) == 2:
                layer = (int(raw_layer[0]), int(raw_layer[1]))
            elif isinstance(raw_layer, str):
                token = raw_layer.strip().lower()
                for prefix in ("metal", "met", "m"):
                    if token.startswith(prefix):
                        token = token[len(prefix):]
                        break
                if not token.isdigit() or int(token) <= 0:
                    raise ValueError(
                        f"terminal claim has unresolved layer: {raw_layer!r}"
                    )
                layer = _normalise_key(tech.metal_key(int(token)))
            else:
                raise ValueError(
                    f"terminal claim has unresolved layer: {raw_layer!r}"
                )
            raw_xy_nm = getattr(pin, "xy_nm", None)
            if raw_xy_nm is None:
                raw_xy_um = getattr(pin, "xy_um", None)
                if raw_xy_um is None or len(raw_xy_um) != 2:
                    raise ValueError(f"terminal claim has no physical point: {pin!r}")
                values = tuple(float(value) for value in raw_xy_um)
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("terminal claim has non-finite coordinates")
                point = tuple(int(round(value * 1000.0)) for value in values)
            else:
                if len(raw_xy_nm) != 2:
                    raise ValueError(f"terminal claim has malformed point: {pin!r}")
                point = tuple(int(value) for value in raw_xy_nm)
            entries.append((net, layer, point))
            if len(entries) > _MAX_TERMINAL_CLAIMS:
                raise ValueError(
                    f"terminal claims exceed bounded limit {_MAX_TERMINAL_CLAIMS}"
                )
    return _normalise_terminal_claims(tuple(entries))


def _topology(
    layout: kdb.Layout,
    cell: kdb.Cell,
    tech: Any,
    metals: Tuple[LayerKey, ...],
    vias: Tuple[Tuple[LayerKey, int, int], ...],
    mim_carves: Sequence[Any] = (),
) -> _Topology:
    islands = tuple(_islands(_layer_region(layout, cell, key)) for key in metals)
    total_islands = sum(len(layer_islands) for layer_islands in islands)
    if total_islands > _MAX_TOPOLOGY_ISLANDS:
        raise ValueError(
            f"routing topology islands exceed bounded limit {_MAX_TOPOLOGY_ISLANDS}"
        )
    offsets = []
    total = 0
    for layer_islands in islands:
        offsets.append(total)
        total += len(layer_islands)

    join_budget = _JoinBudget(_MAX_TOPOLOGY_JOIN_EVALUATIONS)
    island_indexes = []
    for layer_i, layer_islands in enumerate(islands):
        entries = []
        for local, island in enumerate(layer_islands):
            box = island.bbox()
            entries.append(((box.left, box.bottom, box.right, box.top), local))
        island_indexes.append(_BBoxIndex(entries))

    labels_by_fragment = [set() for _ in range(total)]
    try:
        label_key = _normalise_key(tech.label_key(1))
    except Exception:
        label_key = metals[0]
    for name, x, y in _labels(layout, cell, label_key):
        probe = kdb.Region(kdb.Box(x - 1, y - 1, x + 1, y + 1))
        for _bbox, local in island_indexes[0].overlapping(
            (x - 1, y - 1, x + 1, y + 1),
            budget=join_budget,
            what="label-to-metal join",
        ):
            island = islands[0][local]
            if not (island & probe).is_empty():
                labels_by_fragment[offsets[0] + local].add(name)

    device_m1 = set()
    contact_key = _contact_key(tech)
    body_keys = _device_body_keys(tech)
    if contact_key is not None and body_keys:
        contacts = _layer_region(layout, cell, contact_key)
        body = kdb.Region()
        for key in body_keys:
            body += _layer_region(layout, cell, key)
        body.merge()
        device_contacts = contacts.interacting(body)
        for local, island in enumerate(islands[0]):
            if not island.interacting(device_contacts).is_empty():
                device_m1.add(offsets[0] + local)

    union = _UnionFind(total)
    via_cuts: dict[LayerKey, Tuple[_ViaCut, ...]] = {}
    edges = []
    cut_total = 0
    routable_lower_by_connection: dict[Tuple[LayerKey, LayerKey], kdb.Region] = {}
    for carve in mim_carves:
        bottom_key = _normalise_key(carve.bottom)
        riser_key = _normalise_key(carve.riser)
        marker_key = _normalise_key(carve.top_marker)
        if bottom_key not in metals:
            raise ValueError(f"MIM carve bottom {bottom_key} is outside the routing stack")
        raw_bottom = _layer_region(layout, cell, bottom_key)
        marker = _layer_region(layout, cell, marker_key)
        interacting = raw_bottom.interacting(marker)
        oversize_um = float(getattr(carve, "oversize_um", 0.0))
        if oversize_um < 0.0:
            raise ValueError(f"MIM carve has negative oversize: {oversize_um}")
        if oversize_um > 0.0:
            oversize_dbu = int(round(oversize_um / float(layout.dbu)))
            mim_virtual = marker.sized(oversize_dbu) & interacting
        else:
            mim_virtual = interacting
        routable = raw_bottom - mim_virtual
        routable.merge()
        routable_lower_by_connection[(bottom_key, riser_key)] = routable
    for via_key, lower_i, upper_i in vias:
        via_region = _layer_region(layout, cell, via_key)
        routable_lower = routable_lower_by_connection.get(
            (metals[lower_i], via_key)
        )
        cuts = []
        for polygon in via_region.each():
            cut_total += 1
            if cut_total > _MAX_TOPOLOGY_VIA_CUTS:
                raise ValueError(
                    f"routing via cuts exceed bounded limit {_MAX_TOPOLOGY_VIA_CUTS}"
                )
            cut = kdb.Region(polygon)
            box = polygon.bbox()
            bbox = (box.left, box.bottom, box.right, box.top)
            lower_hits = tuple(
                offsets[lower_i] + local
                for _candidate_bbox, local in island_indexes[lower_i].overlapping(
                    bbox,
                    budget=join_budget,
                    what="via-to-lower-metal join",
                )
                for island in (islands[lower_i][local],)
                if not (
                    (island & cut)
                    if routable_lower is None
                    else (island & routable_lower & cut)
                ).is_empty()
            )
            upper_hits = tuple(
                offsets[upper_i] + local
                for _candidate_bbox, local in island_indexes[upper_i].overlapping(
                    bbox,
                    budget=join_budget,
                    what="via-to-upper-metal join",
                )
                for island in (islands[upper_i][local],)
                if not (island & cut).is_empty()
            )
            cuts.append(
                _ViaCut(
                    via_layer=via_key,
                    bbox=bbox,
                    lower_fragments=lower_hits,
                    upper_fragments=upper_hits,
                )
            )
            for lower in lower_hits:
                for upper in upper_hits:
                    union.union(lower, upper)
                    edges.append(
                        _ViaEdge(
                            via_layer=via_key,
                            lower_fragment=lower,
                            upper_fragment=upper,
                        )
                    )
        via_cuts[via_key] = tuple(cuts)

    adjacency: list[list[Tuple[int, Tuple[LayerKey, int, int]]]] = [
        [] for _ in range(total)
    ]
    for edge in edges:
        logical = (edge.via_layer, edge.lower_fragment, edge.upper_fragment)
        adjacency[edge.lower_fragment].append((edge.upper_fragment, logical))
        adjacency[edge.upper_fragment].append((edge.lower_fragment, logical))

    component_labels: dict[int, set[str]] = {}
    for fragment, names in enumerate(labels_by_fragment):
        component_labels.setdefault(union.find(fragment), set()).update(names)
    conflicts = tuple(
        sorted(
            (tuple(sorted(names)) for names in component_labels.values() if len(names) > 1),
            key=lambda names: (len(names), names),
        )
    )
    return _Topology(
        islands=islands,
        offsets=tuple(offsets),
        labels_by_fragment=tuple(tuple(sorted(names)) for names in labels_by_fragment),
        device_m1_fragments=frozenset(device_m1),
        conflicted_components=conflicts,
        via_cuts=via_cuts,
        edges=tuple(edges),
        adjacency=tuple(
            tuple(sorted(neighbours, key=lambda item: (item[0], item[1])))
            for neighbours in adjacency
        ),
    )


def _topology_with_terminal_claims(
    layout: kdb.Layout,
    topology: _Topology,
    metals: Tuple[LayerKey, ...],
    claims: Tuple[Tuple[str, LayerKey, Tuple[int, int]], ...],
) -> Tuple[_Topology, Tuple[str, ...]]:
    """Seed exact emitted-metal fragments from source-derived terminal points."""
    metal_positions = {key: index for index, key in enumerate(metals)}
    indexes = []
    for layer_i, islands in enumerate(topology.islands):
        entries = []
        for local, island in enumerate(islands):
            box = island.bbox()
            entries.append(((box.left, box.bottom, box.right, box.top), local))
        indexes.append(_BBoxIndex(entries))
    scale = float(layout.dbu) * 1000.0
    labels = [set(names) for names in topology.labels_by_fragment]
    errors = []
    budget = _JoinBudget(_MAX_MANIFEST_JOIN_EVALUATIONS)
    for net, key, point_nm in claims:
        layer_i = metal_positions.get(key)
        if layer_i is None:
            errors.append(
                f"terminal claim {net}@{key}:{point_nm} is outside the routing stack"
            )
            continue
        x = round(point_nm[0] / scale)
        y = round(point_nm[1] / scale)
        probe = kdb.Region(kdb.Box(x - 1, y - 1, x + 1, y + 1))
        matches = []
        for _bbox, local in indexes[layer_i].overlapping(
            (x - 1, y - 1, x + 1, y + 1),
            budget=budget,
            what="terminal-claim-to-metal join",
        ):
            if not (topology.islands[layer_i][local] & probe).is_empty():
                matches.append(local)
        if len(matches) != 1:
            errors.append(
                f"terminal claim {net}@{key}:{point_nm} matched {len(matches)} emitted islands"
            )
            continue
        labels[topology.offsets[layer_i] + matches[0]].add(net)

    union = _UnionFind(len(labels))
    for edge in topology.edges:
        union.union(edge.lower_fragment, edge.upper_fragment)
    component_labels: dict[int, set[str]] = {}
    for fragment, names in enumerate(labels):
        component_labels.setdefault(union.find(fragment), set()).update(names)
    conflicts = tuple(sorted(
        (tuple(sorted(names)) for names in component_labels.values() if len(names) > 1),
        key=lambda names: (len(names), names),
    ))
    return replace(
        topology,
        labels_by_fragment=tuple(tuple(sorted(names)) for names in labels),
        conflicted_components=conflicts,
    ), tuple(errors)


def _bbox_matches(left: BBox, right: BBox, tolerance_dbu: int) -> bool:
    return all(abs(int(a) - int(b)) <= tolerance_dbu for a, b in zip(left, right))


def _malformed_observation(index: int, entry: Any) -> ViaProvenanceObservation:
    try:
        expected = str(entry[0]).strip().upper()
    except Exception:
        expected = ""
    return ViaProvenanceObservation(
        status="unmeasurable",
        expected_net=expected,
        lower_nets=(),
        upper_nets=(),
        via_layer=(-1, -1),
        via_bbox=(0, 0, 0, 0),
        reason=f"malformed tagged entry at index {index}",
    )


def _normalise_tagged(
    tagged: Iterable[Any],
) -> Tuple[Tuple[TaggedEntry, ...], Tuple[ViaProvenanceObservation, ...]]:
    out = []
    malformed = []
    for index, entry in enumerate(tagged):
        try:
            net, layer, bbox = entry
            key = _normalise_key(layer)
            box = tuple(int(v) for v in bbox)
            expected = str(net).strip().upper()
            if len(box) != 4 or not expected:
                raise ValueError("empty net or invalid bbox")
            out.append((expected, key, box))
        except Exception:
            malformed.append(_malformed_observation(index, entry))
    return tuple(out), tuple(malformed)


class _ManifestOwnerIndex:
    def __init__(
        self,
        layout: kdb.Layout,
        tagged: Tuple[TaggedEntry, ...],
        metal_keys: frozenset[LayerKey],
    ) -> None:
        scale = float(layout.dbu) * 1000.0
        by_key: dict[LayerKey, list[Tuple[BBox, str]]] = {}
        for net, key, bbox_nm in tagged:
            if key not in metal_keys:
                continue
            bbox = tuple(round(value / scale) for value in bbox_nm)
            by_key.setdefault(key, []).append((bbox, net))
        self._by_key = {
            key: _BBoxIndex(entries) for key, entries in by_key.items()
        }
        self._budget = _JoinBudget(_MAX_MANIFEST_JOIN_EVALUATIONS)

    def owners(self, key: LayerKey, island: kdb.Region) -> Tuple[str, ...]:
        index = self._by_key.get(key)
        if index is None:
            return ()
        box = island.bbox()
        bbox = (box.left, box.bottom, box.right, box.top)
        owners = set()
        for candidate_bbox, net in index.overlapping(
            bbox,
            budget=self._budget,
            what="manifest-to-emitted-metal join",
        ):
            if not (island & kdb.Region(kdb.Box(*candidate_bbox))).is_empty():
                owners.add(net)
        return tuple(sorted(owners))


def _fragment_manifest_owners(
    topology: _Topology,
    owner_index: _ManifestOwnerIndex,
    *,
    metal_key: LayerKey,
    metal_index: int,
    fragment: int,
) -> Tuple[str, ...]:
    """Owners of route-tagged conductors physically covering one GDS island.

    The manifest supplies ownership, not geometry: each bbox is intersected with
    the independently reconstructed emitted-metal island before it can name a
    side of a via.  Device-local M1 remains independently labelled below.
    """
    local = fragment - topology.offsets[metal_index]
    if local < 0 or local >= len(topology.islands[metal_index]):
        return ()
    island = topology.islands[metal_index][local]
    return owner_index.owners(metal_key, island)


def _side_identity(
    labelled: Tuple[str, ...],
    owners: Tuple[str, ...],
    *,
    require_independent_label: bool,
) -> Tuple[Tuple[str, ...], bool]:
    """Resolve one via side and report an explicit label/owner contradiction."""
    if require_independent_label:
        if owners and labelled and owners != labelled:
            return tuple(sorted(set(owners) | set(labelled))), True
        return labelled, False
    if owners and labelled and owners != labelled:
        return tuple(sorted(set(owners) | set(labelled))), True
    return (labelled or owners), False


def _visible_labels(
    topology: _Topology,
    start: int,
    *,
    excluded_logical_via: Tuple[LayerKey, int, int],
    budget: Optional[_JoinBudget] = None,
) -> Tuple[str, ...]:
    """M1 labels reachable from ``start`` with one logical via pair removed.

    All redundant physical cuts connecting the same lower/upper fragments share
    the same logical-via key and are removed together.  This prevents one cut of
    an array from making the other cut feed the expected net back into the lower
    side during provenance adjudication.
    """
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbour, logical_via in topology.adjacency[current]:
            if budget is not None:
                budget.consume("via-provenance graph traversal")
            if logical_via == excluded_logical_via:
                continue
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    names = set()
    for fragment in seen:
        names.update(topology.labels_by_fragment[fragment])
    return tuple(sorted(names))


def check_via_provenance(
    gds_path: str | Path,
    tech: Any,
    *,
    tagged: Optional[Sequence[Any]] = None,
    terminal_claims: Optional[Sequence[Any]] = None,
    top_cell: Optional[str] = None,
    max_metal: int = 8,
    grid_tolerance_nm: int = 1,
) -> ViaProvenanceReport:
    """Check every router-owned routing via in an emitted GDS.

    ``tagged`` is the canonical routed-cell manifest of ``(net, layer, bbox)``
    triples; each bbox is in top-cell coordinates expressed as integer
    nanometres.  When it is absent, no physical via is guessed to be
    router-owned; global multi-label components are still reconstructed and
    reported.
    """

    if kdb is None:
        return ViaProvenanceReport(
            status="unmeasurable", reason="klayout.db is unavailable"
        )

    try:
        metals, vias, mim_carves = _resolved_routing_stack(tech, max_metal)
    except Exception as exc:
        return ViaProvenanceReport(
            status="unsupported", reason=f"incomplete PDK routing stack: {exc}"
        )
    if len(metals) < 2:
        return ViaProvenanceReport(
            status="unsupported", reason="pdk declares fewer than two routing metals"
        )
    if not vias or vias[0][1:] != (0, 1):
        return ViaProvenanceReport(
            status="unsupported", reason="pdk declares no VIA1 between M1 and M2"
        )

    layout = kdb.Layout()
    try:
        layout.read(str(gds_path))
    except Exception as exc:
        return ViaProvenanceReport(
            status="unmeasurable", reason=f"cannot read emitted GDS {gds_path}: {exc}"
        )
    try:
        artifact_digest = emitted_artifact_sha256(gds_path)
    except Exception as exc:
        return ViaProvenanceReport(
            status="unmeasurable",
            reason=f"cannot digest emitted GDS {gds_path}: {exc}",
        )
    if top_cell is not None:
        cell = layout.cell(top_cell)
        if cell is None:
            return ViaProvenanceReport(
                status="unmeasurable", reason=f"top cell {top_cell!r} not found"
            )
    else:
        tops = layout.top_cells()
        if not tops:
            return ViaProvenanceReport(status="unmeasurable", reason="GDS has no top cell")
        if len(tops) != 1:
            return ViaProvenanceReport(
                status="unmeasurable",
                reason=f"multiple top cells ({len(tops)}); select top_cell explicitly",
            )
        cell = tops[0]

    try:
        topology = _topology(
            layout, cell, tech, metals, vias, mim_carves=mim_carves
        )
    except Exception as exc:
        return ViaProvenanceReport(
            status="unmeasurable", reason=f"cannot reconstruct emitted-GDS topology: {exc}"
        )
    claim_digest = ""
    claim_count = 0
    if terminal_claims is not None:
        try:
            canonical_claims = _normalise_terminal_claims(terminal_claims)
            claim_digest = terminal_claims_sha256(canonical_claims)
            claim_count = len(canonical_claims)
            topology, claim_errors = _topology_with_terminal_claims(
                layout, topology, metals, canonical_claims
            )
        except Exception as exc:
            return ViaProvenanceReport(
                status="unmeasurable",
                manifest_available=tagged is not None,
                artifact_sha256=artifact_digest,
                terminal_claim_count=claim_count,
                terminal_claims_sha256=claim_digest,
                reason=f"cannot bind terminal claims to emitted GDS: {exc}",
            )
        if claim_errors:
            return ViaProvenanceReport(
                status="unmeasurable",
                conflicted_components=topology.conflicted_components,
                manifest_available=tagged is not None,
                artifact_sha256=artifact_digest,
                terminal_claim_count=claim_count,
                terminal_claims_sha256=claim_digest,
                reason="; ".join(claim_errors),
            )
    conflicts = topology.conflicted_components
    if tagged is None:
        return ViaProvenanceReport(
            status="fail" if conflicts else "advisory",
            conflicted_components=conflicts,
            manifest_available=False,
            artifact_sha256=artifact_digest,
            terminal_claim_count=claim_count,
            terminal_claims_sha256=claim_digest,
            reason=(
                "multi-label routing component found"
                if conflicts
                else "route manifest unavailable; per-via provenance not adjudicated"
            ),
        )

    if not isinstance(tagged, (tuple, list)):
        return ViaProvenanceReport(
            status="unmeasurable",
            manifest_available=False,
            artifact_sha256=artifact_digest,
            terminal_claim_count=claim_count,
            terminal_claims_sha256=claim_digest,
            reason="route manifest must be a bounded tuple/list",
        )
    if len(tagged) > _MAX_ROUTER_MANIFEST_ENTRIES:
        return ViaProvenanceReport(
            status="unmeasurable",
            manifest_available=False,
            artifact_sha256=artifact_digest,
            terminal_claim_count=claim_count,
            terminal_claims_sha256=claim_digest,
            reason=(
                "route manifest exceeds bounded limit "
                f"{_MAX_ROUTER_MANIFEST_ENTRIES}"
            ),
        )

    manifest_bind_error = ""
    try:
        manifest_digest = route_manifest_sha256(tagged)
    except Exception as exc:
        manifest_digest = ""
        manifest_bind_error = f"cannot bind route manifest: {exc}"
    normalised, malformed = _normalise_tagged(tagged)
    if manifest_bind_error and not malformed:
        malformed = (
            ViaProvenanceObservation(
                status="unmeasurable",
                expected_net="",
                lower_nets=(),
                upper_nets=(),
                via_layer=(-1, -1),
                via_bbox=(0, 0, 0, 0),
                reason=manifest_bind_error,
            ),
        )
    via_spec = {key: (lower_i, upper_i) for key, lower_i, upper_i in vias}
    known_route_keys = frozenset(metals) | frozenset(via_spec)
    unknown = tuple(entry for entry in normalised if entry[1] not in known_route_keys)
    for expected, layer_key, bbox in unknown:
        malformed += (
            ViaProvenanceObservation(
                status="unmeasurable",
                expected_net=expected,
                lower_nets=(),
                upper_nets=(),
                via_layer=layer_key,
                via_bbox=bbox,
                reason=f"router-owned layer {layer_key} is absent from the complete PDK routing stack",
            ),
        )
    entries = tuple(entry for entry in normalised if entry[1] in via_spec)
    tolerance_dbu = max(1, round(float(grid_tolerance_nm) / (layout.dbu * 1000.0)))
    observations = list(malformed)
    owner_index = _ManifestOwnerIndex(layout, normalised, frozenset(metals))
    cut_indexes = {
        key: _BBoxIndex(tuple((cut.bbox, ordinal) for ordinal, cut in enumerate(cuts)))
        for key, cuts in topology.via_cuts.items()
    }
    match_budget = _JoinBudget(_MAX_MANIFEST_JOIN_EVALUATIONS)
    graph_budget = _JoinBudget(_MAX_PROVENANCE_GRAPH_VISITS)
    visible_cache: dict[
        Tuple[LayerKey, int, int], Tuple[Tuple[str, ...], Tuple[str, ...]]
    ] = {}
    owner_cache: dict[Tuple[LayerKey, int], Tuple[str, ...]] = {}
    claimed_cuts: dict[Tuple[LayerKey, int], str] = {}
    for expected, via_key, tagged_box_nm in entries:
        lower_i, upper_i = via_spec[via_key]
        physical = topology.via_cuts.get(via_key, ())
        tagged_box = tuple(
            round(value / (layout.dbu * 1000.0)) for value in tagged_box_nm
        )
        try:
            indexed_matches = cut_indexes.get(via_key, _BBoxIndex(())).overlapping(
                tagged_box,
                tolerance=tolerance_dbu,
                budget=match_budget,
                what="manifest-via-to-emitted-cut join",
            )
        except Exception as exc:
            return ViaProvenanceReport(
                status="unmeasurable",
                observations=tuple(observations),
                unresolved=tuple(
                    item for item in observations if item.status == "unmeasurable"
                ),
                conflicted_components=conflicts,
                manifest_available=True,
                router_owned_vias=len(entries),
                artifact_sha256=artifact_digest,
                route_manifest_sha256=manifest_digest,
                terminal_claim_count=claim_count,
                terminal_claims_sha256=claim_digest,
                reason=str(exc),
            )
        match_ordinals = tuple(
            ordinal
            for _bbox, ordinal in indexed_matches
            if _bbox_matches(physical[ordinal].bbox, tagged_box, tolerance_dbu)
        )
        matches = tuple(physical[ordinal] for ordinal in match_ordinals)
        if len(matches) != 1:
            observations.append(
                ViaProvenanceObservation(
                    status="unmeasurable",
                    expected_net=expected,
                    lower_nets=(),
                    upper_nets=(),
                    via_layer=via_key,
                    via_bbox=tagged_box_nm,
                    reason=f"router via matched {len(matches)} emitted cuts",
                )
            )
            continue
        claim_key = (via_key, match_ordinals[0])
        prior_owner = claimed_cuts.get(claim_key)
        if prior_owner is not None:
            observations.append(
                ViaProvenanceObservation(
                    status="fail" if prior_owner != expected else "unmeasurable",
                    expected_net=expected,
                    lower_nets=(prior_owner,),
                    upper_nets=(),
                    via_layer=via_key,
                    via_bbox=tagged_box_nm,
                    reason=(
                        "one emitted cut is claimed by conflicting router nets"
                        if prior_owner != expected
                        else "one emitted cut is duplicated in the route manifest"
                    ),
                )
            )
            continue
        claimed_cuts[claim_key] = expected
        matched = matches[0]
        if len(matched.lower_fragments) != 1 or len(matched.upper_fragments) != 1:
            observations.append(
                ViaProvenanceObservation(
                    status="unmeasurable",
                    expected_net=expected,
                    lower_nets=(),
                    upper_nets=(),
                    via_layer=via_key,
                    via_bbox=tagged_box_nm,
                    reason=(
                        "emitted cut touches "
                        f"{len(matched.lower_fragments)} lower and "
                        f"{len(matched.upper_fragments)} upper metal islands"
                    ),
                )
            )
            continue
        lower_fragment = matched.lower_fragments[0]
        upper_fragment = matched.upper_fragments[0]
        logical_via = (via_key, lower_fragment, upper_fragment)
        visible = visible_cache.get(logical_via)
        if visible is None:
            try:
                visible = (
                    _visible_labels(
                        topology,
                        lower_fragment,
                        excluded_logical_via=logical_via,
                        budget=graph_budget,
                    ),
                    _visible_labels(
                        topology,
                        upper_fragment,
                        excluded_logical_via=logical_via,
                        budget=graph_budget,
                    ),
                )
            except Exception as exc:
                observations.append(
                    ViaProvenanceObservation(
                        status="unmeasurable",
                        expected_net=expected,
                        lower_nets=(),
                        upper_nets=(),
                        via_layer=via_key,
                        via_bbox=tagged_box_nm,
                        reason=str(exc),
                    )
                )
                continue
            visible_cache[logical_via] = visible
        lower_labels, upper_labels = visible

        lower_owner_key = (metals[lower_i], lower_fragment)
        try:
            if lower_owner_key not in owner_cache:
                owner_cache[lower_owner_key] = _fragment_manifest_owners(
                    topology,
                    owner_index,
                    metal_key=metals[lower_i],
                    metal_index=lower_i,
                    fragment=lower_fragment,
                )
            lower_owners = owner_cache[lower_owner_key]
            upper_owner_key = (metals[upper_i], upper_fragment)
            if upper_owner_key not in owner_cache:
                owner_cache[upper_owner_key] = _fragment_manifest_owners(
                    topology,
                    owner_index,
                    metal_key=metals[upper_i],
                    metal_index=upper_i,
                    fragment=upper_fragment,
                )
            upper_owners = owner_cache[upper_owner_key]
        except Exception as exc:
            observations.append(
                ViaProvenanceObservation(
                    status="unmeasurable",
                    expected_net=expected,
                    lower_nets=lower_labels,
                    upper_nets=upper_labels,
                    via_layer=via_key,
                    via_bbox=tagged_box_nm,
                    reason=str(exc),
                )
            )
            continue
        device_m1_landing = (
            lower_i == 0 and lower_fragment in topology.device_m1_fragments
        )
        lower_nets, lower_conflict = _side_identity(
            lower_labels,
            lower_owners,
            require_independent_label=device_m1_landing,
        )
        upper_nets, upper_conflict = _side_identity(
            upper_labels,
            upper_owners,
            require_independent_label=False,
        )
        if lower_conflict or upper_conflict:
            observations.append(
                ViaProvenanceObservation(
                    status="fail",
                    expected_net=expected,
                    lower_nets=lower_nets,
                    upper_nets=upper_nets,
                    via_layer=via_key,
                    via_bbox=tagged_box_nm,
                    reason="route owner contradicts independently visible emitted net",
                )
            )
            continue
        if len(lower_nets) != 1 or len(upper_nets) != 1:
            observations.append(
                ViaProvenanceObservation(
                    status="unmeasurable",
                    expected_net=expected,
                    lower_nets=lower_nets,
                    upper_nets=upper_nets,
                    via_layer=via_key,
                    via_bbox=tagged_box_nm,
                    reason="lower or upper emitted net is missing or ambiguous",
                )
            )
            continue
        status = (
            "pass"
            if lower_nets == upper_nets == (expected,)
            else "fail"
        )
        observations.append(
            ViaProvenanceObservation(
                status=status,
                expected_net=expected,
                lower_nets=lower_nets,
                upper_nets=upper_nets,
                via_layer=via_key,
                via_bbox=tagged_box_nm,
                reason=(
                    "same-net router-owned via"
                    if status == "pass"
                    else "emitted lower/upper nets or manifest intent disagree"
                ),
            )
        )

    observation_tuple = tuple(observations)
    violations = tuple(o for o in observation_tuple if o.status == "fail")
    unresolved = tuple(o for o in observation_tuple if o.status == "unmeasurable")
    checked = sum(o.status in ("pass", "fail") for o in observation_tuple)
    if violations or conflicts:
        status = "fail"
        reason = "unsafe via landing or multi-label routing component found"
    elif unresolved:
        status = "unmeasurable"
        reason = "one or more router-owned vias lack unique provenance"
    elif checked:
        status = "pass"
        reason = "all router-owned vias are adjudicated same-net"
    else:
        status = "advisory"
        reason = "no router-owned routing via was available to adjudicate"
    return ViaProvenanceReport(
        status=status,
        checked=checked,
        observations=observation_tuple,
        violations=violations,
        unresolved=unresolved,
        conflicted_components=conflicts,
        manifest_available=True,
        router_owned_vias=len(entries),
        artifact_sha256=artifact_digest,
        route_manifest_sha256=manifest_digest,
        terminal_claim_count=claim_count,
        terminal_claims_sha256=claim_digest,
        reason=reason,
    )


__all__ = [
    "ViaProvenanceObservation",
    "ViaProvenanceReport",
    "check_via_provenance",
    "emitted_artifact_sha256",
    "route_manifest_sha256",
    "terminal_claims_from_pins",
    "terminal_claims_sha256",
]
