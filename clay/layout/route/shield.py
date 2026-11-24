"""Coordinate-free materialization of declared route shield fences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

from clay.layout.intent import replay_shield_inventory
from clay.layout.route import Pin


class ShieldMaterializationError(ValueError):
    """A declared shield cannot be placed from the current terminal domains."""


@dataclass(frozen=True)
class ShieldFenceFact:
    declaration_sha256: str
    aggressor_net: str
    victim_net: str
    shield_net: str
    axis: str
    layer_key: tuple[int, int]
    endpoints_nm: tuple[tuple[int, int], tuple[int, int]]
    pin_owner: str


def _installed_shield_layer_numbers(tech: Any):
    from clay.layout.route.topology_compile import (
        TopologyCompileError,
        _pdk_min_width_nm,
        _pdk_transport_layer_closure,
    )

    widths: dict[int, int] = {}
    by_axis: dict[str, list[int]] = {"H": [], "V": []}
    for layer_n in _pdk_transport_layer_closure(tech):
        try:
            widths[layer_n] = _pdk_min_width_nm(tech, layer_n)
        except TopologyCompileError:
            continue
        axis = str(tech.metal_preferred_axis(layer_n)).upper()
        if axis in by_axis:
            by_axis[axis].append(layer_n)
    if any(not layers for layers in by_axis.values()):
        raise ShieldMaterializationError(
            "declared shield requires installed H and V transport layers"
        )
    return by_axis, widths


def installed_shield_layer_policy(
    tech: Any,
) -> tuple[dict[str, tuple[int, int]], int, int]:
    """Resolve the highest H/V layers with installed, cited width rules."""
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

    by_axis, widths = _installed_shield_layer_numbers(tech)
    selected = {axis: layers[-1] for axis, layers in by_axis.items()}
    return (
        {
            axis: tuple(tech.metal_key(layer_n))
            for axis, layer_n in selected.items()
        },
        pdk_manufacturing_grid_nm(tech),
        max(widths[layer_n] for layer_n in selected.values()),
    )


def _mim_shield_blockers(geometry, placement_um, tech):
    """Replay the router's exact whole-component MIM-carve exclusion."""
    import klayout.db as kdb
    from clay.layout.placed_geometry import placed_point_nm
    from clay.layout.route.deck_isolation import DeckConnectStack
    from clay.layout.route.engine_gridless import _marker_selected_conductor_components

    if not isinstance(geometry, Mapping) or not isinstance(placement_um, Mapping):
        raise ShieldMaterializationError('MIM shield admission requires complete geometry and placement')
    carves = DeckConnectStack.from_tech(tech).mim_carves
    relevant = {tuple(key) for carve in carves for key in (carve.bottom,carve.top_marker)}
    literals = []
    for name,stamp in geometry.items():
        for shape in tuple(getattr(stamp,'polygons',()) or ()):
            key = (int(shape.layer),int(shape.datatype))
            if key not in relevant:
                continue
            if name not in placement_um:
                raise ShieldMaterializationError(f'MIM shield geometry has no placement: {name}')
            points = [kdb.Point(*placed_point_nm(point.x/1000.,point.y/1000.,placement_um[name]))
                      for point in shape.polygon.vertices]
            region = kdb.Region(kdb.Polygon(points))
            for polygon in region.decompose_trapezoids_to_region().each():
                box = polygon.bbox()
                if polygon.area() != box.area():
                    raise ShieldMaterializationError('MIM shield domain is not exact rectilinear geometry')
                literals.append((key,None,(int(box.left),int(box.bottom),int(box.right),int(box.top))))
    blockers = {}
    for carve in carves:
        selected = _marker_selected_conductor_components(literals,
            conductor_layer=tuple(carve.bottom),marker_layer=tuple(carve.top_marker))
        boxes = {tuple(box) for _owner,box in selected}
        boxes.update(box for key,_owner,box in literals if key == tuple(carve.top_marker))
        blockers.setdefault(tuple(carve.bottom),set()).update(boxes)
    return {key:tuple(sorted(boxes)) for key,boxes in blockers.items()}


def _admissible_shield_layer(axis, endpoints, *, tech, blockers, width_nm):
    from clay.layout.route.engine_gridless import _closed_rects_touch

    by_axis,_widths = _installed_shield_layer_numbers(tech)
    half = (width_nm+1)//2
    (x0,y0),(x1,y1) = endpoints
    body = (min(x0,x1)-half,min(y0,y1)-half,max(x0,x1)+half,max(y0,y1)+half)
    for number in reversed(by_axis[axis]):
        key = tuple(tech.metal_key(number))
        if not any(_closed_rects_touch(body,box) for box in blockers.get(key,())):
            return key
    raise ShieldMaterializationError('Every installed shield layer intersects a physical MIM carve domain')


def _pin_point(pin: object) -> tuple[int, int]:
    point = getattr(pin, "xy_nm", None)
    if (
        type(point) is not tuple
        or len(point) != 2
        or any(type(value) is not int for value in point)
    ):
        raise ShieldMaterializationError(
            "shield materialization requires exact integer-nm terminal access"
        )
    return point


def _net_alias(
    pins_by_net: Mapping[str, Sequence[object]], declared: str
) -> str:
    normalized = declared.rstrip("!").casefold()
    matches = tuple(
        net
        for net in pins_by_net
        if type(net) is str and net.rstrip("!").casefold() == normalized
    )
    if len(matches) != 1:
        raise ShieldMaterializationError(
            f"shield net {declared!r} does not bind one routed net"
        )
    return matches[0]


def _floor_grid(value: int, grid: int) -> int:
    return (value // grid) * grid


def _ceil_grid(value: int, grid: int) -> int:
    return -((-value) // grid) * grid


def _nearest_grid(value: int, grid: int) -> int:
    low = _floor_grid(value, grid)
    high = _ceil_grid(value, grid)
    return low if value - low < high - value else high


def _domain(points: Sequence[tuple[int, int]]) -> tuple[int, int, int, int]:
    if not points:
        raise ShieldMaterializationError("shield terminal domain is empty")
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _fence(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    *,
    grid_nm: int,
    minimum_width_nm: int,
) -> tuple[str, tuple[tuple[int, int], tuple[int, int]]]:
    candidates: list[tuple[int, str, tuple[tuple[int, int], tuple[int, int]]]] = []
    for axis in ("V", "H"):
        if axis == "V":
            gap_lo, gap_hi = sorted((left[2], right[0])) if left[2] < right[0] else (
                sorted((right[2], left[0])) if right[2] < left[0] else (0, 0)
            )
            span_lo, span_hi = max(left[1], right[1]), min(left[3], right[3])
        else:
            gap_lo, gap_hi = sorted((left[3], right[1])) if left[3] < right[1] else (
                sorted((right[3], left[1])) if right[3] < left[1] else (0, 0)
            )
            span_lo, span_hi = max(left[0], right[0]), min(left[2], right[2])
        gap = gap_hi - gap_lo
        start = _ceil_grid(span_lo, grid_nm)
        stop = _floor_grid(span_hi, grid_nm)
        track = _nearest_grid((gap_lo + gap_hi) // 2, grid_nm)
        half = (minimum_width_nm + 1) // 2
        if (
            gap < 2 * minimum_width_nm
            or stop - start < minimum_width_nm
            or track - half <= gap_lo
            or track + half >= gap_hi
        ):
            continue
        endpoints = (
            ((track, start), (track, stop))
            if axis == "V"
            else ((start, track), (stop, track))
        )
        candidates.append((gap, axis, endpoints))
    if not candidates:
        raise ShieldMaterializationError(
            "shield requires separated terminal domains with one nonvacuous corridor"
        )
    _gap, axis, endpoints = max(candidates, key=lambda item: (item[0], item[1]))
    return axis, endpoints


def _fence_between_terminal_domains(
    left_points: Sequence[tuple[int, int]],
    right_points: Sequence[tuple[int, int]],
    *,
    grid_nm: int,
    minimum_width_nm: int,
) -> tuple[str, tuple[tuple[int, int], tuple[int, int]]]:
    """Prefer whole-domain separation, then the nearest legal source-pin pair."""

    try:
        return _fence(
            _domain(left_points),
            _domain(right_points),
            grid_nm=grid_nm,
            minimum_width_nm=minimum_width_nm,
        )
    except ShieldMaterializationError:
        pass
    low = minimum_width_nm // 2
    high = minimum_width_nm - low

    def access_box(point: tuple[int, int]) -> tuple[int, int, int, int]:
        return (
            point[0] - low,
            point[1] - low,
            point[0] + high,
            point[1] + high,
        )

    candidates = []
    for left in left_points:
        for right in right_points:
            try:
                axis, endpoints = _fence(
                    access_box(left),
                    access_box(right),
                    grid_nm=grid_nm,
                    minimum_width_nm=minimum_width_nm,
                )
            except ShieldMaterializationError:
                continue
            separation = (
                abs(left[0] - right[0])
                if axis == "V"
                else abs(left[1] - right[1])
            )
            candidates.append((separation, axis, endpoints))
    if not candidates:
        raise ShieldMaterializationError(
            "shield requires separated terminal domains with one nonvacuous corridor"
        )
    _separation, axis, endpoints = min(candidates)
    return axis, endpoints


def materialize_declared_shield_fences(
    declarations: object,
    pins_by_net: MutableMapping[str, list[Pin]],
    *,
    layer_by_axis: Mapping[str, tuple[int, int]],
    manufacturing_grid_nm: int,
    minimum_width_nm: int,
    domain_pin_owners: frozenset[str] | None = None,
    emitted_device_geometry: Mapping[str,Any] | None = None,
    emitted_device_placement_um: Mapping[str,Any] | None = None,
    tech: Any = None,
    wire_widths_nm: Mapping[str,int] | None = None,
) -> tuple[ShieldFenceFact, ...]:
    """Add two same-net endpoints that force one route through each shield gap."""

    typed = replay_shield_inventory(declarations)
    if not typed:
        return ()
    if not isinstance(pins_by_net, MutableMapping):
        raise TypeError("shield materializer requires one mutable pin mapping")
    if (
        type(manufacturing_grid_nm) is not int
        or manufacturing_grid_nm <= 0
        or type(minimum_width_nm) is not int
        or minimum_width_nm <= 0
    ):
        raise TypeError("shield grid and width must be positive exact integers")
    if set(layer_by_axis) != {"H", "V"}:
        raise TypeError("shield layer map must name exact H and V axes")
    if domain_pin_owners is not None and (
        type(domain_pin_owners) is not frozenset
        or any(type(owner) is not str or not owner for owner in domain_pin_owners)
    ):
        raise TypeError("shield domain owners must be one exact string frozenset")
    for layer in layer_by_axis.values():
        if (
            type(layer) is not tuple
            or len(layer) != 2
            or any(type(value) is not int for value in layer)
        ):
            raise TypeError("shield layer keys must be exact integer pairs")

    blockers = None
    if any(value is not None for value in
           (emitted_device_geometry,emitted_device_placement_um,tech)):
        if tech is None:
            raise ShieldMaterializationError('MIM shield admission requires native layout technology')
        blockers = _mim_shield_blockers(emitted_device_geometry,emitted_device_placement_um,tech)
        expected_layers,_grid,_width = installed_shield_layer_policy(tech)
        if dict(layer_by_axis) != expected_layers:
            raise ShieldMaterializationError('Shield layer policy differs from installed native rules')
    widths = dict(wire_widths_nm or {})
    if any(type(width) is not int or width <= 0 for width in widths.values()):
        raise ShieldMaterializationError('Shield wire widths require positive exact nanometers')
    facts = []
    for declaration in typed:
        aggressor = _net_alias(pins_by_net, declaration.aggressor)
        victim = _net_alias(pins_by_net, declaration.victim)
        shield = _net_alias(pins_by_net, declaration.shield_net)
        owner = f"__declared_shield__:{declaration.sha256}"
        if any(
            getattr(pin, "device", None) == owner
            for net_pins in pins_by_net.values()
            for pin in net_pins
        ):
            raise ShieldMaterializationError("shield fence is already materialized")
        aggressor_points = tuple(
                _pin_point(pin)
                for pin in pins_by_net[aggressor]
                if domain_pin_owners is None
                or getattr(pin, "device", None) in domain_pin_owners
            )
        victim_points = tuple(
                _pin_point(pin)
                for pin in pins_by_net[victim]
                if domain_pin_owners is None
                or getattr(pin, "device", None) in domain_pin_owners
            )
        axis, endpoints = _fence_between_terminal_domains(
            aggressor_points,
            victim_points,
            grid_nm=manufacturing_grid_nm,
            minimum_width_nm=minimum_width_nm,
        )
        layer = layer_by_axis[axis]
        if blockers is not None:
            declared_width = max((width for net,width in widths.items()
                if net.rstrip('!').casefold() == shield.rstrip('!').casefold()),default=0)
            layer = _admissible_shield_layer(axis,endpoints,tech=tech,blockers=blockers,
                width_nm=max(minimum_width_nm,declared_width))
        for index, point in enumerate(endpoints):
            pins_by_net[shield].append(Pin(
                net=shield,
                device=owner,
                terminal=f"fence_{index}",
                xy_um=(point[0] / 1000.0, point[1] / 1000.0),
                layer=layer,
                xy_nm=point,
            ))
        facts.append(ShieldFenceFact(
            declaration_sha256=declaration.sha256,
            aggressor_net=aggressor,
            victim_net=victim,
            shield_net=shield,
            axis=axis,
            layer_key=layer,
            endpoints_nm=endpoints,
            pin_owner=owner,
        ))
    return tuple(facts)


__all__ = [
    "ShieldFenceFact",
    "ShieldMaterializationError",
    "installed_shield_layer_policy",
    "materialize_declared_shield_fences",
]
