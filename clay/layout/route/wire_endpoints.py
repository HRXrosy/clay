"""Physical endpoint completion for raster routers, before final emission.

The search grid is not a substitute for a compiler-declared terminal. Every
added min-width Manhattan connector is checked against actual foreign routed
metal and the immutable obstacle inventory. A blocked endpoint stays failed.
"""

from __future__ import annotations

import math


def bind_native_endpoint_access(pins_by_net, obstacles):
    """Classify synthetic endpoints using the compiled owned native inventory.

    A pin already on drawn native metal is not a metal-free wire endpoint.
    Routing abstractions and foreign/unowned shapes never grant that access.
    The final writer still independently resolves every point in actual GDS.
    """
    from dataclasses import replace
    from clay.layout.model import ROUTING_ABSTRACTION_KINDS

    native = [obstacle for obstacle in obstacles
              if obstacle.kind is not None
              and obstacle.kind not in ROUTING_ABSTRACTION_KINDS
              and obstacle.owner_net is not None]
    result = {}
    for net, pins in pins_by_net.items():
        result[net] = []
        for pin in pins:
            if getattr(pin, "requires_wire_endpoint", False):
                x, y = pin.xy_nm
                if any(obstacle.owner_net == net and tuple(obstacle.layer) == tuple(pin.layer)
                       and round(obstacle.x0 * 1000) <= x <= round(obstacle.x1 * 1000)
                       and round(obstacle.y0 * 1000) <= y <= round(obstacle.y1 * 1000)
                       for obstacle in native):
                    pin = replace(pin, requires_wire_endpoint=False)
            result[net].append(pin)
    return result


def declared_wire_endpoint_pin(net, device, terminal, xy_um, layer, tech):
    """Materialize a design-derived IO location on the PDK manufacturing grid."""
    from clay.layout.route.emit_gds import _nearest_on_grid, pdk_manufacturing_grid_nm
    from clay.layout.route.engine import Pin

    grid = pdk_manufacturing_grid_nm(tech)
    xy = tuple(_nearest_on_grid(round(float(v) * 1000), grid) for v in xy_um)
    return Pin(net, device, terminal, tuple(v / 1000 for v in xy), layer,
               xy_nm=xy, requires_wire_endpoint=True)


def materialize_required_wire_endpoints(nets, pins_by_net, obstacles, tech):
    """Complete exact physical pins or mark their existing NetRoute unconnected.

    ``tech`` is the raster engine's resolved PDK view. Untagged native terminals
    are unchanged. This runs in the router, not in the GDS writer or probe oracle.
    Existing net failures are never promoted to success.
    """
    if not any(getattr(pin, "requires_wire_endpoint", False)
               for pins in pins_by_net.values() for pin in pins):
        return ()

    import klayout.db as kdb
    from clay.layout.device_cell import _rect

    def region_of(wires, layer):
        region = kdb.Region()
        for wire in wires:
            if (wire.layer, wire.datatype) == layer:
                region.insert(kdb.Polygon([kdb.Point(v.x, v.y)
                                          for v in wire.polygon.vertices]))
        return region.merged()

    failed = []
    for net in sorted(pins_by_net):
        pins = [pin for pin in pins_by_net[net]
                if getattr(pin, "requires_wire_endpoint", False)]
        if not pins:
            continue
        route = nets.get(net)
        if route is None or not route.connected:
            failed.append(net)
            continue
        pending, added_length = [], 0
        for pin in pins:
            layer = tuple(pin.layer)
            layer_n = tech.resolve_layer_to_pdk(layer)
            point = pin.xy_nm
            grid = tech.manufacturing_grid_nm
            if (layer_n is None or point is None
                    or any(value % grid for value in point)):
                route.connected = False
                break
            width = int(math.ceil(tech.min_width_nm[layer_n] / (2 * grid))) * 2 * grid
            half = width // 2
            gap = tech.min_space_nm[layer_n]
            own = region_of([*route.wires, *pending], layer)
            if own.is_empty():
                route.connected = False
                break
            foreign = kdb.Region()
            for other, other_route in nets.items():
                if other != net:
                    foreign += region_of(other_route.wires, layer)
            for obstacle in obstacles:
                if tuple(obstacle.layer) == layer and obstacle.owner_net != net:
                    box = tuple(round(v * 1000) for v in
                                (obstacle.x0, obstacle.y0, obstacle.x1, obstacle.y1))
                    foreign.insert(kdb.Box(*box))
            blocked = foreign.merged().sized(gap)
            x, y = point
            targets = set()
            for polygon in own.each():
                box = polygon.bbox()
                lo_x, hi_x = box.left + half, box.right - half
                lo_y, hi_y = box.bottom + half, box.top - half
                tx = min(max(x, lo_x), hi_x) if lo_x <= hi_x else (box.left + box.right) // 2
                ty = min(max(y, lo_y), hi_y) if lo_y <= hi_y else (box.bottom + box.top) // 2
                tx = int(round(tx / grid)) * grid
                ty = int(round(ty / grid)) * grid
                targets.add((tx, ty))
            accepted = None
            for tx, ty in sorted(targets, key=lambda p: (abs(p[0] - x) + abs(p[1] - y), p)):
                for horizontal_first in (True, False):
                    bend = (tx, y) if horizontal_first else (x, ty)
                    points = ((x, y), bend, (tx, ty))
                    boxes = [
                        (min(a[0], b[0]) - half, min(a[1], b[1]) - half,
                         max(a[0], b[0]) + half, max(a[1], b[1]) + half)
                        for a, b in zip(points, points[1:])
                    ]
                    candidate = kdb.Region()
                    for box in boxes:
                        candidate.insert(kdb.Box(*box))
                    candidate.merge()
                    if (candidate & own).area() <= 0:
                        continue
                    if ((candidate & blocked).area() > 0
                            or (gap == 0 and not candidate.interacting(foreign).is_empty())):
                        continue
                    accepted = (boxes, abs(tx - x) + abs(ty - y))
                    break
                if accepted is not None:
                    break
            if accepted is None:
                route.connected = False
                break
            boxes, length = accepted
            for box in boxes:
                rectangle = kdb.Region(kdb.Box(*box))
                if not (rectangle - own).is_empty():
                    pending.append(_rect(layer, box))
            added_length += length
        if route.connected:
            route.wires.extend(pending)
            route.length_um += added_length / 1000
        else:
            failed.append(net)
    return tuple(failed)
