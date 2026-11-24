"""PDK-derived reservation and placed binding of declared routing channels.

The placement and route consumers share this module; neither invents a second
channel clearance or interprets a remembered layout coordinate.
"""
from __future__ import annotations

from .constraints import ChannelRoute
from .engine_maze import _Tech
from .via_model import via_landing


def channel_clearance_nm(channel, tech) -> int:
    """Track center to the bordering device row, including every via pad."""
    view = _Tech(tech)
    levels = [channel.trunk_layer, channel.branch_layer]
    levels.extend(layer for _net, layer in channel.branch_layers)
    indices = [view.resolve_layer_to_gi(layer) for layer in levels]
    if any(index is None for index in indices):
        raise ValueError("channel names a layer outside the PDK routing stack")
    half = (channel.width_nm + 1) // 2
    spacing = 0
    for gi in range(max(indices) + 1):
        level = view._gi_to_pdk[gi]
        half = max(half, (view.gi_min_width(gi) + 1) // 2)
        spacing = max(spacing, view.min_space_nm[level])
        if gi < max(indices):
            landing = via_landing(tech, f"via{level}", 0, 0)
            if landing is None:
                raise ValueError("channel branch has no PDK via landing rule")
            half = max(half, *map(abs, landing.below_metal_rect),
                       *map(abs, landing.above_metal_rect))
    grid = view.manufacturing_grid_nm
    return ((half + spacing + grid - 1) // grid) * grid


def planned_channel_height_nm(channel, tech) -> int:
    """Required gap for one declaration, measured between actual row boxes."""
    clearance = channel_clearance_nm(channel, tech)
    return (len(channel.nets) - 1) * channel.pitch_nm + 2 * clearance


def bind_channel_routes(intent, placement_um, geometry, tech, placement):
    """Bind source row membership to canonical, already materialized stamps."""
    from .route_cell import _cell_bbox_um

    declarations = tuple(getattr(intent, "channel_routes", ()) or ())
    if not declarations:
        return ()
    from clay.layout.place.ordered_mirror_rows import verify_ordered_row_placement
    verify_ordered_row_placement(placement)
    groups = {group.group: tuple(group.devices)
              for group in getattr(intent, "place_groups", ())}
    result = []
    for declaration in declarations:
        boxes = []
        for name in declaration.between:
            members = groups.get(name)
            if not members or any(member not in geometry or member not in placement_um
                                  for member in members):
                raise ValueError(f"channel group {name!r} lacks actual member geometry")
            bbox = _cell_bbox_um({m: placement_um[m] for m in members},
                                {m: geometry[m] for m in members})
            if bbox is None:
                raise ValueError("channel row has no actual bounding box")
            boxes.append(tuple(round(value * 1000) for value in bbox))
        low, high = boxes[0][3], boxes[1][1]
        if high - low < planned_channel_height_nm(declaration, tech):
            raise ValueError("declared channel does not fit between actual placed rows: "
                             f"available={high-low}nm required="
                             f"{planned_channel_height_nm(declaration, tech)}nm")
        axis2 = None
        if declaration.centered_nets:
            evidence = getattr(placement, "ordered_row_evidence", None)
            if evidence is None:
                raise ValueError("centered channel lacks explicit ordered-row axis evidence")
            axis = getattr(evidence, "axis_x_nm", None)
            if type(axis) is not int:
                raise ValueError("centered channel has no certified shared axis")
            axis2 = 2 * axis
        result.append(ChannelRoute(declaration, low, high, axis2))
    return tuple(result)
