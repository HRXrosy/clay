"""Power grid template generator.

Generates alternating VDD/VSS horizontal rails on a chosen metal layer for
digital/analog placement templates.

Rail layout:
- Rails are horizontal rectangles spanning x in [0, grid_width_nm].
- Rail positions: y = 0 (VDD), y = rail_pitch_nm (VSS),
  y = 2 * rail_pitch_nm (VDD), ... alternating.
- Each rail is centered on its y position with height = rail_width_nm
  (so the rectangle's y range is [y - rail_width_nm/2, y + rail_width_nm/2]).
- Rail emission stops when the next rail position would reach or exceed
  grid_height_nm (i.e., a rail is placed only if y < grid_height_nm).

API:
- PowerGridConfig (dataclass)
- generate_power_grid(config, pdk_layers) -> List[LayerPolygon]
- count_rails(polys) -> int
- estimate_grid_coverage_pct(polys, grid_width_nm, grid_height_nm) -> float
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .backend import (
    LayerMap,
    LayerPolygon,
    LogicalLayer,
    Polygon,
)
from .routing_advanced import metal2_layer_for_pdk_layers
from .routing_m3 import metal3_layer_for_pdk_layers


_SUPPORTED_METAL_LAYERS = ("metal1", "metal2", "metal3")


@dataclass
class PowerGridConfig:
    """Configuration for a power grid template.

    Fields:
        grid_width_nm: total horizontal extent of the grid (nm DBU).
        grid_height_nm: total vertical extent of the grid (nm DBU).
        rail_pitch_nm: vertical spacing between consecutive rails
            (alternates VDD/VSS), e.g. 4000.
        rail_width_nm: rail thickness in the y direction
            (rectangle height), e.g. 800.
        metal_layer_name: logical metal layer name
            ("metal1" / "metal2" / "metal3"). Defaults to "metal2".
    """

    grid_width_nm: int
    grid_height_nm: int
    rail_pitch_nm: int
    rail_width_nm: int
    metal_layer_name: str = "metal2"


def _validate_config(config: PowerGridConfig) -> None:
    """Raise ValueError on out-of-range fields or unknown metal layer."""
    if config.grid_width_nm < 0:
        raise ValueError(
            f"grid_width_nm must be >= 0, got {config.grid_width_nm}"
        )
    if config.grid_height_nm < 0:
        raise ValueError(
            f"grid_height_nm must be >= 0, got {config.grid_height_nm}"
        )
    if config.rail_pitch_nm <= 0:
        raise ValueError(
            f"rail_pitch_nm must be > 0, got {config.rail_pitch_nm}"
        )
    if config.rail_width_nm <= 0:
        raise ValueError(
            f"rail_width_nm must be > 0, got {config.rail_width_nm}"
        )
    if config.metal_layer_name not in _SUPPORTED_METAL_LAYERS:
        raise ValueError(
            f"metal_layer_name must be one of {_SUPPORTED_METAL_LAYERS}, "
            f"got {config.metal_layer_name!r}"
        )


def _metal_layer_lookup(
    metal_layer_name: str, pdk_layers: LayerMap
) -> Tuple[int, int]:
    """Return (gds_layer, gds_datatype) for the chosen logical metal layer."""
    if metal_layer_name == "metal1":
        m1 = pdk_layers.gds_layer(LogicalLayer.METAL1)
        if m1 is None:
            raise ValueError("PDK does not define METAL1 layer")
        return (m1.layer, m1.datatype)
    if metal_layer_name == "metal2":
        return metal2_layer_for_pdk_layers(pdk_layers)
    if metal_layer_name == "metal3":
        return metal3_layer_for_pdk_layers(pdk_layers)
    raise ValueError(
        f"unsupported metal_layer_name: {metal_layer_name!r}"
    )


def _build_rail(
    y_center: int,
    rail_width_nm: int,
    grid_width_nm: int,
    layer: int,
    datatype: int,
) -> LayerPolygon:
    """Build a horizontal rail rectangle centered at y_center."""
    half_w = rail_width_nm // 2
    y_min = y_center - half_w
    y_max = y_center + (rail_width_nm - half_w)  # account for odd widths
    return LayerPolygon(
        layer=layer,
        datatype=datatype,
        polygon=Polygon.rectangle(0, y_min, grid_width_nm, y_max),
    )


def generate_power_grid(
    config: PowerGridConfig, pdk_layers: LayerMap
) -> List[LayerPolygon]:
    """Generate alternating VDD/VSS horizontal rails on the chosen metal.

    Rails are at y = 0 (VDD), y = rail_pitch_nm (VSS),
    y = 2 * rail_pitch_nm (VDD), and so on. Each rail rectangle spans
    x in [0, grid_width_nm] and is centered on its y position with
    height = rail_width_nm. Rail emission stops as soon as the next
    rail position would reach or exceed grid_height_nm.

    Returns:
        List of LayerPolygon rectangles on the chosen metal layer.
        Returns an empty list if either grid dimension is zero.

    Raises:
        ValueError on negative dimensions, non-positive pitch/width, or
        unknown metal_layer_name.
    """
    _validate_config(config)

    if config.grid_width_nm == 0 or config.grid_height_nm == 0:
        return []

    metal_layer_num, metal_dt = _metal_layer_lookup(
        config.metal_layer_name, pdk_layers
    )

    polys: List[LayerPolygon] = []
    y = 0
    while y < config.grid_height_nm:
        polys.append(_build_rail(
            y_center=y,
            rail_width_nm=config.rail_width_nm,
            grid_width_nm=config.grid_width_nm,
            layer=metal_layer_num,
            datatype=metal_dt,
        ))
        y += config.rail_pitch_nm

    return polys


def count_rails(polys: List[LayerPolygon]) -> int:
    """Return the number of rail polygons in a power grid output."""
    return len(polys)


def estimate_grid_coverage_pct(
    polys: List[LayerPolygon],
    grid_width_nm: int,
    grid_height_nm: int,
) -> float:
    """Total rail area / grid area * 100 (rough estimate, ignores edges).

    Returns 0.0 when the grid area is zero (or non-positive) to avoid
    division-by-zero. Rail area is computed from the bounding box of
    each polygon's vertices.
    """
    grid_area = grid_width_nm * grid_height_nm
    if grid_area <= 0:
        return 0.0
    total_rail_area = 0
    for p in polys:
        if not p.polygon.vertices:
            continue
        xs = [v.x for v in p.polygon.vertices]
        ys = [v.y for v in p.polygon.vertices]
        x_extent = max(xs) - min(xs)
        y_extent = max(ys) - min(ys)
        total_rail_area += x_extent * y_extent
    return (total_rail_area / grid_area) * 100.0
