"""Concentric vertical passive stack composer and SKY130 CAPM/M3 DRC contract (Spec P6).

Composes disjoint-layer FEOL/lower-BEOL passives (such as ``cap_var_lvt`` on
FEOL..M2) concentrically beneath upper-BEOL ``cap_mim_m3_1`` capacitors on
M3..M4 without hardcoded coordinates. Enforces:
  * Concentric center alignment within 1 manufacturing grid step (``<= 5 nm``);
  * Layer disjointness across all drawn physical layers;
  * ``via2 == 0`` inside the stacked ``CAPM`` footprint + halo;
  * One solid un-notched ``M3`` bottom-plate rectangle enclosing ``CAPM`` by at
    least ``1400 nm`` on all 4 sides (``>= 140 nm`` ``capm.3`` and ``> 1340 nm``
    ``capm.11`` halo, ``0`` ``capm.2b_a`` notch/slit violations) owned by a
    single non-``None`` net;
  * Unrelated ``M3`` shapes outside the bottom plate clear ``CAPM`` by at least
    ``1340 nm`` (``capm.11``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import klayout.db as kdb

from clay.pdk.sky130_native_geometry import (
    SKY130_MOS_RULESET_NM,
    mim_vertical_stack_m3_surround_nm,
)

_LayerKey = Tuple[int, int]
_BoxNm = Tuple[int, int, int, int]
_EXCLUDED_BOUNDARY_LAYERS = frozenset({(235, 4), (81, 4)})
_LAYER_VIA2: _LayerKey = (69, 44)
_LAYER_M3: _LayerKey = (70, 20)
_LAYER_CAPM: _LayerKey = (89, 44)


@dataclass(frozen=True)
class VerticalStackAudit:
    lower_bbox_nm: _BoxNm
    upper_bbox_nm: _BoxNm
    composite_bbox_nm: _BoxNm
    lower_center_nm: Tuple[int, int]
    upper_center_nm: Tuple[int, int]
    center_delta_nm: Tuple[int, int]
    capm_bbox_nm: _BoxNm
    bottom_plate_m3_bbox_nm: _BoxNm
    m3_surround_margins_nm: Tuple[int, int, int, int]
    bottom_plate_solid_rectangle: bool
    via2_count_in_halo: int
    unrelated_m3_violation_count: int
    bottom_plate_owner: str


def _region_from_shapes(shapes: Sequence, *, offset_nm: Tuple[int, int] = (0, 0)) -> kdb.Region:
    dx, dy = int(offset_nm[0]), int(offset_nm[1])
    region = kdb.Region()
    for item in shapes:
        if type(item) is tuple and len(item) == 4 and all(type(v) is int for v in item):
            x0, y0, x1, y1 = item
            if x1 > x0 and y1 > y0:
                region.insert(kdb.Box(x0 + dx, y0 + dy, x1 + dx, y1 + dy))
        else:
            pts = [kdb.Point(int(round(p[0])) + dx, int(round(p[1])) + dy) for p in item]
            if len(pts) >= 3:
                poly = kdb.Polygon(pts)
                if poly.area() > 0:
                    region.insert(poly)
    region.merge()
    return region


def _artwork_bbox_nm(regions: Mapping[_LayerKey, kdb.Region]) -> _BoxNm:
    total = kdb.Region()
    for layer, region in regions.items():
        if layer in _EXCLUDED_BOUNDARY_LAYERS:
            continue
        total += region
    if total.is_empty():
        raise ValueError("vertical stack member has empty physical artwork")
    bb = total.bbox()
    return (int(bb.left), int(bb.bottom), int(bb.right), int(bb.top))


def verify_vertical_stack_drc_contract(
    lower_polys_nm: Mapping[_LayerKey, Sequence],
    upper_polys_nm: Mapping[_LayerKey, Sequence],
    *,
    lower_offset_nm: Tuple[int, int] = (0, 0),
    upper_offset_nm: Tuple[int, int] = (0, 0),
    unrelated_m3_boxes_nm: Iterable[_BoxNm] = (),
    bottom_plate_owner: str = "VSS",
    min_m3_surround_nm: int | None = None,
    capm_unrelated_m3_clearance_nm: int | None = None,
    max_center_delta_nm: int = 5,
) -> VerticalStackAudit:
    """Verify concentric alignment, layer disjointness, solid M3 plate, and CAPM rules."""
    if type(bottom_plate_owner) is not str or not bottom_plate_owner.strip():
        raise ValueError("vertical stack M3 bottom plate requires a single non-empty net owner")
    req_surround = (
        mim_vertical_stack_m3_surround_nm()
        if min_m3_surround_nm is None
        else int(min_m3_surround_nm)
    )
    req_capm_m3_clear = (
        int(SKY130_MOS_RULESET_NM["capm_to_unrelated_m3_nm"])
        if capm_unrelated_m3_clearance_nm is None
        else int(capm_unrelated_m3_clearance_nm)
    )
    req_capm_enclose = int(SKY130_MOS_RULESET_NM["m3_enclose_capm_nm"])
    req_via2_clear = int(SKY130_MOS_RULESET_NM["capm_to_via2_nm"])

    lower_regions = {
        layer: reg
        for layer, shapes in lower_polys_nm.items()
        if not (reg := _region_from_shapes(shapes, offset_nm=lower_offset_nm)).is_empty()
    }
    upper_regions = {
        layer: reg
        for layer, shapes in upper_polys_nm.items()
        if not (reg := _region_from_shapes(shapes, offset_nm=upper_offset_nm)).is_empty()
    }
    lower_drawn = set(lower_regions) - _EXCLUDED_BOUNDARY_LAYERS
    upper_drawn = set(upper_regions) - _EXCLUDED_BOUNDARY_LAYERS
    shared_layers = lower_drawn & upper_drawn
    if shared_layers:
        raise ValueError(
            f"vertical stack lower and upper passives collide on drawn layers: {sorted(shared_layers)}"
        )

    lower_bb = _artwork_bbox_nm(lower_regions)
    upper_bb = _artwork_bbox_nm(upper_regions)
    lower_cx = (lower_bb[0] + lower_bb[2]) // 2
    lower_cy = (lower_bb[1] + lower_bb[3]) // 2
    upper_cx = (upper_bb[0] + upper_bb[2]) // 2
    upper_cy = (upper_bb[1] + upper_bb[3]) // 2
    dcx = abs(lower_cx - upper_cx)
    dcy = abs(lower_cy - upper_cy)
    if dcx > max_center_delta_nm or dcy > max_center_delta_nm:
        raise ValueError(
            f"vertical stack center misalignment ({dcx}, {dcy}) nm exceeds {max_center_delta_nm} nm"
        )

    capm = upper_regions.get(_LAYER_CAPM, kdb.Region())
    if capm.is_empty():
        raise ValueError("vertical stack upper passive lacks CAPM (89, 44)")
    capm_polys = list(capm.merged().each())
    if len(capm_polys) != 1 or not capm_polys[0].is_box():
        raise ValueError("vertical stack CAPM must be one solid rectangle")
    capm_bb = capm.bbox()
    capm_box_nm = (int(capm_bb.left), int(capm_bb.bottom), int(capm_bb.right), int(capm_bb.top))
    if (
        capm_box_nm[2] - capm_box_nm[0] < int(SKY130_MOS_RULESET_NM["capm_min_width_nm"])
        or capm_box_nm[3] - capm_box_nm[1] < int(SKY130_MOS_RULESET_NM["capm_min_width_nm"])
    ):
        raise ValueError("vertical stack CAPM violates capm.1 minimum width")

    m3 = upper_regions.get(_LAYER_M3, kdb.Region())
    overlapping_m3 = [poly for poly in m3.merged().each() if not (kdb.Region(poly) & capm).is_empty()]
    if len(overlapping_m3) != 1 or not overlapping_m3[0].is_box() or overlapping_m3[0].holes() > 0:
        raise ValueError(
            "vertical stack bottom-plate M3 must be one solid un-notched rectangle (capm.2b_a)"
        )
    plate_bb = overlapping_m3[0].bbox()
    plate_box_nm = (int(plate_bb.left), int(plate_bb.bottom), int(plate_bb.right), int(plate_bb.top))
    margins = (
        capm_box_nm[0] - plate_box_nm[0],
        capm_box_nm[1] - plate_box_nm[1],
        plate_box_nm[2] - capm_box_nm[2],
        plate_box_nm[3] - capm_box_nm[3],
    )
    if min(margins) < max(req_capm_enclose, req_surround):
        raise ValueError(
            f"vertical stack M3 surround margins {margins} nm < required {max(req_capm_enclose, req_surround)} nm"
        )

    via2 = lower_regions.get(_LAYER_VIA2, kdb.Region()) + upper_regions.get(_LAYER_VIA2, kdb.Region())
    capm_via2_halo = kdb.Region(
        kdb.Box(
            capm_bb.left - req_via2_clear,
            capm_bb.bottom - req_via2_clear,
            capm_bb.right + req_via2_clear,
            capm_bb.top + req_via2_clear,
        )
    )
    via2_hits = via2 & capm_via2_halo
    via2_count = sum(1 for _ in via2_hits.each())
    if via2_count != 0:
        raise ValueError(f"vertical stack contains {via2_count} illegal via2 cuts under CAPM")

    # Any M3 polygon in upper_regions not part of the bottom plate + any caller-supplied unrelated M3
    unrelated_m3 = m3 - kdb.Region(overlapping_m3[0])
    for x0, y0, x1, y1 in unrelated_m3_boxes_nm:
        unrelated_m3.insert(kdb.Box(int(x0), int(y0), int(x1), int(y1)))
    unrelated_m3.merge()
    unrelated_violations = 0
    if not unrelated_m3.is_empty():
        # Overlap or Euclidean clearance < 1340 nm to CAPM violates capm.11
        if not (unrelated_m3 & capm).is_empty():
            unrelated_violations += 1
        sep_pairs = capm.separation_check(unrelated_m3, req_capm_m3_clear, True)
        unrelated_violations += sep_pairs.size()
        if unrelated_violations > 0:
            raise ValueError(
                f"unrelated M3 violates capm.11 ({req_capm_m3_clear} nm clearance to CAPM)"
            )

    comp_box_nm = (
        min(lower_bb[0], upper_bb[0]),
        min(lower_bb[1], upper_bb[1]),
        max(lower_bb[2], upper_bb[2]),
        max(lower_bb[3], upper_bb[3]),
    )
    return VerticalStackAudit(
        lower_bbox_nm=lower_bb,
        upper_bbox_nm=upper_bb,
        composite_bbox_nm=comp_box_nm,
        lower_center_nm=(lower_cx, lower_cy),
        upper_center_nm=(upper_cx, upper_cy),
        center_delta_nm=(dcx, dcy),
        capm_bbox_nm=capm_box_nm,
        bottom_plate_m3_bbox_nm=plate_box_nm,
        m3_surround_margins_nm=margins,
        bottom_plate_solid_rectangle=True,
        via2_count_in_halo=via2_count,
        unrelated_m3_violation_count=unrelated_violations,
        bottom_plate_owner=bottom_plate_owner.strip(),
    )


def compose_concentric_vertical_stack(
    lower_polys_nm: Mapping[_LayerKey, Sequence],
    upper_polys_nm: Mapping[_LayerKey, Sequence],
    *,
    grid_nm: int = 5,
    bottom_plate_owner: str = "VSS",
    unrelated_m3_boxes_nm: Iterable[_BoxNm] = (),
    min_m3_surround_nm: int | None = None,
) -> Dict[str, object]:
    """Concentrically align lower + upper passive artwork on the 5nm grid and audit DRC."""
    if type(grid_nm) is not int or grid_nm <= 0:
        raise ValueError("manufacturing grid must be a positive integer")
    lower_raw = {
        layer: reg
        for layer, shapes in lower_polys_nm.items()
        if not (reg := _region_from_shapes(shapes)).is_empty()
    }
    upper_raw = {
        layer: reg
        for layer, shapes in upper_polys_nm.items()
        if not (reg := _region_from_shapes(shapes)).is_empty()
    }
    lb = _artwork_bbox_nm(lower_raw)
    ub = _artwork_bbox_nm(upper_raw)
    lcx, lcy = (lb[0] + lb[2]) // 2, (lb[1] + lb[3]) // 2
    ucx, ucy = (ub[0] + ub[2]) // 2, (ub[1] + ub[3]) // 2

    # Align centers modulo grid_nm and shift composite origin to (0, 0).
    raw_dx = round((ucx - lcx) / grid_nm) * grid_nm
    raw_dy = round((ucy - lcy) / grid_nm) * grid_nm
    min_x = min(lb[0] + raw_dx, ub[0])
    min_y = min(lb[1] + raw_dy, ub[1])
    snap_min_x = (min_x // grid_nm) * grid_nm
    snap_min_y = (min_y // grid_nm) * grid_nm
    lower_offset = (raw_dx - snap_min_x, raw_dy - snap_min_y)
    upper_offset = (-snap_min_x, -snap_min_y)

    audit = verify_vertical_stack_drc_contract(
        lower_polys_nm,
        upper_polys_nm,
        lower_offset_nm=lower_offset,
        upper_offset_nm=upper_offset,
        unrelated_m3_boxes_nm=unrelated_m3_boxes_nm,
        bottom_plate_owner=bottom_plate_owner,
        min_m3_surround_nm=min_m3_surround_nm,
        max_center_delta_nm=grid_nm,
    )
    merged_layers: Dict[_LayerKey, Tuple[Tuple[Tuple[int, int], ...], ...]] = {}
    all_keys = sorted((set(lower_raw) | set(upper_raw)) - _EXCLUDED_BOUNDARY_LAYERS)
    for key in all_keys:
        reg = kdb.Region()
        if key in lower_polys_nm:
            reg += _region_from_shapes(lower_polys_nm[key], offset_nm=lower_offset)
        if key in upper_polys_nm:
            reg += _region_from_shapes(upper_polys_nm[key], offset_nm=upper_offset)
        reg.merge()
        polys = []
        for poly in reg.each():
            hull = tuple((int(pt.x), int(pt.y)) for pt in poly.each_point_hull())
            polys.append(hull)
        if polys:
            merged_layers[key] = tuple(polys)
    return {
        "lower_offset_nm": lower_offset,
        "upper_offset_nm": upper_offset,
        "polygons_by_layer_nm": merged_layers,
        "bottom_plate_m3_owner": audit.bottom_plate_owner,
        "audit": audit,
    }

