"""Full 17-device comparator FLOORPLAN by mirror construction.

Scales the R10/R11a mirror-construction to the complete double-tail comparator:
the 8 declared mirror pairs + the on-axis tail (v173's SYMMETRY_GROUPS), each a
DRC-legal foldable device (coclean_5tota_v163.build_foldable_device), placed as a
LEFT half + its exact x-mirror (backend._mirror_x) about the cell axis, with a
bottom vss rail and top vdd rail. NMOS in one row facing vss, PMOS in one row
facing vdd, so every supply source drops straight to its rail without crossing
another device; the input pair (M5/M6) sources meet on-axis at the tail node
(net069). This yields a >=99% geometrically mirror-symmetric placement.

SCOPE (honest): this is a symmetric PLACEMENT + supply/tail wiring, NOT yet the
fully-routed netlist. The inter-stage differential signal routing (intern/interp,
the cross-coupled crossn/crossp, outm/outp) is NOT wired here -- that needs the
connectivity-aware symmetric router (the cross-coupling crosses the axis and
requires a 2-layer via crossover). Magic extraction therefore shows the supply +
input-stage nets, not the full 12-net comparator topology. Geometric mirror
symmetry is necessary but NOT sufficient for parasitic balance: the routing has
to preserve it too (v183 keeps the symmetry score but loses parasitic balance).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .backend import LayerPolygon, Polygon, _mirror_x, _translate
from .coclean_5tota import build_foldable_device
from .magical_comparator import COMP_DEVICE_PLAN

M1 = (34, 0)
M1_LABEL = (34, 10)

# Authoritative per-device sizing comes from COMP_DEVICE_PLAN (the SAME source the
# v173 reference schematic uses): (label, kind, d_net, g_net, s_net, nf, wf, lu).
# Each device is sized to its true (nf fingers, wf width/finger, lu length) -- the
# tail (nf=8 W=1.7), input pair (nf=8 W=1.0), latch (nf=4 W=1.15), precharge
# (nf=2), reset (nf=2 L=1.0 long-channel) are all DIFFERENT, so the transistors
# are NOT uniform.
_DEV = {d[0]: d for d in COMP_DEVICE_PLAN}
# pair name -> LEFT device label (the n/m-side device; right device is its mirror,
# identical size, on the p-side net).
_PAIR_LEFT = {
    "M5_M6": "M5", "M3_M4": "M4", "M16_M17": "M16", "M0_M22": "M0",
    "M13_M14": "M14", "M18_M2": "M18", "M1_M12": "M1", "M8_M15": "M8",
}
# axis -> outward packing order within each row
_NMOS_PAIRS = ["M5_M6", "M3_M4", "M16_M17", "M0_M22"]
_PMOS_PAIRS = ["M13_M14", "M18_M2", "M1_M12", "M8_M15"]
ROW_PITCH_NM = 6500          # NMOS-row-top -> PMOS-row gap (clears drain stubs)
GAP_NM = 900                 # spacing between adjacent device comps in a row
HALF_GAP_NM = 1600
RAIL_W_NM = 900


def _build_dev(label: str):
    """Build the DRC-legal foldable device for a schematic label at its TRUE size."""
    _, kind, _d, _g, _s, nf, wf, lu = _DEV[label]
    return build_foldable_device(kind=kind, finger_count=int(nf),
                                 gate_length_um=float(lu), channel_width_um=float(wf)), kind


def _rect(layer, box) -> LayerPolygon:
    x0, y0, x1, y1 = (int(v) for v in box)
    return LayerPolygon(layer=layer[0], datatype=layer[1],
                        polygon=Polygon.rectangle(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))


@dataclass
class FullComparatorCell:
    polygons: List[LayerPolygon] = field(default_factory=list)
    labels: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    differential_net_pairs: List[Tuple[str, str]] = field(default_factory=list)
    axis_x: int = 0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)


def _place_pair(dev, right_edge: int, row_y: int):
    """Place the LEFT device so its comp right edge sits at ``right_edge`` and its
    comp bottom at ``row_y``; return (left polys, src bbox, drn bbox, comp_left)."""
    dx = right_edge - dev.comp_bbox[2]
    dy = row_y - dev.comp_bbox[1]
    left = _translate(list(dev.polygons), dx, dy)
    sb = dev.source_rail_bbox
    src = (sb[0] + dx, sb[1] + dy, sb[2] + dx, sb[3] + dy)
    db = dev.drain_rail_bbox
    drn = (db[0] + dx, db[1] + dy, db[2] + dx, db[3] + dy)
    comp_left = dev.comp_bbox[0] + dx
    return left, src, drn, comp_left


def build_full_symmetric_comparator() -> FullComparatorCell:
    """Build the complete 17-device comparator by mirror construction."""
    cell = FullComparatorCell(axis_x=0)
    polys: List[LayerPolygon] = []

    vss_y = 0
    tail_band = vss_y + 1500
    nmos_base = tail_band + ROW_PITCH_NM        # NMOS row above the tail
    pmos_base = nmos_base + ROW_PITCH_NM + 2500
    pmos_top_rail_y = pmos_base + ROW_PITCH_NM + 1500

    drain_trunks: List[Tuple[str, Tuple[int, int, int, int]]] = []
    tail_meet_y = {"y0": None, "y1": None}     # net069: input-pair source meet band

    def add_pair(name, dev, right_edge, row_y, rail_y, *, source_mode="rail"):
        left, src, drn, comp_left = _place_pair(dev, right_edge, row_y)
        trunk = (drn[0], drn[1], drn[2], drn[3] + 1200)   # short drain stub
        left.append(_rect(M1, trunk))
        if source_mode == "tail":
            # route the source rail to the axis (x=0); the mirror brings the right
            # source to meet it -> the shared tail node net069 (common-mode).
            left.append(_rect(M1, (src[2], src[1], 0, src[3])))
            tail_meet_y["y0"], tail_meet_y["y1"] = src[1], src[3]
        polys.extend(left)
        polys.extend(_mirror_x(list(left), 0))
        if source_mode == "rail":
            # narrow strip at the source's INNER edge (clear of the drain stub) so a
            # PMOS source->vdd drop going up past the device does not overlap its own
            # drain rail in metal1 and short the two nets.
            sx0, sx1 = src[0], src[0] + 700
            srcdrop = (sx0, min(src[1], rail_y), sx1, max(src[3], rail_y))
            polys.append(_rect(M1, srcdrop))
            polys.append(_rect(M1, _mirror_box(srcdrop)))
        lcx = (trunk[0] + trunk[2]) // 2
        drain_trunks.append((name, (lcx, trunk[3] - 300)))
        return comp_left

    # NMOS row: pack pairs from the axis outward at their TRUE widths.
    right_edge = -HALF_GAP_NM
    for name in _NMOS_PAIRS:
        dev, _ = _build_dev(_PAIR_LEFT[name])
        mode = "tail" if name == "M5_M6" else "rail"
        comp_left = add_pair(name, dev, right_edge, nmos_base, vss_y, source_mode=mode)
        right_edge = comp_left - GAP_NM
    # on-axis tail (M7) at its true size (nf=8, W=1.7), centred at x=0, above vss
    tail, _ = _build_dev("M7")
    tcx = (tail.comp_bbox[0] + tail.comp_bbox[2]) // 2
    tdy = tail_band - tail.comp_bbox[1]
    polys.extend(_translate(list(tail.polygons), -tcx, tdy))
    ts = tail.source_rail_bbox
    polys.append(_rect(M1, (ts[0] - tcx, vss_y, ts[2] - tcx, ts[3] + tdy)))   # tail src -> vss
    td = tail.drain_rail_bbox
    td = (td[0] - tcx, td[1] + tdy, td[2] - tcx, td[3] + tdy)
    if tail_meet_y["y0"] is not None:                                         # tail drn -> net069
        polys.append(_rect(M1, (td[0], td[1], td[2], tail_meet_y["y1"])))
    # PMOS row: pack pairs from the axis outward at their TRUE widths.
    right_edge = -HALF_GAP_NM
    for name in _PMOS_PAIRS:
        dev, _ = _build_dev(_PAIR_LEFT[name])
        comp_left = add_pair(name, dev, right_edge, pmos_base, pmos_top_rail_y)
        right_edge = comp_left - GAP_NM

    # extents -> rails span full width
    xs = [v.x for lp in polys for v in lp.polygon.vertices]
    ys = [v.y for lp in polys for v in lp.polygon.vertices]
    x_lo, x_hi = min(xs) - 1000, max(xs) + 1000
    span = max(abs(x_lo), abs(x_hi))
    # vss bottom rail
    polys.append(_rect(M1, (-span, vss_y - RAIL_W_NM, span, vss_y)))
    # vdd top rail
    polys.append(_rect(M1, (-span, pmos_top_rail_y, span, pmos_top_rail_y + RAIL_W_NM)))

    cell.polygons = polys
    # differential-net labels: the input pair (M5/M6) and latch (M3/M4) drains are
    # the differential nets to evaluate (outn/outp side).
    for name, (cx, cy) in drain_trunks:
        cell.labels[name + "_L"] = (cx, cy)
        cell.labels[name + "_R"] = (-cx, cy)
    cell.labels["vss"] = (0, vss_y - RAIL_W_NM // 2)
    cell.labels["vdd"] = (0, pmos_top_rail_y + RAIL_W_NM // 2)
    cell.differential_net_pairs = [(n + "_L", n + "_R") for n, _ in drain_trunks]
    xs2 = [v.x for lp in polys for v in lp.polygon.vertices]
    ys2 = [v.y for lp in polys for v in lp.polygon.vertices]
    cell.bbox = (min(xs2), min(ys2), max(xs2), max(ys2))
    return cell


def _mirror_box(box):
    x0, y0, x1, y1 = box
    return (-x1, y0, -x0, y1)


def emit_full_comparator_gds(cell: FullComparatorCell, gds_path: str, *,
                             cell_name: str = "full_sym_comparator", dbu: float = 0.001) -> str:
    import klayout.db as db
    ly = db.Layout(); ly.dbu = dbu
    top = ly.create_cell(cell_name)
    for lp in cell.polygons:
        li = ly.layer(lp.layer, lp.datatype)
        top.shapes(li).insert(db.Polygon([db.Point(int(v.x), int(v.y)) for v in lp.polygon.vertices]))
    lbl = ly.layer(M1_LABEL[0], M1_LABEL[1])
    for net, (x, y) in cell.labels.items():
        top.shapes(lbl).insert(db.Text(net, db.Trans(db.Vector(int(x), int(y)))))
    ly.write(str(gds_path))
    return str(gds_path)
