"""Constructive proof: a parasitically-symmetric differential-pair core.

The user's standard (2026-05-27): a comparator's symmetry must hold at the
PARASITIC level, not just the placement-centroid score (v183 passes the score
but is 18.4% parasitically imbalanced). This module BUILDS a small but real
differential-pair core that IS parasitically symmetric by construction, and
verifies it with the unified gate on real Magic -- the constructive counterpoint
to v183.

Construction (symmetric BY MIRROR, not by hope): build ONE half (a DRC-legal
foldable NMOS via ``coclean_5tota_v163.build_foldable_device`` + its drain output
trunk + a source drop to the vss rail), then produce the other half as the EXACT
x-mirror (``backend._mirror_x``) about the cell axis. Two mirror halves => the two
differential nets (outn / outp) are geometric mirrors => their parasitics balance.
The drain trunks are labelled so Magic names the nets and the gate can run.

This reuses the proven DRC-legal device geometry and the klayout emitter (no Rust
emit / co-clean plumbing), so it is self-contained and additive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .backend import LayerPolygon, Polygon, _mirror_x, _translate
from .coclean_5tota import build_foldable_device

GF180_METAL1_KEY = (34, 0)
GF180_METAL1_LABEL_KEY = (34, 10)


def _rect(layer, box) -> LayerPolygon:
    x0, y0, x1, y1 = box
    return LayerPolygon(layer=layer[0], datatype=layer[1],
                        polygon=Polygon.rectangle(int(x0), int(y0), int(x1), int(y1)))


@dataclass(frozen=True)
class DiffPairCell:
    polygons: Tuple[LayerPolygon, ...]
    labels: Dict[str, Tuple[int, int]]     # net -> (x, y) label position
    differential_nets: Tuple[str, str]
    axis_x: int


def build_symmetric_diff_pair_cell(
    *,
    finger_count: int = 2,
    channel_width_um: float = 4.0,
    gate_length_um: float = 0.5,
    half_gap_nm: int = 4000,
    trunk_up_nm: int = 60000,
) -> DiffPairCell:
    """Build a mirror-symmetric NMOS differential-pair core (outn / outp).

    One half = a foldable NMOS + a metal1 drain output trunk routed up; the other
    half is its exact x-mirror about the axis. The two drain trunks (outn, outp)
    are therefore exact mirror images.
    """
    dev = build_foldable_device(kind="nmos", finger_count=finger_count,
                                gate_length_um=gate_length_um,
                                channel_width_um=channel_width_um)
    # left half: shift the device fully into x < 0 so the mirror axis is x=0.
    dx = -(dev.comp_bbox[2] + half_gap_nm)
    left: List[LayerPolygon] = _translate(list(dev.polygons), dx, 0)
    drain = (dev.drain_rail_bbox[0] + dx, dev.drain_rail_bbox[1],
             dev.drain_rail_bbox[2] + dx, dev.drain_rail_bbox[3])
    # drain output trunk: metal1 from the drain rail straight up (the routed net).
    trunk = (drain[0], drain[1], drain[2], drain[1] + trunk_up_nm)
    left.append(_rect(GF180_METAL1_KEY, trunk))
    left_drain_cx = (trunk[0] + trunk[2]) // 2
    trunk_label_y = trunk[1] + trunk_up_nm - 500

    axis_x = 0
    right = _mirror_x(list(left), axis_x)

    polys = tuple(left) + tuple(right)
    labels = {
        "outn": (left_drain_cx, trunk_label_y),
        "outp": (2 * axis_x - left_drain_cx, trunk_label_y),
    }
    return DiffPairCell(polygons=polys, labels=labels,
                        differential_nets=("outn", "outp"), axis_x=axis_x)


def build_symmetric_diffpair_with_tail(
    *,
    finger_count: int = 2,
    channel_width_um: float = 4.0,
    gate_length_um: float = 0.5,
    half_gap_nm: int = 4000,
    trunk_up_nm: int = 60000,
) -> DiffPairCell:
    """Comparator input front-end (M5/M6 diff pair + M7 on-axis tail), mirror-built.

    The two diff-half sources extend to the axis and meet there (the shared tail
    node); a tail NMOS sits centred on the axis (common-mode, so it does not break
    the outn/outp differential balance) with its drain on the tail node and its
    source on a bottom vss rail. The differential nets outn/outp remain exact
    mirrors -> parasitically balanced.
    """
    dev = build_foldable_device(kind="nmos", finger_count=finger_count,
                                gate_length_um=gate_length_um,
                                channel_width_um=channel_width_um)
    dx = -(dev.comp_bbox[2] + half_gap_nm)
    left: List[LayerPolygon] = _translate(list(dev.polygons), dx, 0)
    drain = (dev.drain_rail_bbox[0] + dx, dev.drain_rail_bbox[1],
             dev.drain_rail_bbox[2] + dx, dev.drain_rail_bbox[3])
    src = (dev.source_rail_bbox[0] + dx, dev.source_rail_bbox[1],
           dev.source_rail_bbox[2] + dx, dev.source_rail_bbox[3])
    # drain output trunk (outn) up
    trunk = (drain[0], drain[1], drain[2], drain[1] + trunk_up_nm)
    left.append(_rect(GF180_METAL1_KEY, trunk))
    # source strip to the axis (x=0) -> after mirror the two sources MEET = tail node
    m1w = max(src[3] - src[1], 230)
    tail_y0 = src[1]
    src_strip = (src[2], tail_y0, 0, tail_y0 + m1w)  # from source rail edge to axis
    left.append(_rect(GF180_METAL1_KEY, src_strip))

    axis_x = 0
    right = _mirror_x(list(left), axis_x)
    polys = list(left) + list(right)

    # on-axis tail NMOS (common-mode), centred at x=0, BELOW the diff devices.
    tail_dev = build_foldable_device(kind="nmos", finger_count=finger_count,
                                     gate_length_um=gate_length_um,
                                     channel_width_um=channel_width_um)
    tcx = (tail_dev.comp_bbox[0] + tail_dev.comp_bbox[2]) // 2
    tail_gap = 3000
    tdy = tail_y0 - (tail_dev.comp_bbox[3] - tail_dev.comp_bbox[1]) - tail_gap
    tail_polys = _translate(list(tail_dev.polygons), -tcx, tdy)
    polys += tail_polys
    td = tail_dev.drain_rail_bbox
    td = (td[0] - tcx, td[1] + tdy, td[2] - tcx, td[3] + tdy)
    # connect tail drain up to the tail node (y=tail_y0)
    polys.append(_rect(GF180_METAL1_KEY, (td[0], td[3], td[2], tail_y0 + m1w)))
    # bottom vss rail spanning the cell + tail source down to it
    ts = tail_dev.source_rail_bbox
    ts = (ts[0] - tcx, ts[1] + tdy, ts[2] - tcx, ts[3] + tdy)
    span = max(abs(trunk[0]), abs(trunk[2])) + 2000
    vss_y = tdy - 2000
    polys.append(_rect(GF180_METAL1_KEY, (-span, vss_y - 600, span, vss_y)))
    polys.append(_rect(GF180_METAL1_KEY, (ts[0], vss_y, ts[2], ts[1])))

    left_drain_cx = (trunk[0] + trunk[2]) // 2
    trunk_label_y = trunk[1] + trunk_up_nm - 500
    labels = {"outn": (left_drain_cx, trunk_label_y),
              "outp": (2 * axis_x - left_drain_cx, trunk_label_y),
              "vss": (0, vss_y - 300)}
    return DiffPairCell(polygons=tuple(polys), labels=labels,
                        differential_nets=("outn", "outp"), axis_x=axis_x)


def emit_cell_with_labels(cell: DiffPairCell, gds_path: str, *, cell_name: str = "sym_diff_pair",
                          dbu: float = 0.001) -> str:
    """Emit a DiffPairCell's polygons (any layers) + net text labels via klayout."""
    import klayout.db as db

    ly = db.Layout()
    ly.dbu = dbu
    top = ly.create_cell(cell_name)
    for lp in cell.polygons:
        li = ly.layer(lp.layer, lp.datatype)
        pts = [db.Point(int(v.x), int(v.y)) for v in lp.polygon.vertices]
        top.shapes(li).insert(db.Polygon(pts))
    lbl = ly.layer(GF180_METAL1_LABEL_KEY[0], GF180_METAL1_LABEL_KEY[1])
    for net, (x, y) in cell.labels.items():
        top.shapes(lbl).insert(db.Text(net, db.Trans(db.Vector(int(x), int(y)))))
    ly.write(str(gds_path))
    return str(gds_path)
