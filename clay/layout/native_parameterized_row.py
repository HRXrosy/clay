"""Rule-driven parameterized SKY130 MOS shared-diffusion row synthesis (Spec P1-P3, P5).

Synthesizes arbitrary ``N``-finger (``N >= 1``) Eulerian shared-diffusion rows
directly from SKY130 design rules (``sky130_native_geometry``) without hardcoded
comparator dimensions, device names, or per-instance whitelists:
  * Variable column pitches derived from local contact demand:
    ``li1_only_nm`` (``440 nm``), ``sd_mcon_nm`` (``460 nm``),
    ``poly_contacted_nm`` (``480 nm``);
  * Contiguous same-net ``poly`` gate merging outside ``active`` with balanced
    splitting at ``max_merged_gates = 5`` (e.g. ``10 -> 5 + 5``);
  * Corridor-safe ``LI1`` comb S/D strapping via 2-track outer interval coloring
    so supply/common-source and internal series columns carry ``0`` ``mcon`` and
    ``0`` ``M1`` bridges across the active diffusion corridor;
  * Width-aligned ``[active.left, active.right]`` horizontal ``tap + licon + li1``
    substrate/well tie bar with ``0 nm`` lateral overhang.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
import re
import shutil
from typing import Dict, List, Mapping, Sequence, Tuple

import klayout.db as kdb

from clay.layout.native_sky130 import (
    NativeGeometryBundle,
    _sha,
    native_source_identity,
    observe_native_primitive,
)
from clay.pdk.sky130_native_geometry import (
    SKY130_MOS_RULESET_NM,
    derive_column_pitches_nm,
    native_geometry_spec,
    sd_licon_layout_nm,
    snap_up_grid_nm,
    tap_rail_layout_nm,
)

_BoxNm = Tuple[int, int, int, int]
_LayerKey = Tuple[int, int]

LAYER_NWELL: _LayerKey = (64, 20)
LAYER_DIFF: _LayerKey = (65, 20)
LAYER_TAP: _LayerKey = (65, 44)
LAYER_POLY: _LayerKey = (66, 20)
LAYER_LICON: _LayerKey = (66, 44)
LAYER_LI1: _LayerKey = (67, 20)
LAYER_MCON: _LayerKey = (67, 44)
LAYER_MET1: _LayerKey = (68, 20)
LAYER_NSDM: _LayerKey = (93, 44)
LAYER_PSDM: _LayerKey = (94, 20)
LAYER_NPC: _LayerKey = (95, 20)
LAYER_LVTN: _LayerKey = (125, 44)

MAX_MERGED_GATES = 5


@dataclass(frozen=True)
class GateSegmentPlan:
    start_finger: int
    end_finger: int
    gate_net: str
    poly_head_box_nm: _BoxNm
    licon_boxes_nm: Tuple[_BoxNm, ...]
    li1_box_nm: _BoxNm
    mcon_box_nm: _BoxNm
    m1_box_nm: _BoxNm
    npc_box_nm: _BoxNm


@dataclass(frozen=True)
class SdNetTrackPlan:
    net: str
    columns: Tuple[int, ...]
    track: str  # "tap_comb" | "gate_comb" | "in_corridor" | "li1_internal"
    spine_li1_box_nm: _BoxNm | None
    mcon_boxes_nm: Tuple[_BoxNm, ...]
    m1_boxes_nm: Tuple[_BoxNm, ...]


@dataclass(frozen=True)
class ParameterizedRowPlan:
    model: str
    family: str  # "pmos" | "nmos"
    is_lvt: bool
    width_nm: int
    length_nm: int
    finger_count: int
    tap_side: str  # "top" | "bottom"
    gate_side: str  # "bottom" | "top"
    column_nets: Tuple[str, ...]
    gate_nets: Tuple[str, ...]
    body_net: str
    finger_pitches_nm: Tuple[int, ...]
    column_xs_nm: Tuple[int, ...]
    gate_spans_x_nm: Tuple[Tuple[int, int], ...]
    active_box_nm: _BoxNm
    tap_box_nm: _BoxNm
    tap_licon_boxes_nm: Tuple[_BoxNm, ...]
    tap_li1_box_nm: _BoxNm
    tap_mcon_box_nm: _BoxNm | None
    tap_m1_box_nm: _BoxNm | None
    tap_label_point_nm: Tuple[int, int]
    gate_segments: Tuple[GateSegmentPlan, ...]
    gate_bridge_m1_boxes_nm: Tuple[_BoxNm, ...]
    sd_tracks: Tuple[SdNetTrackPlan, ...]
    sd_licon_ys_nm: Tuple[Tuple[int, int], ...]
    bbox_nm: _BoxNm


def chain_topology_signature(chain, *, internal_series_nets: Sequence[str] = ()) -> str:
    """Canonical instance-name-free, net-spelling-invariant topology key for caching."""
    net_ids: Dict[str, int] = {}

    def nid(name: str) -> int:
        if name not in net_ids:
            net_ids[name] = len(net_ids)
        return net_ids[name]

    body_id = nid(chain.body)
    col_ids = [nid(col) for col in chain.columns]
    gate_ids = [nid(f.gate) for f in chain.fingers]
    series_ids = sorted(nid(n) for n in set(internal_series_nets))
    payload = {
        "model": chain.model,
        "w_nm": chain.width_nm,
        "l_nm": chain.length_nm,
        "synthesis": getattr(chain, "synthesis", ""),
        "tap_rail": getattr(chain, "tap_rail", ""),
        "body": body_id,
        "columns": col_ids,
        "gates": gate_ids,
        "internal_series": series_ids,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def split_merged_gate_runs(
    gate_nets: Sequence[str],
    *,
    max_merged_gates: int = MAX_MERGED_GATES,
) -> Tuple[Tuple[int, int, str], ...]:
    """Group contiguous identical gate nets and split runs > ``max_merged_gates`` evenly."""
    if max_merged_gates < 1:
        raise ValueError("max_merged_gates must be >= 1")
    nets = tuple(str(g) for g in gate_nets)
    if not nets:
        return ()
    segments: List[Tuple[int, int, str]] = []
    idx = 0
    n = len(nets)
    while idx < n:
        end = idx
        while end + 1 < n and nets[end + 1] == nets[idx]:
            end += 1
        run_len = end - idx + 1
        parts = ceil(run_len / max_merged_gates)
        base, rem = divmod(run_len, parts)
        cur = idx
        for p in range(parts):
            size = base + (1 if p < rem else 0)
            segments.append((cur, cur + size - 1, nets[idx]))
            cur += size
        idx = end + 1
    return tuple(segments)


def _assign_outer_comb_tracks(
    column_nets: Sequence[str],
    *,
    body_net: str,
    supply_nets: frozenset[str],
    internal_series_nets: frozenset[str],
    require_joint_routing_access: bool,
) -> Dict[str, str]:
    """Assign S/D nets to ('tap_comb', 'gate_comb', 'in_corridor', 'li1_internal')."""
    by_net: Dict[str, List[int]] = defaultdict(list)
    for col_idx, net in enumerate(column_nets):
        by_net[str(net)].append(col_idx)

    # Intervals requiring outer LI1 tracks (never crossing active in M1):
    #   1) Multi-column nets (len >= 2);
    #   2) Single-column supply nets (mcon=0 inside active corridor);
    #   3) Single-column internal series nets when joint routing M1 stub is needed.
    candidates: List[Tuple[int, int, int, str]] = []
    track_by_net: Dict[str, str] = {}
    for net, cols in sorted(by_net.items()):
        is_supply = net in supply_nets or net == body_net
        is_series = net in internal_series_nets
        if len(cols) == 1 and is_series and not require_joint_routing_access:
            track_by_net[net] = "li1_internal"
            continue
        if len(cols) >= 2 or is_supply or (len(cols) == 1 and is_series):
            # Priority 0 puts supply/body-tied wide spans onto tap_comb first.
            prio = 0 if is_supply else (1 if len(cols) >= 2 else 2)
            candidates.append((prio, cols[0], cols[-1], net))
        else:
            track_by_net[net] = "in_corridor"

    # Greedy 2-track interval graph coloring: tap_comb first, then gate_comb.
    tap_intervals: List[Tuple[int, int]] = []
    gate_intervals: List[Tuple[int, int]] = []

    def overlaps(intervals: Sequence[Tuple[int, int]], a: int, b: int) -> bool:
        return any(not (b < lo or a > hi) for lo, hi in intervals)

    # Sort by priority, then span width descending so enclosing rails claim tap_comb cleanly.
    candidates.sort(key=lambda item: (item[0], -(item[2] - item[1]), item[1], item[3]))
    for _prio, lo, hi, net in candidates:
        if not overlaps(tap_intervals, lo, hi):
            tap_intervals.append((lo, hi))
            track_by_net[net] = "tap_comb"
        elif not overlaps(gate_intervals, lo, hi):
            gate_intervals.append((lo, hi))
            track_by_net[net] = "gate_comb"
        else:
            # 3-way interleaved fallback: per-column vertical M1 pads (still 0 horizontal M1 walls).
            track_by_net[net] = "in_corridor"
    return track_by_net


def plan_parameterized_row(
    chain,
    *,
    internal_series_nets: Sequence[str] = (),
    supply_nets: Sequence[str] = (),
    require_joint_routing_access: bool = True,
    max_merged_gates: int = MAX_MERGED_GATES,
    rules: Mapping[str, int] | None = None,
) -> ParameterizedRowPlan:
    """Compute exact 5nm-grid geometry for an arbitrary ``N``-finger ``NativeDiffusionChain``."""
    r = SKY130_MOS_RULESET_NM if rules is None else rules
    grid = int(r["grid_nm"])
    n_fingers = len(chain.fingers)
    if n_fingers < 1 or len(chain.columns) != n_fingers + 1:
        raise ValueError("parameterized diffusion chain has invalid finger/column count")

    family = "pmos" if "__pfet_" in chain.model else "nmos"
    is_lvt = "_lvt" in chain.model
    raw_tap = getattr(chain, "tap_rail", "") or ""
    if raw_tap in ("top", "bottom"):
        tap_side = raw_tap
    else:
        tap_side = "top" if family == "pmos" else "bottom"
    gate_side = "bottom" if tap_side == "top" else "top"

    pitches_spec = derive_column_pitches_nm(chain.length_nm, rules=r)
    li1_only_pitch = pitches_spec["li1_only_nm"]
    sd_mcon_pitch = pitches_spec["sd_mcon_nm"]
    poly_contact_pitch = pitches_spec["poly_contacted_nm"]

    gate_nets = tuple(f.gate for f in chain.fingers)
    col_nets = tuple(chain.columns)
    raw_segments = split_merged_gate_runs(gate_nets, max_merged_gates=max_merged_gates)
    seg_index_by_finger = {}
    seg_len_by_finger = {}
    for seg_idx, (s, e, _gnet) in enumerate(raw_segments):
        for f_idx in range(s, e + 1):
            seg_index_by_finger[f_idx] = seg_idx
            seg_len_by_finger[f_idx] = e - s + 1

    supplies = frozenset(str(s) for s in supply_nets) | {str(chain.body)}
    series = frozenset(str(s) for s in internal_series_nets)
    track_by_net = _assign_outer_comb_tracks(
        col_nets,
        body_net=str(chain.body),
        supply_nets=supplies,
        internal_series_nets=series,
        require_joint_routing_access=require_joint_routing_access,
    )

    # Determine per-finger horizontal pitch from local S/D mcon and adjacent gate heads.
    finger_pitches: List[int] = []
    for i in range(n_fingers):
        pitch = li1_only_pitch
        if (
            track_by_net[col_nets[i]] == "in_corridor"
            or track_by_net[col_nets[i + 1]] == "in_corridor"
        ):
            pitch = max(pitch, sd_mcon_pitch)
        # Check if finger i-1 or i+1 has a separate gate contact head whose 270nm poly pad
        # sits next to finger i's gate contact head.
        if i > 0 and seg_index_by_finger[i - 1] != seg_index_by_finger[i]:
            if seg_len_by_finger[i - 1] == 1 or seg_len_by_finger[i] == 1:
                pitch = max(pitch, poly_contact_pitch)
        if i + 1 < n_fingers and seg_index_by_finger[i + 1] != seg_index_by_finger[i]:
            if seg_len_by_finger[i + 1] == 1 or seg_len_by_finger[i] == 1:
                pitch = max(pitch, poly_contact_pitch)
        finger_pitches.append(pitch)

    # Enforce palindromic pitch symmetry when gate-segment lengths and in-corridor M1 flags are symmetric
    in_corridor_flags = tuple(track_by_net[n] == "in_corridor" for n in col_nets)
    seg_lens = tuple(seg_len_by_finger[i] for i in range(n_fingers))
    if in_corridor_flags == in_corridor_flags[::-1] and seg_lens == seg_lens[::-1]:
        finger_pitches = [
            max(finger_pitches[i], finger_pitches[n_fingers - 1 - i])
            for i in range(n_fingers)
        ]

    col_xs = [0]
    for pitch in finger_pitches:
        col_xs.append(col_xs[-1] + pitch)
    gate_spans_x = tuple(
        (
            col_xs[i] + (finger_pitches[i] - chain.length_nm) // 2,
            col_xs[i] + (finger_pitches[i] + chain.length_nm) // 2,
        )
        for i in range(n_fingers)
    )

    cut = int(r["licon_size_nm"])
    half_cut = cut // 2  # 85nm -> wait: 170 // 2 = 85! All coordinates stay on 5nm grid!
    outer_diff_margin = half_cut + int(r["diff_enclose_licon_end_nm"])  # 85 + 60 = 145nm
    active_box: _BoxNm = (-outer_diff_margin, 0, col_xs[-1] + outer_diff_margin, chain.width_nm)
    _, sd_licon_ys = sd_licon_layout_nm(chain.width_nm, y_bottom_nm=0, rules=r)

    # Vertical pad dimensions: 260 x 320 nm (83,200 nm^2 >= 83,000 nm^2) or 320 x 260 nm horizontal
    vpad_w = cut + 2 * int(r["m1_enclose_mcon_sym_nm"])  # 260nm
    vpad_h = snap_up_grid_nm((int(r["m1_min_area_nm2"]) + vpad_w - 1) // vpad_w, grid)  # 320nm
    hpad_w = vpad_h  # 320nm
    hpad_h = vpad_w  # 260nm

    has_tap_comb = any(t == "tap_comb" for t in track_by_net.values())
    has_gate_comb = any(t == "gate_comb" for t in track_by_net.values())

    # Allocate Y bands on tap side and gate side without any LI1/M1/poly/implant collision
    W = chain.width_nm
    if tap_side == "top":
        tap_comb_spine_y = (W + 200, W + 200 + vpad_h) if has_tap_comb else None
        tap_gap = max(
            int(r["diff_to_opposite_tap_space_nm"]),
            (tap_comb_spine_y[1] - W + 200) if tap_comb_spine_y is not None else 0,
        )
    else:
        tap_comb_spine_y = (-200 - vpad_h, -200) if has_tap_comb else None
        tap_gap = max(
            int(r["diff_to_opposite_tap_space_nm"]),
            (-tap_comb_spine_y[0] + 200) if tap_comb_spine_y is not None else 0,
        )

    tap_info = tap_rail_layout_nm(
        active_box[0],
        active_box[2],
        active_bottom_nm=0,
        active_top_nm=W,
        side=tap_side,
        gap_nm=tap_gap,
        rules=r,
    )
    tap_box: _BoxNm = tap_info["tap_box_nm"]
    tap_licons: Tuple[_BoxNm, ...] = tap_info["licon_boxes_nm"]
    tap_li1_box: _BoxNm = tap_info["li1_box_nm"]
    tap_mid_licon = tap_licons[len(tap_licons) // 2]
    tap_cx = ((tap_mid_licon[0] + tap_mid_licon[2]) // 2 // grid) * grid
    tap_cy = ((tap_mid_licon[1] + tap_mid_licon[3]) // 2 // grid) * grid
    tap_label_pt = (tap_cx, tap_cy)
    if require_joint_routing_access:
        # Place tap mcon+M1 centered on the row midpoint (320 x 260 nm horizontal along tap rail)
        row_mid_x = ((active_box[0] + active_box[2]) // 2 // grid) * grid
        tap_mcon_box: _BoxNm | None = (
            row_mid_x - half_cut,
            tap_box[1] + (hpad_h - cut) // 2,
            row_mid_x + half_cut,
            tap_box[1] + (hpad_h + cut) // 2,
        )
        tap_m1_box: _BoxNm | None = (
            row_mid_x - hpad_w // 2,
            tap_box[1],
            row_mid_x + hpad_w // 2,
            tap_box[3],
        )
    else:
        tap_mcon_box = None
        tap_m1_box = None

    if gate_side == "top":
        gate_comb_spine_y = (W + 200, W + 200 + vpad_h) if has_gate_comb else None
        poly_head_y0 = (gate_comb_spine_y[1] + 220) if gate_comb_spine_y is not None else (W + 200)
        poly_head_y1 = poly_head_y0 + cut + 2 * int(r["poly_enclose_licon_end_nm"])  # +330nm
    else:
        gate_comb_spine_y = (-200 - vpad_h, -200) if has_gate_comb else None
        poly_head_y1 = (gate_comb_spine_y[0] - 220) if gate_comb_spine_y is not None else (-200)
        poly_head_y0 = poly_head_y1 - (cut + 2 * int(r["poly_enclose_licon_end_nm"]))

    # Build GateSegmentPlans
    poly_pad_w = int(r["licon_size_nm"]) + 2 * int(r["poly_enclose_licon_side_nm"])  # 270nm
    poly_end_enc = int(r["poly_enclose_licon_end_nm"])  # 80nm
    npc_enc = int(r["npc_enclose_poly_licon_nm"])  # 100nm
    gate_segments: List[GateSegmentPlan] = []
    for s, e, gnet in raw_segments:
        gx0, gx1 = gate_spans_x[s][0], gate_spans_x[e][1]
        gcx = ((gx0 + gx1) // 2 // grid) * grid
        if s == e:
            px0, px1 = gcx - poly_pad_w // 2, gcx + poly_pad_w // 2
            licon_xs = ((gcx - half_cut, gcx + half_cut),)
            li_x0, li_x1 = px0, px1
        else:
            px0 = min(gx0, gcx - poly_pad_w // 2)
            px1 = max(gx1, gcx + poly_pad_w // 2)
            usable = (px1 - px0) - 2 * poly_end_enc
            pitch = cut + int(r["licon_space_nm"])
            k = max(1, (usable + int(r["licon_space_nm"])) // pitch)
            span = k * cut + (k - 1) * int(r["licon_space_nm"])
            lx0 = px0 + (((px1 - px0) - span) // 2 // grid) * grid
            licon_xs = tuple((lx0 + idx * pitch, lx0 + idx * pitch + cut) for idx in range(k))
            li_x0 = min(licon_xs[0][0] - poly_end_enc, gcx - poly_pad_w // 2)
            li_x1 = max(licon_xs[-1][1] + poly_end_enc, gcx + poly_pad_w // 2)
        ly0 = poly_head_y0 + poly_end_enc
        ly1 = ly0 + cut
        g_licons = tuple((x0, ly0, x1, ly1) for x0, x1 in licon_xs)
        li_box: _BoxNm = (li_x0, poly_head_y0, li_x1, poly_head_y1)
        mcon_cy = ((poly_head_y0 + poly_head_y1) // 2 // grid) * grid
        mcon_box: _BoxNm = (gcx - half_cut, mcon_cy - half_cut, gcx + half_cut, mcon_cy + half_cut)
        m1_box: _BoxNm = (gcx - vpad_w // 2, mcon_cy - vpad_h // 2, gcx + vpad_w // 2, mcon_cy + vpad_h // 2)
        npc_box: _BoxNm = (
            g_licons[0][0] - npc_enc,
            ly0 - npc_enc,
            g_licons[-1][2] + npc_enc,
            ly1 + npc_enc,
        )
        gate_segments.append(
            GateSegmentPlan(
                start_finger=s,
                end_finger=e,
                gate_net=gnet,
                poly_head_box_nm=(px0, poly_head_y0, px1, poly_head_y1),
                licon_boxes_nm=g_licons,
                li1_box_nm=li_box,
                mcon_box_nm=mcon_box,
                m1_box_nm=m1_box,
                npc_box_nm=npc_box,
            )
        )

    # Bridge adjacent split segments belonging to the same contiguous gate run (e.g. 10 -> 5+5) in M1
    gate_bridge_m1: List[_BoxNm] = []
    idx = 0
    while idx < len(gate_segments):
        j = idx
        while (
            j + 1 < len(gate_segments)
            and gate_segments[j + 1].gate_net == gate_segments[idx].gate_net
            and gate_segments[j + 1].start_finger == gate_segments[j].end_finger + 1
        ):
            j += 1
        if j > idx:
            cy = ((poly_head_y0 + poly_head_y1) // 2 // grid) * grid
            bx0 = gate_segments[idx].m1_box_nm[0]
            bx1 = gate_segments[j].m1_box_nm[2]
            gate_bridge_m1.append((bx0, cy - hpad_h // 2, bx1, cy + hpad_h // 2))
        idx = j + 1

    # Build S/D track plans
    cols_by_net: Dict[str, List[int]] = defaultdict(list)
    for c_idx, c_net in enumerate(col_nets):
        cols_by_net[str(c_net)].append(c_idx)

    sd_tracks: List[SdNetTrackPlan] = []
    for net, cols in sorted(cols_by_net.items(), key=lambda kv: (kv[1][0], kv[0])):
        t_kind = track_by_net[net]
        if t_kind in ("tap_comb", "gate_comb"):
            spine_y = tap_comb_spine_y if t_kind == "tap_comb" else gate_comb_spine_y
            assert spine_y is not None
            cx_first, cx_last = col_xs[cols[0]], col_xs[cols[-1]]
            mid_x = ((cx_first + cx_last) // 2 // grid) * grid
            # 260 x 320 nm M1 pad (fits inside adjacent 440nm columns with 180nm >= 140nm gap)
            pad_x0 = max(active_box[0], mid_x - vpad_w // 2)
            pad_x1 = pad_x0 + vpad_w
            if pad_x1 > active_box[2]:
                pad_x1 = active_box[2]
                pad_x0 = pad_x1 - vpad_w
            pad_cx = ((pad_x0 + pad_x1) // 2 // grid) * grid
            pad_cy = ((spine_y[0] + spine_y[1]) // 2 // grid) * grid
            # 270 x 320 nm LI1 head (= 86,400 nm^2 >= 56,100 nm^2, 440 - 270 = 170nm >= li.3)
            li_x0 = min(cx_first - half_cut, pad_cx - poly_pad_w // 2)
            li_x1 = max(cx_last + half_cut, pad_cx + poly_pad_w // 2)
            li_x0 = max(active_box[0], li_x0)
            li_x1 = min(active_box[2], li_x1)
            spine_box: _BoxNm = (li_x0, spine_y[0], li_x1, spine_y[1])
            if require_joint_routing_access or net not in series:
                mcon_b = ((pad_cx - half_cut, pad_cy - half_cut, pad_cx + half_cut, pad_cy + half_cut),)
                m1_b = ((pad_x0, spine_y[0], pad_x1, spine_y[1]),)
            else:
                mcon_b = ()
                m1_b = ()
            sd_tracks.append(
                SdNetTrackPlan(
                    net=net,
                    columns=tuple(cols),
                    track=t_kind,
                    spine_li1_box_nm=spine_box,
                    mcon_boxes_nm=mcon_b,
                    m1_boxes_nm=m1_b,
                )
            )
        elif t_kind == "in_corridor":
            cy = ((W // 2) // grid) * grid
            mcons = tuple(
                (col_xs[c] - half_cut, cy - half_cut, col_xs[c] + half_cut, cy + half_cut)
                for c in cols
            )
            m1s = tuple(
                (col_xs[c] - vpad_w // 2, cy - vpad_h // 2, col_xs[c] + vpad_w // 2, cy + vpad_h // 2)
                for c in cols
            )
            sd_tracks.append(
                SdNetTrackPlan(
                    net=net,
                    columns=tuple(cols),
                    track="in_corridor",
                    spine_li1_box_nm=None,
                    mcon_boxes_nm=mcons,
                    m1_boxes_nm=m1s,
                )
            )
        else:
            sd_tracks.append(
                SdNetTrackPlan(
                    net=net,
                    columns=tuple(cols),
                    track="li1_internal",
                    spine_li1_box_nm=None,
                    mcon_boxes_nm=(),
                    m1_boxes_nm=(),
                )
            )

    # Compute exact bounding box from drawn masks
    dummy_plan = ParameterizedRowPlan(
        model=chain.model,
        family=family,
        is_lvt=is_lvt,
        width_nm=W,
        length_nm=chain.length_nm,
        finger_count=n_fingers,
        tap_side=tap_side,
        gate_side=gate_side,
        column_nets=col_nets,
        gate_nets=gate_nets,
        body_net=str(chain.body),
        finger_pitches_nm=tuple(finger_pitches),
        column_xs_nm=tuple(col_xs),
        gate_spans_x_nm=gate_spans_x,
        active_box_nm=active_box,
        tap_box_nm=tap_box,
        tap_licon_boxes_nm=tap_licons,
        tap_li1_box_nm=tap_li1_box,
        tap_mcon_box_nm=tap_mcon_box,
        tap_m1_box_nm=tap_m1_box,
        tap_label_point_nm=tap_label_pt,
        gate_segments=tuple(gate_segments),
        gate_bridge_m1_boxes_nm=tuple(gate_bridge_m1),
        sd_tracks=tuple(sd_tracks),
        sd_licon_ys_nm=sd_licon_ys,
        bbox_nm=(0, 0, 0, 0),
    )
    contacted = build_parameterized_row_regions(dummy_plan, include_contacts=True)
    extent = kdb.Region()
    for reg in contacted.values():
        extent += reg
    bb = extent.bbox()
    return ParameterizedRowPlan(
        **{
            **dummy_plan.__dict__,
            "bbox_nm": (int(bb.left), int(bb.bottom), int(bb.right), int(bb.top)),
        }
    )


def build_parameterized_row_regions(
    plan: ParameterizedRowPlan,
    *,
    include_contacts: bool = True,
) -> Dict[_LayerKey, kdb.Region]:
    """Emit merged KLayout ``Region`` masks for ``raw.gds`` or ``contacted.gds``."""
    r = SKY130_MOS_RULESET_NM
    cut = int(r["licon_size_nm"])
    half_cut = cut // 2
    sdm_enc = int(r["sdm_enclose_diff_nm"])
    nw_enc = int(r["nwell_enclose_diff_nm"])
    lvt_enc = int(r["lvtn_enclose_gate_nm"])
    poly_endcap = int(r["gate_poly_extension_nm"])

    out: Dict[_LayerKey, kdb.Region] = defaultdict(kdb.Region)
    ax0, ay0, ax1, ay1 = plan.active_box_nm
    out[LAYER_DIFF].insert(kdb.Box(ax0, ay0, ax1, ay1))

    diff_sdm_layer = LAYER_PSDM if plan.family == "pmos" else LAYER_NSDM
    tap_sdm_layer = LAYER_NSDM if plan.family == "pmos" else LAYER_PSDM
    out[diff_sdm_layer].insert(kdb.Box(ax0 - sdm_enc, ay0 - sdm_enc, ax1 + sdm_enc, ay1 + sdm_enc))

    # Gate poly fingers + merged poly contact heads are part of raw FEOL so
    # build_native_diffusion_mosaic observes exact gate poly geometry in raw.gds.
    for seg in plan.gate_segments:
        hx0, hy0, hx1, hy1 = seg.poly_head_box_nm
        out[LAYER_POLY].insert(kdb.Box(hx0, hy0, hx1, hy1))
        for f_idx in range(seg.start_finger, seg.end_finger + 1):
            gx0, gx1 = plan.gate_spans_x_nm[f_idx]
            if plan.gate_side == "top":
                py0, py1 = ay0 - poly_endcap, hy1
            else:
                py0, py1 = hy0, ay1 + poly_endcap
            out[LAYER_POLY].insert(kdb.Box(gx0, py0, gx1, py1))

    if plan.is_lvt:
        gx_min = plan.gate_spans_x_nm[0][0] - lvt_enc
        gx_max = plan.gate_spans_x_nm[-1][1] + lvt_enc
        out[LAYER_LVTN].insert(kdb.Box(gx_min, ay0 - lvt_enc, gx_max, ay1 + lvt_enc))

    if plan.family == "pmos":
        tx0, ty0, tx1, ty1 = plan.tap_box_nm
        nw_x0 = min(ax0, tx0) - nw_enc
        nw_x1 = max(ax1, tx1) + nw_enc
        nw_y0 = min(ay0, ty0) - nw_enc
        nw_y1 = max(ay1, ty1) + nw_enc
        if plan.is_lvt:
            nw_y0 = min(nw_y0, ay0 - lvt_enc - nw_enc)
            nw_y1 = max(nw_y1, ay1 + lvt_enc + nw_enc)
        out[LAYER_NWELL].insert(kdb.Box(nw_x0, nw_y0, nw_x1, nw_y1))

    if include_contacts:
        # 1. S/D column licon + vertical LI1 bars
        li_y0 = min(ay0 - 20, plan.sd_licon_ys_nm[0][0] - 85)
        li_y1 = max(ay1 + 20, plan.sd_licon_ys_nm[-1][1] + 85)
        # Ensure single-column LI1 bar >= 56,100 nm^2 (170 * 340 = 57,800)
        if li_y1 - li_y0 < 340:
            mid = (li_y0 + li_y1) // 2
            li_y0 = (mid - 170) // 5 * 5
            li_y1 = li_y0 + 340
        for cx in plan.column_xs_nm:
            for cy0, cy1 in plan.sd_licon_ys_nm:
                out[LAYER_LICON].insert(kdb.Box(cx - half_cut, cy0, cx + half_cut, cy1))
            out[LAYER_LI1].insert(kdb.Box(cx - half_cut, li_y0, cx + half_cut, li_y1))

        # 2. S/D comb tracks & mcon/M1 pads
        for tr in plan.sd_tracks:
            if tr.spine_li1_box_nm is not None:
                sx0, sy0, sx1, sy1 = tr.spine_li1_box_nm
                out[LAYER_LI1].insert(kdb.Box(sx0, sy0, sx1, sy1))
                for col_idx in tr.columns:
                    cx = plan.column_xs_nm[col_idx]
                    py0 = min(li_y0, sy0)
                    py1 = max(li_y1, sy1)
                    out[LAYER_LI1].insert(kdb.Box(cx - half_cut, py0, cx + half_cut, py1))
            elif tr.track == "in_corridor":
                # Widen LI1 around in-corridor mcon if needed (li.5 = 0 on mcon, already covered)
                pass
            for mb in tr.mcon_boxes_nm:
                out[LAYER_MCON].insert(kdb.Box(*mb))
            for m1b in tr.m1_boxes_nm:
                out[LAYER_MET1].insert(kdb.Box(*m1b))

        # 3. Gate licon + npc + LI1 + mcon + M1
        npc_boxes = sorted((seg.npc_box_nm for seg in plan.gate_segments), key=lambda b: b[0])
        merged_npcs: List[List[int]] = []
        for nb in npc_boxes:
            if merged_npcs and nb[0] - merged_npcs[-1][2] < 270 and nb[1] == merged_npcs[-1][1] and nb[3] == merged_npcs[-1][3]:
                merged_npcs[-1][2] = max(merged_npcs[-1][2], nb[2])
            else:
                merged_npcs.append(list(nb))
        for nb in merged_npcs:
            out[LAYER_NPC].insert(kdb.Box(*nb))

        for seg in plan.gate_segments:
            for lb in seg.licon_boxes_nm:
                out[LAYER_LICON].insert(kdb.Box(*lb))
            out[LAYER_LI1].insert(kdb.Box(*seg.li1_box_nm))
            out[LAYER_MCON].insert(kdb.Box(*seg.mcon_box_nm))
            out[LAYER_MET1].insert(kdb.Box(*seg.m1_box_nm))
        for gb in plan.gate_bridge_m1_boxes_nm:
            out[LAYER_MET1].insert(kdb.Box(*gb))

        # 4. Width-aligned horizontal tap rail (0nm lateral X overhang)
        tx0, ty0, tx1, ty1 = plan.tap_box_nm
        out[LAYER_TAP].insert(kdb.Box(tx0, ty0, tx1, ty1))
        out[tap_sdm_layer].insert(
            kdb.Box(tx0 - sdm_enc, ty0 - sdm_enc, tx1 + sdm_enc, ty1 + sdm_enc)
        )
        for lb in plan.tap_licon_boxes_nm:
            out[LAYER_LICON].insert(kdb.Box(*lb))
        out[LAYER_LI1].insert(kdb.Box(*plan.tap_li1_box_nm))
        if plan.tap_mcon_box_nm is not None:
            out[LAYER_MCON].insert(kdb.Box(*plan.tap_mcon_box_nm))
        if plan.tap_m1_box_nm is not None:
            out[LAYER_MET1].insert(kdb.Box(*plan.tap_m1_box_nm))

    return {layer: reg.merged() for layer, reg in sorted(out.items()) if not reg.is_empty()}


def write_regions_gds(
    regions: Mapping[_LayerKey, kdb.Region],
    path: Path | str,
    *,
    cell_name: str = "primitive",
) -> None:
    """Write nm-unit KLayout ``Region`` masks to a canonical ``DBU = 0.001`` GDS file."""
    layout = kdb.Layout()
    layout.dbu = 0.001
    top = layout.create_cell(cell_name)
    for (layer, datatype), reg in sorted(regions.items()):
        lidx = layout.layer(int(layer), int(datatype))
        for poly in reg.merged().each():
            top.shapes(lidx).insert(poly)
    layout.write(str(path))


def audit_corridor_safety(plan: ParameterizedRowPlan, regions: Mapping[_LayerKey, kdb.Region] | None = None) -> Dict[str, object]:
    """Verify Spec P2/P3/P5 corridor-safety and zero-lateral-overhang invariants."""
    regs = build_parameterized_row_regions(plan, include_contacts=True) if regions is None else regions
    ax0, ay0, ax1, ay1 = plan.active_box_nm
    corridor = kdb.Region(kdb.Box(ax0, ay0, ax1, ay1))
    m1_in_corridor = regs.get(LAYER_MET1, kdb.Region()) & corridor
    # No M1 polygon inside the active corridor may span across >= 2 columns
    min_pitch = min(plan.finger_pitches_nm)
    horizontal_m1_walls = [
        poly.bbox()
        for poly in m1_in_corridor.merged().each()
        if poly.bbox().width() >= min_pitch
    ]
    if horizontal_m1_walls:
        raise ValueError(f"illegal horizontal M1 wall(s) inside active corridor: {horizontal_m1_walls}")

    # Supply and comb tracks must have 0 mcon inside active corridor
    for tr in plan.sd_tracks:
        if tr.track in ("tap_comb", "gate_comb", "li1_internal"):
            for mb in tr.mcon_boxes_nm:
                if not (kdb.Region(kdb.Box(*mb)) & corridor).is_empty():
                    raise ValueError(f"comb/supply net {tr.net!r} placed mcon inside active corridor")

    # Tap rail lateral overhang relative to active diffusion must be 0 nm
    tx0, _ty0, tx1, _ty1 = plan.tap_box_nm
    overhang_left = max(0, ax0 - tx0)
    overhang_right = max(0, tx1 - ax1)
    if overhang_left != 0 or overhang_right != 0:
        raise ValueError(f"tap rail has non-zero lateral overhang ({overhang_left}, {overhang_right}) nm")

    # Every M1 polygon must satisfy met1.6 >= 83,000 nm^2 and every LI1 polygon li.6 >= 56,100 nm^2,
    # and intra-row spacing rules poly.2 (210nm), li.3 (170nm), met1.2 (140nm) must be clean.
    m1_merged = regs.get(LAYER_MET1, kdb.Region()).merged()
    li1_merged = regs.get(LAYER_LI1, kdb.Region()).merged()
    poly_merged = regs.get(LAYER_POLY, kdb.Region()).merged()
    for poly in m1_merged.each():
        if poly.area() < int(SKY130_MOS_RULESET_NM["m1_min_area_nm2"]):
            raise ValueError(f"M1 polygon area {poly.area()} < 83000 nm^2")
    for poly in li1_merged.each():
        if poly.area() < int(SKY130_MOS_RULESET_NM["li1_min_area_nm2"]):
            raise ValueError(f"LI1 polygon area {poly.area()} < 56100 nm^2")
    if not m1_merged.space_check(int(SKY130_MOS_RULESET_NM["m1_space_nm"])).is_empty():
        raise ValueError("parameterized row violates intra-row met1.2 spacing")
    if not li1_merged.space_check(int(SKY130_MOS_RULESET_NM["li1_space_nm"])).is_empty():
        raise ValueError("parameterized row violates intra-row li.3 spacing")
    if not poly_merged.space_check(int(SKY130_MOS_RULESET_NM["poly_space_nm"])).is_empty():
        raise ValueError("parameterized row violates intra-row poly.2 spacing")

    return {
        "horizontal_m1_walls_in_corridor": 0,
        "tap_lateral_overhang_nm": (overhang_left, overhang_right),
        "gate_segment_count": len(plan.gate_segments),
        "finger_pitches_nm": plan.finger_pitches_nm,
        "bbox_nm": plan.bbox_nm,
    }


def internal_series_nets_for_chain(cell, chain) -> Tuple[str, ...]:
    """Identify nets internal to a CMOS series stack (appearing on == 2 MOS S/D pins and nowhere else)."""
    ports = {str(p) for p in (getattr(cell, "ports", ()) or ())}
    chain_owners = {f.owner for f in chain.fingers}
    net_refs: Dict[str, int] = defaultdict(int)
    outside_refs: set[str] = set()
    for dev in getattr(cell, "devices", ()) or ():
        pins = getattr(dev, "pins", None) or {}
        if dev.instance_name in chain_owners:
            for role, net in pins.items():
                if role in ("d", "s"):
                    net_refs[str(net)] += 1
                else:
                    outside_refs.add(str(net))
        else:
            for net in pins.values():
                outside_refs.add(str(net))
    col_counts = defaultdict(int)
    for col in chain.columns:
        col_counts[str(col)] += 1
    return tuple(
        sorted(
            net
            for net, cnt in col_counts.items()
            if cnt == 1
            and net not in ports
            and net != chain.body
            and net not in outside_refs
            and net_refs.get(net, 0) >= 2
        )
    )


def supply_nets_for_cell(cell) -> Tuple[str, ...]:
    roles = getattr(cell, "port_roles", None) or {}
    supplies = {
        str(net)
        for net, role in roles.items()
        if str(role) in ("power_p", "power_n", "ground")
    }
    if not supplies:
        for p in getattr(cell, "ports", ()) or ():
            if str(p).lower() in ("vdd", "vss", "gnd", "vcc"):
                supplies.add(str(p))
    return tuple(sorted(supplies))


def _verify_parameterized_magic_extraction(
    folder: Path,
    plan: ParameterizedRowPlan,
    contact_regions: Mapping[Tuple[int, int], kdb.Region],
) -> Dict[str, object]:
    """Verify live Magic full DRC, integer 5nm scale, SPICE W/L/count, and EXT channel/tap anchors."""
    from decimal import Decimal
    import shlex
    from clay.layout.native_sky130 import _raw_anchors, _raw_devices

    drc_text = (folder / "drc.txt").read_text(encoding="utf-8")
    if drc_text.strip():
        raise ValueError(
            f"parameterized native row ({plan.model}, N={plan.finger_count}, W={plan.width_nm}nm) "
            f"failed Magic drc(full):\n{drc_text.strip()[:800]}"
        )
    scale_nm = Decimal((folder / "scale_um.txt").read_text(encoding="utf-8").strip()) * 1000
    grid_nm = int(scale_nm.to_integral_value())
    if grid_nm != 5 or abs(scale_nm - grid_nm) >= Decimal("0.000001"):
        raise ValueError(f"parameterized native extraction scale differs from 5nm: {scale_nm!r}")

    spice_text = (folder / "native.spice").read_text(encoding="utf-8")
    rows = _raw_devices(spice_text, plan.model, 4)
    if len(rows) != plan.finger_count:
        raise ValueError(
            f"parameterized native SPICE extracted {len(rows)} devices, expected {plan.finger_count}"
        )
    for _nodes, params in rows:
        if Decimal(params["w"]) * 1000 != plan.width_nm or Decimal(params["l"]) * 1000 != plan.length_nm:
            raise ValueError("parameterized native SPICE extracted W/L differs from source plan")
    bodies = {nodes[3] for nodes, _ in rows}
    if len(bodies) != 1:
        raise ValueError(f"parameterized native SPICE extracted multiple body nodes: {sorted(bodies)}")
    body_node = next(iter(bodies))

    ext_text = (folder / "primitive_flat.ext").read_text(encoding="utf-8")
    anchors = _raw_anchors(ext_text, grid_nm)
    if body_node not in anchors:
        raise ValueError(f"parameterized native EXT missing body anchor {body_node!r}")
    if plan.family == "nmos" and anchors[body_node][2] != "ppd":
        raise ValueError(
            f"parameterized NMOS body anchor material is {anchors[body_node][2]!r}, expected 'ppd'"
        )

    active = contact_regions[LAYER_DIFF].merged()
    poly = contact_regions[LAYER_POLY].merged()
    channels = sorted((active & poly).merged().each(), key=lambda p: p.bbox().left)
    if len(channels) != plan.finger_count:
        raise ValueError("parameterized contacted GDS channel denominator differs")

    ext_channels = []
    for line in ext_text.splitlines():
        fields = shlex.split(line)
        if len(fields) >= 11 and fields[:3] == ["device", "msubckt", plan.model]:
            ext_channels.append((int(fields[3]) * grid_nm, int(fields[4]) * grid_nm))
    if len(ext_channels) != plan.finger_count:
        raise ValueError("parameterized native EXT channel count differs from finger count")
    hit_indices = set()
    for ex, ey in ext_channels:
        probe = kdb.Region(kdb.Box(ex, ey, ex + 1, ey + 1))
        hits = [idx for idx, ch in enumerate(channels) if not (kdb.Region(ch) & probe).is_empty()]
        if len(hits) != 1 or hits[0] in hit_indices:
            raise ValueError("parameterized native EXT channel anchor is ambiguous")
        hit_indices.add(hits[0])

    return {
        "model": plan.model,
        "physical_occurrences": len(rows),
        "body_node": body_node,
        "body_anchor": anchors[body_node],
        "magic_drc_clean": True,
    }


def _magic_extract_and_drc_tcl(plan: ParameterizedRowPlan) -> str:
    bx, by = plan.tap_label_point_nm
    contact_type = "nsc" if plan.family == "pmos" else "psc"
    return f"""crashbackups stop
drc off
gds read /work/contacted.gds
load primitive
box {bx / 1000:.3f}um {by / 1000:.3f}um {bx / 1000:.3f}um {by / 1000:.3f}um
label B c {contact_type}
set h [open scale_um.txt w]
puts $h [cif scale out]
close $h
flatten primitive_flat
load primitive_flat
select top cell
drc euclidean on
drc style drc(full)
drc check
drc catchup
set h [open drc.txt w]
puts $h [drc listall why]
close $h
extract path .
extract no capacitance
extract no coupling
extract no resistance
extract no length
extract all
ext2spice lvs
ext2spice merge none
ext2spice -p . -o native.spice
puts PARAMETERIZED_ROW_OBSERVED
quit -noprompt
"""


def materialize_parameterized_diffusion_template(
    cell,
    bundle,
    chain,
    *,
    tech,
    output_dir,
    timeout_seconds: int = 60,
):
    """Synthesize GDS, verify corridor safety, run Magic full DRC + SPICE/EXT verification."""
    from clay.layout.native_diffusion_pcell import _check_chain, _magic
    from clay.layout.native_diffusion_source import _require_canonical_native_arrays

    if type(bundle) is not NativeGeometryBundle:
        raise TypeError("parameterized diffusion template requires the complete native bundle")
    if getattr(tech, "pdk_id", None) != "sky130":
        raise ValueError("parameterized diffusion template requires SKY130 tech")
    bundle.require_current(cell)
    _check_chain(cell, chain)
    source_before, canonical = _require_canonical_native_arrays(cell)
    _check_chain(canonical, chain)

    command = json.loads((bundle.output_dir / "command.json").read_text(encoding="utf-8"))
    images = [
        arg for arg in command
        if isinstance(arg, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", arg)
    ]
    if len(images) != 1:
        raise ValueError("native bundle image authority is missing")
    image = images[0]
    runtime = json.loads((bundle.output_dir / "toolchain.json").read_text(encoding="utf-8"))
    pdk_path = runtime["pdk_root"]
    if not pdk_path.endswith("/sky130A"):
        raise ValueError("native bundle PDK root differs")
    pdk_root = pdk_path[: -len("/sky130A")]

    folder = Path(output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=False)

    series_nets = internal_series_nets_for_chain(cell, chain)
    supplies = supply_nets_for_cell(cell)
    plan = plan_parameterized_row(
        chain,
        internal_series_nets=series_nets,
        supply_nets=supplies,
        require_joint_routing_access=True,
    )
    raw_regions = build_parameterized_row_regions(plan, include_contacts=False)
    contact_regions = build_parameterized_row_regions(plan, include_contacts=True)
    safety = audit_corridor_safety(plan, contact_regions)

    write_regions_gds(raw_regions, folder / "primitive.gds")
    shutil.copyfile(folder / "primitive.gds", folder / "raw.gds")
    write_regions_gds(contact_regions, folder / "contacted.gds")

    tcl_text = _magic_extract_and_drc_tcl(plan)
    (folder / "observe.tcl").write_text(tcl_text, encoding="utf-8", newline="\n")
    _magic(folder, image, pdk_root, "observe.tcl", "observe", timeout_seconds)

    observed = _verify_parameterized_magic_extraction(folder, plan, contact_regions)

    bundle.require_current(cell)
    source_sha, _ = _require_canonical_native_arrays(cell)
    if source_sha != source_before:
        raise ValueError("native diffusion source changed during parameterized generation")

    topo_sig = chain_topology_signature(chain, internal_series_nets=series_nets)
    plan_summary = {
        "topology_signature": topo_sig,
        "finger_pitches_nm": list(plan.finger_pitches_nm),
        "tap_side": plan.tap_side,
        "gate_side": plan.gate_side,
        "corridor_safety": safety,
    }
    (folder / "parameterized-plan.json").write_text(
        json.dumps(plan_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    outputs = {
        path.name: _sha(path)
        for path in sorted(folder.iterdir())
        if path.is_file() and path.name != "template-receipt.json"
    }
    receipt = {
        "source_identity": native_source_identity(cell),
        "source_sha256": source_sha,
        "model": chain.model,
        "unit_width_nm": chain.width_nm,
        "length_nm": chain.length_nm,
        "synthesis": chain.synthesis,
        "tap_rail": plan.tap_side,
        "topology_signature": topo_sig,
        "image": image,
        "native_bundle_manifest_sha256": _sha(bundle.output_dir / "geometry.json"),
        "producer_sha256": _sha(__file__),
        "inputs": {str(Path(cell.source_span["file"]).resolve()): source_sha},
        "outputs": outputs,
    }
    (folder / "template-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return observed, receipt


def verify_parameterized_receipt(cell, chain, folder: Path | str, receipt: Mapping[str, object], tech) -> None:
    """Independently replay the parameterized plan and verify receipt bindings."""
    folder = Path(folder)
    if getattr(tech, "pdk_id", None) != "sky130":
        raise ValueError("parameterized native row receipt requires SKY130 tech")
    if receipt.get("synthesis") != chain.synthesis or chain.synthesis not in ("parameterized", "cmos_gate"):
        raise ValueError("parameterized native row receipt synthesis differs")
    series_nets = internal_series_nets_for_chain(cell, chain)
    supplies = supply_nets_for_cell(cell)
    expected_sig = chain_topology_signature(chain, internal_series_nets=series_nets)
    if receipt.get("topology_signature") != expected_sig:
        raise ValueError("parameterized native row topology signature differs")
    plan = plan_parameterized_row(
        chain,
        internal_series_nets=series_nets,
        supply_nets=supplies,
        require_joint_routing_access=True,
    )
    if receipt.get("tap_rail") != plan.tap_side:
        raise ValueError("parameterized native row tap_rail differs")
    audit_corridor_safety(plan)
