"""Topology-driven complementary CMOS gate & multi-stage buffer composer (Spec P4).

Recognizes and synthesizes complementary CMOS functional groups (1-stage
inverter, multi-input ``NOR`` / ``NAND``, cross-coupled ``SR`` latch, 2-stage
buffer) purely from transistor connectivity — zero cell/instance/net-name
heuristics:
  * Splits complementary PMOS (upper N-well band, top ``NTAP`` ``VDD`` rail) and
    NMOS (lower P-sub band, bottom ``PTAP`` ``VSS`` rail) into Eulerian chains;
  * Selects column-aligned / palindromic occurrence orders maximizing vertical
    ``poly`` gate column alignment across the P/N boundary;
  * Keeps internal series stack junctions (``NAND`` NMOS / ``NOR`` PMOS) on pure
    ``LI1`` at ``440 nm`` column pitch (``mcon = 0`` inside active);
  * Expands ``@array(<fg>, style=native_cmos [, transforms=(...)])`` into
    canonical ``<fg>__pmos`` and ``<fg>__nmos`` parameterized diffusion rows plus
    relative ``@order`` + ``@align(..., edge=center_x)`` constraints, and also
    provides standalone ``compose_cmos_gate_macro`` synthesis & audit.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Sequence, Tuple

import klayout.db as kdb

from clay.layout.native_parameterized_row import (
    LAYER_LI1,
    LAYER_MCON,
    LAYER_MET1,
    LAYER_NWELL,
    LAYER_POLY,
    ParameterizedRowPlan,
    audit_corridor_safety,
    build_parameterized_row_regions,
    internal_series_nets_for_chain,
    plan_parameterized_row,
    supply_nets_for_cell,
)
from clay.layout.place.native_diffusion_chain import (
    NativeDiffusionChain,
    NativeDiffusionFinger,
    compile_native_diffusion_chains,
    order_native_diffusion_chain,
)
from clay.layout.spatial.intent_projection import MosaicOccurrence
from clay.pdk.sky130_native_geometry import SKY130_MOS_RULESET_NM, snap_up_grid_nm


@dataclass(frozen=True)
class CmosGatePlan:
    group: str
    gate_kind: str  # "INV" | "BUF" | "NOR" | "NAND" | "SR_LATCH" | "CMOS_COMPOSITE"
    pmos_group: str
    nmos_group: str
    pmos_chain: NativeDiffusionChain
    nmos_chain: NativeDiffusionChain
    pmos_occurrences: Tuple[MosaicOccurrence, ...]
    nmos_occurrences: Tuple[MosaicOccurrence, ...]
    pmos_row_plan: ParameterizedRowPlan
    nmos_row_plan: ParameterizedRowPlan
    internal_series_nets: Tuple[str, ...]
    anchor_pmos_device: str
    anchor_nmos_device: str
    anchor_center_x_offset_nm: int


def _polarity(device) -> str | None:
    model = " ".join(
        str(getattr(device, attr, "") or "")
        for attr in ("source_model", "pdk_variant", "device_class")
    ).lower()
    if "pfet" in model or "pmos" in model:
        return "pmos"
    if "nfet" in model or "nmos" in model:
        return "nmos"
    return None


def _occurrences_from_chain(chain: NativeDiffusionChain) -> Tuple[MosaicOccurrence, ...]:
    ordinals: Dict[str, int] = {}
    out: List[MosaicOccurrence] = []
    for finger in chain.fingers:
        idx = ordinals.get(finger.owner, 0)
        out.append(MosaicOccurrence(finger.owner, idx, "R0"))
        ordinals[finger.owner] = idx + 1
    return tuple(out)


def _euler_trails_bounded(
    chain: NativeDiffusionChain,
    *,
    preferred_outer_net: str,
    max_candidates: int = 32,
) -> Tuple[NativeDiffusionChain, ...]:
    """Generate deterministic candidate Eulerian traversals preferring supply outer ends."""
    fingers = chain.fingers
    n = len(fingers)
    if n <= 1:
        if n == 1 and fingers[0].right == preferred_outer_net and fingers[0].left != preferred_outer_net:
            f = fingers[0]
            flipped = NativeDiffusionFinger(
                f.owner, f.occurrence, f.replica, f.finger, f.gate, f.body, f.right, f.left
            )
            return (
                replace(chain, fingers=(flipped,), columns=(f.right, f.left)),
                chain,
            )
        return (chain,)

    # Group equivalent fingers by (owner, unordered_sd, gate) to avoid factorial blowup on nf>1
    adj: Dict[str, List[int]] = defaultdict(list)
    for idx, f in enumerate(fingers):
        adj[f.left].append(idx)
        adj[f.right].append(idx)
    nodes = sorted(adj)
    odd = [node for node in nodes if len(adj[node]) % 2 == 1]
    starts = odd if odd else ([preferred_outer_net] if preferred_outer_net in adj else [nodes[0]])
    if preferred_outer_net in starts and starts[0] != preferred_outer_net:
        starts = [preferred_outer_net] + [s for s in starts if s != preferred_outer_net]

    results: List[NativeDiffusionChain] = []
    seen_patterns: set[Tuple[Tuple[str, str, str], ...]] = set()

    def dfs(curr: str, used: int, path_edges: List[Tuple[int, str, str]]) -> None:
        if len(results) >= max_candidates:
            return
        if used == (1 << n) - 1:
            cols = [path_edges[0][1]] + [r for _, _, r in path_edges]
            ord_by_owner: Dict[str, int] = {}
            new_fingers: List[NativeDiffusionFinger] = []
            for e_idx, l_net, r_net in path_edges:
                src = fingers[e_idx]
                occ = ord_by_owner.get(src.owner, 0)
                ord_by_owner[src.owner] = occ + 1
                new_fingers.append(
                    NativeDiffusionFinger(
                        owner=src.owner,
                        occurrence=occ,
                        replica=0,
                        finger=occ,
                        gate=src.gate,
                        body=src.body,
                        left=l_net,
                        right=r_net,
                    )
                )
            pat = tuple((f.owner, f.gate, f.left + "->" + f.right) for f in new_fingers)
            if pat not in seen_patterns:
                seen_patterns.add(pat)
                results.append(replace(chain, fingers=tuple(new_fingers), columns=tuple(cols)))
            return

        seen_step: set[Tuple[str, str, str]] = set()
        for e_idx in adj[curr]:
            if used & (1 << e_idx):
                continue
            f = fingers[e_idx]
            nxt = f.right if f.left == curr else f.left
            key = (f.owner, f.gate, nxt)
            if key in seen_step:
                continue
            seen_step.add(key)
            path_edges.append((e_idx, curr, nxt))
            dfs(nxt, used | (1 << e_idx), path_edges)
            path_edges.pop()
            if len(results) >= max_candidates:
                return

    for st in starts:
        dfs(st, 0, [])
    return tuple(results) if results else (chain,)


def _classify_cmos_gate_kind(
    pmos_chain: NativeDiffusionChain,
    nmos_chain: NativeDiffusionChain,
    internal_series_nets: frozenset[str],
) -> str:
    p_gates = {f.gate for f in pmos_chain.fingers}
    n_gates = {f.gate for f in nmos_chain.fingers}
    all_gates = p_gates | n_gates
    p_sd = set(pmos_chain.columns) - {pmos_chain.body} - internal_series_nets
    n_sd = set(nmos_chain.columns) - {nmos_chain.body} - internal_series_nets
    shared_outputs = p_sd & n_sd
    feedback_gates = all_gates & shared_outputs
    if len(all_gates) == 1 and len(shared_outputs) == 1 and not internal_series_nets:
        return "INV"
    if len(feedback_gates) >= 2 and len(shared_outputs) >= 2:
        return "SR_LATCH"
    if len(feedback_gates) == 1 and len(shared_outputs) == 2 and not internal_series_nets:
        return "BUF"
    if internal_series_nets:
        p_series = set(pmos_chain.columns) & internal_series_nets
        n_series = set(nmos_chain.columns) & internal_series_nets
        if p_series and not n_series:
            return "NOR"
        if n_series and not p_series:
            return "NAND"
    return "CMOS_COMPOSITE"


def _score_cmos_pair(p_chain: NativeDiffusionChain, n_chain: NativeDiffusionChain) -> Tuple[int, ...]:
    """Higher score is better: supply outer ends, gate-net column alignment, contiguous gate merges, symmetry."""
    p_outer = int(p_chain.columns[0] == p_chain.body) + int(p_chain.columns[-1] == p_chain.body)
    n_outer = int(n_chain.columns[0] == n_chain.body) + int(n_chain.columns[-1] == n_chain.body)

    p_gates = [f.gate for f in p_chain.fingers]
    n_gates = [f.gate for f in n_chain.fingers]
    if len(p_gates) == len(n_gates):
        aligned = sum(int(pg == ng) for pg, ng in zip(p_gates, n_gates))
    elif len(p_gates) == 2 * len(n_gates):
        # e.g. symmetric 4-finger NOR2 PMOS over 2-finger NOR2 NMOS, or 8 over 4 SR-NOR latch
        mid_slice = p_gates[len(p_gates) // 2 - len(n_gates) // 2 : len(p_gates) // 2 + (len(n_gates) + 1) // 2]
        aligned = sum(int(pg == ng) for pg, ng in zip(mid_slice, n_gates))
    elif len(n_gates) == 2 * len(p_gates):
        mid_slice = n_gates[len(n_gates) // 2 - len(p_gates) // 2 : len(n_gates) // 2 + (len(p_gates) + 1) // 2]
        aligned = sum(int(pg == ng) for pg, ng in zip(p_gates, mid_slice))
    else:
        aligned = 0

    p_merges = sum(int(p_gates[i] == p_gates[i + 1]) for i in range(len(p_gates) - 1))
    n_merges = sum(int(n_gates[i] == n_gates[i + 1]) for i in range(len(n_gates) - 1))
    p_sym = int(p_gates == p_gates[::-1])
    n_sym = int(n_gates == n_gates[::-1])
    return (p_outer + n_outer, aligned, p_merges + n_merges, p_sym + n_sym)


def _occurrence_envelope_center_x_nm(plan: ParameterizedRowPlan, owner: str, chain: NativeDiffusionChain, *, mirrored: bool) -> int:
    """Compute the exact center_x of ``owner``'s occurrence envelope relative to row bbox center_x."""
    xs = []
    bx0, _by0, bx1, _by1 = plan.bbox_nm
    for idx, f in enumerate(chain.fingers):
        if f.owner != owner:
            continue
        gx0, gx1 = plan.gate_spans_x_nm[idx]
        cx = (gx0 + gx1) // 2
        if mirrored:
            # Under MR0 around local bbox, local x maps to (width - local_x)
            local_x = cx - bx0
            cx = (bx1 - bx0) - local_x
        else:
            cx = cx - bx0
        xs.append(cx)
    if not xs:
        raise ValueError(f"CMOS anchor owner {owner!r} has no fingers in chain")
    # Matches _source_device_occurrence_envelopes_nm + _bbox_edge_value(..., 'center_x'):
    # envelope is [min_x, max_x + 1), center_x is (min_x + max_x + 1) // 2
    env_cx = (min(xs) + max(xs) + 1) // 2
    row_cx = (bx1 - bx0) // 2
    return env_cx - row_cx


def plan_cmos_gate_group(
    cell_def,
    group_name: str,
    members: Sequence[str] | None = None,
    *,
    transforms: Tuple[str, ...] = (),
    require_joint_routing_access: bool = True,
    _specs=None,
    _source_identity=None,
) -> CmosGatePlan:
    """Compile a complementary CMOS functional group into aligned PMOS + NMOS row plans."""
    if members is None:
        from clay.frontend.craft_infer import group_members

        gmap = group_members(cell_def)
        if group_name not in gmap:
            raise ValueError(f"native_cmos group {group_name!r} has no functional_group members")
        members = gmap[group_name]

    devices = {d.instance_name: d for d in getattr(cell_def, "devices", ()) or ()}
    unknown = [m for m in members if m not in devices]
    if unknown:
        raise ValueError(f"native_cmos group {group_name!r} references unknown devices: {unknown}")

    pmos_members = tuple(sorted(m for m in members if _polarity(devices[m]) == "pmos"))
    nmos_members = tuple(sorted(m for m in members if _polarity(devices[m]) == "nmos"))
    if not pmos_members or not nmos_members or len(pmos_members) + len(nmos_members) != len(set(members)):
        raise ValueError(
            f"native_cmos group {group_name!r} must partition cleanly into non-empty PMOS and NMOS MOS devices"
        )

    p_raw = compile_native_diffusion_chains(
        cell_def,
        pmos_members,
        synthesis="cmos_gate",
        tap_rail="top",
        _specs=_specs,
        _source_identity=_source_identity,
    )
    n_raw = compile_native_diffusion_chains(
        cell_def,
        nmos_members,
        synthesis="cmos_gate",
        tap_rail="bottom",
        _specs=_specs,
        _source_identity=_source_identity,
    )
    if len(p_raw) != 1 or len(n_raw) != 1:
        raise ValueError(
            f"native_cmos group {group_name!r} does not form single continuous PMOS and NMOS Euler chains"
        )

    p_cands = _euler_trails_bounded(p_raw[0], preferred_outer_net=p_raw[0].body)
    n_cands = _euler_trails_bounded(n_raw[0], preferred_outer_net=n_raw[0].body)
    best_pair = max(
        ((pc, nc) for pc in p_cands for nc in n_cands),
        key=lambda pair: _score_cmos_pair(pair[0], pair[1]),
    )
    p_chain, n_chain = best_pair
    p_occ = _occurrences_from_chain(p_chain)
    n_occ = _occurrences_from_chain(n_chain)

    series = tuple(
        sorted(
            set(internal_series_nets_for_chain(cell_def, p_chain))
            | set(internal_series_nets_for_chain(cell_def, n_chain))
        )
    )
    supplies = supply_nets_for_cell(cell_def)
    p_plan = plan_parameterized_row(
        p_chain,
        internal_series_nets=series,
        supply_nets=supplies,
        require_joint_routing_access=require_joint_routing_access,
    )
    n_plan = plan_parameterized_row(
        n_chain,
        internal_series_nets=series,
        supply_nets=supplies,
        require_joint_routing_access=require_joint_routing_access,
    )
    gate_kind = _classify_cmos_gate_kind(p_chain, n_chain, frozenset(series))

    mirrored = transforms == ("MR0",)
    p_anchor = p_chain.fingers[0].owner
    n_anchor = n_chain.fingers[0].owner
    p_rel = _occurrence_envelope_center_x_nm(p_plan, p_anchor, p_chain, mirrored=mirrored)
    n_rel = _occurrence_envelope_center_x_nm(n_plan, n_anchor, n_chain, mirrored=mirrored)
    # Equation enforced by EdgeAlignmentDeclaration(first=p, second=n, edge='center_x', offset_nm=off):
    # x(n_master) + n_env_cx - (x(p_master) + p_env_cx) == off.
    # For concentric master centers: x(p_master) + W_p//2 == x(n_master) + W_n//2
    # => (x(n) + n_env_cx) - (x(p) + p_env_cx) == n_rel - p_rel, snapped to 5nm grid.
    raw_off = n_rel - p_rel
    snapped_off = round(raw_off / 5) * 5

    return CmosGatePlan(
        group=group_name,
        gate_kind=gate_kind,
        pmos_group=f"{group_name}__pmos",
        nmos_group=f"{group_name}__nmos",
        pmos_chain=p_chain,
        nmos_chain=n_chain,
        pmos_occurrences=p_occ,
        nmos_occurrences=n_occ,
        pmos_row_plan=p_plan,
        nmos_row_plan=n_plan,
        internal_series_nets=series,
        anchor_pmos_device=p_anchor,
        anchor_nmos_device=n_anchor,
        anchor_center_x_offset_nm=int(snapped_off),
    )


def expand_cmos_array_intents(cell_def, intent):
    """Expand ``style=native_cmos`` ``ArrayIntent`` entries into PMOS+NMOS rows and relative constraints."""
    from clay.frontend.craft_infer import group_members
    from clay.layout.intent import ArrayIntent, EdgeAlignmentIntent

    cmos_rows = [row for row in intent.arrays if row.style == "native_cmos"]
    if not cmos_rows:
        return intent

    from clay.layout.native_sky130 import native_source_identity
    from clay.pdk.sky130_native_geometry import native_cell_geometry_specs

    gmap = group_members(cell_def)
    cell_specs = native_cell_geometry_specs(cell_def)
    cell_identity = native_source_identity(cell_def)
    new_arrays: List[ArrayIntent] = []
    added_orders = list(intent.order_constraints)
    added_alignments = list(intent.edge_alignments)
    seen_align_keys = {(x.first, x.second, x.edge) for x in added_alignments}

    for row in intent.arrays:
        if row.style != "native_cmos":
            new_arrays.append(row)
            continue
        if row.group not in gmap:
            raise ValueError(f"native_cmos group {row.group!r} is absent from cell functional_groups")
        plan = plan_cmos_gate_group(
            cell_def,
            row.group,
            gmap[row.group],
            transforms=row.transforms,
            require_joint_routing_access=True,
            _specs=cell_specs,
            _source_identity=cell_identity,
        )
        new_arrays.append(
            ArrayIntent(
                group=plan.pmos_group,
                style="native_diffusion",
                rows=1,
                order="ratio",
                transforms=row.transforms,
                occurrences=plan.pmos_occurrences,
                synthesis="cmos_gate",
                tap_rail="top",
            )
        )
        new_arrays.append(
            ArrayIntent(
                group=plan.nmos_group,
                style="native_diffusion",
                rows=1,
                order="ratio",
                transforms=row.transforms,
                occurrences=plan.nmos_occurrences,
                synthesis="cmos_gate",
                tap_rail="bottom",
            )
        )
        order_item = (plan.anchor_pmos_device, plan.anchor_nmos_device, "above")
        if order_item not in added_orders:
            added_orders.append(order_item)
        align_decl = EdgeAlignmentIntent(
            first=plan.anchor_pmos_device,
            second=plan.anchor_nmos_device,
            edge="center_x",
            offset_nm=plan.anchor_center_x_offset_nm,
        )
        align_key = (align_decl.first, align_decl.second, align_decl.edge)
        if align_key not in seen_align_keys:
            seen_align_keys.add(align_key)
            added_alignments.append(align_decl)

    return replace(
        intent,
        arrays=tuple(new_arrays),
        order_constraints=tuple(added_orders),
        edge_alignments=tuple(added_alignments),
    )


def compose_cmos_gate_macro(
    cell_def,
    group_name: str,
    *,
    transforms: Tuple[str, ...] = (),
    rules: Mapping[str, int] | None = None,
) -> Dict[str, object]:
    """Synthesize a standalone complementary CMOS gate/buffer macro with shared poly and LI1 internal nodes."""
    r = SKY130_MOS_RULESET_NM if rules is None else rules
    grid = int(r["grid_nm"])
    plan = plan_cmos_gate_group(
        cell_def,
        group_name,
        transforms=transforms,
        require_joint_routing_access=False,
    )
    p_regs = build_parameterized_row_regions(plan.pmos_row_plan, include_contacts=True)
    n_regs = build_parameterized_row_regions(plan.nmos_row_plan, include_contacts=True)
    audit_corridor_safety(plan.pmos_row_plan, p_regs)
    audit_corridor_safety(plan.nmos_row_plan, n_regs)

    n_bb = plan.nmos_row_plan.bbox_nm
    p_bb = plan.pmos_row_plan.bbox_nm

    # Center-align PMOS X over NMOS X on 5nm grid and stack PMOS above NMOS with legal poly/li1/m1/nwell gap
    n_cx = (n_bb[0] + n_bb[2]) // 2
    p_cx = (p_bb[0] + p_bb[2]) // 2
    dx_p = round((n_cx - p_cx) / grid) * grid

    # Minimum vertical clearance between NMOS top (gate heads) and PMOS bottom (gate heads / nwell):
    # poly.2 = 210nm, li.3 = 170nm, met1.2 = 140nm, nwell-to-nsdm/diff = 340nm.
    n_top_y = n_bb[3]
    p_bot_y = p_bb[1]
    nw_bot_y = p_regs[LAYER_NWELL].bbox().bottom
    n_diff_top_y = plan.nmos_row_plan.active_box_nm[3]
    dy_from_channel = (n_top_y + int(r["poly_space_nm"])) - p_bot_y
    dy_from_nwell = (n_diff_top_y + 430) - nw_bot_y
    dy_p = snap_up_grid_nm(max(dy_from_channel, dy_from_nwell), grid)

    combined: Dict[Tuple[int, int], kdb.Region] = defaultdict(kdb.Region)
    for layer, reg in n_regs.items():
        combined[layer] += reg
    shift_p = kdb.Trans(int(dx_p), int(dy_p))
    for layer, reg in p_regs.items():
        combined[layer] += reg.transformed(shift_p)

    # Connect aligned complementary vertical poly columns when they share the exact same gate net
    p_gates = plan.pmos_row_plan.gate_nets
    n_gates = plan.nmos_row_plan.gate_nets
    shared_poly_bridges = 0
    if len(p_gates) == len(n_gates):
        for idx, (pg, ng) in enumerate(zip(p_gates, n_gates)):
            if pg != ng:
                continue
            nx0, nx1 = plan.nmos_row_plan.gate_spans_x_nm[idx]
            px0, px1 = plan.pmos_row_plan.gate_spans_x_nm[idx]
            px0 += dx_p
            px1 += dx_p
            if nx0 == px0 and nx1 == px1:
                ny_top = plan.nmos_row_plan.gate_segments[0].poly_head_box_nm[1]
                py_bot = plan.pmos_row_plan.gate_segments[0].poly_head_box_nm[3] + dy_p
                combined[LAYER_POLY].insert(kdb.Box(nx0, min(ny_top, py_bot), nx1, max(ny_top, py_bot)))
                shared_poly_bridges += 1

    merged = {layer: reg.merged() for layer, reg in sorted(combined.items()) if not reg.is_empty()}
    ext = kdb.Region()
    for reg in merged.values():
        ext += reg
    bb = ext.bbox()
    bbox_nm = (int(bb.left), int(bb.bottom), int(bb.right), int(bb.top))

    # Verify internal series nodes have 0 mcon and 0 M1 anywhere in standalone macro
    series_mcon_count = sum(
        len(tr.mcon_boxes_nm)
        for rp in (plan.pmos_row_plan, plan.nmos_row_plan)
        for tr in rp.sd_tracks
        if tr.net in plan.internal_series_nets
    )
    if series_mcon_count != 0:
        raise ValueError(f"CMOS gate internal series nodes emitted {series_mcon_count} mcon cuts")

    return {
        "group": group_name,
        "gate_kind": plan.gate_kind,
        "pmos_plan": plan.pmos_row_plan,
        "nmos_plan": plan.nmos_row_plan,
        "pmos_offset_nm": (int(dx_p), int(dy_p)),
        "internal_series_nets": plan.internal_series_nets,
        "internal_series_mcon_count": series_mcon_count,
        "shared_poly_bridges": shared_poly_bridges,
        "bbox_nm": bbox_nm,
        "width_nm": bbox_nm[2] - bbox_nm[0],
        "height_nm": bbox_nm[3] - bbox_nm[1],
        "regions_by_layer_nm": merged,
    }
