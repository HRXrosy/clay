"""Conservative multilayer KLayout Region reachability pre-check, peer-only
escape-mouth lookahead, and static terminal escapability gate for GridlessEngine.
"""
from __future__ import annotations

import math
import time
import klayout.db as db


def _insert_strict_violation_boxes(
    excluded: db.Region,
    rect: tuple[int, int, int, int],
    half: int,
    sp: int,
) -> None:
    """Insert ONLY integer centerline points (x, y) as unit squares [x, x+1]x[y, y+1]
    whose candidate box [x-half, y-half, x+half, y+half] STRICTLY violates Euclidean
    spacing `sp` (`sx*sx + sy*sy < sp*sp`, or `sx==0 and sy==0` when `sp<=0`) against `rect`.
    """
    rx0, ry0, rx1, ry1 = rect
    if rx0 > rx1:
        rx0, rx1 = rx1, rx0
    if ry0 > ry1:
        ry0, ry1 = ry1, ry0
    if sp <= 0:
        excluded.insert(db.Box(rx0 - half, ry0 - half, rx1 + half + 1, ry1 + half + 1))
        return
    excluded.insert(
        db.Box(rx0 - half - sp + 1, ry0 - half, rx1 + half + sp, ry1 + half + 1)
    )
    excluded.insert(
        db.Box(rx0 - half, ry0 - half - sp + 1, rx1 + half + 1, ry1 + half + sp)
    )
    diag = math.isqrt((sp * sp - 1) // 2)
    if diag > 0:
        excluded.insert(
            db.Box(
                rx0 - half - diag,
                ry0 - half - diag,
                rx1 + half + diag + 1,
                ry1 + half + diag + 1,
            )
        )


def _polygon_has_mfg_grid_point(poly: db.Polygon, grid: int) -> bool:
    if grid <= 1:
        return True
    box = poly.bbox()
    gx0 = -(-box.left // grid) * grid
    gx1 = ((box.right - 1) // grid) * grid
    gy0 = -(-box.bottom // grid) * grid
    gy1 = ((box.top - 1) // grid) * grid
    if gx0 > gx1 or gy0 > gy1:
        return False
    for trap in poly.decompose_trapezoids(db.Polygon.TD_vtrapezoids):
        tb = trap.bbox()
        tx0 = -(-tb.left // grid) * grid
        tx1 = ((tb.right - 1) // grid) * grid
        ty0 = -(-tb.bottom // grid) * grid
        ty1 = ((tb.top - 1) // grid) * grid
        if tx0 <= tx1 and ty0 <= ty1:
            return True
    return False


def _build_conservative_free_regions(
    engine,
    net: str,
    ordered_layers: list[int],
    bounds: tuple[int, int, int, int],
    *,
    min_width_only: bool = False,
    static_only: bool = False,
) -> tuple[dict[int, db.Region], dict[int, db.Region]]:
    bx0, by0, bx1, by1 = bounds
    if bx0 > bx1:
        bx0, bx1 = bx1, bx0
    if by0 > by1:
        by0, by1 = by1, by0
    envelope = getattr(engine, "_routing_envelope", None)
    env_bbox = envelope.bbox_nm if envelope is not None else None
    local_closure = bool(getattr(engine, "_active_group_route_local_closure", False))
    local_iface = (
        getattr(engine, "_active_group_route_local_interface", None)
        if local_closure
        else None
    )
    neg_legalize = bool(getattr(engine, "_negotiation_port_legalize", False))
    grid = max(1, int(getattr(engine, "_manufacturing_grid_nm", 1)))
    layer_set = set(ordered_layers)
    wide_map = engine._wide_indexes(not static_only)

    F_wire: dict[int, db.Region] = {}
    for gi in ordered_layers:
        base_hw = int(engine._half_w[gi])
        half = base_hw if min_width_only else int(engine._hw(net, gi))
        sp = int(engine._space[gi])
        lx0, ly0, lx1, ly1 = bx0, by0, bx1, by1
        if env_bbox is not None:
            lx0 = max(lx0, env_bbox[0] + half)
            ly0 = max(ly0, env_bbox[1] + half)
            lx1 = min(lx1, env_bbox[2] - half)
            ly1 = min(ly1, env_bbox[3] - half)
        if local_iface is not None:
            ex0, ey0, ex1, ey1 = (int(v) for v in local_iface[0])
            px, py = int(local_iface[1][0]), int(local_iface[1][1])
            lx0 = max(lx0, ex0)
            ly0 = max(ly0, ey0)
            lx1 = min(lx1, ex1)
            ly1 = min(ly1, ey1)
            if lx0 > lx1 or ly0 > ly1:
                F_wire[gi] = db.Region()
                continue
            outer = db.Region()
            ix0 = max(lx0, ex0 + half)
            iy0 = max(ly0, ey0 + half)
            ix1 = min(lx1, ex1 - half)
            iy1 = min(ly1, ey1 - half)
            if ix0 <= ix1 and iy0 <= iy1:
                outer.insert(db.Box(ix0, iy0, ix1 + 1, iy1 + 1))
            sx0 = max(lx0, ex0 if px == ex0 else ex0 + half, px - half)
            sy0 = max(ly0, ey0 if py == ey0 else ey0 + half, py - half)
            sx1 = min(lx1, ex1 if px == ex1 else ex1 - half, px + half)
            sy1 = min(ly1, ey1 if py == ey1 else ey1 - half, py + half)
            if sx0 <= sx1 and sy0 <= sy1:
                outer.insert(db.Box(sx0, sy0, sx1 + 1, sy1 + 1))
            outer.merge()
            if outer.is_empty():
                F_wire[gi] = db.Region()
                continue
        else:
            if lx0 > lx1 or ly0 > ly1:
                F_wire[gi] = db.Region()
                continue
            outer = db.Region(db.Box(lx0, ly0, lx1 + 1, ly1 + 1))
        excl = db.Region()
        q = (
            lx0 - half - sp - 5,
            ly0 - half - sp - 5,
            lx1 + half + sp + 5,
            ly1 + half + sp + 5,
        )
        obs_idx = getattr(engine, "_obs_idx", {}).get(gi)
        if obs_idx is not None:
            for owner, rect in obs_idx.query(q):
                if owner != net:
                    _insert_strict_violation_boxes(excl, rect, half, sp)
        if not static_only:
            for owner, rect in engine._book.via_pads.get(gi, ()):
                if owner != net:
                    if (
                        rect[2] >= q[0]
                        and rect[0] <= q[2]
                        and rect[3] >= q[1]
                        and rect[1] <= q[3]
                    ):
                        _insert_strict_violation_boxes(excl, rect, half, sp)
            for owner, rect in engine._book.metal.get(gi, ()):
                if owner != net:
                    if (
                        rect[2] >= q[0]
                        and rect[0] <= q[2]
                        and rect[3] >= q[1]
                        and rect[1] <= q[3]
                    ):
                        pers_hard = getattr(
                            engine, "_persistent_metal_requires_hard_space", None
                        )
                        enforce_sp = (
                            (
                                bool(pers_hard(gi, owner, rect))
                                if callable(pers_hard)
                                else (
                                    owner in engine._book.persistent
                                    and (local_closure or neg_legalize)
                                )
                            )
                            or half > base_hw
                            or (
                                owner is not None
                                and int(engine._hw(str(owner), gi)) > base_hw
                            )
                        )
                        _insert_strict_violation_boxes(
                            excl, rect, half, sp if enforce_sp else 0
                        )
        for wspace, core, _win, _attached in wide_map.get(gi, ()):
            wq = (
                lx0 - half - int(wspace) - 5,
                ly0 - half - int(wspace) - 5,
                lx1 + half + int(wspace) + 5,
                ly1 + half + int(wspace) + 5,
            )
            for owner, rect in core.query(wq):
                if owner != net:
                    _insert_strict_violation_boxes(excl, rect, half, int(wspace))
        free = outer - excl.merged()
        free.min_coherence = True
        free.merge()
        free.min_coherence = True
        F_wire[gi] = free

    F_via: dict[int, db.Region] = {}
    use_single_via = bool(
        getattr(engine, "_single_via_escape_allowed", lambda: False)()
    )
    for lo in ordered_layers:
        if (lo + 1) not in layer_set or lo not in engine._via_pad_half:
            continue
        bh, ah = (
            engine._via_pad_half1[lo]
            if use_single_via
            else engine._via_pad_half[lo]
        )
        bh, ah = int(bh), int(ah)
        ext = int(engine._via_cut_extent(lo, 1 if use_single_via else None))
        vsp = int(engine._via_space[lo])
        max_h = max(bh, ah, ext)
        vx0, vy0, vx1, vy1 = bx0, by0, bx1, by1
        if env_bbox is not None:
            vx0 = max(vx0, env_bbox[0] + max_h)
            vy0 = max(vy0, env_bbox[1] + max_h)
            vx1 = min(vx1, env_bbox[2] - max_h)
            vy1 = min(vy1, env_bbox[3] - max_h)
        if local_iface is not None:
            ex0, ey0, ex1, ey1 = (int(v) for v in local_iface[0])
            vx0 = max(vx0, ex0 + max_h)
            vy0 = max(vy0, ey0 + max_h)
            vx1 = min(vx1, ex1 - max_h)
            vy1 = min(vy1, ey1 - max_h)
        if vx0 > vx1 or vy0 > vy1:
            F_via[lo] = db.Region()
            continue
        v_outer = db.Region(db.Box(vx0, vy0, vx1 + 1, vy1 + 1))
        v_excl = db.Region()
        for gi, ph in ((lo, bh), (lo + 1, ah)):
            sp = int(engine._space[gi])
            vq = (
                vx0 - ph - sp - 5,
                vy0 - ph - sp - 5,
                vx1 + ph + sp + 5,
                vy1 + ph + sp + 5,
            )
            obs_idx = getattr(engine, "_obs_idx", {}).get(gi)
            if obs_idx is not None:
                for owner, rect in obs_idx.query(vq):
                    if owner != net:
                        _insert_strict_violation_boxes(v_excl, rect, ph, sp)
            if not static_only:
                for rows in (
                    engine._book.metal.get(gi, ()),
                    engine._book.via_pads.get(gi, ()),
                ):
                    for owner, rect in rows:
                        if owner != net:
                            if (
                                rect[2] >= vq[0]
                                and rect[0] <= vq[2]
                                and rect[3] >= vq[1]
                                and rect[1] <= vq[3]
                            ):
                                _insert_strict_violation_boxes(v_excl, rect, ph, sp)
            for wspace, core, _win, _attached in wide_map.get(gi, ()):
                wq = (
                    vx0 - ph - int(wspace) - 5,
                    vy0 - ph - int(wspace) - 5,
                    vx1 + ph + int(wspace) + 5,
                    vy1 + ph + int(wspace) + 5,
                )
                for owner, rect in core.query(wq):
                    if owner != net:
                        _insert_strict_violation_boxes(v_excl, rect, ph, int(wspace))
        cq = (
            vx0 - ext - vsp - 5,
            vy0 - ext - vsp - 5,
            vx1 + ext + vsp + 5,
            vy1 + ext + vsp + 5,
        )
        if not static_only:
            cut_idx = engine._cut_idx.get(lo)
            if cut_idx is not None:
                for _owner, rect in cut_idx.query(cq):
                    _insert_strict_violation_boxes(v_excl, rect, ext, vsp)
            for rect in getattr(engine, "_attempt_cuts", {}).get(lo, ()):
                _insert_strict_violation_boxes(v_excl, rect, ext, vsp)
        for rect in engine._local_fixed_cuts(lo, cq):
            if (rect[2] - rect[0]) != 2 * ext or (rect[3] - rect[1]) != 2 * ext:
                _insert_strict_violation_boxes(v_excl, rect, ext, vsp)
        raw_v = v_outer - v_excl.merged()
        raw_v.min_coherence = True
        raw_v.merge()
        raw_v.min_coherence = True
        if grid > 1 and not raw_v.is_empty():
            filtered = db.Region()
            for poly in raw_v.each():
                if _polygon_has_mfg_grid_point(poly, grid):
                    filtered.insert(poly)
            filtered.min_coherence = True
            F_via[lo] = filtered
        else:
            F_via[lo] = raw_v

    return F_wire, F_via


def fast_planar_via_precheck(
    engine,
    starts: list,
    goals: dict,
    net: str,
    ordered_layers: list[int],
    target_domain,
    effective_target_regions: tuple,
    *,
    xs: list[int] = (),
    ys: list[int] = (),
    source_domain=None,
    effective_source_regions: tuple = (),
    harvest_samples_on_fail: bool = True,
) -> tuple[bool, float]:
    """Return (is_potentially_reachable, elapsed_ms) with zero false positives."""
    t0 = time.perf_counter()
    if not starts or (not goals and not effective_target_regions):
        return False, (time.perf_counter() - t0) * 1000.0

    bx0, by0, bx1, by1 = engine._bounds
    if bx0 > bx1:
        bx0, bx1 = bx1, bx0
    if by0 > by1:
        by0, by1 = by1, by0
    if xs:
        bx0, bx1 = min(bx0, min(xs)), max(bx1, max(xs))
    if ys:
        by0, by1 = min(by0, min(ys)), max(by1, max(ys))
    for (sx, sy, _sgi), _v in starts:
        bx0, by0 = min(bx0, sx), min(by0, sy)
        bx1, by1 = max(bx1, sx), max(by1, sy)
    for gx, gy, _ggi in goals:
        bx0, by0 = min(bx0, gx), min(by0, gy)
        bx1, by1 = max(bx1, gx), max(by1, gy)
    for _atom, (rx0, ry0, rx1, ry1) in effective_target_regions:
        bx0, by0 = min(bx0, rx0), min(by0, ry0)
        bx1, by1 = max(bx1, rx1), max(by1, ry1)
    for _atom, (rx0, ry0, rx1, ry1) in effective_source_regions:
        bx0, by0 = min(bx0, rx0), min(by0, ry0)
        bx1, by1 = max(bx1, rx1), max(by1, ry1)

    grid = max(1, int(getattr(engine, "_manufacturing_grid_nm", 1)))
    F_wire, F_via = _build_conservative_free_regions(
        engine, net, ordered_layers, (bx0, by0, bx1, by1)
    )

    reached_pts = {gi: db.Region() for gi in ordered_layers}
    reached_wire = {gi: db.Region() for gi in ordered_layers}
    for (sx, sy, sgi), _v in starts:
        if sgi in reached_pts:
            reached_pts[sgi].insert(db.Box(sx, sy, sx + 1, sy + 1))
    if effective_source_regions and source_domain is not None:
        sgi = source_domain.node[2]
        if sgi in reached_pts:
            for _atom, (rx0, ry0, rx1, ry1) in effective_source_regions:
                reached_pts[sgi].insert(db.Box(rx0, ry0, rx1 + 1, ry1 + 1))
    for gi in ordered_layers:
        if not reached_pts[gi].is_empty():
            reached_pts[gi].merge()
            w = F_wire[gi].interacting(reached_pts[gi])
            w.min_coherence = True
            reached_wire[gi] = w

    goal_reg = {gi: db.Region() for gi in ordered_layers}
    for gx, gy, ggi in goals:
        if ggi in goal_reg:
            goal_reg[ggi].insert(db.Box(gx, gy, gx + 1, gy + 1))
    if effective_target_regions and target_domain is not None:
        tgi = target_domain.node[2]
        if tgi in goal_reg:
            for _atom, (rx0, ry0, rx1, ry1) in effective_target_regions:
                goal_reg[tgi].insert(db.Box(rx0, ry0, rx1 + 1, ry1 + 1))
    for gi in ordered_layers:
        if not goal_reg[gi].is_empty():
            goal_reg[gi].merge()

    used_via = {
        (src_gi, dst_gi): db.Region()
        for lo in F_via
        for src_gi, dst_gi in ((lo, lo + 1), (lo + 1, lo))
    }
    while True:
        for gi in ordered_layers:
            if goal_reg[gi].is_empty():
                continue
            cur = reached_wire[gi] + reached_pts[gi]
            if not cur.is_empty() and not (cur & goal_reg[gi]).is_empty():
                return True, (time.perf_counter() - t0) * 1000.0
        progressed = False
        for lo, f_v in F_via.items():
            if f_v.is_empty():
                continue
            for src_gi, dst_gi in ((lo, lo + 1), (lo + 1, lo)):
                src = reached_wire[src_gi] + reached_pts[src_gi]
                if src.is_empty():
                    continue
                v_cand = (f_v & src) - used_via[(src_gi, dst_gi)]
                if v_cand.is_empty():
                    continue
                v_cand.merge()
                if grid > 1:
                    v_hit = db.Region()
                    for poly in v_cand.each():
                        if _polygon_has_mfg_grid_point(poly, grid):
                            v_hit.insert(poly)
                    if v_hit.is_empty():
                        used_via[(src_gi, dst_gi)] = (
                            used_via[(src_gi, dst_gi)] + v_cand
                        ).merged()
                        continue
                else:
                    v_hit = v_cand
                v_hit.min_coherence = True
                used_via[(src_gi, dst_gi)] = (
                    used_via[(src_gi, dst_gi)] + v_cand
                ).merged()
                reached_pts[dst_gi] = (reached_pts[dst_gi] + v_hit).merged()
                cand_w = F_wire[dst_gi] - reached_wire[dst_gi]
                if not cand_w.is_empty():
                    cand_w.min_coherence = True
                    nw = cand_w.interacting(v_hit)
                    if not nw.is_empty():
                        nw.min_coherence = True
                        reached_wire[dst_gi] = (reached_wire[dst_gi] + nw).merged()
                        reached_wire[dst_gi].min_coherence = True
                progressed = True
        if not progressed:
            break

    if harvest_samples_on_fail:
        def _multilayer_reach(seed_pts: dict[int, db.Region]) -> dict[int, db.Region]:
            r_pts = {gi: seed_pts[gi].dup() for gi in ordered_layers}
            r_wire = {gi: db.Region() for gi in ordered_layers}
            for gi in ordered_layers:
                if not r_pts[gi].is_empty():
                    r_pts[gi].merge()
                    w = F_wire[gi].interacting(r_pts[gi])
                    w.min_coherence = True
                    r_wire[gi] = w
            u_via = {
                (s_gi, d_gi): db.Region()
                for lo in F_via
                for s_gi, d_gi in ((lo, lo + 1), (lo + 1, lo))
            }
            while True:
                prog = False
                for lo, f_v in F_via.items():
                    if f_v.is_empty():
                        continue
                    for s_gi, d_gi in ((lo, lo + 1), (lo + 1, lo)):
                        s_cur = r_wire[s_gi] + r_pts[s_gi]
                        if s_cur.is_empty():
                            continue
                        vc = (f_v & s_cur) - u_via[(s_gi, d_gi)]
                        if vc.is_empty():
                            continue
                        vc.merge()
                        if grid > 1:
                            vh = db.Region()
                            for poly in vc.each():
                                if _polygon_has_mfg_grid_point(poly, grid):
                                    vh.insert(poly)
                            if vh.is_empty():
                                u_via[(s_gi, d_gi)] = (
                                    u_via[(s_gi, d_gi)] + vc
                                ).merged()
                                continue
                        else:
                            vh = vc
                        vh.min_coherence = True
                        u_via[(s_gi, d_gi)] = (u_via[(s_gi, d_gi)] + vc).merged()
                        r_pts[d_gi] = (r_pts[d_gi] + vh).merged()
                        cw = F_wire[d_gi] - r_wire[d_gi]
                        if not cw.is_empty():
                            cw.min_coherence = True
                            nw = cw.interacting(vh)
                            if not nw.is_empty():
                                nw.min_coherence = True
                                r_wire[d_gi] = (r_wire[d_gi] + nw).merged()
                                r_wire[d_gi].min_coherence = True
                        prog = True
                if not prog:
                    break
            return {
                gi: (r_wire[gi] + r_pts[gi]).merged()
                for gi in ordered_layers
                if not (r_wire[gi] + r_pts[gi]).is_empty()
            }

        start_comp = {
            gi: (reached_wire[gi] + reached_pts[gi]).merged()
            for gi in ordered_layers
            if not (reached_wire[gi] + reached_pts[gi]).is_empty()
        }
        goal_comp = _multilayer_reach(goal_reg)
        start_area = sum(r.area() for r in start_comp.values())
        goal_area = sum(r.area() for r in goal_comp.values())
        ordered_comps = (
            [goal_comp, start_comp]
            if (0 < goal_area <= start_area or start_area == 0)
            else [start_comp, goal_comp]
        )

        def _has_committed_rejection() -> bool:
            return any(
                str(cat).startswith("committed_")
                for cat, _owner in getattr(engine, "_hard_rejections", {}).keys()
            )

        for comp in ordered_comps:
            for gi, reg in comp.items():
                if reg.is_empty():
                    continue
                bb = reg.bbox()
                half = int(engine._hw(net, gi))
                sp = int(engine._space[gi])
                reg_grown = reg.sized(grid)
                wq = (
                    bb.left - half - sp - grid - 5,
                    bb.bottom - half - sp - grid - 5,
                    bb.right + half + sp + grid + 5,
                    bb.top + half + sp + grid + 5,
                )
                for rows in (
                    engine._book.via_pads.get(gi, ()),
                    engine._book.metal.get(gi, ()),
                ):
                    for owner, rect in rows:
                        if (
                            owner != net
                            and rect[2] >= wq[0]
                            and rect[0] <= wq[2]
                            and rect[3] >= wq[1]
                            and rect[1] <= wq[3]
                        ):
                            v_reg = db.Region()
                            _insert_strict_violation_boxes(v_reg, rect, half, sp)
                            hit = reg_grown & v_reg
                            if not hit.is_empty():
                                for poly in hit.each():
                                    for trap in poly.decompose_trapezoids(
                                        db.Polygon.TD_vtrapezoids
                                    ):
                                        tb = trap.bbox()
                                        mx = -(-tb.left // grid) * grid
                                        my = -(-tb.bottom // grid) * grid
                                        if mx < tb.right and my < tb.top:
                                            wb = (
                                                mx - half,
                                                my - half,
                                                mx + half,
                                                my + half,
                                            )
                                            engine._reject_foreign_committed_metal_spacing(
                                                gi,
                                                wb,
                                                net,
                                                candidate_half=half,
                                                require_supra_minimum_endpoint=True,
                                            )
                                            engine._reject_foreign_committed_metal_contact(
                                                gi,
                                                wb,
                                                net,
                                                candidate_half=half,
                                            )
                                            break
                for lo in (gi, gi - 1):
                    if lo not in F_via:
                        continue
                    bh, ah = engine._via_pad_half[lo]
                    bh, ah = int(bh), int(ah)
                    ext = int(engine._via_cut_extent(lo, None))
                    vsp = int(engine._via_space[lo])
                    cand_choke = db.Region()
                    for m_gi, ph in ((lo, bh), (lo + 1, ah)):
                        msp = int(engine._space[m_gi])
                        mq = (
                            bb.left - ph - msp - 5,
                            bb.bottom - ph - msp - 5,
                            bb.right + ph + msp + 5,
                            bb.top + ph + msp + 5,
                        )
                        for rows in (
                            engine._book.via_pads.get(m_gi, ()),
                            engine._book.metal.get(m_gi, ()),
                        ):
                            for owner, rect in rows:
                                if (
                                    owner != net
                                    and rect[2] >= mq[0]
                                    and rect[0] <= mq[2]
                                    and rect[3] >= mq[1]
                                    and rect[1] <= mq[3]
                                ):
                                    _insert_strict_violation_boxes(
                                        cand_choke, rect, ph, msp
                                    )
                    cq = (
                        bb.left - ext - vsp - 5,
                        bb.bottom - ext - vsp - 5,
                        bb.right + ext + vsp + 5,
                        bb.top + ext + vsp + 5,
                    )
                    cut_idx = engine._cut_idx.get(lo)
                    if cut_idx is not None:
                        for owner, rect in cut_idx.query(cq):
                            if owner != net:
                                _insert_strict_violation_boxes(
                                    cand_choke, rect, ext, vsp
                                )
                    choked = reg & cand_choke.merged()
                    if not choked.is_empty():
                        probed_v = 0
                        for poly in choked.each():
                            for trap in poly.decompose_trapezoids(
                                db.Polygon.TD_vtrapezoids
                            ):
                                tb = trap.bbox()
                                px = -(-tb.left // grid) * grid
                                py = -(-tb.bottom // grid) * grid
                                if px < tb.right and py < tb.top:
                                    if engine._evaluate_via_static_legality(
                                        px, py, lo, net, None
                                    ):
                                        engine._via_legal(px, py, lo, net)
                                        probed_v += 1
                                        if probed_v >= 16:
                                            break
                            if probed_v >= 16:
                                break
                        if probed_v == 0:
                            for poly in choked.each():
                                for trap in poly.decompose_trapezoids(
                                    db.Polygon.TD_vtrapezoids
                                ):
                                    tb = trap.bbox()
                                    px = -(-tb.left // grid) * grid
                                    py = -(-tb.bottom // grid) * grid
                                    if px < tb.right and py < tb.top:
                                        for m_gi, ph in ((lo, bh), (lo + 1, ah)):
                                            pad = (px - ph, py - ph, px + ph, py + ph)
                                            engine._reject_committed_via_pad_spacing(
                                                m_gi, pad, net, candidate_half=ph
                                            )
                                            engine._reject_foreign_committed_metal_contact(
                                                m_gi, pad, net, candidate_half=ph
                                            )
                                        probed_v += 1
                                        if probed_v >= 8:
                                            break
                                if probed_v >= 8:
                                    break
            if _has_committed_rejection():
                break

        if not getattr(engine, "_hard_rejections", None):
            for comp in ordered_comps:
                for gi, reg in comp.items():
                    if not reg.is_empty():
                        bb = reg.bbox()
                        mid_x = ((bb.left + bb.right - 1) // (2 * grid)) * grid
                        mid_y = ((bb.bottom + bb.top - 1) // (2 * grid)) * grid
                        if gi in F_via:
                            engine._via_legal(mid_x, mid_y, gi, net)
                        engine._wire_legal(gi, mid_x, mid_y, mid_x + grid, mid_y, net)
                        break
                if getattr(engine, "_hard_rejections", None):
                    break

    return False, (time.perf_counter() - t0) * 1000.0


def _add_cells(
    target: dict,
    rc: int,
    resource_gi: int,
    box: tuple[int, int, int, int],
    penalty: float,
) -> None:
    x0, y0, x1, y1 = box
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    rgi = int(resource_gi)
    val = float(penalty)
    for cx in range(x0 // rc, x1 // rc + 1):
        for cy in range(y0 // rc, y1 // rc + 1):
            key = (cx, cy, rgi)
            if val > target.get(key, 0.0):
                target[key] = val


def precompute_group_escape_mouths(
    engine,
    group_ports: tuple,
    *,
    primary_penalty: float = 0.20,
    secondary_penalty: float = 0.10,
    scarce_on_pin_limit: int = 1,
) -> dict[str, dict[tuple[int, int, int], float]]:
    """Precompute peer-only lookahead congestion cells for group port escapes.

    Accesses without an in-pin escape via get a stem + nearest via mouth;
    accesses with only a few spatially independent in-pin escape via slots get
    those pads priced by scarcity.  The cells are applied only to peers under
    negotiation.
    """
    rc = int(getattr(engine, "_res_cell", 0) or 0)
    n_layers = int(getattr(engine, "_n_layers", 5) or 5)
    grid = max(1, int(getattr(engine, "_manufacturing_grid_nm", 1)))
    if rc <= 0 or not hasattr(engine, "_tech") or not group_ports:
        return {}

    min_hw = max(1, min(int(hw) for hw in engine._half_w)) if engine._half_w else 70
    fine_layers = [
        g for g in range(len(engine._half_w)) if int(engine._half_w[g]) < 4 * min_hw
    ] or list(range(len(engine._half_w)))
    max_fine = max(fine_layers) if fine_layers else 0
    max_pad = max(
        (
            max(pair)
            for lo, pair in engine._via_pad_half.items()
            if int(lo) < max_fine
        ),
        default=max((int(engine._half_w[g]) for g in fine_layers), default=min_hw),
    )
    halo = (
        max((int(engine._half_w[g]) for g in fine_layers), default=min_hw)
        + max((int(engine._space[g]) for g in fine_layers), default=0)
        + max_pad
    )
    global_bounds = tuple(engine._bounds)

    outer_rej = getattr(engine, "_hard_rejections", {})
    outer_samp = getattr(engine, "_hard_rejection_samples", {})
    outer_gen = getattr(engine, "_hard_rejection_generation", 0)
    engine._hard_rejections = {}
    engine._hard_rejection_samples = {}
    by_net: dict[str, dict[tuple[int, int, int], float]] = {}
    rail_nets = set(getattr(engine, "_rail_by_net", {}) or ())
    try:
        for port in group_ports:
            is_rail_port = port.net in rail_nets
            if is_rail_port:
                continue
            ex0, ey0, ex1, ey1 = port.envelope_nm
            lbounds = (
                max(global_bounds[0], ex0 - halo),
                max(global_bounds[1], ey0 - halo),
                min(global_bounds[2], ex1 + halo),
                min(global_bounds[3], ey1 + halo),
            )
            net_cells: dict[tuple[int, int, int], float] = {}
            for access in port.accesses:
                pgi = int(engine._tech.resolve_layer_to_gi(access.layer_key))
                if pgi < 0 or pgi >= n_layers - 1:
                    continue
                px, py = int(access.xy_nm[0]), int(access.xy_nm[1])
                if is_rail_port and engine._escape_landing(port.net, px, py, pgi) is not None:
                    continue
                on_pin = engine._owned_component_escape_states(port.net, px, py, pgi)
                if on_pin:
                    if is_rail_port:
                        continue
                    # Event-grid sampling yields dozens of +/-1nm centres per
                    # finger; thin to spatially independent pad slots (>= 1.5*ph
                    # apart) so scarce finger accesses are priced while wide
                    # multi-slot structures stay unpriced.
                    raw_sites = sorted({
                        (
                            abs(int(site.x) - px) + abs(int(site.y) - py),
                            int(site.x),
                            int(site.y),
                            int(site.lo),
                        )
                        for _state, site in on_pin
                        if site is not None and 0 <= int(site.lo) < n_layers - 1
                    })
                    indep: list[tuple[int, int, int]] = []
                    for _dist, sx, sy, slo in raw_sites:
                        pads = engine._via_pad_half.get(slo, ())
                        ph = int(max(pads)) if pads else int(max_pad)
                        sep_req = max(360, (3 * ph) // 2)
                        if all(
                            slo != olo or max(abs(sx - ox), abs(sy - oy)) >= sep_req
                            for ox, oy, olo in indep
                        ):
                            indep.append((sx, sy, slo))
                            if len(indep) > scarce_on_pin_limit:
                                break
                    if 0 < len(indep) <= scarce_on_pin_limit:
                        pen_s = primary_penalty * min(2.0, 2.5 / len(indep))
                        lo_mult = 1.5 if len(indep) <= 2 else 0.75
                        for sx, sy, slo in indep:
                            pads = engine._via_pad_half.get(slo, ())
                            bh = int(pads[0]) if len(pads) >= 1 else int(max_pad)
                            ah = int(pads[1]) if len(pads) >= 2 else bh
                            sp_lo = int(engine._space[slo]) // 2
                            sp_hi = int(engine._space[slo + 1]) // 2
                            box_lo = (sx - bh - sp_lo, sy - bh - sp_lo, sx + bh + sp_lo, sy + bh + sp_lo)
                            box_hi = (sx - ah - sp_hi, sy - ah - sp_hi, sx + ah + sp_hi, sy + ah + sp_hi)
                            _add_cells(net_cells, rc, slo, box_lo, pen_s * lo_mult)
                            _add_cells(net_cells, rc, slo + 1, box_hi, pen_s * 0.25)
                            _add_cells(net_cells, rc, n_layers + slo, box_hi, pen_s * 1.5)
                    continue
                ax0, ay0, ax1, ay1 = px, py, px, py
                oidx_pgi = getattr(engine, "_obs_idx", {}).get(pgi)
                if oidx_pgi is not None:
                    changed_pin = True
                    seen_rects: set[tuple[int, int, int, int]] = set()
                    while changed_pin:
                        changed_pin = False
                        for owner_r, r_raw in oidx_pgi.query((ax0 - 1, ay0 - 1, ax1 + 1, ay1 + 1)):
                            if owner_r != port.net:
                                continue
                            rx0, ry0 = min(int(r_raw[0]), int(r_raw[2])), min(int(r_raw[1]), int(r_raw[3]))
                            rx1, ry1 = max(int(r_raw[0]), int(r_raw[2])), max(int(r_raw[1]), int(r_raw[3]))
                            rk = (rx0, ry0, rx1, ry1)
                            if rk in seen_rects:
                                continue
                            if rx0 <= ax1 and rx1 >= ax0 and ry0 <= ay1 and ry1 >= ay0:
                                seen_rects.add(rk)
                                ax0, ay0 = min(ax0, rx0), min(ay0, ry0)
                                ax1, ay1 = max(ax1, rx1), max(ay1, ry1)
                                changed_pin = True
                win = (
                    max(lbounds[0], ax0 - 2600),
                    max(lbounds[1], ay0 - 2600),
                    min(lbounds[2], ax1 + 2600),
                    min(lbounds[3], ay1 + 2600),
                )
                F_wire, F_via = _build_conservative_free_regions(
                    engine, port.net, [pgi, pgi + 1], win
                )
                seed = db.Region(db.Box(px, py, px + 1, py + 1))
                comp = F_wire[pgi].interacting(seed)
                if comp.is_empty():
                    continue
                comp.min_coherence = True
                hw_lo = int(engine._hw(port.net, pgi))
                sp_full = int(engine._space[pgi])
                sp_lo = sp_full // 2
                sp_hi = int(engine._space[pgi + 1]) // 2
                pads = engine._via_pad_half.get(pgi, ())
                bh = int(pads[0]) if len(pads) >= 1 else int(max_pad)
                ah = int(pads[1]) if len(pads) >= 2 else bh
                stem_pad = hw_lo + sp_lo
                if not is_rail_port:
                    inner_ext = hw_lo + sp_lo
                    collar_ext = inner_ext + 2 * sp_full + hw_lo
                    pin_core = db.Region(
                        db.Box(
                            ax0 - inner_ext,
                            ay0 - inner_ext,
                            ax1 + inner_ext + 1,
                            ay1 + inner_ext + 1,
                        )
                    )
                    collar = (
                        db.Region(
                            db.Box(
                                ax0 - collar_ext,
                                ay0 - collar_ext,
                                ax1 + collar_ext + 1,
                                ay1 + collar_ext + 1,
                            )
                        )
                        - pin_core
                    )
                    throats = comp & collar
                    throats.merge()
                    n_throats = throats.count()
                    if 0 < n_throats <= 4:
                        for tpoly in throats.each():
                            tb = tpoly.bbox()
                            min_span = min(int(tb.width()), int(tb.height()))
                            if n_throats <= 2 and min_span <= 2 * (bh + sp_full):
                                throat_pen = (
                                    max(primary_penalty * 9.0, 1.8)
                                    if n_throats == 1
                                    else max(primary_penalty * 6.0, 1.2)
                                )
                            else:
                                throat_pen = primary_penalty * (
                                    2.0 if n_throats == 1 else 1.5 if n_throats == 2 else 1.0
                                )
                            t_box = (
                                int(tb.left),
                                int(tb.bottom),
                                int(tb.right),
                                int(tb.top),
                            )
                            _add_cells(net_cells, rc, pgi, t_box, throat_pen)
                            _add_cells(net_cells, rc, n_layers + pgi, t_box, throat_pen * 0.5)
                v_reach = F_via.get(pgi, db.Region()) & comp
                if v_reach.is_empty():
                    continue
                cand_vias: list[tuple[int, int, int]] = []
                for poly in v_reach.each():
                    for trap in poly.decompose_trapezoids(db.Polygon.TD_vtrapezoids):
                        tb = trap.bbox()
                        tx0 = -(-tb.left // grid) * grid
                        tx1 = ((tb.right - 1) // grid) * grid
                        ty0 = -(-tb.bottom // grid) * grid
                        ty1 = ((tb.top - 1) // grid) * grid
                        if tx0 > tx1 or ty0 > ty1:
                            continue
                        ref_x = min(max((tx0 + tx1) // 2, ax0), ax1)
                        ref_y = min(max((ty0 + ty1) // 2, ay0), ay1)
                        cx_p = min(max(ref_x, tx0), tx1)
                        cy_p = min(max(ref_y, ty0), ty1)
                        cx_p = (
                            -(-cx_p // grid) * grid
                            if cx_p < ref_x
                            else (cx_p // grid) * grid
                        )
                        cy_p = (
                            -(-cy_p // grid) * grid
                            if cy_p < ref_y
                            else (cy_p // grid) * grid
                        )
                        cx_p = min(max(cx_p, tx0), tx1)
                        cy_p = min(max(cy_p, ty0), ty1)
                        if engine._via_legal(cx_p, cy_p, pgi, port.net):
                            dx_pin = max(0, ax0 - cx_p, cx_p - ax1)
                            dy_pin = max(0, ay0 - cy_p, cy_p - ay1)
                            cand_vias.append((dx_pin + dy_pin, cx_p, cy_p))
                cand_vias.sort()
                chosen: list[tuple[int, int]] = []
                mid_x, mid_y = (ax0 + ax1) // 2, (ay0 + ay1) // 2
                if cand_vias:
                    chosen.append((cand_vias[0][1], cand_vias[0][2]))
                    v0x, v0y = chosen[0]
                    for _dist, vx, vy in cand_vias[1:]:
                        if abs(vx - v0x) + abs(vy - v0y) < 600:
                            continue
                        if (vx - mid_x) * (v0x - mid_x) <= 0 and (vy - mid_y) * (v0y - mid_y) <= 0:
                            chosen.append((vx, vy))
                            break
                    if len(chosen) < 2:
                        for _dist, vx, vy in cand_vias[1:]:
                            if all(abs(vx - ox) + abs(vy - oy) >= 600 for ox, oy in chosen):
                                chosen.append((vx, vy))
                                break
                for idx_m, (vx, vy) in enumerate(chosen):
                    pen = primary_penalty if idx_m == 0 else secondary_penalty
                    sx_pin = min(max(vx, ax0), ax1)
                    sy_pin = min(max(vy, ay0), ay1)
                    dx_s = abs(vx - sx_pin)
                    dy_s = abs(vy - sy_pin)
                    if dx_s + dy_s > 3 * rc:
                        continue
                    if min(dx_s, dy_s) <= stem_pad:
                        stem_box = (
                            min(sx_pin, vx) - stem_pad,
                            min(sy_pin, vy) - stem_pad,
                            max(sx_pin, vx) + stem_pad,
                            max(sy_pin, vy) + stem_pad,
                        )
                        _add_cells(net_cells, rc, pgi, stem_box, pen)
                    box_lo = (vx - bh - sp_lo, vy - bh - sp_lo, vx + bh + sp_lo, vy + bh + sp_lo)
                    box_hi = (vx - ah - sp_hi, vy - ah - sp_hi, vx + ah + sp_hi, vy + ah + sp_hi)
                    _add_cells(net_cells, rc, pgi, box_lo, pen)
                    _add_cells(net_cells, rc, pgi + 1, box_hi, pen)
                    _add_cells(net_cells, rc, n_layers + pgi, box_hi, pen * 1.5)
            if net_cells:
                by_net[port.net] = net_cells
    finally:
        engine._hard_rejections = outer_rej
        engine._hard_rejection_samples = outer_samp
        engine._hard_rejection_generation = outer_gen
    return by_net


def precompute_group_sealed_escape_vias(
    engine,
    group_ports: tuple,
    *,
    max_pocket_vias: int = 2048,
) -> dict[str, tuple[tuple[tuple[int, int, int], ...], ...]]:
    """Return complete static escape-via site lists per sealed terminal pocket.

    Each entry maps ``port.net`` to one or more statically disconnected access
    pockets, where each pocket is a tuple of ``(vx, vy, lo)`` candidate centres.
    A pocket is included only when all accesses in that static fixed-conductor
    component lie on a single layer inside a bounded planar pocket that neither
    reaches the group envelope boundary, nor connects planarly to a sibling
    component, nor climbs through fixed cuts or multi-layer terminal stacks.
    Every valid route for the port must emit at least one legal via inside each
    included pocket.
    """
    n_layers = int(getattr(engine, "_n_layers", 5) or 5)
    grid = max(1, int(getattr(engine, "_manufacturing_grid_nm", 5) or 5))
    coarse_stride = max(grid, (85 // grid) * grid)
    if not hasattr(engine, "_tech") or not hasattr(engine, "_via_pad_half") or not group_ports:
        return {}

    atoms_by_net: dict[str, list[tuple[int, tuple[int, int, int, int]]]] = {}
    for atom in getattr(engine, "_owned_terminal_atoms", ()):
        atoms_by_net.setdefault(str(atom.owner), []).append(
            (int(atom.gi), tuple(int(v) for v in atom.rect))
        )
    rail_nets = set(getattr(engine, "_rail_by_net", {}) or ())
    reach = 1600
    deck_layout = None
    deck_top = None

    missing = object()
    prev_closure = getattr(engine, "_active_group_route_local_closure", missing)
    outer_rej = getattr(engine, "_hard_rejections", {})
    outer_samp = getattr(engine, "_hard_rejection_samples", {})
    outer_gen = getattr(engine, "_hard_rejection_generation", 0)
    prev_metal_scans = getattr(engine, "_active_static_metal_scans", None)
    prev_cut_scans = getattr(engine, "_active_static_cut_scans", None)
    prev_obs_scans = getattr(engine, "_active_static_obstacle_scans", None)
    engine._active_group_route_local_closure = True
    engine._hard_rejections = {}
    engine._hard_rejection_samples = {}
    if prev_metal_scans is None:
        engine._active_static_metal_scans = {}
    if prev_cut_scans is None:
        engine._active_static_cut_scans = {}
    if prev_obs_scans is None:
        engine._active_static_obstacle_scans = {}
    watched: dict[str, tuple[tuple[tuple[int, int, int], ...], ...]] = {}
    try:
        for port in group_ports:
            accesses = tuple(getattr(port, "accesses", ()))
            if not accesses or port.net in rail_nets:
                continue
            ex0, ey0, ex1, ey1 = port.envelope_nm
            all_access_nodes: list[tuple[int, int, int]] = []
            valid_port = True
            for acc in accesses:
                pgi_acc = int(engine._tech.resolve_layer_to_gi(acc.layer_key))
                if pgi_acc < 0 or pgi_acc >= n_layers:
                    valid_port = False
                    break
                all_access_nodes.append((int(acc.xy_nm[0]), int(acc.xy_nm[1]), pgi_acc))
            if not valid_port:
                continue

            if all(
                getattr(acc, "terminal_binding", None) is not None
                and hasattr(acc.terminal_binding, "stamp_id")
                and hasattr(acc.terminal_binding, "component_id")
                for acc in accesses
            ):
                by_bkey: dict[tuple[str, str], list[int]] = {}
                bkeys_order: list[tuple[str, str]] = []
                for idx, acc in enumerate(accesses):
                    b = acc.terminal_binding
                    bkey = (str(b.stamp_id), str(b.component_id))
                    if bkey not in by_bkey:
                        by_bkey[bkey] = []
                        bkeys_order.append(bkey)
                    by_bkey[bkey].append(idx)

                if (
                    len(bkeys_order) > 1
                    and getattr(engine, "_deck_connect_stack", None) is not None
                    and getattr(engine, "_fixed_deck_literal_rects", ())
                ):
                    from clay.layout.route.deck_isolation import (
                        NetProbe,
                        check_deck_net_isolation,
                    )

                    if deck_layout is None:
                        deck_layout = db.Layout()
                        deck_layout.dbu = 0.001
                        deck_top = deck_layout.create_cell("SEALED_ESCAPE_WATCH")
                        layer_idx_cache: dict[tuple[int, int], int] = {}
                        for lk, _owner, rect in getattr(engine, "_fixed_deck_literal_rects", ()):
                            li = layer_idx_cache.get(lk)
                            if li is None:
                                li = deck_layout.layer(int(lk[0]), int(lk[1]))
                                layer_idx_cache[lk] = li
                            deck_top.shapes(li).insert(
                                db.Box(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
                            )
                        contact_rects = getattr(engine, "_fixed_contact_rects", ())
                        if contact_rects:
                            try:
                                ck = tuple(engine._tech.pdk.contact_key())
                            except Exception:
                                ck = None
                            if ck is not None:
                                cli = deck_layout.layer(int(ck[0]), int(ck[1]))
                                for rect in contact_rects:
                                    deck_top.shapes(cli).insert(
                                        db.Box(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
                                    )

                    probes = [
                        NetProbe(
                            f"a_{idx}",
                            tuple(acc.layer_key),
                            int(acc.xy_nm[0]),
                            int(acc.xy_nm[1]),
                        )
                        for idx, acc in enumerate(accesses)
                    ]
                    deck_obs = check_deck_net_isolation(
                        layout=deck_layout,
                        cell=deck_top,
                        stack=engine._deck_connect_stack,
                        probes=probes,
                        include_probe_partition=True,
                    )
                    comp_of_probe: dict[str, int] = {}
                    for comp_id, grp in enumerate(deck_obs.resolved_probe_groups):
                        for pr in grp:
                            if isinstance(pr, tuple) and len(pr) == 4:
                                comp_of_probe[str(pr[0])] = comp_id
                    if any(f"a_{idx}" not in comp_of_probe for idx in range(len(accesses))):
                        subgroup_indices = [list(range(len(accesses)))]
                    else:
                        parent = {bk: bk for bk in bkeys_order}

                        def _find_bk(x: tuple[str, str]) -> tuple[str, str]:
                            while parent[x] != x:
                                parent[x] = parent[parent[x]]
                                x = parent[x]
                            return x

                        first_bk_for_comp: dict[int, tuple[str, str]] = {}
                        for idx, acc in enumerate(accesses):
                            bk = (
                                str(acc.terminal_binding.stamp_id),
                                str(acc.terminal_binding.component_id),
                            )
                            cid = comp_of_probe[f"a_{idx}"]
                            if cid in first_bk_for_comp:
                                ra = _find_bk(first_bk_for_comp[cid])
                                rb = _find_bk(bk)
                                if ra != rb:
                                    parent[rb] = ra
                            else:
                                first_bk_for_comp[cid] = bk
                        merged_groups: dict[tuple[str, str], list[int]] = {}
                        merged_order: list[tuple[str, str]] = []
                        for bk in bkeys_order:
                            root = _find_bk(bk)
                            if root not in merged_groups:
                                merged_groups[root] = []
                                merged_order.append(root)
                            merged_groups[root].extend(by_bkey[bk])
                        subgroup_indices = [merged_groups[r] for r in merged_order]
                else:
                    subgroup_indices = [by_bkey[bk] for bk in bkeys_order]
            else:
                subgroup_indices = [list(range(len(accesses)))]

            net_atoms = atoms_by_net.get(port.net, ())
            pockets_for_port: list = []

            for sub_idxs in subgroup_indices:
                sub_idx_set = set(sub_idxs)
                access_nodes: list[tuple[int, int, int]] = []
                seed_boxes: list[tuple[int, int, int, int]] = []
                layers_seen: set[int] = set()
                skip_sub = False
                for idx in sub_idxs:
                    px, py, pgi_acc = all_access_nodes[idx]
                    access_nodes.append((px, py, pgi_acc))
                    layers_seen.add(pgi_acc)
                    seed_boxes.append((px, py, px, py))
                    if hasattr(engine, "_owned_terminal_stack_access"):
                        comp_atoms, fixed_states = engine._owned_terminal_stack_access(
                            port.net, px, py, pgi_acc
                        )
                        if fixed_states:
                            skip_sub = True
                            break
                        for atom in comp_atoms:
                            layers_seen.add(int(atom.gi))
                            seed_boxes.append(tuple(int(v) for v in atom.rect))
                if skip_sub or len(layers_seen) != 1 or not seed_boxes:
                    continue
                pgi = next(iter(layers_seen))
                if len(subgroup_indices) == 1:
                    for agi, arect in net_atoms:
                        if (
                            arect[2] >= ex0
                            and arect[0] <= ex1
                            and arect[3] >= ey0
                            and arect[1] <= ey1
                        ):
                            if agi != pgi:
                                skip_sub = True
                                break
                            seed_boxes.append(arect)
                    if skip_sub:
                        continue

                half_pgi = int(engine._half_w[pgi])
                sp_pgi = int(engine._space[pgi])
                margin = half_pgi + sp_pgi + 10
                ux0 = min(b[0] for b in seed_boxes)
                uy0 = min(b[1] for b in seed_boxes)
                ux1 = max(b[2] for b in seed_boxes)
                uy1 = max(b[3] for b in seed_boxes)
                if (
                    ux0 <= ex0 + margin
                    or ux1 >= ex1 - margin
                    or uy0 <= ey0 + margin
                    or uy1 >= ey1 - margin
                    or (ux1 - ux0) > 3600
                    or (uy1 - uy0) > 3600
                ):
                    continue

                layers_local = sorted(
                    {pgi}
                    | ({pgi + 1} if pgi + 1 < n_layers else set())
                    | ({pgi - 1} if pgi > 0 else set())
                )
                win = (ux0 - reach, uy0 - reach, ux1 + reach, uy1 + reach)
                F_wire, F_via = _build_conservative_free_regions(
                    engine,
                    port.net,
                    layers_local,
                    win,
                    min_width_only=True,
                    static_only=True,
                )
                own_reg = db.Region()
                static_own_boxes: list[tuple[int, int, int, int]] = []
                for agi, arect in net_atoms:
                    if (
                        agi == pgi
                        and arect[2] >= win[0]
                        and arect[0] <= win[2]
                        and arect[3] >= win[1]
                        and arect[1] <= win[3]
                    ):
                        own_reg.insert(db.Box(arect[0], arect[1], arect[2] + 1, arect[3] + 1))
                oidx_pgi = getattr(engine, "_obs_idx", {}).get(pgi)
                if oidx_pgi is not None:
                    for owner_r, r_raw in oidx_pgi.query(win):
                        if owner_r == port.net:
                            rx0, ry0 = min(int(r_raw[0]), int(r_raw[2])), min(int(r_raw[1]), int(r_raw[3]))
                            rx1, ry1 = max(int(r_raw[0]), int(r_raw[2])), max(int(r_raw[1]), int(r_raw[3]))
                            own_reg.insert(db.Box(rx0, ry0, rx1 + 1, ry1 + 1))
                if not own_reg.is_empty():
                    own_reg.merge()
                    for poly in own_reg.each():
                        pb = poly.bbox()
                        static_own_boxes.append((pb.left, pb.bottom, pb.right - 1, pb.top - 1))

                seed_reg = db.Region()
                for bx0, by0, bx1, by1 in seed_boxes:
                    seed_reg.insert(db.Box(bx0, by0, bx1 + 1, by1 + 1))
                seed_reg.merge()
                if not own_reg.is_empty():
                    conn_own = own_reg.interacting(seed_reg)
                    if not conn_own.is_empty():
                        seed_reg = (seed_reg + conn_own).merged()

                sibling_pts: list[tuple[int, int]] = []
                other_layer_atoms: list[tuple[int, int, int, int]] = []
                seed_touches_other = False
                if len(subgroup_indices) > 1:
                    for oidx, (qx, qy, qgi) in enumerate(all_access_nodes):
                        if oidx in sub_idx_set or qgi != pgi:
                            continue
                        sibling_pts.append((qx, qy))
                        if not (seed_reg & db.Region(db.Box(qx, qy, qx + 1, qy + 1))).is_empty():
                            seed_touches_other = True
                            break
                    if seed_touches_other:
                        continue
                    sbb = seed_reg.bbox()
                    for agi, arect in net_atoms:
                        if (
                            agi != pgi
                            and arect[2] >= win[0] - 5
                            and arect[0] <= win[2] + 5
                            and arect[3] >= win[1] - 5
                            and arect[1] <= win[3] + 5
                        ):
                            other_layer_atoms.append(arect)
                            if (
                                arect[2] >= sbb.left - 5
                                and arect[0] <= sbb.right + 5
                                and arect[3] >= sbb.bottom - 5
                                and arect[1] <= sbb.top + 5
                                and not (
                                    seed_reg
                                    & db.Region(db.Box(arect[0], arect[1], arect[2] + 1, arect[3] + 1))
                                ).is_empty()
                            ):
                                seed_touches_other = True
                                break
                    if seed_touches_other:
                        continue

                reach_reg = (F_wire[pgi].interacting(seed_reg) + seed_reg).merged()
                while not own_reg.is_empty():
                    joined = (reach_reg + own_reg.interacting(reach_reg)).merged()
                    expanded = (joined + F_wire[pgi].interacting(joined)).merged()
                    if (expanded - reach_reg).is_empty():
                        break
                    reach_reg = expanded

                bb = reach_reg.bbox()
                cq_win = (win[0] - 5, win[1] - 5, win[2] + 5, win[3] + 5)
                fixed_cuts: list[tuple[int, tuple[int, int, int, int]]] = []
                seed_has_fixed_cut = False
                reach_has_fixed_cut = False
                for lo in (pgi, pgi - 1):
                    if lo >= 0:
                        for crect in engine._local_fixed_cuts(lo, cq_win):
                            ctup = (int(crect[0]), int(crect[1]), int(crect[2]), int(crect[3]))
                            fixed_cuts.append((lo, ctup))
                            cbox = db.Region(
                                db.Box(ctup[0], ctup[1], ctup[2] + 1, ctup[3] + 1)
                            )
                            if not (seed_reg & cbox).is_empty():
                                seed_has_fixed_cut = True
                                break
                            if not reach_has_fixed_cut and not (reach_reg & cbox).is_empty():
                                reach_has_fixed_cut = True
                    if seed_has_fixed_cut:
                        break
                if seed_has_fixed_cut:
                    continue

                touches_win_or_env = (
                    bb.left <= max(ex0 + half_pgi + 5, win[0] + 10)
                    or bb.right >= min(ex1 - half_pgi - 5, win[2] - 10)
                    or bb.bottom <= max(ey0 + half_pgi + 5, win[1] + 10)
                    or bb.top >= min(ey1 - half_pgi - 5, win[3] - 10)
                )

                sibling_reached = False
                if len(subgroup_indices) > 1 and not touches_win_or_env:
                    for qx, qy in sibling_pts:
                        if not (reach_reg & db.Region(db.Box(qx, qy, qx + 1, qy + 1))).is_empty():
                            sibling_reached = True
                            break
                    if not sibling_reached:
                        for arect in other_layer_atoms:
                            if (
                                arect[2] >= bb.left - 5
                                and arect[0] <= bb.right + 5
                                and arect[3] >= bb.bottom - 5
                                and arect[1] <= bb.top + 5
                                and not (
                                    reach_reg
                                    & db.Region(db.Box(arect[0], arect[1], arect[2] + 1, arect[3] + 1))
                                ).is_empty()
                            ):
                                sibling_reached = True
                                break

                def _collect_vias_on_region(
                    target_reg: db.Region, *, include_fine: bool
                ) -> tuple[tuple[int, int, int], ...]:
                    ov: list[tuple[int, int, int]] = []
                    sv: set[tuple[int, int, int]] = set()
                    if pgi < n_layers - 1 and hasattr(engine, "_owned_component_escape_states"):
                        for px, py, apgi in access_nodes:
                            for _state, site in engine._owned_component_escape_states(
                                port.net, px, py, apgi
                            ):
                                if site is not None:
                                    vk = (int(site.x), int(site.y), int(site.lo))
                                    if vk not in sv and engine._via_legal(
                                        vk[0], vk[1], vk[2], port.net
                                    ):
                                        sv.add(vk)
                                        ov.append(vk)
                                        if not include_fine and len(ov) >= 256:
                                            break
                            if not include_fine and len(ov) >= 256:
                                break
                    cv: list[tuple[int, int, int]] = []
                    fv: list[tuple[int, int, int]] = []
                    ovf = False
                    step_xy = grid if include_fine else coarse_stride
                    for lo in (pgi, pgi - 1):
                        if lo not in F_via:
                            continue
                        v_cand = F_via[lo] & target_reg
                        if v_cand.is_empty():
                            continue
                        for poly in v_cand.each():
                            for trap in poly.decompose_trapezoids(db.Polygon.TD_vtrapezoids):
                                tb = trap.bbox()
                                tx0 = -(-tb.left // grid) * grid
                                tx1 = ((tb.right - 1) // grid) * grid
                                ty0 = -(-tb.bottom // grid) * grid
                                ty1 = ((tb.top - 1) // grid) * grid
                                if tx0 > tx1 or ty0 > ty1:
                                    continue
                                if not include_fine:
                                    xs_iter = list(range(tx0, tx1 + 1, step_xy))
                                    if xs_iter[-1] != tx1:
                                        xs_iter.append(tx1)
                                    ys_iter = list(range(ty0, ty1 + 1, step_xy))
                                    if ys_iter[-1] != ty1:
                                        ys_iter.append(ty1)
                                    for vx in xs_iter:
                                        for vy in ys_iter:
                                            vk = (vx, vy, lo)
                                            if vk not in sv and engine._via_legal(
                                                vx, vy, lo, port.net
                                            ):
                                                sv.add(vk)
                                                cv.append(vk)
                                                if len(ov) + len(cv) >= 512:
                                                    ovf = True
                                                    break
                                        if ovf:
                                            break
                                    if ovf:
                                        break
                                    continue
                                vx = tx0
                                while vx <= tx1:
                                    is_coarse_x = (
                                        vx == tx0
                                        or vx == tx1
                                        or (vx - tx0) % coarse_stride == 0
                                    )
                                    vy = ty0
                                    while vy <= ty1:
                                        vk = (vx, vy, lo)
                                        if vk not in sv and engine._via_legal(
                                            vx, vy, lo, port.net
                                        ):
                                            sv.add(vk)
                                            if is_coarse_x and (
                                                vy == ty0
                                                or vy == ty1
                                                or (vy - ty0) % coarse_stride == 0
                                            ):
                                                cv.append(vk)
                                            else:
                                                fv.append(vk)
                                            if len(ov) + len(cv) + len(fv) > max_pocket_vias:
                                                ovf = True
                                                break
                                        vy += grid
                                    if ovf:
                                        break
                                    vx += grid
                                if ovf:
                                    break
                            if ovf:
                                break
                        if ovf:
                            break
                    if ovf and include_fine:
                        return ()
                    ov.extend(cv)
                    ov.extend(fv)
                    return tuple(ov)

                static_vias = (
                    _collect_vias_on_region(reach_reg, include_fine=True)
                    if (not touches_win_or_env and not sibling_reached and not reach_has_fixed_cut)
                    else ()
                )
                if static_vias:
                    pockets_for_port.append(
                        (
                            "static",
                            static_vias,
                            {"pgi": int(pgi), "win": win},
                        )
                    )
                else:
                    seed_vias = _collect_vias_on_region(seed_reg, include_fine=False)
                    if seed_vias or touches_win_or_env or sibling_reached or reach_has_fixed_cut:
                        pockets_for_port.append(
                            (
                                "dynamic",
                                seed_vias,
                                {
                                    "pgi": int(pgi),
                                    "half_pgi": int(half_pgi),
                                    "layers_local": tuple(layers_local),
                                    "win": win,
                                    "env_nm": (int(ex0), int(ey0), int(ex1), int(ey1)),
                                    "seed_boxes": tuple(seed_boxes),
                                    "own_reg": own_reg,
                                    "static_reach": reach_reg.dup(),
                                    "sibling_pts": tuple(sibling_pts),
                                    "other_layer_atoms": tuple(other_layer_atoms),
                                    "fixed_cuts": tuple(fixed_cuts),
                                },
                            )
                        )

            if pockets_for_port:
                watched[port.net] = tuple(pockets_for_port)
    finally:
        engine._hard_rejections = outer_rej
        engine._hard_rejection_samples = outer_samp
        engine._hard_rejection_generation = outer_gen
        engine._active_static_metal_scans = prev_metal_scans
        engine._active_static_cut_scans = prev_cut_scans
        engine._active_static_obstacle_scans = prev_obs_scans
        if prev_closure is missing:
            delattr(engine, "_active_group_route_local_closure")
        else:
            engine._active_group_route_local_closure = prev_closure
    return watched


def check_watched_pocket_starved(
    engine,
    net: str,
    pocket,
    *,
    choke_boxes_out: list[tuple[str, int, tuple[int, int, int, int]]] | None = None,
) -> bool:
    """Return True (and populate _hard_rejections/_hard_rejection_samples) iff pocket is 100% starved."""
    if isinstance(pocket, tuple) and len(pocket) == 3 and pocket[0] in ("static", "dynamic"):
        kind, vias, meta = pocket
    else:
        kind, vias, meta = "static", pocket, None
    if kind != "dynamic" and not vias:
        return False
    if vias and any(engine._via_legal(vx, vy, lo, net) for vx, vy, lo in vias):
        return False

    if kind == "static":
        if meta is not None:
            pgi_s = int(meta["pgi"])
            win_s = meta["win"]
            for owner_m, rect_m in engine._book.metal.get(pgi_s, ()):
                if (
                    owner_m == net
                    and rect_m[2] >= win_s[0]
                    and rect_m[0] <= win_s[2]
                    and rect_m[3] >= win_s[1]
                    and rect_m[1] <= win_s[3]
                ):
                    return False
        engine._hard_rejections = {}
        engine._hard_rejection_samples = {}
        for vx, vy, lo in vias:
            engine._via_legal(vx, vy, lo, net)
        return True

    pgi = int(meta["pgi"])
    half_pgi = int(meta["half_pgi"])
    layers_local = list(meta["layers_local"])
    win = meta["win"]
    ex0, ey0, ex1, ey1 = meta["env_nm"]
    grid = max(1, int(getattr(engine, "_manufacturing_grid_nm", 5) or 5))

    if not vias:
        pad_q = (win[0] - 600, win[1] - 600, win[2] + 600, win[3] + 600)
        has_foreign = False
        for lgi in layers_local:
            for rows in (engine._book.metal.get(lgi, ()), engine._book.via_pads.get(lgi, ())):
                for owner_f, rect_f in rows:
                    if (
                        owner_f != net
                        and rect_f[2] >= pad_q[0]
                        and rect_f[0] <= pad_q[2]
                        and rect_f[3] >= pad_q[1]
                        and rect_f[1] <= pad_q[3]
                    ):
                        has_foreign = True
                        break
                if has_foreign:
                    break
            if has_foreign:
                break
        if not has_foreign:
            for lo_c in (pgi, pgi - 1):
                cidx = engine._cut_idx.get(lo_c) if lo_c >= 0 else None
                if cidx is not None and any(o_c != net for o_c, _ in cidx.query(pad_q)):
                    has_foreign = True
                    break
        if not has_foreign:
            return False

    F_wire, F_via = _build_conservative_free_regions(
        engine,
        net,
        layers_local,
        win,
        min_width_only=True,
        static_only=False,
    )
    own_reg = meta["own_reg"].dup()
    added_same_net = False
    for rows in (engine._book.metal.get(pgi, ()), engine._book.via_pads.get(pgi, ())):
        for owner_m, rect_m in rows:
            if (
                owner_m == net
                and rect_m[2] >= win[0]
                and rect_m[0] <= win[2]
                and rect_m[3] >= win[1]
                and rect_m[1] <= win[3]
            ):
                own_reg.insert(
                    db.Box(int(rect_m[0]), int(rect_m[1]), int(rect_m[2]) + 1, int(rect_m[3]) + 1)
                )
                added_same_net = True
    if added_same_net:
        own_reg.merge()

    seed_reg = db.Region()
    for bx0, by0, bx1, by1 in meta["seed_boxes"]:
        seed_reg.insert(db.Box(bx0, by0, bx1 + 1, by1 + 1))
    seed_reg.merge()
    if not own_reg.is_empty():
        conn_own = own_reg.interacting(seed_reg)
        if not conn_own.is_empty():
            seed_reg = (seed_reg + conn_own).merged()

    reach_dyn = (F_wire[pgi].interacting(seed_reg) + seed_reg).merged()
    while not own_reg.is_empty():
        joined = (reach_dyn + own_reg.interacting(reach_dyn)).merged()
        expanded = (joined + F_wire[pgi].interacting(joined)).merged()
        if (expanded - reach_dyn).is_empty():
            break
        reach_dyn = expanded

    bb = reach_dyn.bbox()
    if (
        bb.left <= max(ex0 + half_pgi + 5, win[0] + 10)
        or bb.right >= min(ex1 - half_pgi - 5, win[2] - 10)
        or bb.bottom <= max(ey0 + half_pgi + 5, win[1] + 10)
        or bb.top >= min(ey1 - half_pgi - 5, win[3] - 10)
    ):
        return False

    for qx, qy in meta["sibling_pts"]:
        if not (reach_dyn & db.Region(db.Box(qx, qy, qx + 1, qy + 1))).is_empty():
            return False
    for arect in meta["other_layer_atoms"]:
        if (
            arect[2] >= bb.left - 5
            and arect[0] <= bb.right + 5
            and arect[3] >= bb.bottom - 5
            and arect[1] <= bb.top + 5
            and not (
                reach_dyn & db.Region(db.Box(arect[0], arect[1], arect[2] + 1, arect[3] + 1))
            ).is_empty()
        ):
            return False
    for _lo_fc, crect in meta["fixed_cuts"]:
        if (
            crect[2] >= bb.left - 5
            and crect[0] <= bb.right + 5
            and crect[3] >= bb.bottom - 5
            and crect[1] <= bb.top + 5
            and not (
                reach_dyn & db.Region(db.Box(crect[0], crect[1], crect[2] + 1, crect[3] + 1))
            ).is_empty()
        ):
            return False

    cq_dyn = (bb.left - 5, bb.bottom - 5, bb.right + 5, bb.top + 5)
    for lo in (pgi, pgi - 1):
        cut_idx = engine._cut_idx.get(lo) if lo >= 0 else None
        if cut_idx is not None:
            for owner_c, crect in cut_idx.query(cq_dyn):
                if owner_c == net:
                    if not (
                        reach_dyn
                        & db.Region(
                            db.Box(int(crect[0]), int(crect[1]), int(crect[2]) + 1, int(crect[3]) + 1)
                        )
                    ).is_empty():
                        return False

    checked_vias = set(vias)
    extra_illegal_vias: list[tuple[int, int, int]] = []
    for lo in (pgi, pgi - 1):
        if lo not in F_via:
            continue
        v_cand = F_via[lo] & reach_dyn
        if v_cand.is_empty():
            continue
        for poly in v_cand.each():
            for trap in poly.decompose_trapezoids(db.Polygon.TD_vtrapezoids):
                tb = trap.bbox()
                tx0 = -(-tb.left // grid) * grid
                tx1 = ((tb.right - 1) // grid) * grid
                ty0 = -(-tb.bottom // grid) * grid
                ty1 = ((tb.top - 1) // grid) * grid
                vx = tx0
                while vx <= tx1:
                    vy = ty0
                    while vy <= ty1:
                        vk = (vx, vy, lo)
                        if vk not in checked_vias:
                            checked_vias.add(vk)
                            if engine._via_legal(vx, vy, lo, net):
                                return False
                            if len(extra_illegal_vias) < 64:
                                extra_illegal_vias.append(vk)
                        vy += grid
                    vx += grid

    engine._hard_rejections = {}
    engine._hard_rejection_samples = {}
    for vx, vy, lo in vias:
        engine._via_legal(vx, vy, lo, net)
    for vx, vy, lo in extra_illegal_vias:
        engine._via_legal(vx, vy, lo, net)

    half = int(engine._hw(net, pgi))
    sp = int(engine._space[pgi])
    reg_grown = reach_dyn.sized(grid)
    rc_choke = max(140, int(getattr(engine, "_res_cell", 280) or 280))
    narrow_htraps: list[tuple[int, int, int]] = []
    narrow_vtraps: list[tuple[int, int, int]] = []
    static_reach = meta.get("static_reach") if choke_boxes_out is not None else None
    if isinstance(static_reach, db.Region) and not static_reach.is_empty():
        lost_all = (static_reach - reach_dyn).merged()
        true_lost = db.Region()
        lim_l = max(ex0 + half_pgi + 5, win[0] + 10)
        lim_r = min(ex1 - half_pgi - 5, win[2] - 10)
        lim_b = max(ey0 + half_pgi + 5, win[1] + 10)
        lim_t = min(ey1 - half_pgi - 5, win[3] - 10)
        fvia_escape = db.Region()
        for lo_v in (pgi, pgi - 1):
            if lo_v in F_via:
                fvia_escape += F_via[lo_v]
        for lpoly in lost_all.each():
            lbb = lpoly.bbox()
            lreg = db.Region(lpoly)
            escapes = (
                lbb.left <= lim_l
                or lbb.right >= lim_r
                or lbb.bottom <= lim_b
                or lbb.top >= lim_t
                or not (fvia_escape & lreg).is_empty()
                or any(
                    not (lreg & db.Region(db.Box(qx, qy, qx + 1, qy + 1))).is_empty()
                    for qx, qy in meta["sibling_pts"]
                )
                or any(
                    not (lreg & db.Region(db.Box(ar[0], ar[1], ar[2] + 1, ar[3] + 1))).is_empty()
                    for ar in meta["other_layer_atoms"]
                )
            )
            if escapes:
                true_lost.insert(lpoly)
        true_lost.merge()
        if not true_lost.is_empty():
            for spoly in static_reach.each():
                for htrap in spoly.decompose_trapezoids(db.Polygon.TD_htrapezoids):
                    htb = htrap.bbox()
                    if int(htb.height()) <= 0:
                        continue
                    ym_h = (int(htb.bottom) + int(htb.top)) // 2
                    mslice = db.Region(htrap) & db.Region(db.Box(htb.left, ym_h, htb.right, ym_h + 1))
                    if mslice.is_empty():
                        continue
                    msb = mslice.bbox()
                    if 0 < int(msb.width()) <= 2 * rc_choke:
                        w_l = db.Region(db.Box(msb.left - 10, ym_h, msb.left - 1, ym_h + 1))
                        w_r = db.Region(db.Box(msb.right + 1, ym_h, msb.right + 10, ym_h + 1))
                        if (
                            (static_reach & w_l).is_empty()
                            and (static_reach & w_r).is_empty()
                            and not (db.Region(htrap) & true_lost).is_empty()
                        ):
                            narrow_htraps.append(
                                ((int(msb.left) + int(msb.right)) // 2, int(htb.bottom), int(htb.top))
                            )
                for vtrap in spoly.decompose_trapezoids(db.Polygon.TD_vtrapezoids):
                    vtb = vtrap.bbox()
                    if int(vtb.width()) <= 0:
                        continue
                    xm_v = (int(vtb.left) + int(vtb.right)) // 2
                    mslice = db.Region(vtrap) & db.Region(db.Box(xm_v, vtb.bottom, xm_v + 1, vtb.top))
                    if mslice.is_empty():
                        continue
                    msb = mslice.bbox()
                    if 0 < int(msb.height()) <= 2 * rc_choke:
                        w_b = db.Region(db.Box(xm_v, msb.bottom - 10, xm_v + 1, msb.bottom - 1))
                        w_t = db.Region(db.Box(xm_v, msb.top + 1, xm_v + 1, msb.top + 10))
                        if (
                            (static_reach & w_b).is_empty()
                            and (static_reach & w_t).is_empty()
                            and not (db.Region(vtrap) & true_lost).is_empty()
                        ):
                            narrow_vtraps.append(
                                ((int(msb.bottom) + int(msb.top)) // 2, int(vtb.left), int(vtb.right))
                            )

    def _record_throat_chokes(owner_c: str, rect_c: tuple[int, int, int, int], v_reg_c: db.Region) -> None:
        if choke_boxes_out is None or (not narrow_htraps and not narrow_vtraps):
            return
        rx0, ry0 = min(int(rect_c[0]), int(rect_c[2])), min(int(rect_c[1]), int(rect_c[3]))
        rx1, ry1 = max(int(rect_c[0]), int(rect_c[2])), max(int(rect_c[1]), int(rect_c[3]))
        ex0_b = rx0 - sp // 2 + 5
        ex1_b = rx1 + sp // 2 - 5
        ey0_b = ry0 - sp // 2 + 5
        ey1_b = ry1 + sp // 2 - 5
        for xm_raw, hy0, hy1 in narrow_htraps:
            t_strip = db.Region(db.Box(xm_raw - 5, hy0, xm_raw + 6, hy1 + 1))
            inter_h = t_strip & v_reg_c
            if inter_h.is_empty():
                continue
            ib = inter_h.bbox()
            xm = xm_raw
            if ex0_b <= ex1_b:
                xm = min(max(xm, ex0_b), ex1_b)
            cy0 = max(hy0, int(ib.bottom) - rc_choke)
            cy1 = min(hy1, int(ib.top) + rc_choke)
            choke_boxes_out.append((str(owner_c), int(pgi), (xm, cy0, xm, cy1)))
        for ym_raw, vx0, vx1 in narrow_vtraps:
            t_strip = db.Region(db.Box(vx0, ym_raw - 5, vx1 + 1, ym_raw + 6))
            inter_v = t_strip & v_reg_c
            if inter_v.is_empty():
                continue
            ib = inter_v.bbox()
            ym = ym_raw
            if ey0_b <= ey1_b:
                ym = min(max(ym, ey0_b), ey1_b)
            cx0 = max(vx0, int(ib.left) - rc_choke)
            cx1 = min(vx1, int(ib.right) + rc_choke)
            choke_boxes_out.append((str(owner_c), int(pgi), (cx0, ym, cx1, ym)))

    wq = (
        bb.left - half - sp - grid - 5,
        bb.bottom - half - sp - grid - 5,
        bb.right + half + sp + grid + 5,
        bb.top + half + sp + grid + 5,
    )
    for rows in (engine._book.via_pads.get(pgi, ()), engine._book.metal.get(pgi, ())):
        for owner, rect in rows:
            if (
                owner != net
                and rect[2] >= wq[0]
                and rect[0] <= wq[2]
                and rect[3] >= wq[1]
                and rect[1] <= wq[3]
            ):
                v_reg = db.Region()
                _insert_strict_violation_boxes(v_reg, rect, half, sp)
                hit = reg_grown & v_reg
                if not hit.is_empty():
                    _record_throat_chokes(owner, rect, v_reg)
                    for poly in hit.each():
                        for trap in poly.decompose_trapezoids(db.Polygon.TD_vtrapezoids):
                            tb = trap.bbox()
                            mx = -(-tb.left // grid) * grid
                            my = -(-tb.bottom // grid) * grid
                            if mx < tb.right and my < tb.top:
                                wb = (mx - half, my - half, mx + half, my + half)
                                engine._reject_foreign_committed_metal_spacing(
                                    pgi,
                                    wb,
                                    net,
                                    candidate_half=half,
                                    require_supra_minimum_endpoint=True,
                                )
                                engine._reject_foreign_committed_metal_contact(
                                    pgi,
                                    wb,
                                    net,
                                    candidate_half=half,
                                )
                                break
    for wspace, core, _win_w, _attached in engine._wide_indexes(True).get(pgi, ()):
        wq_wide = (
            bb.left - half - int(wspace) - grid - 5,
            bb.bottom - half - int(wspace) - grid - 5,
            bb.right + half + int(wspace) + grid + 5,
            bb.top + half + int(wspace) + grid + 5,
        )
        for owner, rect in core.query(wq_wide):
            if owner != net:
                v_reg = db.Region()
                _insert_strict_violation_boxes(v_reg, rect, half, int(wspace))
                hit = reg_grown & v_reg
                if not hit.is_empty():
                    _record_throat_chokes(owner, rect, v_reg)
                    for poly in hit.each():
                        for trap in poly.decompose_trapezoids(db.Polygon.TD_vtrapezoids):
                            tb = trap.bbox()
                            mx = -(-tb.left // grid) * grid
                            my = -(-tb.bottom // grid) * grid
                            if mx < tb.right and my < tb.top:
                                wb = (mx - half, my - half, mx + half, my + half)
                                engine._reject_wide_metal_spacing(
                                    pgi, wb, net, candidate_half=half
                                )
                                break
    return True


def verify_static_terminal_escapability(
    engine,
    pins_by_net: dict,
    net_nodes: dict[str, list[tuple[int, int, int]]],
    net_node_pins: dict[str, list],
    net_layers: dict[str, list[int]],
) -> None:
    """Fast (<20ms) zero-false-positive PDK-driven static escapability gate.

    Raises ValueError if a multi-component net with PhysicalTerminalPinBinding has
    an isolated fixed-terminal component whose pocket is completely sealed by static
    fixed device metal (0 legal via_redundancy=N escapes and 0 planar wire exits).
    Mutates zero engine caches or counters.
    """
    envelope = getattr(engine, "_routing_envelope", None)
    env_bbox = envelope.bbox_nm if envelope is not None else None
    wide_fixed = engine._wide_indexes(False)
    atoms_by_net_gi: dict[tuple[str, int], list[tuple[int, int, int, int]]] = {}
    for atom in getattr(engine, "_owned_terminal_atoms", ()):
        atoms_by_net_gi.setdefault((str(atom.owner), int(atom.gi)), []).append(
            tuple(atom.rect)
        )

    reach = 1600
    deck_layout = None
    deck_top = None

    for net, pins in pins_by_net.items():
        if not any(getattr(p, "terminal_binding", None) is not None for p in pins):
            continue
        nodes = net_nodes.get(net, ())
        node_pins = net_node_pins.get(net, ())
        lg = net_layers.get(net, ())
        rails = engine._rail_by_net.get(net, ())
        if not lg or len(nodes) != len(node_pins) or (len(nodes) < 2 and not rails):
            continue
        lg_set = set(lg)
        suspects: list[tuple[int, object, tuple[int, int, int], db.Region]] = []

        for idx, (pin, node) in enumerate(zip(node_pins, nodes)):
            if getattr(pin, "terminal_binding", None) is None:
                continue
            px, py, gi = int(node[0]), int(node[1]), int(node[2])
            if gi not in lg_set:
                continue
            half = int(engine._half_w[gi])
            sp = int(engine._space[gi])
            obs_idx = getattr(engine, "_obs_idx", {}).get(gi)
            wide_rows = wide_fixed.get(gi, ())

            # Fast 4-ray check (~3us): if any straight 1600nm cardinal ray is clear, pin escapes
            ray_escaped = False
            for rx0, ry0, rx1, ry1 in (
                (px, py, px + reach, py),
                (px - reach, py, px, py),
                (px, py, px, py + reach),
                (px, py - reach, px, py),
            ):
                if env_bbox is not None and (
                    rx0 - half < env_bbox[0]
                    or ry0 - half < env_bbox[1]
                    or rx1 + half > env_bbox[2]
                    or ry1 + half > env_bbox[3]
                ):
                    continue
                blocked = False
                if obs_idx is not None:
                    q = (
                        rx0 - half - sp,
                        ry0 - half - sp,
                        rx1 + half + sp,
                        ry1 + half + sp,
                    )
                    for owner, _rect in obs_idx.query(q):
                        if owner != net:
                            blocked = True
                            break
                if not blocked and wide_rows:
                    for wspace, core, _win, _att in wide_rows:
                        wq = (
                            rx0 - half - int(wspace),
                            ry0 - half - int(wspace),
                            rx1 + half + int(wspace),
                            ry1 + half + int(wspace),
                        )
                        for owner, _rect in core.query(wq):
                            if owner != net:
                                blocked = True
                                break
                        if blocked:
                            break
                if not blocked:
                    ray_escaped = True
                    break
            if ray_escaped:
                continue

            # Conservative Region pocket check in +-1600nm window
            layers_local = sorted(
                {gi} | ({gi + 1} & lg_set) | ({gi - 1} & lg_set)
            )
            win = (px - reach, py - reach, px + reach, py + reach)
            F_wire, F_via = _build_conservative_free_regions(
                engine,
                net,
                layers_local,
                win,
                min_width_only=True,
                static_only=True,
            )
            seed = db.Region(db.Box(px, py, px + 1, py + 1))
            owned_rects = atoms_by_net_gi.get((net, gi), ())
            own_reg = db.Region()
            if owned_rects:
                for ox0, oy0, ox1, oy1 in owned_rects:
                    own_reg.insert(db.Box(ox0, oy0, ox1 + 1, oy1 + 1))
                own_reg.merge()
                conn_own = own_reg.interacting(seed)
                if not conn_own.is_empty():
                    seed = (seed + conn_own).merged()

            comp = F_wire[gi].interacting(seed)
            reach_reg = (comp + seed).merged()
            # A reached free-centerline region can touch another already drawn
            # same-net terminal component, which can in turn reach another free
            # region.  This is an optimistic precheck: over-approximate that
            # union to a fixed point and leave actual legality to the router.
            # A single interacting() step can otherwise reject a real escape.
            iteration = 0
            while not own_reg.is_empty():
                joined = (reach_reg + own_reg.interacting(reach_reg)).merged()
                expanded = (joined + F_wire[gi].interacting(joined)).merged()
                if (expanded - reach_reg).is_empty():
                    break
                reach_reg = expanded
                iteration += 1
                if iteration % 256 == 0 and hasattr(engine, '_watchdog_check'):
                    engine._watchdog_check(net, 'static_terminal_escape_closure')
            bb = reach_reg.bbox()
            # Escape A: reaches +-1590nm window boundary
            if (
                bb.left <= px - reach + 10
                or bb.right >= px + reach - 10
                or bb.bottom <= py - reach + 10
                or bb.top >= py + reach - 10
            ):
                continue
            # Escape B: has >= 1 conservative N=via_redundancy via in reach_reg
            has_via = False
            for lo in (gi, gi - 1):
                if lo in F_via and not (F_via[lo] & reach_reg).is_empty():
                    has_via = True
                    break
            if has_via:
                continue
            # Escape C: fixed cut directly climbs from reach_reg to an allowed adjacent layer
            has_fixed_cut = False
            cq = (bb.left - 5, bb.bottom - 5, bb.right + 5, bb.top + 5)
            for lo, adj_gi in ((gi, gi + 1), (gi - 1, gi - 1)):
                if adj_gi in lg_set and lo >= 0:
                    for crect in engine._local_fixed_cuts(lo, cq):
                        cbox = db.Region(
                            db.Box(crect[0], crect[1], crect[2] + 1, crect[3] + 1)
                        )
                        if not (reach_reg & cbox).is_empty():
                            has_fixed_cut = True
                            break
                if has_fixed_cut:
                    break
            if has_fixed_cut:
                continue
            # Escape D: all other endpoints & rails of `net` already lie inside reach_reg on `gi`
            if not rails and all(
                qgi == gi
                and not (reach_reg & db.Region(db.Box(qx, qy, qx + 1, qy + 1))).is_empty()
                for qx, qy, qgi in nodes
            ):
                continue
            suspects.append((idx, pin, node, reach_reg))

        if not suspects:
            continue

        # Tier 2: Verify via DeckConnectStack that the suspect belongs to a disconnected
        # fixed-metal component and no co-component pin escapes.
        from clay.layout.route.deck_isolation import NetProbe, check_deck_net_isolation

        if deck_layout is None:
            deck_layout = db.Layout()
            deck_layout.dbu = 0.001
            deck_top = deck_layout.create_cell("STATIC_ESCAPE_GATE")
            layer_idx_cache: dict[tuple[int, int], int] = {}
            for lk, _owner, rect in getattr(engine, "_fixed_deck_literal_rects", ()):
                li = layer_idx_cache.get(lk)
                if li is None:
                    li = deck_layout.layer(int(lk[0]), int(lk[1]))
                    layer_idx_cache[lk] = li
                deck_top.shapes(li).insert(
                    db.Box(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
                )
            contact_rects = getattr(engine, "_fixed_contact_rects", ())
            if contact_rects:
                try:
                    ck = tuple(engine._tech.pdk.contact_key())
                except Exception:
                    ck = None
                if ck is not None:
                    cli = deck_layout.layer(int(ck[0]), int(ck[1]))
                    for rect in contact_rects:
                        deck_top.shapes(cli).insert(
                            db.Box(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
                        )

        probes = []
        for idx, (p, n) in enumerate(zip(node_pins, nodes)):
            lk = (
                tuple(p.layer)
                if isinstance(p.layer, tuple) and len(p.layer) == 2
                else tuple(engine._tech.gi_to_metal_key(n[2]))
            )
            probes.append(NetProbe(f"p_{idx}", lk, int(n[0]), int(n[1])))
        for r_idx, (r_gi, r_rect) in enumerate(rails):
            rk = tuple(engine._tech.gi_to_metal_key(r_gi))
            mx = (int(r_rect[0]) + int(r_rect[2])) // 2
            my = (int(r_rect[1]) + int(r_rect[3])) // 2
            probes.append(NetProbe(f"r_{r_idx}", rk, mx, my))

        deck_obs = check_deck_net_isolation(
            layout=deck_layout,
            cell=deck_top,
            stack=engine._deck_connect_stack,
            probes=probes,
            include_probe_partition=True,
        )
        comp_of_probe: dict[str, int] = {}
        for comp_id, group in enumerate(deck_obs.resolved_probe_groups):
            # Each component is a group of full probe identities
            # (name, layer, x, y), not a list of names in group[0].
            for probe in group:
                if isinstance(probe, tuple) and len(probe) == 4:
                    comp_of_probe[str(probe[0])] = comp_id
        # An unresolved probe is not an independent component: its connection
        # to any escape or rail is unknown.  Defer this net to the normal route
        # and physical checks rather than reject it as statically disconnected.
        if any(p_obj.net not in comp_of_probe for p_obj in probes):
            continue

        parent = {cid: cid for cid in set(comp_of_probe.values())}

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[rb] = ra

        by_binding: dict[tuple[str, str], int] = {}
        for idx, p in enumerate(node_pins):
            b = getattr(p, "terminal_binding", None)
            if b is not None and hasattr(b, "stamp_id") and hasattr(b, "component_id"):
                bkey = (str(b.stamp_id), str(b.component_id))
                cid = comp_of_probe[f"p_{idx}"]
                if bkey in by_binding:
                    _union(by_binding[bkey], cid)
                else:
                    by_binding[bkey] = cid
        for k in list(comp_of_probe.keys()):
            comp_of_probe[k] = _find(comp_of_probe[k])

        distinct_comps = set(comp_of_probe.values())
        if len(distinct_comps) <= 1:
            continue

        suspect_by_idx = {idx: (pin, node, reg) for idx, pin, node, reg in suspects}
        for idx, pin, node, reach_reg in suspects:
            my_comp = comp_of_probe[f"p_{idx}"]
            # If any rail belongs to my_comp and there are no other components, already handled;
            # check if ANY other pin in my_comp is NOT a suspect (i.e. can escape!)
            sibling_escapes = any(
                comp_of_probe.get(f"p_{other_i}") == my_comp
                and other_i not in suspect_by_idx
                for other_i in range(len(nodes))
            )
            if sibling_escapes:
                continue
            # Also check if a rail is already in my_comp and all other pins can reach rails
            if any(
                comp_of_probe.get(f"r_{r_i}") == my_comp for r_i in range(len(rails))
            ):
                continue
            # Check if reach_reg on `node[2]` touches any pin of another component
            touches_other_comp = any(
                comp_of_probe.get(f"p_{other_i}") != my_comp
                and nodes[other_i][2] == node[2]
                and not (
                    reach_reg
                    & db.Region(
                        db.Box(
                            nodes[other_i][0],
                            nodes[other_i][1],
                            nodes[other_i][0] + 1,
                            nodes[other_i][1] + 1,
                        )
                    )
                ).is_empty()
                for other_i in range(len(nodes))
            )
            if touches_other_comp:
                continue
            bb = reach_reg.bbox()
            raise ValueError(
                f"terminal static escape gate rejected net {net!r}: bound terminal "
                f"{pin.device}.{pin.terminal} at ({node[0]}, {node[1]}) on layer "
                f"{pin.layer} is sealed inside fixed-metal pocket "
                f"[{bb.left}, {bb.bottom}, {bb.right}, {bb.top}] nm with 0 legal "
                f"via_redundancy={engine.via_redundancy} escapes and 0 planar wire exits"
            )
