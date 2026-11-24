"""Bind group landing replay to the active routing engine and committed world."""
import hashlib
import json
from types import SimpleNamespace

from clay.layout.route.group_fallback_order import ordered_alternatives
from clay.layout.route.canonical_input_capture import _json_value
from clay.layout.route.planar_precheck import (
    check_watched_pocket_starved,
    precompute_group_escape_mouths,
    precompute_group_sealed_escape_vias,
)


class OverlayAdapter:
    def __init__(self, engine, book, local, regenerate, close_port, *, checkpoint,
                 physical_feedback, observer=None):
        self.engine, self.book, self.local = engine, book, local
        self.regenerate, self.close_port = regenerate, close_port
        self.checkpoints, self.feedback, self.observer = checkpoint, physical_feedback, observer
        self._group_ports_by_name = {}
        self._mouth_cache = {}
        self._escape_watch_cache = {}
        self._neg_base_history = {}
        self._neg_group_history = {}
        self._reservation_history = {}

    def snapshot(self):
        return self.checkpoints.capture(self.book, self.engine, self.local)

    def restore(self, snapshot):
        self.checkpoints.restore(snapshot, self.book, self.engine, self.local)

    def snapshot_key(self, snapshot):
        return self.checkpoints.digest(snapshot)

    def checkpoint(self, port, phase):
        self.engine._watchdog_check(port.net, phase)

    def candidates(self, port):
        return self.regenerate(port)

    def via_redundancy(self):
        return self.engine.via_redundancy

    def supports_group_negotiation(self):
        return (hasattr(self.engine, '_history')
                and hasattr(self.engine, '_res_cell')
                and int(getattr(self.engine, '_res_cell', 0) or 0) > 0
                and hasattr(self.engine, '_book'))

    def _set_history(self, new_history):
        if not hasattr(self.engine, '_history') or new_history is None:
            return
        updated = dict(new_history)
        if self.engine._history != updated:
            self.engine._history = updated
            if hasattr(self.engine, '_history_generation'):
                self.engine._history_generation = int(self.engine._history_generation) + 1

    def save_history(self):
        if not hasattr(self.engine, '_history') or self.engine._history is None:
            self._neg_base_history = {}
            self._neg_group_history = {}
            return None
        res = dict(self.engine._history)
        self._neg_base_history = dict(res)
        self._neg_group_history = {}
        return res

    def restore_history(self, saved):
        self._neg_group_history = {}
        if saved is not None:
            self._neg_base_history = dict(saved)
            self._set_history(saved)

    def apply_negotiated_history(self, base_history, group_history):
        if not hasattr(self.engine, '_history'):
            return
        self._neg_base_history = {} if base_history is None else dict(base_history)
        self._neg_group_history = {} if group_history is None else dict(group_history)
        merged = dict(self._neg_base_history)
        collapsed = {}
        for key, penalty in self._neg_group_history.items():
            if type(key) is tuple and len(key) == 5 and key[0] == '__victim__':
                cell = (key[2], key[3], key[4])
                fval = float(penalty)
                if fval > collapsed.get(cell, 0.0):
                    collapsed[cell] = fval
            elif type(key) is tuple and len(key) == 6 and key[0] == '__owner_victim__':
                cell = (key[3], key[4], key[5])
                fval = float(penalty)
                if fval > collapsed.get(cell, 0.0):
                    collapsed[cell] = fval
            else:
                merged[key] = merged.get(key, 0.0) + float(penalty)
        for cell, val in collapsed.items():
            merged[cell] = merged.get(cell, 0.0) + val
        self._set_history(merged)

    def prepare_escape_starvation_watch(self, group_name, group_ports, clean_start_snapshot=None):
        self._group_ports_by_name[group_name] = tuple(group_ports)
        if (group_name not in self._escape_watch_cache
                and self.supports_group_negotiation()
                and hasattr(self.engine, '_tech')
                and hasattr(self.engine, '_obs_idx')):
            if clean_start_snapshot is not None:
                cur_snap = self.snapshot()
                try:
                    self.restore(clean_start_snapshot)
                    self._escape_watch_cache[group_name] = precompute_group_sealed_escape_vias(
                        self.engine, tuple(group_ports)
                    )
                finally:
                    self.restore(cur_snap)
            else:
                self._escape_watch_cache[group_name] = precompute_group_sealed_escape_vias(
                    self.engine, tuple(group_ports)
                )

    def prepare_group_negotiation(self, group_name, group_ports, clean_start_snapshot=None):
        self._group_ports_by_name[group_name] = tuple(group_ports)
        need_mouths = group_name not in self._mouth_cache
        need_watch = group_name not in self._escape_watch_cache
        if ((need_mouths or need_watch)
                and self.supports_group_negotiation()
                and hasattr(self.engine, '_tech')
                and hasattr(self.engine, '_obs_idx')):
            if clean_start_snapshot is not None:
                cur_snap = self.snapshot()
                try:
                    self.restore(clean_start_snapshot)
                    if need_mouths:
                        self._mouth_cache[group_name] = precompute_group_escape_mouths(
                            self.engine, tuple(group_ports)
                        )
                    if need_watch:
                        self._escape_watch_cache[group_name] = precompute_group_sealed_escape_vias(
                            self.engine, tuple(group_ports)
                        )
                finally:
                    self.restore(cur_snap)
            else:
                if need_mouths:
                    self._mouth_cache[group_name] = precompute_group_escape_mouths(
                        self.engine, tuple(group_ports)
                    )
                if need_watch:
                    self._escape_watch_cache[group_name] = precompute_group_sealed_escape_vias(
                        self.engine, tuple(group_ports)
                    )

    def first_starved_remaining_port(self, remaining_ports, frames=()):
        """Detect in <1ms if prior same-group routes choked every escape via of a sealed pocket."""
        if not remaining_ports or not self.supports_group_negotiation():
            return None
        group_name = remaining_ports[0].group
        watch_by_net = self._escape_watch_cache.get(group_name)
        if not watch_by_net or not hasattr(self.engine, '_via_legal'):
            return None
        prior_group_nets = {
            f.port.net for f in frames if getattr(f.port, 'group', None) == group_name
        }
        if not prior_group_nets:
            return None
        if getattr(self.engine, '_book', None) is not self.book and hasattr(self.engine, '_rebuild_metal_index'):
            self.engine._book = self.book
            self.engine._rebuild_metal_index()
        missing = object()
        prev_closure = getattr(self.engine, '_active_group_route_local_closure', missing)
        outer_rej = getattr(self.engine, '_hard_rejections', {})
        outer_samp = getattr(self.engine, '_hard_rejection_samples', {})
        outer_gen = getattr(self.engine, '_hard_rejection_generation', 0)
        prev_metal_scans = getattr(self.engine, '_active_static_metal_scans', None)
        prev_cut_scans = getattr(self.engine, '_active_static_cut_scans', None)
        prev_obs_scans = getattr(self.engine, '_active_static_obstacle_scans', None)
        prev_att_cuts = getattr(self.engine, '_attempt_cuts', {})
        self.engine._active_group_route_local_closure = True
        self.engine._attempt_cuts = {}
        if prev_metal_scans is None:
            self.engine._active_static_metal_scans = {}
        if prev_cut_scans is None:
            self.engine._active_static_cut_scans = {}
        if prev_obs_scans is None:
            self.engine._active_static_obstacle_scans = {}
        try:
            from clay.layout.route.group_fallback_controller import LocalInconclusive

            found: list[LocalInconclusive] = []
            for rem in remaining_ports:
                pockets = watch_by_net.get(rem.net)
                if not pockets:
                    continue
                rem_samples: list = []
                rem_rejs: dict[tuple[str, str], int] = {}
                rem_chokes: list[tuple[str, int, tuple[int, int, int, int]]] = []
                for pocket in pockets:
                    pocket_chokes: list[tuple[str, int, tuple[int, int, int, int]]] = []
                    if not check_watched_pocket_starved(
                        self.engine, rem.net, pocket, choke_boxes_out=pocket_chokes
                    ):
                        continue
                    samples = (
                        self.engine._hard_rejection_sample_evidence()
                        if hasattr(self.engine, '_hard_rejection_sample_evidence')
                        else ()
                    )
                    has_prior_owner = any(
                        str(cat).startswith('committed_')
                        and str(owner) in prior_group_nets
                        and int(cnt) > 0
                        for (cat, owner), cnt in self.engine._hard_rejections.items()
                    )
                    if not has_prior_owner:
                        continue
                    rem_samples.extend(samples)
                    rem_chokes.extend(
                        row for row in pocket_chokes if row[0] in prior_group_nets
                    )
                    for (cat, owner), cnt in self.engine._hard_rejections.items():
                        if int(cnt) > 0:
                            rk = (str(cat), str(owner))
                            rem_rejs[rk] = rem_rejs.get(rk, 0) + int(cnt)
                if rem_rejs:
                    witness_rows = tuple(
                        (cat, owner, cnt)
                        for (cat, owner), cnt in sorted(rem_rejs.items())
                    )
                    details_rows = [('sealed_pocket_escape_starved', True)]
                    if rem_chokes:
                        details_rows.append(('throat_choke_boxes', tuple(dict.fromkeys(rem_chokes))))
                    probe = LocalInconclusive(
                        str(rem.group),
                        str(rem.net),
                        ('static_disconnected', 0, 0, 0, 0, 0, witness_rows),
                        tuple(details_rows),
                        RuntimeError(f'sealed pocket escape starved for {rem.group}/{rem.net}'),
                        tuple(rem_samples),
                    )
                    if probe.committed_owners & prior_group_nets:
                        found.append(probe)
            if not found:
                return None
            if len(found) == 1:
                return found[0]
            first = found[0]
            return LocalInconclusive(
                first.group,
                first.net,
                first.edge_fail,
                first.details,
                first.original,
                first.samples,
                tuple(found[1:]),
            )
        finally:
            self.engine._hard_rejections = outer_rej
            self.engine._hard_rejection_samples = outer_samp
            self.engine._hard_rejection_generation = outer_gen
            self.engine._active_static_metal_scans = prev_metal_scans
            self.engine._active_static_cut_scans = prev_cut_scans
            self.engine._active_static_obstacle_scans = prev_obs_scans
            self.engine._attempt_cuts = prev_att_cuts
            if prev_closure is missing:
                delattr(self.engine, '_active_group_route_local_closure')
            else:
                self.engine._active_group_route_local_closure = prev_closure

    def _refresh_port_lookahead_history(self, port):
        if (not bool(getattr(self.engine, '_negotiation_port_legalize', False))
                or not self.supports_group_negotiation()):
            return
        mouth_by_net = self._mouth_cache.get(port.group, {})
        merged = dict(self._neg_base_history)
        hist_fac = float(getattr(self.engine, 'hist_fac', 1.0) or 1.0)
        committed = self.local.get('committed_groups', set())
        per_victim_cell = {}
        for key, val in self._neg_group_history.items():
            if type(key) is tuple and len(key) == 5 and key[0] == '__victim__':
                if key[1] == port.net or (port.group, key[1]) in committed:
                    continue
                vc_key = (key[1], key[2], key[3], key[4])
                fval = float(val)
                if fval > per_victim_cell.get(vc_key, 0.0):
                    per_victim_cell[vc_key] = fval
            elif type(key) is tuple and len(key) == 6 and key[0] == '__owner_victim__':
                if key[1] != port.net or key[2] == port.net or (port.group, key[2]) in committed:
                    continue
                vc_key = (key[2], key[3], key[4], key[5])
                fval = float(val)
                if fval > per_victim_cell.get(vc_key, 0.0):
                    per_victim_cell[vc_key] = fval
            else:
                merged[key] = merged.get(key, 0.0) + float(val)
        own_fixed_cells = set()
        rc = int(getattr(self.engine, '_res_cell', 0) or 0)
        if rc > 0:
            for pgi, rx0, ry0, rx1, ry1 in self._port_terminal_rects(port):
                for cx in range(rx0 // rc, rx1 // rc + 1):
                    for cy in range(ry0 // rc, ry1 // rc + 1):
                        own_fixed_cells.add((cx, cy, pgi))
        v_cap = 90.0 * hist_fac
        victim_cells = {}
        for (_v_net, cx, cy, rgi), val in per_victim_cell.items():
            cell = (cx, cy, rgi)
            if cell in own_fixed_cells:
                continue
            victim_cells[cell] = min(v_cap, victim_cells.get(cell, 0.0) + val)
        for cell, val in victim_cells.items():
            merged[cell] = merged.get(cell, 0.0) + val
        peer_max = {}
        for peer in self._group_ports_by_name.get(port.group, ()):
            if peer.net == port.net or (peer.group, peer.net) in committed:
                continue
            peer_cells = mouth_by_net.get(peer.net)
            if peer_cells:
                for key, val in peer_cells.items():
                    fval = float(val)
                    if fval > peer_max.get(key, 0.0):
                        peer_max[key] = fval
        for key, val in peer_max.items():
            if key in own_fixed_cells:
                continue
            merged[key] = merged.get(key, 0.0) + val
        for key, val in self._reservation_history.items():
            merged[key] = merged.get(key, 0.0) + val
        self._set_history(merged)

    def _port_terminal_rects(self, port):
        if not (hasattr(port, 'accesses') and hasattr(getattr(self.engine, '_tech', None), 'resolve_layer_to_gi')):
            return ()
        obs_map = getattr(self.engine, '_obs_idx', None) or {}
        out = []
        seen_all = set()
        for access in port.accesses:
            pgi = int(self.engine._tech.resolve_layer_to_gi(access.layer_key))
            px, py = int(access.xy_nm[0]), int(access.xy_nm[1])
            pt_key = (pgi, px, py, px, py)
            if pt_key not in seen_all:
                seen_all.add(pt_key)
                out.append(pt_key)
            oidx_pgi = obs_map.get(pgi) if hasattr(obs_map, 'get') else None
            if oidx_pgi is not None and hasattr(oidx_pgi, 'query'):
                frontier = [(px, py, px, py)]
                seen_rects = set()
                while frontier:
                    bx0, by0, bx1, by1 = frontier.pop()
                    for owner_r, r_raw in oidx_pgi.query((bx0 - 1, by0 - 1, bx1 + 1, by1 + 1)):
                        if owner_r != port.net:
                            continue
                        rx0, ry0 = min(int(r_raw[0]), int(r_raw[2])), min(int(r_raw[1]), int(r_raw[3]))
                        rx1, ry1 = max(int(r_raw[0]), int(r_raw[2])), max(int(r_raw[1]), int(r_raw[3]))
                        rk = (rx0, ry0, rx1, ry1)
                        if rk in seen_rects:
                            continue
                        if rx0 <= bx1 and rx1 >= bx0 and ry0 <= by1 and ry1 >= by0:
                            seen_rects.add(rk)
                            frontier.append(rk)
                for rx0, ry0, rx1, ry1 in seen_rects:
                    full_k = (pgi, rx0, ry0, rx1, ry1)
                    if full_k not in seen_all:
                        seen_all.add(full_k)
                        out.append(full_k)
        return tuple(out)

    def _congestion_margin_nm(self, net, gi):
        hw = self.engine._hw(net, gi) if hasattr(self.engine, '_hw') else 70
        sp = (int(self.engine._tech.min_space_for_gi(gi))
              if hasattr(getattr(self.engine, '_tech', None), 'min_space_for_gi')
              else 140)
        pad = 160
        return max(140, int(hw) + int(sp) + pad)

    def _add_box_congestion(self, target_map, resource_gi, box, penalty, *, cap=None, victim_net=None):
        rc = int(getattr(self.engine, '_res_cell', 0) or 0)
        if rc <= 0:
            return False
        x0, y0, x1, y1 = box
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        added = False
        for cx in range(x0 // rc, x1 // rc + 1):
            for cy in range(y0 // rc, y1 // rc + 1):
                key = (
                    ('__victim__', str(victim_net), cx, cy, int(resource_gi))
                    if victim_net is not None
                    else (cx, cy, int(resource_gi))
                )
                prev = target_map.get(key, 0.0)
                nxt = prev + float(penalty)
                if cap is not None and nxt > float(cap):
                    nxt = max(prev, float(cap))
                if nxt > prev:
                    target_map[key] = nxt
                    added = True
        return added

    def deposit_causal_congestion(self, target_ports, victim_port, failure,
                                  group_history, round_index):
        if not self.supports_group_negotiation():
            return False
        samples = getattr(failure, 'samples', ())
        if type(samples) is not tuple:
            samples = ()
        target_by_net = {p.net: p for p in target_ports}
        if not target_by_net:
            return False
        hist_fac = float(getattr(self.engine, 'hist_fac', 1.0) or 1.0)
        pen = hist_fac * min(9.0, 3.5 + 1.5 * max(0, int(round_index) - 1))
        cell_cap = 84.0 * hist_fac
        details_rows = getattr(failure, 'details', ())
        is_starved_escape = False
        throat_chokes = ()
        if type(details_rows) is tuple:
            for drow in details_rows:
                if type(drow) is tuple and len(drow) == 2:
                    if drow[0] == 'sealed_pocket_escape_starved' and bool(drow[1]):
                        is_starved_escape = True
                    elif drow[0] == 'throat_choke_boxes' and type(drow[1]) is tuple:
                        throat_chokes = drow[1]
        m1_cong_scale = 1.0
        n_layers = int(getattr(self.engine, '_n_layers', 5) or 5)
        obs_map = getattr(self.engine, '_obs_idx', None) or {}
        rc = int(getattr(self.engine, '_res_cell', 0) or 0)
        victim_rects = [
            (rx0, ry0, rx1, ry1)
            for _pgi, rx0, ry0, rx1, ry1 in self._port_terminal_rects(victim_port)
        ]
        if not victim_rects:
            victim_rects = [
                (int(access.xy_nm[0]), int(access.xy_nm[1]), int(access.xy_nm[0]), int(access.xy_nm[1]))
                for access in getattr(victim_port, 'accesses', ())
                if hasattr(access, 'xy_nm')
            ]
        round_deposits = {}
        round_cell_rank = {}
        owner_sample_deposits = {}
        owner_sample_rank = {}
        owner_round_deposits = {}
        priced_cells = 0

        def record_owner_box(owner_str, resource_gi, box, amount):
            nonlocal priced_cells
            if rc <= 0:
                return
            x0, y0, x1, y1 = box
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            rgi = int(resource_gi)
            amt = float(amount) * (m1_cong_scale if rgi == 0 else 1.0)
            for cx in range(x0 // rc, x1 // rc + 1):
                for cy in range(y0 // rc, y1 // rc + 1):
                    priced_cells += 1
                    if priced_cells % 1024 == 0 and hasattr(self.engine, '_watchdog_check'):
                        self.engine._watchdog_check(victim_port.net, 'group_causal_congestion')
                    ock = (owner_str, cx, cy, rgi)
                    if amt > owner_round_deposits.get(ock, 0.0):
                        owner_round_deposits[ock] = amt

        def record_round_box(owner_str, resource_gi, box, amount, rej_count=1):
            nonlocal priced_cells
            if rc <= 0:
                return
            x0, y0, x1, y1 = box
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            rgi = int(resource_gi)
            amt = float(amount) * (m1_cong_scale if rgi == 0 else 1.0)
            rej_i = int(rej_count) if type(rej_count) is int and rej_count > 0 else 1
            for cx in range(x0 // rc, x1 // rc + 1):
                mx = int((cx + 0.5) * rc)
                for cy in range(y0 // rc, y1 // rc + 1):
                    priced_cells += 1
                    if priced_cells % 1024 == 0 and hasattr(self.engine, '_watchdog_check'):
                        self.engine._watchdog_check(victim_port.net, 'group_causal_congestion')
                    my = int((cy + 0.5) * rc)
                    d_nm = (
                        min(
                            max(rx0 - mx, 0, mx - rx1) + max(ry0 - my, 0, my - ry1)
                            for rx0, ry0, rx1, ry1 in victim_rects
                        )
                        if victim_rects
                        else 0
                    )
                    rank = (0 if d_nm <= 2400 else 1, -rej_i, d_nm, cx, cy, rgi)
                    ck = (cx, cy, rgi)
                    if amt > round_deposits.get(ck, 0.0):
                        round_deposits[ck] = amt
                    prev_rank = round_cell_rank.get(ck)
                    if prev_rank is None or rank < prev_rank:
                        round_cell_rank[ck] = rank
                    ock = (owner_str, cx, cy, rgi)
                    if amt > owner_sample_deposits.get(ock, 0.0):
                        owner_sample_deposits[ock] = amt
                    prev_o_rank = owner_sample_rank.get(ock)
                    if prev_o_rank is None or rank < prev_o_rank:
                        owner_sample_rank[ock] = rank

        changed = False
        for raw in samples:
            if (type(raw) is not tuple or any(type(row) is not tuple or len(row) != 2
                    or type(row[0]) is not str for row in raw)
                    or len({row[0] for row in raw}) != len(raw)):
                continue
            sample = dict(raw)
            owner = sample.get('owner')
            category = sample.get('category')
            gi = sample.get('gi')
            blocker = sample.get('blocker_rect')
            if (owner not in target_by_net or type(category) is not str
                    or not category.startswith('committed_')
                    or type(gi) is not int or type(blocker) is not tuple
                    or len(blocker) != 4 or any(type(v) is not int for v in blocker)):
                continue
            owner_port = target_by_net[owner]
            if not self.owns_committed_witness(owner_port, SimpleNamespace(samples=(raw,))):
                continue
            cand = sample.get('candidate_rect')
            req = sample.get('required_space')
            rej_cnt = sample.get('rejection_count', 1)
            req_nm = int(req) if type(req) is int and req > 0 else 0
            locus = blocker
            victim_pad_rejected = False
            if (type(cand) is tuple and len(cand) == 4
                    and all(type(v) is int for v in cand)):
                candidate_half = sample.get('candidate_half')
                owner_half = (int(self.engine._hw(owner, gi))
                              if hasattr(self.engine, '_hw') else 70)
                if (category.startswith(('committed_metal_', 'committed_via_pad_', 'committed_cut_'))
                        and type(candidate_half) is int
                        and candidate_half > owner_half):
                    victim_pad_rejected = True
                    # The recorder aggregates rejected candidates by blocker.
                    # Price their extent as a soft signal for the next owner
                    # route; it is neither a single via nor a hard reservation.
                    locus = (cand[0] - req_nm, cand[1] - req_nm,
                             cand[2] + req_nm, cand[3] + req_nm)
                else:
                    band = req_nm + 40
                    lx0 = max(blocker[0], cand[0] - band)
                    ly0 = max(blocker[1], cand[1] - band)
                    lx1 = min(blocker[2], cand[2] + band)
                    ly1 = min(blocker[3], cand[3] + band)
                    if lx0 < lx1 and ly0 < ly1:
                        locus = (lx0, ly0, lx1, ly1)
            owner_str = str(owner)
            eff_via_pen = max(pen, 8.0 * hist_fac) if is_starved_escape else pen
            if category.startswith('committed_cut_'):
                grown = (locus[0] - 140, locus[1] - 140,
                         locus[2] + 140, locus[3] + 140)
                v_lo = min(gi, max(0, n_layers - 2))
                record_round_box(owner_str, n_layers + v_lo, grown, eff_via_pen, rej_cnt)
                record_round_box(owner_str, v_lo, grown, eff_via_pen * 0.75, rej_cnt)
                if v_lo + 1 < n_layers:
                    record_round_box(owner_str, v_lo + 1, grown, eff_via_pen * 0.75, rej_cnt)
            elif category.startswith('committed_via_pad_'):
                grown = (locus[0] - 140, locus[1] - 140,
                         locus[2] + 140, locus[3] + 140)
                if gi < n_layers - 1:
                    record_round_box(owner_str, n_layers + gi, grown, eff_via_pen, rej_cnt)
                if gi > 0:
                    record_round_box(owner_str, n_layers + gi - 1, grown, eff_via_pen, rej_cnt)
                record_round_box(owner_str, gi, grown, eff_via_pen * 0.75, rej_cnt)
            else:
                loci = [locus]
                oidx_gi = obs_map.get(gi) if hasattr(obs_map, 'get') else None
                if oidx_gi is not None and hasattr(oidx_gi, 'query'):
                    for owner_r, r_raw in oidx_gi.query(locus):
                        if owner_r != owner or not loci:
                            continue
                        cx0, cy0 = min(int(r_raw[0]), int(r_raw[2])), min(int(r_raw[1]), int(r_raw[3]))
                        cx1, cy1 = max(int(r_raw[0]), int(r_raw[2])), max(int(r_raw[1]), int(r_raw[3]))
                        nxt_loci = []
                        for bx0, by0, bx1, by1 in loci:
                            if cx1 <= bx0 or cx0 >= bx1 or cy1 <= by0 or cy0 >= by1:
                                nxt_loci.append((bx0, by0, bx1, by1))
                                continue
                            if bx0 < cx0:
                                nxt_loci.append((bx0, by0, cx0, by1))
                            if cx1 < bx1:
                                nxt_loci.append((cx1, by0, bx1, by1))
                            ix0, ix1 = max(bx0, cx0), min(bx1, cx1)
                            if ix0 < ix1:
                                if by0 < cy0:
                                    nxt_loci.append((ix0, by0, ix1, cy0))
                                if cy1 < by1:
                                    nxt_loci.append((ix0, cy1, ix1, by1))
                        loci = nxt_loci
                hw_owner = int(self.engine._hw(owner, gi)) if hasattr(self.engine, '_hw') else 70
                for sub_l in loci:
                    grown = (sub_l[0] - hw_owner, sub_l[1] - hw_owner,
                             sub_l[2] + hw_owner, sub_l[3] + hw_owner)
                    record_round_box(owner_str, gi, grown, pen, rej_cnt)
                    if victim_pad_rejected and gi < n_layers - 1:
                        record_round_box(owner_str, n_layers + gi, grown, pen * 0.5, rej_cnt)
                if (is_starved_escape and victim_pad_rejected
                        and type(cand) is tuple and len(cand) == 4):
                    ix0 = max(blocker[0], cand[0] - req_nm)
                    iy0 = max(blocker[1], cand[1] - req_nm)
                    ix1 = min(blocker[2], cand[2] + req_nm)
                    iy1 = min(blocker[3], cand[3] + req_nm)
                    if ix0 <= ix1 and iy0 <= iy1:
                        record_owner_box(
                            str(owner), gi, (ix0, iy0, ix1, iy1), max(pen, 18.0 * hist_fac)
                        )
        if throat_chokes and rc > 0:
            from clay.layout.route.engine_gridless import (
                _CONG_WEIGHT,
                _STEP_COST_PER_NM,
                _VIA_COST,
            )
            unit_cell_cost = float(max(1, rc) * _STEP_COST_PER_NM * _CONG_WEIGHT)
            two_via_hv = (2.0 * float(_VIA_COST)) / unit_cell_cost
            choke_amt = min(cell_cap, max(pen, 0.95 * two_via_hv * hist_fac))
            owner_term_cache = {}
            choke_tiles_by_owner = {}
            for ch_row in throat_chokes:
                if (type(ch_row) is not tuple or len(ch_row) != 3
                        or ch_row[0] not in target_by_net
                        or type(ch_row[1]) is not int
                        or type(ch_row[2]) is not tuple or len(ch_row[2]) != 4):
                    continue
                ch_owner, ch_gi, cbox = ch_row
                used_for_owner = choke_tiles_by_owner.setdefault(ch_owner, set())
                o_rects = owner_term_cache.get((ch_owner, ch_gi))
                if o_rects is None:
                    o_rects = [
                        (rx0, ry0, rx1, ry1)
                        for pgi_o, rx0, ry0, rx1, ry1 in self._port_terminal_rects(target_by_net[ch_owner])
                        if pgi_o == ch_gi
                    ]
                    owner_term_cache[(ch_owner, ch_gi)] = o_rects
                sp_o = (
                    int(self.engine._tech.min_space_for_gi(ch_gi))
                    if hasattr(getattr(self.engine, '_tech', None), 'min_space_for_gi')
                    else 140
                )
                hw_o = int(self.engine._hw(ch_owner, ch_gi)) if hasattr(self.engine, '_hw') else 70
                pin_guard = sp_o + hw_o
                cx0, cx1 = min(int(cbox[0]), int(cbox[2])), max(int(cbox[0]), int(cbox[2]))
                cy0, cy1 = min(int(cbox[1]), int(cbox[3])), max(int(cbox[1]), int(cbox[3]))
                for t_cx in range(cx0 // rc, cx1 // rc + 1):
                    for t_cy in range(cy0 // rc, cy1 // rc + 1):
                        if len(used_for_owner) >= 12 and (t_cx, t_cy) not in used_for_owner:
                            continue
                        pt_x = int((t_cx + 0.5) * rc)
                        pt_y = int((t_cy + 0.5) * rc)
                        if any(
                            rx0 - pin_guard <= pt_x <= rx1 + pin_guard
                            and ry0 - pin_guard <= pt_y <= ry1 + pin_guard
                            for rx0, ry0, rx1, ry1 in o_rects
                        ):
                            continue
                        used_for_owner.add((t_cx, t_cy))
                        record_owner_box(
                            str(ch_owner), ch_gi, (pt_x, pt_y, pt_x, pt_y), choke_amt
                        )
        by_tile = {}
        tile_rank = {}
        for ck in round_deposits:
            tile = (ck[0], ck[1])
            by_tile.setdefault(tile, []).append(ck)
            rk = round_cell_rank[ck]
            prev_rk = tile_rank.get(tile)
            if prev_rk is None or rk < prev_rk:
                tile_rank[tile] = rk
        if len(by_tile) > 48:
            kept_tiles = set(sorted(by_tile, key=lambda t: tile_rank[t])[:48])
            selected_keys = [ck for t in by_tile if t in kept_tiles for ck in by_tile[t]]
        else:
            selected_keys = list(round_deposits.keys())
        v_net_str = str(victim_port.net)
        peer_hint_cap = 0.5 * hist_fac
        for (cx, cy, rgi) in selected_keys:
            amt = min(0.25 * hist_fac, round_deposits[(cx, cy, rgi)])
            key = ('__victim__', v_net_str, cx, cy, rgi)
            prev = group_history.get(key, 0.0)
            nxt = max(prev, min(peer_hint_cap, prev + amt))
            if nxt > prev:
                group_history[key] = nxt
                changed = True
        owner_by_tile = {}
        owner_tile_rank = {}
        for ock in owner_sample_deposits:
            otile = (ock[0], ock[1], ock[2])
            owner_by_tile.setdefault(otile, []).append(ock)
            ork = owner_sample_rank[ock]
            prev_ork = owner_tile_rank.get(otile)
            if prev_ork is None or ork < prev_ork:
                owner_tile_rank[otile] = ork
        tiles_per_owner = {}
        for otile in owner_by_tile:
            tiles_per_owner.setdefault(otile[0], []).append(otile)
        selected_owner_keys = []
        for _o_str, otiles in tiles_per_owner.items():
            if len(otiles) > 48:
                kept_ot = set(sorted(otiles, key=lambda t: owner_tile_rank[t])[:48])
                selected_owner_keys.extend(
                    ock for t in otiles if t in kept_ot for ock in owner_by_tile[t]
                )
            else:
                selected_owner_keys.extend(ock for t in otiles for ock in owner_by_tile[t])
        for (o_net_str, cx, cy, rgi) in selected_owner_keys:
            amt = owner_sample_deposits[(o_net_str, cx, cy, rgi)]
            o_key = ('__owner_victim__', o_net_str, v_net_str, cx, cy, rgi)
            prev_o = group_history.get(o_key, 0.0)
            eff_cap = float(cell_cap) * (m1_cong_scale if rgi == 0 else 1.0)
            nxt_o = max(prev_o, min(eff_cap, prev_o + amt))
            if nxt_o > prev_o:
                group_history[o_key] = nxt_o
                changed = True
        for (o_net_str, cx, cy, rgi), amt in owner_round_deposits.items():
            o_key = ('__owner_victim__', o_net_str, v_net_str, cx, cy, rgi)
            prev_o = group_history.get(o_key, 0.0)
            eff_cap = float(cell_cap) * (m1_cong_scale if rgi == 0 else 1.0)
            nxt_o = max(prev_o, min(eff_cap, prev_o + amt))
            if nxt_o > prev_o:
                group_history[o_key] = nxt_o
                changed = True
        mouth_by_net = self._mouth_cache.setdefault(victim_port.group, {})
        v_cells = mouth_by_net.setdefault(victim_port.net, {})
        mouth_cap = 4.5 * hist_fac
        for key in list(v_cells.keys()):
            v_key = ('__victim__', str(victim_port.net), key[0], key[1], key[2])
            prev_m = group_history.get(v_key, 0.0)
            nxt_m = min(mouth_cap, prev_m + 0.75 * hist_fac)
            if nxt_m > prev_m:
                group_history[v_key] = nxt_m
                changed = True
        if (not changed and not v_cells and hasattr(victim_port, 'accesses')
                and hasattr(getattr(self.engine, '_tech', None), 'resolve_layer_to_gi')):
            for access in victim_port.accesses:
                a_gi = int(self.engine._tech.resolve_layer_to_gi(access.layer_key))
                px, py = access.xy_nm
                changed |= self._add_box_congestion(
                    v_cells, a_gi, (px - 280, py - 280, px + 280, py + 280), 1.5 * hist_fac,
                    cap=mouth_cap,
                )
                if 0 <= a_gi < n_layers - 1:
                    changed |= self._add_box_congestion(
                        v_cells, n_layers + a_gi, (px - 280, py - 280, px + 280, py + 280),
                        1.5 * hist_fac, cap=mouth_cap,
                    )
        return changed

    def recoverable_edge_budget_failure(self, port, exc, frames):
        from clay.layout.route.engine_gridless import GroupRouteBudgetStop
        from clay.layout.route.group_fallback_controller import LocalInconclusive

        if type(exc) is not GroupRouteBudgetStop:
            return None
        diag = exc.flow_error_data()
        if diag.get('reason') not in {'EDGE_EXPANSION_BUDGET', 'RESOURCE_BUDGET_EXCEEDED'}:
            return None
        if int(getattr(self.engine, 'max_expansions', 0) or 0) < 100000:
            return None
        if (getattr(self.engine, '_solve_expansions_total', 0)
                >= getattr(self.engine, '_solve_expansion_budget', 0)):
            return None
        if (not any(prior.port.group == port.group for prior in frames)
                and not bool(getattr(self.engine, '_history', None))):
            return None
        samples = (
            self.engine._hard_rejection_sample_evidence()
            if hasattr(self.engine, '_hard_rejection_sample_evidence')
            else ()
        )
        probe = SimpleNamespace(samples=samples)
        owners = tuple(sorted({
            prior.port.net
            for prior in frames
            if prior.port.group == port.group
            and self.owns_committed_witness(prior.port, probe)
        })) if samples else ()
        if not owners:
            return LocalInconclusive(
                str(port.group),
                str(port.net),
                ('edge_budget_incomplete', diag),
                getattr(self.engine, '_route_failure_details', ()),
                exc,
                samples,
            )
        witness_rows = tuple(('committed_metal_contact', owner, 1) for owner in owners)
        return LocalInconclusive(
            str(port.group),
            str(port.net),
            ('symbolic_domain_incomplete', ('exhausted', 1, 1, 1, 1, witness_rows), witness_rows),
            getattr(self.engine, '_route_failure_details', ()),
            exc,
            samples,
        )

    def _candidate_congestion_cost(self, port, row):
        if type(row) is not tuple or len(row) < 11:
            return 0
        state = row[7]
        via = row[8]
        portal = row[10]
        cost = 0
        if via is not None and hasattr(self.engine, '_via_congestion'):
            cost += int(self.engine._via_congestion(
                via.x, via.y, via.lo, port.net, via.n
            ))
        elif hasattr(self.engine, '_box_congestion'):
            hw = int(self.engine._hw(port.net, state[2])) if hasattr(self.engine, '_hw') else 70
            cost += int(self.engine._box_congestion(
                state[2],
                (state[0] - hw, state[1] - hw, state[0] + hw, state[1] + hw),
                port.net,
            ))
        if hasattr(self.engine, '_box_congestion'):
            phw = int(self.engine._hw(port.net, portal[2])) if hasattr(self.engine, '_hw') else 70
            cost += int(self.engine._box_congestion(
                portal[2],
                (portal[0] - phw, portal[1] - phw, portal[0] + phw, portal[1] + phw),
                port.net,
            ))
        return cost

    @staticmethod
    def _is_gridless_landing_row(row):
        return (type(row) is tuple and len(row) >= 11
                and all(type(row[i]) is int for i in range(4)))

    def _portal_target_gi(self, port):
        pref = getattr(self.engine, '_prototype_portal_preference', None)
        gi_to_pdk = getattr(getattr(self.engine, '_tech', None), '_gi_to_pdk', ())
        if pref is not None and gi_to_pdk:
            if isinstance(pref, dict) and (port.group, port.net) in pref:
                target_lvl = pref[(port.group, port.net)]
                return next((gi for gi, lvl in enumerate(gi_to_pdk) if lvl == target_lvl), None)
            if isinstance(pref, tuple) and len(pref) >= 3 and (port.group, port.net) == pref[:2]:
                target_lvl = pref[2]
                return next((gi for gi, lvl in enumerate(gi_to_pdk) if lvl == target_lvl), None)
        if getattr(self.engine, '_rail_by_net', {}) and port.net in self.engine._rail_by_net:
            return None
        promoted = getattr(self.engine, '_narrow_corridor_portal_gi', None) or {}
        pgi = promoted.get((port.group, port.net))
        if type(pgi) is int:
            return int(pgi)
        trunks = getattr(self.engine, '_declared_trunk_gi_by_net', None) or {}
        tgi = trunks.get(port.net)
        return int(tgi) if type(tgi) is int else None

    def _matched_portal_penalty_nm(self, port, row):
        fn = getattr(self.engine, "_matched_portal_layer_penalty_nm", None)
        if fn is None:
            return 0
        landings = ()
        if isinstance(getattr(self, "local", None), dict):
            landings = self.local.get("landings", ())
        return int(fn(port, row, landings))

    def _portal_trunk_penalty_nm(self, port, row):
        target_gi = self._portal_target_gi(port)
        if target_gi is None or not self._is_gridless_landing_row(row):
            return 0
        portal = row[10]
        if type(portal) is not tuple or len(portal) < 3 or type(portal[2]) is not int:
            return 0
        return abs(int(portal[2]) - target_gi) * 900

    def _is_floating_matched_port(self, port) -> bool:
        floating = getattr(self.engine, "_floating_matched_escape_anchors", None)
        return bool(
            floating
            and str(getattr(port, "net", "")) in floating
            and getattr(self.engine, "_matched_partners_by_net", {}).get(str(getattr(port, "net", "")))
        )

    def _gridless_landing_sort_key(
        self,
        port,
        row,
        *,
        include_congestion: bool = False,
        alt_order: bool = False,
    ):
        total_cost = (
            row[2]
            + self._portal_trunk_penalty_nm(port, row)
            + self._matched_portal_penalty_nm(port, row)
            + (min(32000, self._candidate_congestion_cost(port, row) // 4) if include_congestion else 0)
        )
        # Single-group floating matched pairs have no external target; let length/layer
        # balance decide before directional projection rank `row[1]`.
        rank_primary = 0 if self._is_floating_matched_port(port) else row[1]
        if alt_order:
            return (1 if row[0] >= 3 else 0, rank_primary, total_cost, row[0], row[1])
        return (1 if row[0] >= 3 else 0, row[0], rank_primary, total_cost, row[1])

    def default_candidate(self, port, rows):
        if (bool(getattr(self.engine, '_negotiation_port_legalize', False))
                and self.supports_group_negotiation()):
            self._refresh_port_lookahead_history(port)
            if (getattr(self.engine, '_experimental_group_demand_guide', None) is not None
                    and not self._neg_group_history
                    and not self._reservation_history):
                return self.engine._select_group_route_landing(
                    port, rows, self.book, self.local['landings']
                )
            if bool(getattr(self.engine, '_history', None)):
                scored = [
                    (
                        self._gridless_landing_sort_key(port, row, include_congestion=True)
                        if self._is_gridless_landing_row(row)
                        else (self._candidate_congestion_cost(port, row),),
                        row,
                    )
                    for row in rows
                ]
                min_cost = min(c for c, _r in scored)
                best_rows = [r for c, r in scored if c == min_cost]
                if getattr(self.engine, '_experimental_group_demand_guide', None) is not None:
                    return self.engine._select_group_route_landing(
                        port, best_rows, self.book, self.local['landings']
                    )
                return min(best_rows)
        if getattr(self.engine, '_experimental_group_demand_guide', None) is not None:
            return self.engine._select_group_route_landing(
                port, rows, self.book, self.local['landings'])
        has_match_partners = bool(
            getattr(self.engine, '_matched_partners_by_net', {}).get(str(port.net))
        )
        if self._portal_target_gi(port) is not None or has_match_partners:
            return min(
                rows,
                key=lambda r: (
                    (self._gridless_landing_sort_key(port, r), r)
                    if self._is_gridless_landing_row(r)
                    else (0, r)
                ),
            )
        return min(rows)

    def negotiated_alternate_candidates(self, port, rows, selected, limit=2):
        if limit <= 0 or len(rows) <= 1:
            return ()
        self._refresh_port_lookahead_history(port)
        alts = list(self.alternatives(port, rows, selected))
        if not alts:
            return ()
        portal_involved = True
        landing_involved = True
        raw_details = getattr(self.engine, '_route_failure_details', ())
        if (self._is_gridless_landing_row(selected)
                and type(raw_details) is tuple and raw_details):
            try:
                detail_map = dict(raw_details)
                endpoints = [
                    dict(detail_map[k]).get('device')
                    for k in ('source', 'goal')
                    if k in detail_map and type(detail_map[k]) is tuple
                ]
            except (TypeError, ValueError):
                endpoints = []
            portal_dev = f'__route_group__{port.group}'
            landing_dev = f'__route_group_landing__{port.group}'
            if endpoints and portal_dev not in endpoints:
                portal_involved = False
                if landing_dev not in endpoints:
                    landing_involved = False
        if not portal_involved and not landing_involved:
            return ()
        if not portal_involved and self._is_gridless_landing_row(selected):
            limit = min(limit, 1)
            sel_internal = (
                tuple(selected[6]),
                selected[7][0] // 280,
                selected[7][1] // 280,
                selected[7][2],
            )
            alts = [
                row for row in alts
                if not self._is_gridless_landing_row(row)
                or (
                    tuple(row[6]),
                    row[7][0] // 280,
                    row[7][1] // 280,
                    row[7][2],
                ) != sel_internal
            ]
            if not alts:
                return ()
        def _alt_bucket(row):
            if not portal_involved and self._is_gridless_landing_row(row):
                return (
                    tuple(row[6]),
                    row[9],
                    (row[7][0] // 560, row[7][1] // 560, row[7][2]),
                )
            return self.candidate_coarse_bucket(port, row)
        sel_bucket = _alt_bucket(selected)
        scored = [
            (
                self._gridless_landing_sort_key(
                    port, row, include_congestion=True, alt_order=True
                )
                if self._is_gridless_landing_row(row)
                else (self._candidate_congestion_cost(port, row),),
                idx,
                row,
            )
            for idx, row in enumerate(alts)
        ]
        scored.sort(key=lambda item: (item[0], item[1]))
        chosen = []
        used_keys = set()
        seen_buckets = {sel_bucket} if sel_bucket is not None else set()
        for _cost, _idx, row in scored:
            bucket = _alt_bucket(row)
            if bucket is not None and bucket in seen_buckets:
                continue
            if bucket is not None:
                seen_buckets.add(bucket)
            chosen.append(row)
            used_keys.add(self.candidate_key(port, row))
            if len(chosen) >= limit:
                return tuple(chosen)
        if sel_bucket is None:
            for _cost, _idx, row in scored:
                key = self.candidate_key(port, row)
                if key in used_keys:
                    continue
                chosen.append(row)
                used_keys.add(key)
                if len(chosen) >= limit:
                    break
        return tuple(chosen)

    def candidate_coarse_bucket(self, port, row):
        if (not self.supports_group_negotiation()
                or type(row) is not tuple or len(row) < 11):
            return None
        node = tuple(row[6])
        state = tuple(row[7])
        ikind = row[9]
        portal = tuple(row[10])
        return (
            node,
            ikind,
            (state[0] // 560, state[1] // 560, state[2]),
            (portal[0] // 560, portal[1] // 560, portal[2]),
        )

    def candidate_key(self, port, row):
        raw = json.dumps(_json_value(row), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(raw.encode()).hexdigest()

    def alternatives(self, port, rows, selected):
        return ordered_alternatives(
            port.net, rows, selected, getattr(port, 'envelope_nm', None)
        )

    def completed_route_reservation(self, port):
        """Return live, incremental metal for a completed group route.

        This is a search hint only.  The consuming trial must reroute both
        ports and run the ordinary same-invocation physical feedback.
        """
        key = (port.group, port.net)
        route = self.local['baselines'].get(port.net)
        contribution = self.local['group_contributions'].get(key)
        if (key not in self.local['committed_groups']
                or port.net not in self.book.persistent
                or route is None or not getattr(route, 'connected', False)
                or contribution is None):
            return ()
        incremental = contribution.get('incremental', {})
        boxes = set()
        for kind, collection in (('metal', self.book.metal),
                                 ('via_pad', self.book.via_pads)):
            for item in incremental.get(kind, ()):
                if (type(item) is not tuple or len(item) != 2
                        or type(item[0]) is not int or type(item[1]) is not tuple
                        or len(item[1]) != 4 or any(type(x) is not int for x in item[1])):
                    return ()
                gi, rect = item
                if (gi not in collection or rect[0] >= rect[2] or rect[1] >= rect[3]
                        or (port.net, rect) not in collection[gi]):
                    return ()
                boxes.add((gi, rect))
        return (port.net, tuple(sorted(boxes))) if boxes else ()

    def causal_reservations(self, port, failure):
        """Project exact same-group metal-contact samples into temporary trials."""
        if not self.owns_committed_witness(port, failure):
            return ()
        contribution = self.local['group_contributions'][(port.group, port.net)]['incremental']['metal']
        reservations = set()
        for raw in failure.samples:
            if (type(raw) is not tuple or any(type(part) is not tuple or len(part) != 2
                    or type(part[0]) is not str for part in raw)
                    or len({part[0] for part in raw}) != len(raw)):
                continue
            try:
                item = dict(raw)
            except (TypeError, ValueError):
                continue
            gi, obstacle, candidate = item.get('gi'), item.get('blocker_rect'), item.get('candidate_rect')
            if (item.get('category') != 'committed_metal_contact' or item.get('owner') != port.net
                    or type(gi) is not int or type(obstacle) is not tuple or len(obstacle) != 4
                    or any(type(v) is not int for v in obstacle) or (gi, obstacle) not in contribution
                    or type(candidate) is not tuple or len(candidate) != 4
                    or any(type(v) is not int for v in candidate)):
                continue
            x0, y0 = max(obstacle[0], candidate[0]), max(obstacle[1], candidate[1])
            x1, y1 = min(obstacle[2], candidate[2]), min(obstacle[3], candidate[3])
            if x0 < x1 and y0 < y1:
                reservations.add((gi, (x0, y0, x1, y1)))
        return (failure.net, tuple(sorted(reservations))) if reservations else ()

    def route_avoiding(self, port, selected, reservation):
        """Temporarily reserve victim overlap while rerouting its causal owner."""
        from clay.layout.route.group_fallback_controller import _PhysicalRejection
        owner, boxes = reservation
        if (type(owner) is not str or not owner or owner == port.net or type(boxes) is not tuple
                or not boxes):
            raise ValueError('invalid causal group reservation')
        temporary_owner = '__clay_group_causal_reservation__'
        if (temporary_owner in self.book.persistent
                or any(net == temporary_owner for rows in self.book.metal.values() for net, _ in rows)
                or any(net == temporary_owner for rows in self.book.via_pads.values() for net, _ in rows)
                or any(net == temporary_owner for net, _, _ in self.book.cuts)):
            raise ValueError('causal reservation owner is already present')
        entries = []
        prior_base_history = dict(self._neg_base_history)
        prior_group_history = dict(self._neg_group_history)
        prior_reservation_history = dict(self._reservation_history)
        current_history = getattr(self.engine, '_history', None)
        prior_history = None if current_history is None else dict(current_history)
        try:
            for gi, box in boxes:
                if (type(gi) is not int or gi not in self.book.metal or type(box) is not tuple
                        or len(box) != 4 or any(type(v) is not int for v in box)
                        or box[0] >= box[2] or box[1] >= box[3]):
                    raise ValueError('invalid causal reservation rectangle')
                entry = (temporary_owner, box)
                self.book.metal[gi].append(entry)
                entries.append((gi, entry))
            if prior_history is not None and self.supports_group_negotiation():
                hist_fac = float(getattr(self.engine, 'hist_fac', 1.0) or 1.0)
                boosted = dict(prior_history)
                for gi, box in boxes:
                    margin = self._congestion_margin_nm(port.net, gi)
                    self._add_box_congestion(
                        boosted, gi,
                        (box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin),
                        2.5 * hist_fac,
                        cap=4.0 * hist_fac,
                    )
                # route() refreshes peer/victim history before searching. Keep
                # this temporary increment separate so that refresh preserves
                # victim exemptions and does not discard the reservation price.
                if not prior_base_history and not prior_group_history:
                    self._neg_base_history = dict(prior_history)
                self._reservation_history = dict(prior_reservation_history)
                for key, val in boosted.items():
                    increment = val - prior_history.get(key, 0.0)
                    if increment > 0:
                        self._reservation_history[key] = (
                            self._reservation_history.get(key, 0.0) + increment
                        )
                self._set_history(boosted)
            self.engine._book = self.book
            self.engine._rebuild_metal_index()
            self.route(port, selected)
            # The chosen escape via can be published before search. Ensure no
            # retained owner metal or pad has bypassed the temporary resource.
            for gi, box in boxes:
                for rows in (self.book.metal[gi], self.book.via_pads[gi]):
                    for net, rect in rows:
                        if net == port.net and max(rect[0], box[0]) < min(rect[2], box[2]) and max(rect[1], box[1]) < min(rect[3], box[3]):
                            raise _PhysicalRejection()
        finally:
            self._neg_base_history = prior_base_history
            self._neg_group_history = prior_group_history
            self._reservation_history = prior_reservation_history
            if prior_history is not None:
                self._set_history(prior_history)
            for gi, entry in entries:
                self.book.metal[gi][:] = [row for row in self.book.metal[gi] if row[0] != temporary_owner]
            self.engine._book = self.book
            self.engine._rebuild_metal_index()

    def route(self, port, selected):
        if bool(getattr(self.engine, '_negotiation_port_legalize', False)):
            self._refresh_port_lookahead_history(port)
        def owner_world():
            metal={(int(gi),tuple(rect)) for gi,rows in self.book.metal.items()
                   for owner,rect in rows if owner==port.net}
            pads={(int(gi),tuple(rect)) for gi,rows in self.book.via_pads.items()
                  for owner,rect in rows if owner==port.net}
            cuts={(int(lo),tuple(rect)) for owner,rect,lo in self.book.cuts if owner==port.net}
            return {'metal':metal,'via_pad':pads,'via_cut':cuts}
        before_world=owner_world()
        old = self.local['baselines'].get(port.net)
        prior_baseline_geometry=bool(old is not None and (old.segs or old.vias))
        before = None if old is None else _json_value(old)
        try:
            self.close_port(port, selected)
        except ValueError as exc:
            edge_fail = getattr(self.engine, '_edge_fail', None)
            if (type(exc) is ValueError
                    and 'cannot close inside its PDK-derived local envelope' in str(exc)
                    and type(edge_fail) is tuple and len(edge_fail) >= 1
                    and edge_fail[0] in {
                        'candidate_source_union_unknown',
                        'disconnected',
                        'static_disconnected',
                        'planar_shared_endpoint_proof_failed',
                    }):
                from clay.layout.route.group_fallback_controller import LocalInconclusive
                samples = (
                    self.engine._hard_rejection_sample_evidence()
                    if hasattr(self.engine, '_hard_rejection_sample_evidence')
                    else ()
                )
                raise LocalInconclusive(
                    str(port.group),
                    str(port.net),
                    edge_fail,
                    getattr(self.engine, '_route_failure_details', ()),
                    exc,
                    samples,
                ) from exc
            raise
        after = _json_value(self.local['baselines'][port.net])
        if before != after:
            self.local['committed_groups'].add((port.group, port.net))
            after_world=owner_world()
            self.local['group_contributions'][(port.group,port.net)]={
                'had_prior_committed':any(before_world.values()) or prior_baseline_geometry,
                'incremental':{kind:frozenset(after_world[kind]-before_world[kind])
                               for kind in before_world}}

    def owns_committed(self, port, failure):
        key=(port.group,port.net)
        if key not in self.local['committed_groups'] or port.net not in self.book.persistent:
            return False
        contribution=self.local['group_contributions'].get(key)
        if contribution is None:return False
        if not contribution['had_prior_committed']:
            return True
        # An owner name alone cannot distinguish old same-net artwork from the
        # incremental group contribution. Admit only an exact retained blocker
        # sample that matches newly committed group geometry on the same layer.
        return self.owns_committed_witness(port, failure)

    def owns_committed_witness(self, port, failure):
        """Require an exact current-group blocker, even with no prior owner metal."""
        key = (port.group, port.net)
        if key not in self.local['committed_groups'] or port.net not in self.book.persistent:
            return False
        contribution = self.local['group_contributions'].get(key)
        if contribution is None:
            return False
        if type(failure.samples) is not tuple:
            return False
        for raw in failure.samples:
            if (type(raw) is not tuple or any(type(row) is not tuple or len(row) != 2
                or type(row[0]) is not str for row in raw)
                or len({row[0] for row in raw}) != len(raw)):
                continue
            sample=dict(raw);category=sample.get('category')
            if (sample.get('owner')!=port.net or type(category) is not str
                or not category.startswith('committed_')):
                continue
            kind=('via_cut' if category.startswith('committed_cut_') else
                  'via_pad' if category.startswith('committed_via_pad_') else
                  'metal' if category.startswith('committed_metal_') else None)
            rect=sample.get('blocker_rect');gi=sample.get('gi')
            if (kind is not None and type(gi) is int and type(rect) is tuple
                and len(rect)==4 and all(type(value) is int for value in rect)
                and (gi,rect) in contribution['incremental'][kind]):
                return True
        return False

    def physical_nonworsening(self, snapshot):
        return self.feedback(self.engine, self.local, snapshot)

    def record_outcome(self, port, key, rejected):
        if self.observer is not None:
            self.observer(port, key, rejected)

    def finish(self):
        return None
