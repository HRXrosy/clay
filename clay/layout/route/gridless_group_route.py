"""Group-route boundary portals, landing candidates, and transactional reservation mixin."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from clay.layout.route.engine import Pin
from clay.layout.route.engine_maze import _NM_PER_UM as _base_NM_PER_UM
from clay.layout.route.gridless_types import (
    GroupRouteBudgetStop,
    GroupRouteFullFeedbackUnavailable,
    GroupRouteLandingUnavailable,
    NegotiationRouteStop,
    NegotiationStopReason,
    Rect,
    _Book,
    _RoutedNet as _base_RoutedNet,
    _ViaSite,
    _WireSeg,
    _closed_rects_touch,
    _eg_attr,
    _group_route_pin_key,
    _norm,
    _pin_order_key,
    _rects_clear as _base_rects_clear,
    _seg_bbox,
    _sep as _base_sep,
)


class _GridlessGroupRouteMixin:
    """Group-route boundary portals and ordinary/transactional landing reservation."""

    def _prepare_group_route_reservation_context(
        self,
        contract: object,
        pins_by_net: Mapping[str, Sequence[Pin]],
        net_nodes: Dict[str, List[Tuple[int, int, int]]],
        net_node_pins: Dict[str, List[Pin]],
    ):
        """Validate contract/pin inventory and build shared group-reservation closures."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        from clay.layout.route.group_ports import GroupRouteRoutingContract

        if type(contract) is not GroupRouteRoutingContract:
            raise TypeError("group-route landing allocator requires an exact contract")
        pin_nodes: Dict[Tuple[object, ...], Tuple[Pin, Tuple[int, int, int]]] = {}
        for net in sorted(pins_by_net):
            nodes = tuple(net_nodes.get(net, ()))
            pins = tuple(net_node_pins.get(net, ()))
            if len(nodes) != len(pins):
                raise ValueError("group-route landing pin/node inventory is incomplete")
            for pin, node in zip(pins, nodes):
                if type(pin) is not Pin or pin.xy_nm is None:
                    continue
                layer_key = pin.layer
                if (
                    type(layer_key) is not tuple
                    or len(layer_key) != 2
                    or any(type(value) is not int for value in layer_key)
                ):
                    continue
                key = _group_route_pin_key(str(net), pin)
                if key in pin_nodes:
                    raise ValueError("group-route landing has duplicate exact Pins")
                pin_nodes[key] = (pin, tuple(node))

        required_states: Dict[
            Tuple[str, Tuple[int, int, int]],
            Tuple[int, int, int],
        ] = {}
        baselines: Dict[str, _RoutedNet] = {}
        landings = []
        grouped_pin_keys: Set[Tuple[object, ...]] = set()
        portals_by_net: Dict[
            str,
            List[Tuple[Tuple[int, int, int], Pin]],
        ] = {}

        def physical_layer_key(gi: int) -> Tuple[int, int]:
            layer_n = self._tech._gi_to_pdk[int(gi)]
            key = tuple(self._tech.pdk.metal_key(layer_n))
            if len(key) != 2 or any(type(value) is not int for value in key):
                raise ValueError("group-route landing lacks exact PDK metal authority")
            return key

        grouped_contract_pin_keys = frozenset(
            _group_route_pin_key(port.net, access)
            for port in contract.ports
            for access in port.accesses
        )

        def external_target_rects(port):
            return self._group_route_external_target_rects(
                port, contract, net_nodes, net_node_pins, grouped_contract_pin_keys)
        return (
            pin_nodes,
            required_states,
            baselines,
            landings,
            grouped_pin_keys,
            portals_by_net,
            physical_layer_key,
            external_target_rects,
        )

    def _finalize_group_route_landing_certificate(
        self,
        *,
        contract: object,
        landings: list,
        portals_by_net: Dict[str, List[Tuple[Tuple[int, int, int], Pin]]],
        net_nodes: Dict[str, List[Tuple[int, int, int]]],
        net_node_pins: Dict[str, List[Pin]],
        grouped_pin_keys: Set[Tuple[object, ...]],
        required_states: Dict[Tuple[str, Tuple[int, int, int]], Tuple[int, int, int]],
        baselines: Dict[str, _RoutedNet],
        physical_layer_key,
    ):
        """Verify landing certificate and publish global portals + baseline state."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        from clay.layout.route.group_ports import (
            GroupRouteLandingCertificate,
            verify_group_route_landing_certificate,
        )

        certificate = GroupRouteLandingCertificate(
            routing_contract_sha256=contract.sha256,
            landings=tuple(sorted(landings)),
        )
        verify_group_route_landing_certificate(
            contract,
            certificate,
            expected_via_redundancy=self.via_redundancy,
            allow_single_escape=self._single_via_escape_allowed(),
        )

        # Replace every grouped physical access in the global endpoint inventory
        # with exactly one upper-layer portal per (group, external net).  Ungrouped
        # endpoints remain byte-for-byte present and are the only global peers.
        for net, portal_rows in sorted(portals_by_net.items()):
            prior_nodes = tuple(net_nodes.get(net, ()))
            prior_pins = tuple(net_node_pins.get(net, ()))
            if len(prior_nodes) != len(prior_pins):
                raise ValueError("group-route global endpoint inventory is incomplete")
            retained = []
            for pin, node in zip(prior_pins, prior_nodes):
                if pin.xy_nm is None:
                    raise ValueError(
                        "group-route global endpoint lacks exact integer coordinates"
                    )
                pin_key = _group_route_pin_key(net, pin)
                if pin_key not in grouped_pin_keys:
                    retained.append((tuple(node), pin))
            rows = retained + portal_rows
            rows.sort(key=lambda row: _pin_order_key(row[1]))
            net_nodes[net] = [tuple(row[0]) for row in rows]
            net_node_pins[net] = [row[1] for row in rows]

        # Layered portals must depart as a real same-layer global wire.
        layered = set()
        for landing in certificate.landings:
            if landing.portal_layer_key is None:
                continue
            gi = next(
                gi
                for gi in range(self._n_layers)
                if physical_layer_key(gi) == tuple(landing.portal_layer_key)
            )
            node = (*landing.portal_xy_nm, gi)
            key = (landing.net, node)
            if key in required_states and tuple(required_states[key]) != node:
                raise ValueError("layered portal conflicts with selected access")
            required_states[key] = node
            layered.add(key)
        self._layered_portal_wire_states = layered
        self._group_route_required_states = required_states
        self._group_route_landing_baselines = baselines
        self._group_route_global_interfaces_by_net = {
            net: tuple(sorted(
                (
                    tuple(port.envelope_nm),
                    tuple(landing.portal_xy_nm),
                )
                for port in contract.ports
                if port.net == net
                for landing in certificate.landings
                if (landing.group, landing.net) == (port.group, port.net)
            ))
            for net in sorted(portals_by_net)
        }
        self._group_route_local_segment_keys_by_net = {
            net: frozenset(
                (
                    min((seg.ax, seg.ay), (seg.bx, seg.by)),
                    max((seg.ax, seg.ay), (seg.bx, seg.by)),
                    int(seg.gi),
                    int(self._segment_half_width(net, seg)),
                )
                for seg in baseline.segs
            )
            for net, baseline in sorted(baselines.items())
        }
        self._group_route_local_via_keys_by_net = {
            net: frozenset(
                (int(via.x), int(via.y), int(via.lo), int(via.n))
                for via in baseline.vias
            )
            for net, baseline in sorted(baselines.items())
        }
        self._group_route_global_nodes_by_net = {
            net: tuple(net_nodes[net]) for net in sorted(portals_by_net)
        }
        return certificate

    def _portal_corner_margins(
        self, net: str, portal_gi: int
    ) -> Tuple[int, int, int]:
        """Return ``(full_margin_nm, wire_margin_nm, grid_nm)`` for corner-inset portals."""
        half_w = getattr(self, "_half_w", None)
        if not half_w:
            return 0, 0, 1
        grid = max(1, int(getattr(self, "_manufacturing_grid_nm", 1) or 1))

        def ceil_grid(val: int) -> int:
            return -(-int(val) // grid) * grid

        pgi = max(0, min(int(portal_gi), len(half_w) - 1))
        hw_portal = int(self._hw(str(net), pgi))
        wire_margin = ceil_grid(max(hw_portal, grid))
        min_hw = max(1, min(int(h) for h in half_w))
        active_gis = [
            g
            for g in range(len(half_w))
            if g <= pgi or int(half_w[g]) < 4 * min_hw
        ]
        max_active = max(active_gis) if active_gis else pgi
        max_hw = max(
            [hw_portal, *(int(half_w[g]) for g in active_gis)],
            default=hw_portal,
        )
        use_single = bool(
            getattr(self, "_single_via_escape_allowed", lambda: False)()
        )
        via_pad_half = getattr(self, "_via_pad_half", None) or {}
        via_pad_half1 = getattr(self, "_via_pad_half1", None) or via_pad_half
        max_via = 0
        for lo, pair in via_pad_half.items():
            lo_i = int(lo)
            if lo_i >= max_active and lo_i not in (pgi - 1, pgi):
                continue
            bh, ah = via_pad_half1.get(lo_i, pair) if use_single else pair
            cut_fn = getattr(self, "_via_cut_extent", None)
            ext = (
                int(cut_fn(lo_i, 1 if use_single else None))
                if cut_fn is not None
                else 0
            )
            max_via = max(max_via, int(bh), int(ah), ext)
        full_margin = ceil_grid(max(wire_margin, max_hw, max_via))
        return full_margin, wire_margin, grid

    @staticmethod
    def _group_route_boundary_portals(
        envelope: Tuple[int, int, int, int],
        landing_state: Tuple[int, int, int],
        external_target_rects: Tuple[Tuple[int, int, int, int], ...],
        *,
        corner_margin_nm: int = 0,
        wire_margin_nm: int = 0,
        grid_nm: int = 1,
    ) -> Tuple[Tuple[int, int, int], ...]:
        """Enumerate ranked boundary portals toward obligations outside a group.

        A locally nearest side is not necessarily a usable global interface: a
        bottom-selected signal portal can face away from an explicit right-edge
        anchor, and a top-selected supply portal can face away from its bottom
        rail.  Targets are derived only from current routed pins, declared rails,
        and other group envelopes.  Returning every side lets the live PDK
        legality world reject a blocked nearest portal without importing a
        reference coordinate or widening the group-local routing envelope.
        """

        if (
            type(envelope) is not tuple
            or len(envelope) != 4
            or any(type(value) is not int for value in envelope)
        ):
            raise TypeError("group-route envelope must be one exact integer bbox")
        x0, y0, x1, y1 = envelope
        if x1 <= x0 or y1 <= y0:
            raise ValueError("group-route envelope must have positive area")
        if (
            type(landing_state) is not tuple
            or len(landing_state) != 3
            or any(type(value) is not int for value in landing_state)
        ):
            raise TypeError("group-route landing state must be one exact integer row")
        x, y, gi = landing_state
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            raise ValueError("group-route landing is outside its certified envelope")
        if type(external_target_rects) is not tuple:
            raise TypeError("group-route external targets must be one exact tuple")
        if len(external_target_rects) > 4096:
            raise ValueError("group-route external target inventory is unbounded")
        targets = []
        for rect in external_target_rects:
            if (
                type(rect) is not tuple
                or len(rect) != 4
                or any(type(value) is not int for value in rect)
            ):
                raise TypeError("group-route external target must be an integer bbox")
            tx0, ty0, tx1, ty1 = rect
            if tx1 < tx0 or ty1 < ty0:
                raise ValueError("group-route external target bbox is inverted")
            targets.append(rect)

        grid = max(1, int(grid_nm or 1))
        use_inset = int(corner_margin_nm) > 0 or int(wire_margin_nm) > 0

        def ceil_grid(val: int) -> int:
            return -(-int(val) // grid) * grid

        def floor_grid(val: int) -> int:
            return (int(val) // grid) * grid

        def inset_interval(side_lo: int, side_hi: int) -> Tuple[int, int]:
            if not use_inset:
                return side_lo, side_hi
            for margin in (int(corner_margin_nm), int(wire_margin_nm), grid):
                if margin <= 0:
                    continue
                lo = ceil_grid(side_lo + margin)
                hi = floor_grid(side_hi - margin)
                if side_lo < lo <= hi < side_hi:
                    return lo, hi
            if side_hi - side_lo >= 2:
                mid = (side_lo + side_hi) // 2
                if side_lo < mid < side_hi:
                    return mid, mid
            return side_lo, side_hi

        eff_x0, eff_x1 = inset_interval(x0, x1)
        eff_y0, eff_y1 = inset_interval(y0, y1)

        def clamp(value: int, lower: int, upper: int) -> int:
            return min(max(value, lower), upper)

        if not targets:
            px_proj = clamp(x, eff_x0, eff_x1) if use_inset else x
            py_proj = clamp(y, eff_y0, eff_y1) if use_inset else y
            choices = (
                (x - x0, 0, (x0, py_proj, gi)),
                (x1 - x, 1, (x1, py_proj, gi)),
                (y - y0, 2, (px_proj, y0, gi)),
                (y1 - y, 3, (px_proj, y1, gi)),
            )
            if any(distance < 0 for distance, _side, _portal in choices):
                raise ValueError("group-route landing cannot project to its envelope")
            deduped = []
            seen_empty = set()
            for _distance, _side, portal in sorted(choices):
                if portal not in seen_empty:
                    seen_empty.add(portal)
                    deduped.append(portal)
            return tuple(deduped)

        def closest_on_interval(
            local: int,
            side_lower: int,
            side_upper: int,
            target_lower: int,
            target_upper: int,
        ) -> int:
            overlap_lower = max(side_lower, target_lower)
            overlap_upper = min(side_upper, target_upper)
            if overlap_lower <= overlap_upper:
                if overlap_lower <= local <= overlap_upper:
                    return local
                if local < overlap_lower:
                    return (
                        clamp(ceil_grid(overlap_lower), side_lower, side_upper)
                        if use_inset and grid > 1
                        else overlap_lower
                    )
                return (
                    clamp(floor_grid(overlap_upper), side_lower, side_upper)
                    if use_inset and grid > 1
                    else overlap_upper
                )
            if target_upper < side_lower:
                return side_lower
            return side_upper

        def rect_distance(px: int, py: int, rect: Tuple[int, int, int, int]) -> int:
            tx0, ty0, tx1, ty1 = rect
            dx = max(tx0 - px, 0, px - tx1)
            dy = max(ty0 - py, 0, py - ty1)
            return dx + dy

        candidates = []
        for target_index, target in enumerate(targets):
            tx0, ty0, tx1, ty1 = target
            side_portals = [
                (0, (x0, closest_on_interval(y, eff_y0, eff_y1, ty0, ty1), gi)),
                (1, (x1, closest_on_interval(y, eff_y0, eff_y1, ty0, ty1), gi)),
                (2, (closest_on_interval(x, eff_x0, eff_x1, tx0, tx1), y0, gi)),
                (3, (closest_on_interval(x, eff_x0, eff_x1, tx0, tx1), y1, gi)),
            ]
            if use_inset:
                px_proj = clamp(x, eff_x0, eff_x1)
                py_proj = clamp(y, eff_y0, eff_y1)
                side_portals.extend((
                    (0, (x0, py_proj, gi)),
                    (1, (x1, py_proj, gi)),
                    (2, (px_proj, y0, gi)),
                    (3, (px_proj, y1, gi)),
                ))
            for side_rank, portal in side_portals:
                px, py, _pgi = portal
                candidates.append((
                    rect_distance(px, py, target),
                    abs(px - x) + abs(py - y),
                    side_rank,
                    target_index,
                    portal,
                ))
        ordered = []
        seen = set()
        for _distance, _landing_distance, _side, _target, portal in sorted(
            candidates
        ):
            if portal in seen:
                continue
            seen.add(portal)
            ordered.append(portal)
        if not ordered:
            raise ValueError("group-route landing has no boundary portal candidate")
        return tuple(ordered)

    @staticmethod
    def _group_route_boundary_portal(
        envelope: Tuple[int, int, int, int],
        landing_state: Tuple[int, int, int],
        external_target_rects: Tuple[Tuple[int, int, int, int], ...],
    ) -> Tuple[int, int, int]:
        """Return the highest-ranked portal for legacy direct callers."""

        return _eg_attr("GridlessEngine", _GridlessGroupRouteMixin)._group_route_boundary_portals(
            envelope,
            landing_state,
            external_target_rects,
        )[0]

    def _ranked_legal_group_route_boundary_portals(
        self,
        envelope: Tuple[int, int, int, int],
        landing_state: Tuple[int, int, int],
        external_target_rects: Tuple[Tuple[int, int, int, int], ...],
        net: str,
        *,
        preserve_topology_rank: bool = False,
    ) -> Tuple[Tuple[int, int, Tuple[int, int, int]], ...]:
        """Return legal boundary portals with either topology-true or dense rank."""

        has_engine_geom = bool(getattr(self, "_half_w", None))
        if has_engine_geom:
            full_margin, wire_margin, grid = self._portal_corner_margins(
                str(net), int(landing_state[2])
            )
            portals = _eg_attr(
                "GridlessEngine", _GridlessGroupRouteMixin
            )._group_route_boundary_portals(
                envelope,
                landing_state,
                external_target_rects,
                corner_margin_nm=full_margin,
                wire_margin_nm=wire_margin,
                grid_nm=grid,
            )
        else:
            grid = 1
            portals = _eg_attr(
                "GridlessEngine", _GridlessGroupRouteMixin
            )._group_route_boundary_portals(
                envelope,
                landing_state,
                external_target_rects,
            )
        x0, y0, x1, y1 = (int(v) for v in envelope)
        missing = object()
        previous = getattr(self, "_active_group_route_local_closure", missing)
        memo = getattr(self, "_portal_legality_memo", None)
        self._active_group_route_local_closure = True
        try:
            def _portal_point_legal(pt: Tuple[int, int, int]) -> bool:
                key = (str(net), tuple(int(v) for v in envelope), pt)
                if memo is not None and key in memo:
                    return bool(memo[key])
                if has_engine_geom:
                    px_i, py_i, pgi_i = int(pt[0]), int(pt[1]), int(pt[2])
                    if (
                        (x1 - x0) >= 2 * grid
                        and (y1 - y0) >= 2 * grid
                        and px_i in (x0, x1)
                        and py_i in (y0, y1)
                    ):
                        if memo is not None:
                            memo[key] = False
                        return False
                    hw_pt = int(self._hw(str(net), pgi_i))
                    if (
                        px_i in (x0, x1)
                        and (y1 - y0) >= 2 * hw_pt
                        and not (y0 + hw_pt <= py_i <= y1 - hw_pt)
                    ) or (
                        py_i in (y0, y1)
                        and (x1 - x0) >= 2 * hw_pt
                        and not (x0 + hw_pt <= px_i <= x1 - hw_pt)
                    ):
                        if memo is not None:
                            memo[key] = False
                        return False
                legal = bool(
                    self._wire_legal_single(
                        pt[2],
                        pt[0],
                        pt[1],
                        pt[0],
                        pt[1],
                        str(net),
                    )
                )
                if legal and getattr(self, "_tech", None) is not None:
                    cap_box = _seg_bbox(
                        pt[0],
                        pt[1],
                        pt[0],
                        pt[1],
                        self._hw(str(net), pt[2]),
                    )
                    if self._mim_metal_changes(
                        str(net), {}, {pt[2]: (cap_box,)}
                    ):
                        legal = False
                if memo is not None:
                    memo[key] = legal
                return legal

            accepted_with_rank: List[Tuple[int, Tuple[int, int, int]]] = []
            rejected_portals: List[Tuple[int, Tuple[int, int, int]]] = []
            seen_accepted: Set[Tuple[int, int, int]] = set()
            for orig_rank, portal in enumerate(portals):
                if _portal_point_legal(portal):
                    if portal not in seen_accepted:
                        seen_accepted.add(portal)
                        accepted_with_rank.append((orig_rank, portal))
                else:
                    rejected_portals.append((orig_rank, portal))
            if (
                rejected_portals
                and preserve_topology_rank
                and getattr(self, "_half_w", None) is not None
                and getattr(self, "_space", None) is not None
                and bool(getattr(self, "_matched_partners_by_net", {}).get(str(net)))
            ):
                accepted_edge_layers = {
                    (r, int(pt[2])) for r, pt in accepted_with_rank
                }
                ex0, ey0, ex1, ey1 = envelope
                added_edge_layers: Set[Tuple[int, int]] = set()
                hw_fn = getattr(self, "_hw", None)
                for orig_rank, portal in rejected_portals:
                    pgi = int(portal[2])
                    edge_layer = (orig_rank, pgi)
                    if (
                        edge_layer in accepted_edge_layers
                        or edge_layer in added_edge_layers
                    ):
                        continue
                    hw = (
                        int(hw_fn(net, pgi))
                        if hw_fn is not None
                        else int(self._half_w[pgi])
                    )
                    sp = int(self._space[pgi])
                    margin = hw + sp
                    step_pitch = max(100, ((2 * hw + sp + 9) // 10) * 10)
                    px, py = int(portal[0]), int(portal[1])
                    shifted_cands: List[Tuple[int, int, int]] = []
                    if px in (ex0, ex1):
                        for mult in (1, -1, 2, -2, 3, -3, 4, -4):
                            ny = py + mult * step_pitch
                            if ey0 + margin <= ny <= ey1 - margin:
                                shifted_cands.append((px, ny, pgi))
                    elif py in (ey0, ey1):
                        for mult in (1, -1, 2, -2, 3, -3, 4, -4):
                            nx = px + mult * step_pitch
                            if ex0 + margin <= nx <= ex1 - margin:
                                shifted_cands.append((nx, py, pgi))
                    added_for_pair = 0
                    for cand in shifted_cands:
                        if cand in seen_accepted:
                            continue
                        if _portal_point_legal(cand):
                            seen_accepted.add(cand)
                            accepted_with_rank.append((orig_rank, cand))
                            added_edge_layers.add(edge_layer)
                            added_for_pair += 1
                            if added_for_pair >= 2:
                                break
            floating_anchors = getattr(self, "_floating_matched_escape_anchors", None)
            if (
                not external_target_rects
                and floating_anchors
                and str(net) in floating_anchors
                and getattr(self, "_half_w", None) is not None
                and getattr(self, "_space", None) is not None
            ):
                grp, _self_anchor = floating_anchors[str(net)]
                ex0, ey0, ex1, ey1 = envelope
                hw_fn = getattr(self, "_hw", None)
                partners = getattr(self, "_matched_partners_by_net", {}).get(str(net), ())
                for orig_rank, portal in list(accepted_with_rank):
                    px, py, pgi = int(portal[0]), int(portal[1]), int(portal[2])
                    if px not in (ex0, ex1) and py not in (ey0, ey1):
                        continue
                    self_edge_dist = abs(landing_state[0] - px) + abs(landing_state[1] - py)
                    hw = int(hw_fn(net, pgi)) if hw_fn is not None else int(self._half_w[pgi])
                    sp = int(self._space[pgi])
                    margin = hw + sp
                    step_pitch = max(100, ((2 * hw + sp + 9) // 10) * 10)
                    grid = max(1, int(getattr(self, "_manufacturing_grid_nm", 5)))
                    for mate in partners:
                        mate_info = floating_anchors.get(str(mate))
                        if not mate_info or mate_info[0] != grp:
                            continue
                        mx, my = mate_info[1]
                        mate_dist = (abs(mx - px) if px in (ex0, ex1) else abs(my - py))
                        if mate_dist <= self_edge_dist + 400:
                            continue
                        delta = (((mate_dist - self_edge_dist) + grid - 1) // grid) * grid
                        balance_cands: List[Tuple[int, int, int]] = []
                        if px in (ex0, ex1):
                            for sign in (-1, 1):
                                for k in (0, 1, -1, 2, -2, 3, -3):
                                    ny = py + sign * delta + k * step_pitch
                                    ny = int(round(ny / grid) * grid)
                                    if ey0 + margin <= ny <= ey1 - margin:
                                        balance_cands.append((px, ny, pgi))
                        else:
                            for sign in (-1, 1):
                                for k in (0, 1, -1, 2, -2, 3, -3):
                                    nx = px + sign * delta + k * step_pitch
                                    nx = int(round(nx / grid) * grid)
                                    if ex0 + margin <= nx <= ex1 - margin:
                                        balance_cands.append((nx, py, pgi))
                        added_bal = 0
                        for cand in balance_cands:
                            if cand in seen_accepted:
                                continue
                            if _portal_point_legal(cand):
                                seen_accepted.add(cand)
                                accepted_with_rank.append((orig_rank, cand))
                                added_bal += 1
                                if added_bal >= 4:
                                    break
            if not preserve_topology_rank:
                return _eg_attr("GridlessEngine", _GridlessGroupRouteMixin)._group_route_ranked_portal_rows(
                    landing_state,
                    tuple(pt for _r, pt in accepted_with_rank),
                )
            rows = []
            for orig_rank, portal in accepted_with_rank:
                rows.append((
                    orig_rank,
                    abs(portal[0] - landing_state[0])
                    + abs(portal[1] - landing_state[1]),
                    portal,
                ))
            return tuple(rows)
        finally:
            if previous is missing:
                delattr(self, "_active_group_route_local_closure")
            else:
                self._active_group_route_local_closure = previous

    def _legal_group_route_boundary_portals(
        self,
        envelope: Tuple[int, int, int, int],
        landing_state: Tuple[int, int, int],
        external_target_rects: Tuple[Tuple[int, int, int, int], ...],
        net: str,
    ) -> Tuple[Tuple[int, int, int], ...]:
        """Return ranked boundary portals that fit the live routing world."""

        ranked_fn = getattr(self, "_ranked_legal_group_route_boundary_portals", None)
        rows = (
            ranked_fn(
                envelope,
                landing_state,
                external_target_rects,
                net,
                preserve_topology_rank=False,
            )
            if ranked_fn is not None
            else _eg_attr(
                "GridlessEngine", _GridlessGroupRouteMixin
            )._ranked_legal_group_route_boundary_portals(
                self,
                envelope,
                landing_state,
                external_target_rects,
                net,
                preserve_topology_rank=False,
            )
        )
        return tuple(portal for _rank, _dist, portal in rows)

    @staticmethod
    def _group_route_ranked_portal_rows(
        landing_state: Tuple[int, int, int],
        portal_states: Tuple[Tuple[int, int, int], ...],
    ) -> Tuple[Tuple[int, int, Tuple[int, int, int]], ...]:
        """Keep target-derived portal rank ahead of local escape distance."""

        if (
            type(landing_state) is not tuple
            or len(landing_state) != 3
            or any(type(value) is not int for value in landing_state)
        ):
            raise TypeError("group-route landing state must be one integer row")
        if type(portal_states) is not tuple:
            raise TypeError("group-route portal states must be one exact tuple")
        rows = []
        for rank, portal in enumerate(portal_states):
            if (
                type(portal) is not tuple
                or len(portal) != 3
                or any(type(value) is not int for value in portal)
                or portal[2] != landing_state[2]
            ):
                raise TypeError("group-route portal state is malformed")
            rows.append((
                rank,
                abs(portal[0] - landing_state[0])
                + abs(portal[1] - landing_state[1]),
                portal,
            ))
        return tuple(rows)

    @staticmethod
    def _group_route_local_interface_rows(
        landing_state: Tuple[int, int, int],
        landing_pin: Pin,
        portal_state: Tuple[int, int, int],
        portal_pin: Pin,
    ) -> Tuple[Tuple[Tuple[int, int, int], Pin], ...]:
        """Name the local landing/portal endpoints without a zero-length edge."""

        if tuple(landing_state) == tuple(portal_state):
            return ((tuple(portal_state), portal_pin),)
        return (
            (tuple(landing_state), landing_pin),
            (tuple(portal_state), portal_pin),
        )

    @staticmethod
    def _group_route_unlanded_access_row(
        selected_source_node: Tuple[int, int, int],
        access_node: Tuple[int, int, int],
        access_pin: Pin,
    ) -> Optional[Tuple[Tuple[int, int, int], Pin]]:
        """Keep only access geometry not already closed by the landing node.

        Distinct terminal obligations may replay one physical pin node (for
        example a PMOS source/body short).  The contract retains both terminal
        records, while the local route graph must contain that conductor only
        once.
        """

        if tuple(access_node) == tuple(selected_source_node):
            return None
        return (tuple(access_node), access_pin)

    def set_experimental_group_demand_guidance(self, enabled: bool) -> None:
        """Opt into coarse source-demand ranking for subsequent group routes.

        The ranking is a 2-D heuristic; original candidate generation, route
        legality, group certificates and physical verification still apply.
        This is independent of and does not replace the source route policy.
        """
        if type(enabled) is not bool:
            raise TypeError("experimental group demand guidance needs an exact bool")
        self._experimental_group_demand_guidance = enabled
        self.__dict__.pop('_group_demand_guidance_last_report', None)

    def set_experimental_group_full_feedback(self, enabled: bool) -> None:
        """Opt into physical feedback for each complete group prefix."""
        if type(enabled) is not bool:
            raise TypeError('experimental group full feedback needs an exact bool')
        self._experimental_group_full_feedback = enabled

    def _select_group_route_landing(self, port, rows, book, landings=()):
        guide = getattr(self, '_experimental_group_demand_guide', None)
        if guide is None:
            floating_anchors = getattr(self, "_floating_matched_escape_anchors", None)
            if (
                floating_anchors
                and str(getattr(port, "net", "")) in floating_anchors
                and getattr(self, "_matched_partners_by_net", {}).get(str(port.net))
            ):
                return min(
                    rows,
                    key=lambda r: (
                        (
                            1 if r[0] >= 3 else 0,
                            r[0],
                            r[2] + self._matched_portal_layer_penalty_nm(port, r, landings),
                            r[1],
                            r,
                        )
                        if isinstance(r, tuple) and len(r) >= 11
                        else (0, r)
                    ),
                )
            return min(rows)
        from clay.layout.route.group_demand_guidance import UnsupportedGuide
        try:
            return guide.choose(port, rows, book, landings)
        except UnsupportedGuide as error:
            self._group_demand_guidance_fallbacks.append((port.group, port.net, str(error)))
            return min(rows)

    def _order_group_ports_for_matched_escapes(
        self,
        ports: Sequence[object],
        net_nodes: Mapping[str, Sequence[Tuple[int, int, int]]],
    ) -> Tuple[object, ...]:
        matched_map = getattr(self, "_matched_partners_by_net", None)
        if not matched_map:
            return tuple(ports)
        rails = set(getattr(self, "_rail_by_net", {}).keys())
        group_rank: Dict[str, int] = {}
        for port in ports:
            grp = str(getattr(port, "group", ""))
            if grp not in group_rank:
                group_rank[grp] = len(group_rank)

        def _sort_key(item: Tuple[int, object]) -> Tuple[int, int, int, int]:
            idx, port = item
            grp = str(getattr(port, "group", ""))
            net = str(getattr(port, "net", ""))
            edge_dist = 0
            if net in rails:
                tier = 2
            else:
                env = getattr(port, "envelope_nm", None)
                nodes = net_nodes.get(net, ()) if env is not None else ()
                inside_xy = {
                    (int(xy[0]), int(xy[1]))
                    for xy in nodes
                    if env[0] <= int(xy[0]) <= env[2] and env[1] <= int(xy[1]) <= env[3]
                }
                tier = 0 if (net in matched_map and len(inside_xy) <= 1) else 1
                if tier == 0 and env is not None and inside_xy:
                    ex0, ey0, ex1, ey1 = (int(v) for v in env)
                    px, py = next(iter(inside_xy))
                    edge_dist = min(
                        abs(px - ex0),
                        abs(ex1 - px),
                        abs(py - ey0),
                        abs(ey1 - py),
                    )
            return (group_rank[grp], tier, edge_dist, idx)

        return tuple(port for _idx, port in sorted(enumerate(ports), key=_sort_key))

    def _reserve_group_route_landings(
        self,
        contract: object,
        pins_by_net: Mapping[str, Sequence[Pin]],
        net_nodes: Dict[str, List[Tuple[int, int, int]]],
        net_node_pins: Dict[str, List[Pin]],
        net_layers: Mapping[str, Sequence[int]],
        book: _Book,
        *,
        ordered_nets: Optional[Sequence[str]] = None,
        topology_plan: Optional[object] = None,
    ) -> object:
        """Allocate local group interfaces under the actual rail width proof.

        The materialized rail, not a second group-local wire, carries a wide
        supply Width.  Use the same actual-rail proof as ordinary local drops;
        let other local ports close first while retaining negotiation rank
        within each class, and freeze segments before the scope ends.
        """
        from clay.layout.route.group_ports import GroupRouteRoutingContract
        from clay.layout.route.planar_precheck import verify_static_terminal_escapability

        if type(contract) is not GroupRouteRoutingContract:
            raise TypeError("group-route landing allocator requires an exact contract")
        verify_static_terminal_escapability(
            self, pins_by_net, net_nodes, net_node_pins, net_layers
        )
        overrides = {}
        proven_rail_nets = set()
        for net in sorted({port.net for port in contract.ports}):
            rail = tuple(self._rail_by_net.get(net, ()))
            local_overrides = self._rail_local_half_width_overrides(
                net, net_layers.get(net, ()), rail,
            )
            if local_overrides:
                proven_rail_nets.add(net)
                overrides.update(local_overrides)
        if ordered_nets is None:
            ports = contract.ports
        else:
            ranks = {str(net): rank for rank, net in enumerate(ordered_nets)}
            ports = tuple(sorted(contract.ports, key=lambda port: (
                port.group, port.net in proven_rail_nets,
                ranks.get(port.net, len(ranks)), port.net,
            )))
        def _call_group_helper(name: str, *args: Any):
            helper = getattr(self, name, None)
            if callable(helper):
                return helper(*args)
            return getattr(_GridlessGroupRouteMixin, name)(self, *args)

        ports = _call_group_helper("_order_group_ports_for_matched_escapes", ports, net_nodes)
        matched_map = getattr(self, "_matched_partners_by_net", {}) or {}
        floating_matched_anchors: Dict[str, Tuple[str, Tuple[int, int]]] = {}
        if matched_map:
            for port in ports:
                net_s = str(getattr(port, "net", ""))
                if net_s not in matched_map:
                    continue
                env = getattr(port, "envelope_nm", None)
                if env is None:
                    continue
                nodes = net_nodes.get(net_s, ())
                inside_pts = [
                    (int(xy[0]), int(xy[1]))
                    for xy in nodes
                    if env[0] <= int(xy[0]) <= env[2] and env[1] <= int(xy[1]) <= env[3]
                ]
                outside_pts = [
                    xy for xy in nodes
                    if not (env[0] <= int(xy[0]) <= env[2] and env[1] <= int(xy[1]) <= env[3])
                ]
                if len(set(inside_pts)) == 1 and not outside_pts:
                    floating_matched_anchors[net_s] = (str(port.group), inside_pts[0])
        promoted_portals = _call_group_helper(
            "_compute_narrow_corridor_portal_promotions", contract, net_layers
        )
        prior_promoted = getattr(self, "_narrow_corridor_portal_gi", None)
        prior_floating_anchors = getattr(self, "_floating_matched_escape_anchors", None)
        self._narrow_corridor_portal_gi = promoted_portals
        self._floating_matched_escape_anchors = floating_matched_anchors or None
        try:
            if not getattr(self, '_experimental_group_demand_guidance', False):
                with self._half_width_scope(overrides):
                    return self._reserve_group_route_landings_once(
                        contract, pins_by_net, net_nodes, net_node_pins, net_layers,
                        book, ordered_ports=ports,
                    )

            from clay.layout.route.group_demand_guidance import UnsupportedGuide, build_guide
            if getattr(self, '_experimental_group_demand_guide', None) is not None:
                raise ValueError('experimental group demand guidance cannot nest')
            self._group_demand_guidance_fallbacks = []
            guide = None
            completed = False
            def finish_guidance():
                try:
                    self._group_demand_guidance_last_report = {
                        'schema': 'clay.experimental-group-demand-guidance/v2',
                        'scope': '2d source-demand ranking only; not a route, capacity or physical admission',
                        'status': ('UNKNOWN' if guide is None else
                                   'COMPLETE' if completed and not self._group_demand_guidance_fallbacks
                                   and not guide.unknown else 'PARTIAL'),
                        'ledger': {} if guide is None else dict(guide.ledger),
                        'topology_sha256': None if guide is None else guide.topology_sha256,
                        'demands': {} if guide is None else guide.report_demands(),
                        'selections': [] if guide is None else list(guide.selections),
                        'unknown': [] if guide is None else list(guide.unknown),
                        'fallbacks': list(self._group_demand_guidance_fallbacks),
                    }
                finally:
                    self.__dict__.pop('_experimental_group_demand_guide', None)
            try:
                try:
                    guide = build_guide(net_nodes, net_node_pins, net_layers,
                                        self._rail_by_net, contract, self,
                                        topology_plan, pins_by_net)
                except UnsupportedGuide as error:
                    self._group_demand_guidance_fallbacks.append(('', '', str(error)))
                # The immutable demand sketch does not own the transactional Book.
                # Keep it outside checkpoint's _active_/_group_route_ mutable-state
                # namespace: every choice receives the restored live Book and its
                # counters/selections are observation only, never ranking state.
                self._experimental_group_demand_guide = guide
                with self._half_width_scope(overrides):
                    result = self._reserve_group_route_landings_once(
                        contract, pins_by_net, net_nodes, net_node_pins, net_layers,
                        book, ordered_ports=ports,
                    )
                completed = True
            except BaseException as original:
                try:
                    finish_guidance()
                except BaseException as ending:
                    from clay.flow.control_stop import record_unwind_failure
                    try:
                        record_unwind_failure(original, ending)
                    except BaseException:
                        pass
                raise
            finish_guidance()
            return result
        finally:
            if prior_promoted is None:
                self.__dict__.pop("_narrow_corridor_portal_gi", None)
            else:
                self._narrow_corridor_portal_gi = prior_promoted
            if prior_floating_anchors is None:
                self.__dict__.pop("_floating_matched_escape_anchors", None)
            else:
                self._floating_matched_escape_anchors = prior_floating_anchors

    def _direct_group_route_landing_legal(
        self,
        node: Tuple[int, int, int],
        net: str,
    ) -> bool:
        missing = object()
        previous = getattr(self, "_active_group_route_local_closure", missing)
        outer_rejections = getattr(self, "_hard_rejections", {})
        outer_samples = getattr(self, "_hard_rejection_samples", {})
        outer_gen = getattr(self, "_hard_rejection_generation", 0)
        self._active_group_route_local_closure = True
        self._hard_rejections = {}
        self._hard_rejection_samples = {}
        try:
            return bool(
                self._wire_legal_single(
                    node[2],
                    node[0],
                    node[1],
                    node[0],
                    node[1],
                    str(net),
                )
            )
        finally:
            self._hard_rejections = outer_rejections
            self._hard_rejection_samples = outer_samples
            self._hard_rejection_generation = outer_gen
            if previous is missing:
                delattr(self, "_active_group_route_local_closure")
            else:
                self._active_group_route_local_closure = previous

    def _matched_portal_layer_penalty_nm(self, port, row, landings) -> int:
        partners_map = getattr(self, "_matched_partners_by_net", None)
        tech = getattr(self, "_tech", None)
        if not partners_map or tech is None or type(row) is not tuple or len(row) < 11:
            return 0
        mates = partners_map.get(str(port.net))
        if not mates:
            return 0
        portal = row[10]
        if type(portal) is not tuple or len(portal) < 3:
            return 0
        pgi = int(portal[2])
        mate_set = set(mates)
        floating_anchors = getattr(self, "_floating_matched_escape_anchors", None)
        net_s = str(port.net)
        mate_layer_pen = 0
        balance_pen = 0
        seen_landed_mates: Set[str] = set()
        for landing in landings:
            if str(getattr(landing, "group", "")) != str(port.group):
                continue
            mate_net = str(getattr(landing, "net", ""))
            if mate_net not in mate_set:
                continue
            seen_landed_mates.add(mate_net)
            m_layer_key = (
                getattr(landing, "portal_layer_key", None)
                or getattr(landing, "upper_layer_key", None)
            )
            m_gi = tech.resolve_layer_to_gi(m_layer_key) if m_layer_key is not None else None
            if m_gi is not None and int(m_gi) != pgi:
                mate_layer_pen = max(mate_layer_pen, 1800 * abs(pgi - int(m_gi)))
            if floating_anchors and net_s in floating_anchors and mate_net in floating_anchors:
                lxy = getattr(landing, "xy_nm", None)
                lp = getattr(landing, "portal_xy_nm", None)
                if lxy is not None and lp is not None:
                    mate_span = abs(int(lp[0]) - int(lxy[0])) + abs(int(lp[1]) - int(lxy[1]))
                    balance_pen = max(balance_pen, 2 * abs(int(row[2]) - mate_span))
        if floating_anchors and net_s in floating_anchors and getattr(port, "envelope_nm", None) is not None:
            ex0, ey0, ex1, ey1 = (int(v) for v in port.envelope_nm)
            px, py = int(portal[0]), int(portal[1])
            if px in (ex0, ex1) or py in (ey0, ey1):
                for mate_net in mate_set:
                    if mate_net in seen_landed_mates or mate_net not in floating_anchors:
                        continue
                    m_grp, (mx, my) = floating_anchors[mate_net]
                    if m_grp != str(port.group):
                        continue
                    mate_edge_dist = abs(mx - px) if px in (ex0, ex1) else abs(my - py)
                    if mate_edge_dist > int(row[2]):
                        balance_pen = max(balance_pen, 2 * (mate_edge_dist - int(row[2])))
        return mate_layer_pen + balance_pen

    def _preferred_portal_gi(
        self, port: object, allowed: Set[int]
    ) -> Optional[int]:
        """Resolve an explicit test preference or transactional @trunk layer."""
        preference = getattr(self, "_prototype_portal_preference", None)
        if preference is not None:
            target_level = None
            if isinstance(preference, dict) and (port.group, port.net) in preference:
                target_level = preference[(port.group, port.net)]
            elif (
                isinstance(preference, tuple)
                and len(preference) >= 3
                and (port.group, port.net) == preference[:2]
            ):
                target_level = preference[2]
            if target_level is not None:
                target_gi = next(
                    (
                        gi
                        for gi, level in enumerate(self._tech._gi_to_pdk)
                        if level == target_level
                    ),
                    None,
                )
                if target_gi is not None and target_gi in allowed:
                    return int(target_gi)
                return None
        if not bool(getattr(self, "_transactional_group_portal_layers", False)):
            return None
        if port.net in getattr(self, "_rail_by_net", {}):
            return None
        promoted_gi = (
            getattr(self, "_narrow_corridor_portal_gi", None) or {}
        ).get((port.group, port.net))
        if type(promoted_gi) is int and promoted_gi in allowed:
            return int(promoted_gi)
        trunk_gi = getattr(self, "_declared_trunk_gi_by_net", {}).get(port.net)
        if type(trunk_gi) is int and trunk_gi in allowed:
            return int(trunk_gi)
        return None

    def _compute_narrow_corridor_portal_promotions(
        self,
        contract: object,
        net_layers: Mapping[str, Sequence[int]],
    ) -> Dict[Tuple[str, str], int]:
        """Promote colliding same-layer signal portals across single-track inter-group corridors."""
        ports = getattr(contract, "ports", ())
        if not ports or not self._half_w:
            return {}
        min_hw = max(1, min(int(h) for h in self._half_w))
        rails = getattr(self, "_rail_by_net", {}) or {}
        declared_trunks = getattr(self, "_declared_trunk_gi_by_net", {}) or {}
        envelopes_by_group: Dict[str, Rect] = {}
        ports_by_group: Dict[str, Dict[str, object]] = {}
        for port in ports:
            grp = str(port.group)
            env = tuple(int(v) for v in port.envelope_nm)
            envelopes_by_group[grp] = env
            ports_by_group.setdefault(grp, {})[str(port.net)] = port
        groups = sorted(envelopes_by_group)
        if len(groups) < 2:
            return {}
        promoted: Dict[Tuple[str, str], int] = {}
        for idx_a, g_a in enumerate(groups):
            ax0, ay0, ax1, ay1 = envelopes_by_group[g_a]
            for g_b in groups[idx_a + 1:]:
                bx0, by0, bx1, by1 = envelopes_by_group[g_b]
                gap: Optional[int] = None
                trans_axis: Optional[int] = None
                if (ax1 < bx0 or bx1 < ax0) and max(ay0, by0) < min(ay1, by1):
                    c_lo, c_hi = (ax1, bx0) if ax1 < bx0 else (bx1, ax0)
                    t_lo, t_hi = max(ay0, by0), min(ay1, by1)
                    blocked_between = any(
                        g_c not in (g_a, g_b)
                        and max(c_lo, envelopes_by_group[g_c][0])
                        < min(c_hi, envelopes_by_group[g_c][2])
                        and max(t_lo, envelopes_by_group[g_c][1])
                        < min(t_hi, envelopes_by_group[g_c][3])
                        for g_c in groups
                    )
                    if not blocked_between:
                        gap = c_hi - c_lo
                        trans_axis = 1
                elif (ay1 < by0 or by1 < ay0) and max(ax0, bx0) < min(ax1, bx1):
                    c_lo, c_hi = (ay1, by0) if ay1 < by0 else (by1, ay0)
                    t_lo, t_hi = max(ax0, bx0), min(ax1, bx1)
                    blocked_between = any(
                        g_c not in (g_a, g_b)
                        and max(c_lo, envelopes_by_group[g_c][1])
                        < min(c_hi, envelopes_by_group[g_c][3])
                        and max(t_lo, envelopes_by_group[g_c][0])
                        < min(t_hi, envelopes_by_group[g_c][2])
                        for g_c in groups
                    )
                    if not blocked_between:
                        gap = c_hi - c_lo
                        trans_axis = 0
                if gap is None or trans_axis is None or gap <= 0:
                    continue
                shared_nets = sorted(
                    set(ports_by_group[g_a]).intersection(ports_by_group[g_b])
                )
                by_base_gi: Dict[
                    int, List[Tuple[str, Tuple[int, int], Set[int]]]
                ] = {}
                for net in shared_nets:
                    if net in rails:
                        continue
                    allowed = {int(v) for v in net_layers.get(net, ())}
                    if not allowed:
                        continue
                    t_gi = declared_trunks.get(net)
                    if type(t_gi) is int and t_gi in allowed:
                        base_gi = int(t_gi)
                    elif 1 in allowed:
                        base_gi = 1
                    else:
                        base_gi = min(allowed)
                    p_a = ports_by_group[g_a][net]
                    p_b = ports_by_group[g_b][net]
                    coords = [
                        int(acc.xy_nm[trans_axis])
                        for acc in (*p_a.accesses, *p_b.accesses)
                    ]
                    if not coords:
                        continue
                    by_base_gi.setdefault(base_gi, []).append(
                        (net, (min(coords), max(coords)), allowed)
                    )
                for base_gi, entries in sorted(by_base_gi.items()):
                    if len(entries) < 2 or base_gi >= len(self._half_w):
                        continue
                    hw = int(self._half_w[base_gi])
                    sp = int(self._space[base_gi])
                    if gap >= 4 * hw + sp:
                        continue
                    assigned: List[Tuple[int, Tuple[int, int]]] = []
                    for net, span, allowed in sorted(entries, key=lambda r: r[0]):
                        chosen_gi = base_gi
                        for cand_gi in range(base_gi, self._n_layers):
                            if cand_gi not in allowed:
                                continue
                            if (
                                cand_gi > base_gi
                                and int(self._half_w[cand_gi]) >= 4 * min_hw
                            ):
                                continue
                            cand_pad = 2 * int(self._half_w[cand_gi]) + int(
                                self._space[cand_gi]
                            )
                            conflict = any(
                                prev_gi == cand_gi
                                and max(span[0], prev_span[0])
                                <= min(span[1], prev_span[1]) + cand_pad
                                for prev_gi, prev_span in assigned
                            )
                            if not conflict:
                                chosen_gi = cand_gi
                                break
                        assigned.append((chosen_gi, span))
                        if chosen_gi > base_gi:
                            for grp_name in (g_a, g_b):
                                key = (grp_name, net)
                                prev_val = promoted.get(key)
                                if prev_val is None or chosen_gi > prev_val:
                                    promoted[key] = chosen_gi
        return promoted

    def _group_route_active_local_halo_and_layers(
        self,
        port_net: str,
        allowed_layers: Sequence[int],
        required_gis: Sequence[int],
    ) -> Tuple[int, List[int]]:
        """Scope local group closure away from thick top RDL layers unless needed."""
        allowed = [int(g) for g in allowed_layers]
        if not allowed or not self._half_w:
            max_pad = max(
                (max(pair) for pair in self._via_pad_half.values()),
                default=max(self._half_w) if self._half_w else 0,
            )
            halo = (max(self._half_w) if self._half_w else 0) + (
                max(self._space) if self._space else 0
            ) + max_pad
            return halo, allowed
        min_hw = max(1, min(int(hw) for hw in self._half_w))
        req_max = max([0, *(int(g) for g in required_gis)])
        trunk_gi = getattr(self, "_declared_trunk_gi_by_net", {}).get(port_net)
        if type(trunk_gi) is int:
            req_max = max(req_max, trunk_gi)
        active = [
            g
            for g in allowed
            if g <= req_max or int(self._half_w[g]) < 4 * min_hw
        ]
        if not active:
            active = list(allowed)
        max_active = max(active)
        active_span = range(min(len(self._half_w), max_active + 1))
        max_hw = max((int(self._half_w[g]) for g in active_span), default=min_hw)
        max_sp = max((int(self._space[g]) for g in active_span), default=0)
        max_pad = max(
            (
                max(pair)
                for lo, pair in self._via_pad_half.items()
                if int(lo) < max_active
            ),
            default=max_hw,
        )
        return max_hw + max_sp + max_pad, active

    def _group_route_has_legal_via_climb(
        self,
        port: object,
        allowed: Set[int],
        from_gi: int,
        to_gi: int,
        seed_xy: Tuple[int, int],
    ) -> bool:
        """Verify intermediate layers are allowed and admit at least one legal via."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        if from_gi >= to_gi:
            return False
        if not all(g in allowed for g in range(from_gi, to_gi + 1)):
            return False
        memo = getattr(self, "_portal_climb_memo", None)
        halo, _active = self._group_route_active_local_halo_and_layers(
            port.net, sorted(allowed), (from_gi, to_gi)
        )
        global_bounds = tuple(self._bounds)
        x0, y0, x1, y1 = port.envelope_nm
        if x1 <= x0 or y1 <= y0:
            return False
        grid = max(1, int(getattr(self, "_manufacturing_grid_nm", 5) or 5))
        step = max(grid, ((280 + grid - 1) // grid) * grid)
        use_single_via = bool(
            getattr(self, "_single_via_escape_allowed", lambda: False)()
        )

        missing = object()
        previous_closure = getattr(
            self, "_active_group_route_local_closure", missing
        )
        outer_rejections = self._hard_rejections
        outer_samples = self._hard_rejection_samples
        outer_gen = self._hard_rejection_generation
        self._active_group_route_local_closure = True
        self._hard_rejections = {}
        self._hard_rejection_samples = {}
        try:
            def _climb_via_legal(vx_nm: int, vy_nm: int, via_lo: int) -> bool:
                if not self._via_legal(vx_nm, vy_nm, via_lo, str(port.net)):
                    return False
                if getattr(self, "_tech", None) is not None:
                    via_metal = self._net_metal(
                        _RoutedNet(str(port.net), vias=[_ViaSite(vx_nm, vy_nm, via_lo, 0)])
                    )[0]
                    if self._mim_metal_changes(str(port.net), {}, via_metal):
                        return False
                return True

            sx, sy = int(seed_xy[0]), int(seed_xy[1])
            for lo in range(from_gi, to_gi):
                mkey = (port.group, port.net, lo)
                if memo is not None and memo.get(mkey) is False:
                    return False
                if memo is not None and memo.get(mkey) is True:
                    continue
                if lo not in self._via_pad_half:
                    if memo is not None:
                        memo[mkey] = False
                    return False
                bh, ah = (
                    self._via_pad_half1[lo]
                    if use_single_via
                    else self._via_pad_half[lo]
                )
                ext = self._via_cut_extent(lo, 1 if use_single_via else None)
                max_h = max(int(bh), int(ah), int(ext))
                bx0 = max(global_bounds[0], x0 + max_h)
                by0 = max(global_bounds[1], y0 + max_h)
                bx1 = min(global_bounds[2], x1 - max_h)
                by1 = min(global_bounds[3], y1 - max_h)
                if bx1 < bx0 or by1 < by0:
                    if memo is not None:
                        memo[mkey] = False
                    return False
                if (
                    bx0 <= sx <= bx1
                    and by0 <= sy <= by1
                    and sx % grid == 0
                    and sy % grid == 0
                    and _climb_via_legal(sx, sy, lo)
                ):
                    if memo is not None:
                        memo[mkey] = True
                    continue
                found = False
                probes = 0
                start_x = ((bx0 + grid - 1) // grid) * grid
                start_y = ((by0 + grid - 1) // grid) * grid
                vx = start_x
                while vx <= bx1 and not found:
                    vy = start_y
                    while vy <= by1:
                        probes += 1
                        if probes % 128 == 0:
                            self._watchdog_check(port.net, "group_portal_via_climb")
                        if _climb_via_legal(vx, vy, lo):
                            found = True
                            break
                        vy += step
                    vx += step
                if memo is not None:
                    memo[mkey] = found
                if not found:
                    return False
            return True
        finally:
            self._hard_rejections = outer_rejections
            self._hard_rejection_samples = outer_samples
            self._hard_rejection_generation = outer_gen
            if previous_closure is missing:
                delattr(self, "_active_group_route_local_closure")
            else:
                self._active_group_route_local_closure = previous_closure

    def _legalize_group_port_baseline_with_interface(
        self,
        port,
        local_port_net: _RoutedNet,
        portal_node: Tuple[int, int, int],
    ) -> None:
        """Legalize a certified group baseline while keeping envelope confinement active."""
        missing = object()
        prev_closure = getattr(self, "_active_group_route_local_closure", missing)
        prev_iface = getattr(self, "_active_group_route_local_interface", missing)
        self._active_group_route_local_closure = True
        self._active_group_route_local_interface = (
            tuple(int(v) for v in port.envelope_nm),
            (int(portal_node[0]), int(portal_node[1]), int(portal_node[2])),
        )
        try:
            self._legalize_group_port_baseline(
                port.net, local_port_net, port.envelope_nm
            )
        finally:
            if prev_iface is missing:
                self.__dict__.pop("_active_group_route_local_interface", None)
            else:
                self._active_group_route_local_interface = prev_iface
            if prev_closure is missing:
                self.__dict__.pop("_active_group_route_local_closure", None)
            else:
                self._active_group_route_local_closure = prev_closure

    def _group_local_interface_allows_landing(
        self,
        envelope_nm: Tuple[int, int, int, int],
        state: Tuple[int, int, int],
        via: Optional[_ViaSite],
        portal_node: Tuple[int, int, int],
        net: str,
    ) -> bool:
        """Check that a landing state/via overhangs ``envelope_nm`` only at ``portal_node``."""
        x0, y0, x1, y1 = (int(v) for v in envelope_nm)
        px, py, pgi = int(portal_node[0]), int(portal_node[1]), int(portal_node[2])
        sx, sy, sgi = int(state[0]), int(state[1]), int(state[2])
        if not (x0 <= sx <= x1 and y0 <= sy <= y1):
            return False
        portal_l = px == x0
        portal_r = px == x1
        portal_b = py == y0
        portal_t = py == y1
        shw = int(self._hw(net, sgi))
        sc0, sc1, sc2, sc3 = sx - shw, sy - shw, sx + shw, sy + shw
        if sc0 < x0 or sc1 < y0 or sc2 > x1 or sc3 > y1:
            if not (sc0 <= px <= sc2 and sc1 <= py <= sc3):
                return False
            if (
                (sc0 < x0 and not portal_l)
                or (sc2 > x1 and not portal_r)
                or (sc1 < y0 and not portal_b)
                or (sc3 > y1 and not portal_t)
            ):
                return False
        if via is not None and via.lo in self._via_pad_half:
            bh, ah = (
                self._via_pad_half1[via.lo]
                if via.n == 1
                else self._via_pad_half[via.lo]
            )
            ext = self._via_cut_extent(via.lo, 1 if via.n == 1 else None)
            max_h = max(int(bh), int(ah), int(ext))
            vc0, vc1, vc2, vc3 = (
                int(via.x) - max_h,
                int(via.y) - max_h,
                int(via.x) + max_h,
                int(via.y) + max_h,
            )
            if vc0 < x0 or vc1 < y0 or vc2 > x1 or vc3 > y1:
                if pgi != sgi:
                    return False
                if not (vc0 <= px <= vc2 and vc1 <= py <= vc3):
                    return False
                if (
                    (vc0 < x0 and not portal_l)
                    or (vc2 > x1 and not portal_r)
                    or (vc1 < y0 and not portal_b)
                    or (vc3 > y1 and not portal_t)
                ):
                    return False
        return True

    def _group_route_landing_candidates(
        self, port, pin_nodes, net_layers, book, physical_layer_key,
        external_target_rects,
    ):
        """Generate the current port candidates in the active routing world."""
        candidates = []
        portal_targets = external_target_rects(port)
        has_external_targets = bool(portal_targets)
        allowed = set(int(value) for value in net_layers.get(port.net, ()))
        target_gi = (
            self._preferred_portal_gi(port, allowed)
            if has_external_targets
            else None
        )
        preserve_rank = (target_gi is not None) or bool(
            getattr(self, "_negotiation_port_legalize", False)
        )
        self._book = book
        self._rebuild_metal_index()
        self._portal_legality_memo = {}
        self._portal_climb_memo = {}
        seed_portals_memo: Dict[
            Tuple[int, int, int],
            Tuple[Tuple[int, int, Tuple[int, int, int]], ...],
        ] = {}
        component_escape_memo: Dict[
            Tuple[object, ...],
            Tuple[
                Tuple[Tuple[Tuple[int, int, int], Optional[_ViaSite]], ...],
                Tuple[Any, ...],
                Tuple[Any, ...],
                Tuple[Any, ...],
                int,
            ],
        ] = {}
        prev_metal_scans = self._active_static_metal_scans
        prev_cut_scans = self._active_static_cut_scans
        prev_obs_scans = self._active_static_obstacle_scans
        if prev_metal_scans is None:
            self._active_static_metal_scans = {}
        if prev_cut_scans is None:
            self._active_static_cut_scans = {}
        if prev_obs_scans is None:
            self._active_static_obstacle_scans = {}

        def _cached_seed_portals(
            seed_state: Tuple[int, int, int],
        ) -> Tuple[Tuple[int, int, Tuple[int, int, int]], ...]:
            pt_key = (int(seed_state[0]), int(seed_state[1]), int(seed_state[2]))
            cached_pts = seed_portals_memo.get(pt_key)
            if cached_pts is not None:
                return cached_pts
            computed_pts = self._ranked_legal_group_route_boundary_portals(
                port.envelope_nm,
                pt_key,
                portal_targets,
                port.net,
                preserve_topology_rank=preserve_rank,
            )
            seed_portals_memo[pt_key] = computed_pts
            return computed_pts

        try:
            with self._component_access.escape_scope():
                for access in port.accesses:
                    key = _group_route_pin_key(port.net, access)
                    pair = pin_nodes.get(key)
                    if pair is None:
                        raise ValueError(
                            "group-route landing source Pin does not replay routing input"
                        )
                    _pin, node = pair
                    x0, y0, x1, y1 = port.envelope_nm
                    if (
                        node[2] in allowed
                        and physical_layer_key(node[2]) == access.layer_key
                        and x0 <= node[0] <= x1
                        and y0 <= node[1] <= y1
                    ):
                        boundary_depth = min(
                            node[0] - x0,
                            x1 - node[0],
                            node[1] - y0,
                            y1 - node[1],
                        )
                        direct_legal = self._direct_group_route_landing_legal(
                            tuple(node), str(port.net)
                        )
                        for portal_rank, portal_distance, portal_node in (
                            _cached_seed_portals(tuple(node))
                        ):
                            if not self._group_local_interface_allows_landing(
                                port.envelope_nm,
                                tuple(node),
                                None,
                                portal_node,
                                str(port.net),
                            ):
                                continue
                            candidates.append((
                                2 if (direct_legal or tuple(node) == tuple(portal_node)) else 3,
                                portal_rank,
                                portal_distance,
                                boundary_depth,
                                access,
                                (0, tuple(node)),
                                node,
                                node,
                                None,
                                "direct",
                                portal_node,
                            ))
                    if node[2] < self._n_layers - 1:
                        comp_stack = self._owned_terminal_stack_access(
                            port.net, *node
                        )
                        cached_comp = component_escape_memo.get(comp_stack)
                        if cached_comp is not None:
                            (
                                c_states,
                                c_rej,
                                c_samp,
                                c_details,
                                c_rej_cnt,
                            ) = cached_comp
                            self._merge_component_access_rejections(c_rej, c_samp)
                            if c_rej_cnt:
                                self._hard_rejection_generation = (
                                    int(getattr(self, "_hard_rejection_generation", 0))
                                    + c_rej_cnt
                                )
                            cur_details = dict(
                                getattr(self, "_route_failure_details", ())
                            )
                            cur_details.update(c_details)
                            self._route_failure_details = tuple(
                                sorted(cur_details.items())
                            )
                            escape_states = list(c_states)
                        else:
                            outer_c_rej = getattr(self, "_hard_rejections", {})
                            outer_c_samp = getattr(
                                self, "_hard_rejection_samples", {}
                            )
                            outer_c_gen = int(
                                getattr(self, "_hard_rejection_generation", 0)
                            )
                            outer_c_det = tuple(
                                getattr(self, "_route_failure_details", ())
                            )
                            self._hard_rejections = {}
                            self._hard_rejection_samples = {}
                            self._route_failure_details = ()
                            comp_done = False
                            try:
                                escape_states = self._owned_component_escape_states(
                                    port.net, *node
                                )
                                comp_done = True
                            finally:
                                loc_c_rej = tuple(sorted(
                                    (c, int(cnt))
                                    for c, cnt in self._hard_rejections.items()
                                ))
                                loc_c_samp = tuple(
                                    (c, int(cnt), tuple(ext))
                                    for c, (cnt, ext) in sorted(
                                        self._hard_rejection_samples.items()
                                    )
                                )
                                loc_c_det = tuple(
                                    (nm, val)
                                    for nm, val in self._route_failure_details
                                    if nm.startswith("component_escape_")
                                )
                                loc_c_cnt = sum(int(cnt) for _c, cnt in loc_c_rej)
                                self._hard_rejections = outer_c_rej
                                self._hard_rejection_samples = outer_c_samp
                                self._hard_rejection_generation = (
                                    outer_c_gen + loc_c_cnt
                                )
                                merged_det = dict(outer_c_det)
                                merged_det.update(loc_c_det)
                                self._route_failure_details = tuple(
                                    sorted(merged_det.items())
                                )
                                self._merge_component_access_rejections(
                                    loc_c_rej, loc_c_samp
                                )
                            if comp_done:
                                component_escape_memo[comp_stack] = (
                                    tuple(escape_states),
                                    loc_c_rej,
                                    loc_c_samp,
                                    loc_c_det,
                                    loc_c_cnt,
                                )
                    else:
                        escape_states = ()
                    for state, via in escape_states:
                        if state[2] not in allowed:
                            continue
                        x, y, _upper_gi = state
                        if not (x0 <= x <= x1 and y0 <= y <= y1):
                            continue
                        boundary_depth = min(x - x0, x1 - x, y - y0, y1 - y)
                        for portal_rank, portal_distance, portal_node in (
                            _cached_seed_portals(tuple(state))
                        ):
                            if not self._group_local_interface_allows_landing(
                                port.envelope_nm,
                                tuple(state),
                                via,
                                portal_node,
                                str(port.net),
                            ):
                                continue
                            candidates.append((
                                0 if via is None else 1,
                                portal_rank,
                                portal_distance,
                                boundary_depth,
                                access,
                                (1, self._access_state_key((state, via))),
                                node,
                                state,
                                via,
                                (
                                    "fixed_device_stack"
                                    if via is None
                                    else "emitted_via"
                                ),
                                portal_node,
                            ))
                upper_portal_gis: List[int] = []
                if target_gi is not None:
                    upper_portal_gis.append(int(target_gi))
                if (
                    has_external_targets
                    and bool(getattr(self, "_transactional_group_portal_layers", False))
                    and port.net not in getattr(self, "_rail_by_net", {})
                    and self._half_w
                ):
                    min_hw = max(1, min(int(h) for h in self._half_w))
                    if target_gi is not None or bool(
                        getattr(self, "_negotiation_port_legalize", False)
                    ):
                        lower_bound = (
                            int(target_gi) if target_gi is not None else 2
                        )
                        for g_cand in sorted(allowed):
                            if (
                                g_cand >= max(2, lower_bound)
                                and int(self._half_w[g_cand]) < 4 * min_hw
                                and g_cand not in upper_portal_gis
                            ):
                                upper_portal_gis.append(int(g_cand))
                if upper_portal_gis:
                    extra = []
                    seen_extra = set()
                    climb_state_memo: Dict[Tuple[Tuple[int, int, int], int], bool] = {}
                    base_rows = list(candidates)
                    for upper_gi in upper_portal_gis:
                        for ordinal, row in enumerate(base_rows):
                            if ordinal % 128 == 0:
                                self._watchdog_check(
                                    port.net, "group_portal_layer_candidates"
                                )
                            state = tuple(row[7])
                            via_row = row[8]
                            if row[0] not in (0, 1) or state[2] >= upper_gi:
                                continue
                            memo_key = (state, int(upper_gi))
                            climb_ok = climb_state_memo.get(memo_key)
                            if climb_ok is None:
                                climb_ok = self._group_route_has_legal_via_climb(
                                    port,
                                    allowed,
                                    int(state[2]),
                                    int(upper_gi),
                                    (state[0], state[1]),
                                )
                                climb_state_memo[memo_key] = climb_ok
                            if not climb_ok:
                                continue
                            seed = (state[0], state[1], int(upper_gi))
                            for rank, distance, portal in _cached_seed_portals(seed):
                                if not self._group_local_interface_allows_landing(
                                    port.envelope_nm,
                                    state,
                                    via_row,
                                    portal,
                                    str(port.net),
                                ):
                                    continue
                                added = (row[0], rank, distance, *row[3:10], portal)
                                if added in seen_extra:
                                    continue
                                seen_extra.add(added)
                                extra.append(added)
                    candidates.extend(extra)
        finally:
            self._active_static_metal_scans = prev_metal_scans
            self._active_static_cut_scans = prev_cut_scans
            self._active_static_obstacle_scans = prev_obs_scans
            self.__dict__.pop("_portal_legality_memo", None)
            self.__dict__.pop("_portal_climb_memo", None)
        if not candidates:
            raise GroupRouteLandingUnavailable(
                str(port.group), str(port.net), self.via_redundancy
            )
        return candidates

    def _group_route_external_target_rects(self, port, contract, net_nodes,
                                           net_node_pins, grouped_contract_pin_keys):
        # A materialized rail is the complete routing target for its supply
        # net.  Otherwise an explicit edge anchor is stronger than ordinary
        # device peers.  With neither, retain every ungrouped exact peer and
        # other same-net group envelope.
        rail_targets = tuple(
            tuple(_norm(rect))
            for _gi, rect in self._rail_by_net.get(port.net, ())
        )
        if rail_targets:
            return tuple(sorted(set(rail_targets)))

        peers = []
        anchors = []
        prior_nodes = tuple(net_nodes.get(port.net, ()))
        prior_pins = tuple(net_node_pins.get(port.net, ()))
        if len(prior_nodes) != len(prior_pins):
            raise ValueError("group-route external endpoint inventory is incomplete")
        for pin, node in zip(prior_pins, prior_nodes):
            if pin.xy_nm is None:
                raise ValueError(
                    "group-route external endpoint lacks integer coordinates"
                )
            key = _group_route_pin_key(port.net, pin)
            if key in grouped_contract_pin_keys:
                continue
            point = (int(node[0]), int(node[1]), int(node[0]), int(node[1]))
            peers.append(point)
            if pin.device == "__declared_anchor__":
                anchors.append(point)
        if anchors:
            return tuple(sorted(set(anchors)))
        peers.extend(
            tuple(other.envelope_nm)
            for other in contract.ports
            if other.net == port.net and other.group != port.group
        )
        return tuple(sorted(set(peers)))

    def _reserve_group_route_landings_once(
        self,
        contract: object,
        pins_by_net: Mapping[str, Sequence[Pin]],
        net_nodes: Dict[str, List[Tuple[int, int, int]]],
        net_node_pins: Dict[str, List[Pin]],
        net_layers: Mapping[str, Sequence[int]],
        book: _Book,
        *,
        ordered_ports: Sequence[object],
    ) -> object:
        """Close each group port locally and export one exact global portal."""

        _NM_PER_UM = _eg_attr("_NM_PER_UM", _base_NM_PER_UM)
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        if getattr(self, '_experimental_group_full_feedback', False):
            raise GroupRouteFullFeedbackUnavailable(
                'group full feedback requires the transactional physical feedback caller')

        from clay.layout.route.group_ports import (
            GroupRouteLanding,
            GroupRouteLandingCertificate,
            GroupRouteRoutingContract,
            verify_group_route_landing_certificate,
        )

        (
            pin_nodes,
            required_states,
            baselines,
            landings,
            grouped_pin_keys,
            portals_by_net,
            physical_layer_key,
            external_target_rects,
        ) = self._prepare_group_route_reservation_context(
            contract, pins_by_net, net_nodes, net_node_pins,
        )

        for port in ordered_ports:
            candidates = self._group_route_landing_candidates(
                port, pin_nodes, net_layers, book, physical_layer_key,
                external_target_rects,
            )
            if (
                getattr(self, "_experimental_group_demand_guide", None) is not None
            ):
                chosen_landing_row = self._select_group_route_landing(
                    port, candidates, book, landings
                )
            else:
                pref_gi = self._preferred_portal_gi(
                    port, set(int(v) for v in net_layers.get(port.net, ()))
                )
                has_match_partners = bool(
                    getattr(self, "_matched_partners_by_net", {}).get(str(port.net))
                )
                if pref_gi is not None or has_match_partners:
                    chosen_landing_row = min(
                        candidates,
                        key=lambda r: (
                            1 if r[0] >= 3 else 0,
                            r[0],
                            r[1],
                            r[2]
                            + (0 if pref_gi is None else abs(int(r[10][2]) - pref_gi) * 900)
                            + self._matched_portal_layer_penalty_nm(port, r, landings),
                            r,
                        ),
                    )
                else:
                    chosen_landing_row = min(candidates)
            (
                _kind_rank,
                _portal_rank,
                _portal_distance,
                _depth,
                access,
                _key,
                node,
                state,
                via,
                interface_kind,
                portal_node,
            ) = chosen_landing_row
            required_key = (port.net, tuple(node))
            if required_key in required_states:
                raise ValueError(
                    "group-route ports cannot share one physical source node"
                )
            required_states[required_key] = tuple(state)
            if portal_node[2] != state[2]:
                portal_req_key = (port.net, tuple(portal_node))
                if (
                    portal_req_key in required_states
                    and tuple(required_states[portal_req_key]) != tuple(portal_node)
                ):
                    raise ValueError(
                        "layered portal conflicts with selected access"
                    )
                required_states[portal_req_key] = tuple(portal_node)
            # The local solve consumes this exact selected interface.  Publish it
            # before entering _route_net so the source pin cannot silently choose
            # a different escape from the same owned conductor component.
            self._group_route_required_states = required_states
            baseline = baselines.setdefault(port.net, _RoutedNet(name=port.net))
            if via is not None:
                self._require_policy_via(via)
                if any(
                    (item.x, item.y, item.lo, item.n)
                    == (via.x, via.y, via.lo, via.n)
                    for item in baseline.vias
                ):
                    raise ValueError("group-route ports cannot share one emitted via")
                baseline.vias.append(via)
            metal_by_gi, cuts = self._net_metal(baseline)
            book.set_persistent(
                port.net,
                metal_by_gi,
                cuts,
                self._net_via_pads(baseline),
            )
            self._book = book
            self._rebuild_metal_index()

            local_rows = []
            for port_access in port.accesses:
                pin_key = _group_route_pin_key(port.net, port_access)
                pin_row = pin_nodes.get(pin_key)
                if pin_row is None:
                    raise ValueError(
                        "group-route local closure lacks an exact physical Pin"
                    )
                if pin_key in grouped_pin_keys:
                    raise ValueError(
                        "one physical Pin cannot belong to two group-route ports"
                    )
                grouped_pin_keys.add(pin_key)
                # The selected source is already physically joined to the upper
                # portal by the persistent landing via.  Re-adding the same
                # interface as a logical endpoint would ask the candidate checker
                # to prove a zero-length route that emits no new conductor.
                access_row = self._group_route_unlanded_access_row(
                    tuple(node),
                    tuple(pin_row[1]),
                    pin_row[0],
                )
                if access_row is not None:
                    local_rows.append(access_row)

            upper_key = physical_layer_key(state[2])
            landing_pin = Pin(
                net=port.net,
                device=f"__route_group_landing__{port.group}",
                terminal="landing",
                xy_um=(state[0] / _NM_PER_UM, state[1] / _NM_PER_UM),
                layer=upper_key,
                xy_nm=(state[0], state[1]),
            )
            portal_pin = Pin(
                net=port.net,
                device=f"__route_group__{port.group}",
                terminal="port",
                xy_um=(
                    portal_node[0] / _NM_PER_UM,
                    portal_node[1] / _NM_PER_UM,
                ),
                layer=physical_layer_key(portal_node[2]),
                xy_nm=(portal_node[0], portal_node[1]),
            )
            local_rows.extend(self._group_route_local_interface_rows(
                tuple(state),
                landing_pin,
                tuple(portal_node),
                portal_pin,
            ))

            # The local authority is the exact final group envelope, grown only
            # by current PDK wire-space and via-pad requirements.  It neither
            # imports reference coordinates nor permits a whole-cell detour.
            global_bounds = tuple(self._bounds)
            max_pad = max(
                (max(pair) for pair in self._via_pad_half.values()),
                default=max(self._half_w),
            )
            halo = max(self._half_w) + max(self._space) + max_pad
            x0, y0, x1, y1 = port.envelope_nm
            local_bounds = (
                max(global_bounds[0], x0 - halo),
                max(global_bounds[1], y0 - halo),
                min(global_bounds[2], x1 + halo),
                min(global_bounds[3], y1 + halo),
            )
            if (
                local_bounds[2] <= local_bounds[0]
                or local_bounds[3] <= local_bounds[1]
            ):
                raise ValueError("group-route local routing envelope is empty")

            prior_rail = self._rail_by_net.pop(port.net, None)
            self._bounds = local_bounds
            self._edge_fail = None
            self._hard_rejections = {}
            self._hard_rejection_samples = {}
            missing_baseline = object()
            previous_baseline = getattr(self, '_active_group_route_union_baseline', missing_baseline)
            previous_local_iface = getattr(self, '_active_group_route_local_interface', missing_baseline)
            previous_source_union_required = getattr(self, '_active_group_source_union_required', missing_baseline)
            previous_source_union_scratch = {
                name: getattr(self, name, missing_baseline)
                for name in ('_last_accepted_group_source_union', '_last_accepted_group_source_candidate')
            }
            try:
                previous_local_closure = getattr(
                    self, "_active_group_route_local_closure", False
                )
                self._active_group_route_local_closure = True
                self._active_group_route_local_interface = (
                    tuple(int(v) for v in port.envelope_nm),
                    (int(portal_node[0]), int(portal_node[1]), int(portal_node[2])),
                )
                self._active_group_route_union_baseline = baseline
                self._active_group_source_union_required = True
                if getattr(self, '_post_selection_observation', None) is not None:
                    from clay.layout.route.group_endpoint_observation import observe_selected
                    observe_selected(self, port, access, node, state, portal_node, interface_kind, book)
                local_route = self._route_net(
                    port.net,
                    [row[0] for row in local_rows],
                    list(net_layers[port.net]),
                    [row[1] for row in local_rows],
                )
            finally:
                if getattr(self, '_post_selection_observation', None) is None:
                    for name, previous_scratch in previous_source_union_scratch.items():
                        if previous_scratch is missing_baseline:
                            self.__dict__.pop(name, None)
                        else:
                            setattr(self, name, previous_scratch)
                    if previous_source_union_required is missing_baseline:
                        delattr(self, '_active_group_source_union_required')
                    else:
                        self._active_group_source_union_required = previous_source_union_required
                    if previous_baseline is missing_baseline:
                        delattr(self, '_active_group_route_union_baseline')
                    else:
                        self._active_group_route_union_baseline = previous_baseline
                    if previous_local_iface is missing_baseline:
                        self.__dict__.pop('_active_group_route_local_interface', None)
                    else:
                        self._active_group_route_local_interface = previous_local_iface
                    self._active_group_route_local_closure = previous_local_closure
                    self._bounds = global_bounds
                    if prior_rail is not None:
                        self._rail_by_net[port.net] = prior_rail
                else:
                    import sys
                    original = sys.exception()
                    try:
                        for name, previous_scratch in previous_source_union_scratch.items():
                            if previous_scratch is missing_baseline:
                                self.__dict__.pop(name, None)
                            else:
                                setattr(self, name, previous_scratch)
                        if previous_source_union_required is missing_baseline:
                            delattr(self, '_active_group_source_union_required')
                        else:
                            self._active_group_source_union_required = previous_source_union_required
                        if previous_baseline is missing_baseline:
                            delattr(self, '_active_group_route_union_baseline')
                        else:
                            self._active_group_route_union_baseline = previous_baseline
                        if previous_local_iface is missing_baseline:
                            self.__dict__.pop('_active_group_route_local_interface', None)
                        else:
                            self._active_group_route_local_interface = previous_local_iface
                        self._active_group_route_local_closure = previous_local_closure
                        self._bounds = global_bounds
                        if prior_rail is not None:
                            self._rail_by_net[port.net] = prior_rail
                    except BaseException as ending:
                        if original is None:
                            raise
                        from clay.layout.route.group_endpoint_observation import record_observation_unwind
                        record_observation_unwind(original, ending)
            if local_route is None or not local_route.connected:
                message = (
                    f"group-route port {port.group!r}:{port.net!r} "
                    "cannot close inside its PDK-derived local envelope; "
                    f"edge_fail={getattr(self, '_edge_fail', None)!r}; "
                    f"details={getattr(self, '_route_failure_details', ())!r}"
                )
                edge_fail = getattr(self, "_edge_fail", None)
                full_budget = (type(edge_fail) is tuple and len(edge_fail) in (8, 9)
                               and edge_fail[0] in {"budget", "solve_budget"}
                               and all(type(value) is int and value >= 0 for value in edge_fail[1:8]))
                exhausted_solve = (type(edge_fail) is tuple and len(edge_fail) == 6
                                   and edge_fail[0] == "solve_budget"
                                   and all(type(value) is int and value >= 0 for value in edge_fail[1:]))
                region_budget = (type(edge_fail) is tuple and len(edge_fail) == 6
                                 and edge_fail[0] in {"budget", "solve_budget"}
                                 and all(type(value) is int and value >= 0 for value in edge_fail[1:5])
                                 and edge_fail[5] == "planar_region_geometry")
                is_graph = (
                    full_budget
                    and len(edge_fail) == 9
                    and edge_fail[8] == "causal_frontier_graph"
                )
                if full_budget or exhausted_solve or region_budget:
                    reason = ("SOLVE_BUDGET_EXCEEDED" if edge_fail[0] == "solve_budget"
                              else "RESOURCE_BUDGET_EXCEEDED" if is_graph or region_budget
                              else "EDGE_EXPANSION_BUDGET")
                    diagnostic = {
                        "schema": "clay.group-route-budget-stop.v1",
                        "reason": reason,
                        "group": str(port.group),
                        "net": str(port.net),
                        "phase": "group_route_local_closure",
                        "failure_kind": edge_fail[0],
                    }
                    if full_budget:
                        diagnostic.update({
                            "edge_expansions": edge_fail[1],
                            "edge_budget": edge_fail[2],
                            "solve_work_before_edge": edge_fail[3],
                            "solve_work_budget": edge_fail[4],
                            "graph_axes": [edge_fail[5], edge_fail[6]],
                            "allowed_layer_count": edge_fail[7],
                        })
                    elif region_budget:
                        diagnostic.update({"resource_kind": "planar_region_geometry",
                                           "region_work": edge_fail[1],
                                           "region_work_budget": edge_fail[2],
                                           "solve_work_budget": edge_fail[4]})
                    else:
                        diagnostic.update({
                            "solve_work_before_edge": edge_fail[1],
                            "solve_work_budget": edge_fail[2],
                            "graph_axes": [edge_fail[3], edge_fail[4]],
                            "allowed_layer_count": edge_fail[5],
                        })
                    actual_work = getattr(self, "_solve_expansions_total", None)
                    if type(actual_work) is int and actual_work >= 0:
                        diagnostic["solve_work_actual"] = actual_work
                    if is_graph:
                        diagnostic["resource_kind"] = "causal_frontier_graph"
                        diagnostic["estimated_graph_states"] = edge_fail[5] * edge_fail[6] * max(1, edge_fail[7]) * 3
                        diagnostic["graph_state_cap"] = edge_fail[2]
                    if len(edge_fail) == 9 and edge_fail[8] == "planar_region_events":
                        diagnostic["resource_kind"] = "planar_region_events"
                        event_work = dict(getattr(self, "_route_failure_details", ())).get("planar_region_event_work")
                        if type(event_work) is int and event_work >= 0:
                            diagnostic["event_work"] = event_work
                            diagnostic["total_edge_work"] = edge_fail[1] + event_work
                            breakdown = dict(getattr(self, "_route_failure_details", ())).get("planar_region_event_work_by_kind")
                            if (type(breakdown) is tuple and all(type(row) is tuple and len(row) == 2
                                    and type(row[0]) is str and type(row[1]) is int and row[1] > 0 for row in breakdown)
                                    and sum(row[1] for row in breakdown) == event_work):
                                diagnostic["event_work_by_kind"] = dict(breakdown)
                    raise GroupRouteBudgetStop(message, diagnostic)
                raise ValueError(message)

            # Some local routing paths construct raw segments.  Freeze their
            # scoped width before publishing occupancy for the next port.
            active_widths = getattr(self, "_active_half_width_overrides", {})
            port_segs = [
                _WireSeg(seg.gi, seg.ax, seg.ay, seg.bx, seg.by,
                         self._segment_half_width(port.net, seg))
                if not seg.half_width_nm and (str(port.net), int(seg.gi)) in active_widths
                else seg
                for seg in local_route.segs
            ]
            port_vias = list(local_route.vias)
            if via is not None and not any(
                (item.x, item.y, item.lo, item.n) == (via.x, via.y, via.lo, via.n)
                for item in port_vias
            ):
                port_vias.append(via)
            local_port_net = _RoutedNet(
                name=port.net,
                segs=port_segs,
                vias=port_vias,
                connected=True,
            )
            self._legalize_group_port_baseline_with_interface(
                port, local_port_net, portal_node
            )
            baseline.segs.extend(local_port_net.segs)
            seen_vias = {
                (item.x, item.y, item.lo, item.n) for item in baseline.vias
            }
            for local_via in local_port_net.vias:
                local_key = (
                    local_via.x,
                    local_via.y,
                    local_via.lo,
                    local_via.n,
                )
                if local_key in seen_vias:
                    continue
                seen_vias.add(local_key)
                baseline.vias.append(local_via)
            baseline.connected = True
            metal_by_gi, cuts = self._net_metal(baseline)
            book.set_persistent(
                port.net,
                metal_by_gi,
                cuts,
                self._net_via_pads(baseline),
            )
            self._book = book
            self._rebuild_metal_index()
            portals_by_net.setdefault(port.net, []).append(
                (portal_node, portal_pin)
            )
            actual_count = (
                0
                if via is None
                else self.via_redundancy if via.n == 0 else via.n
            )
            landings.append(GroupRouteLanding(
                group=port.group,
                net=port.net,
                source_access=access,
                xy_nm=(state[0], state[1]),
                portal_xy_nm=(portal_node[0], portal_node[1]),
                lower_layer_key=(
                    access.layer_key
                    if via is None
                    else physical_layer_key(via.lo)
                ),
                upper_layer_key=physical_layer_key(state[2]),
                portal_layer_key=(
                    physical_layer_key(portal_node[2])
                    if portal_node[2] != state[2]
                    else None
                ),
                interface_kind=interface_kind,
                via_count=actual_count,
            ))
        return self._finalize_group_route_landing_certificate(
            contract=contract,
            landings=landings,
            portals_by_net=portals_by_net,
            net_nodes=net_nodes,
            net_node_pins=net_node_pins,
            grouped_pin_keys=grouped_pin_keys,
            required_states=required_states,
            baselines=baselines,
            physical_layer_key=physical_layer_key,
        )

    def _reserve_group_route_landings_transactional(
        self,
        contract: object,
        pins_by_net: Mapping[str, Sequence[Pin]],
        net_nodes: Dict[str, List[Tuple[int, int, int]]],
        net_node_pins: Dict[str, List[Pin]],
        net_layers: Mapping[str, Sequence[int]],
        book: _Book,
        *,
        ordered_ports: Sequence[object],
        adapter_factory: object,
    ) -> object:
        """Close each group port locally and export one exact global portal."""
        _NM_PER_UM = _eg_attr("_NM_PER_UM", _base_NM_PER_UM)
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        from clay.layout.route.group_fallback_controller import allocate, LocalExhaustion, LocalInconclusive

        from clay.layout.route.group_ports import (
            GroupRouteLanding,
            GroupRouteLandingCertificate,
            GroupRouteRoutingContract,
            verify_group_route_landing_certificate,
        )

        (
            pin_nodes,
            required_states,
            baselines,
            landings,
            grouped_pin_keys,
            portals_by_net,
            physical_layer_key,
            external_target_rects,
        ) = self._prepare_group_route_reservation_context(
            contract, pins_by_net, net_nodes, net_node_pins,
        )

        def regenerate(port):
            prev_flag = getattr(self, "_transactional_group_portal_layers", False)
            self._transactional_group_portal_layers = True
            try:
                return self._group_route_landing_candidates(
                    port, pin_nodes, net_layers, book, physical_layer_key,
                    external_target_rects,
                )
            finally:
                if prev_flag:
                    self._transactional_group_portal_layers = prev_flag
                else:
                    self.__dict__.pop("_transactional_group_portal_layers", None)

        def close_port(port, selected):
            (
                _kind_rank,
                _portal_rank,
                _portal_distance,
                _depth,
                access,
                _key,
                node,
                state,
                via,
                interface_kind,
                portal_node,
            ) = selected
            required_key = (port.net, tuple(node))
            if required_key in required_states:
                raise ValueError(
                    "group-route ports cannot share one physical source node"
                )
            required_states[required_key] = tuple(state)
            if portal_node[2] != state[2]:
                portal_req_key = (port.net, tuple(portal_node))
                if (
                    portal_req_key in required_states
                    and tuple(required_states[portal_req_key]) != tuple(portal_node)
                ):
                    raise ValueError(
                        "layered portal conflicts with selected access"
                    )
                required_states[portal_req_key] = tuple(portal_node)
            # The local solve consumes this exact selected interface.  Publish it
            # before entering _route_net so the source pin cannot silently choose
            # a different escape from the same owned conductor component.
            self._group_route_required_states = required_states
            baseline = baselines.setdefault(port.net, _RoutedNet(name=port.net))
            if via is not None:
                self._require_policy_via(via)
                if any(
                    (item.x, item.y, item.lo, item.n)
                    == (via.x, via.y, via.lo, via.n)
                    for item in baseline.vias
                ):
                    raise ValueError("group-route ports cannot share one emitted via")
                baseline.vias.append(via)
            metal_by_gi, cuts = self._net_metal(baseline)
            book.set_persistent(
                port.net,
                metal_by_gi,
                cuts,
                self._net_via_pads(baseline),
            )
            self._book = book
            self._rebuild_metal_index()

            local_rows = []
            for port_access in port.accesses:
                pin_key = _group_route_pin_key(port.net, port_access)
                pin_row = pin_nodes.get(pin_key)
                if pin_row is None:
                    raise ValueError(
                        "group-route local closure lacks an exact physical Pin"
                    )
                if pin_key in grouped_pin_keys:
                    raise ValueError(
                        "one physical Pin cannot belong to two group-route ports"
                    )
                grouped_pin_keys.add(pin_key)
                # The selected source is already physically joined to the upper
                # portal by the persistent landing via.  Re-adding the same
                # interface as a logical endpoint would ask the candidate checker
                # to prove a zero-length route that emits no new conductor.
                access_row = self._group_route_unlanded_access_row(
                    tuple(node),
                    tuple(pin_row[1]),
                    pin_row[0],
                )
                if access_row is not None:
                    local_rows.append(access_row)

            upper_key = physical_layer_key(state[2])
            landing_pin = Pin(
                net=port.net,
                device=f"__route_group_landing__{port.group}",
                terminal="landing",
                xy_um=(state[0] / _NM_PER_UM, state[1] / _NM_PER_UM),
                layer=upper_key,
                xy_nm=(state[0], state[1]),
            )
            portal_pin = Pin(
                net=port.net,
                device=f"__route_group__{port.group}",
                terminal="port",
                xy_um=(
                    portal_node[0] / _NM_PER_UM,
                    portal_node[1] / _NM_PER_UM,
                ),
                layer=physical_layer_key(portal_node[2]),
                xy_nm=(portal_node[0], portal_node[1]),
            )
            local_rows.extend(self._group_route_local_interface_rows(
                tuple(state),
                landing_pin,
                tuple(portal_node),
                portal_pin,
            ))

            # The local authority is the exact final group envelope, grown only
            # by active PDK wire-space and via-pad requirements (excluding thick
            # top redistribution layers unless explicitly needed by this port).
            global_bounds = tuple(self._bounds)
            if bool(getattr(self, "_negotiation_port_legalize", False)):
                halo, active_local_layers = self._group_route_active_local_halo_and_layers(
                    port.net,
                    net_layers[port.net],
                    (
                        state[2],
                        portal_node[2],
                        *(row[0][2] for row in local_rows),
                    ),
                )
            else:
                max_pad = max(
                    (max(pair) for pair in self._via_pad_half.values()),
                    default=max(self._half_w),
                )
                halo = max(self._half_w) + max(self._space) + max_pad
                active_local_layers = net_layers[port.net]
            x0, y0, x1, y1 = port.envelope_nm
            local_bounds = (
                max(global_bounds[0], x0 - halo),
                max(global_bounds[1], y0 - halo),
                min(global_bounds[2], x1 + halo),
                min(global_bounds[3], y1 + halo),
            )
            if (
                local_bounds[2] <= local_bounds[0]
                or local_bounds[3] <= local_bounds[1]
            ):
                raise ValueError("group-route local routing envelope is empty")

            prior_rail = self._rail_by_net.pop(port.net, None)
            self._bounds = local_bounds
            self._edge_fail = None
            self._hard_rejections = {}
            self._hard_rejection_samples = {}
            missing_baseline = object()
            previous_baseline = getattr(self, '_active_group_route_union_baseline', missing_baseline)
            previous_local_iface = getattr(self, '_active_group_route_local_interface', missing_baseline)
            previous_source_union_required = getattr(self, '_active_group_source_union_required', missing_baseline)
            previous_source_union_scratch = {
                name: getattr(self, name, missing_baseline)
                for name in ('_last_accepted_group_source_union', '_last_accepted_group_source_candidate')
            }
            try:
                previous_local_closure = getattr(
                    self, "_active_group_route_local_closure", False
                )
                self._active_group_route_local_closure = True
                self._active_group_route_local_interface = (
                    tuple(int(v) for v in port.envelope_nm),
                    (int(portal_node[0]), int(portal_node[1]), int(portal_node[2])),
                )
                self._active_group_route_union_baseline = baseline
                self._active_group_source_union_required = True
                if getattr(self, '_post_selection_observation', None) is not None:
                    from clay.layout.route.group_endpoint_observation import observe_selected
                    observe_selected(self, port, access, node, state, portal_node, interface_kind, book)
                local_route = self._route_net(
                    port.net,
                    [row[0] for row in local_rows],
                    active_local_layers,
                    [row[1] for row in local_rows],
                )
            finally:
                if getattr(self, '_post_selection_observation', None) is None:
                    for name, previous_scratch in previous_source_union_scratch.items():
                        if previous_scratch is missing_baseline:
                            self.__dict__.pop(name, None)
                        else:
                            setattr(self, name, previous_scratch)
                    if previous_source_union_required is missing_baseline:
                        delattr(self, '_active_group_source_union_required')
                    else:
                        self._active_group_source_union_required = previous_source_union_required
                    if previous_baseline is missing_baseline:
                        delattr(self, '_active_group_route_union_baseline')
                    else:
                        self._active_group_route_union_baseline = previous_baseline
                    if previous_local_iface is missing_baseline:
                        self.__dict__.pop('_active_group_route_local_interface', None)
                    else:
                        self._active_group_route_local_interface = previous_local_iface
                    self._active_group_route_local_closure = previous_local_closure
                    self._bounds = global_bounds
                    if prior_rail is not None:
                        self._rail_by_net[port.net] = prior_rail
                else:
                    import sys
                    original = sys.exception()
                    try:
                        for name, previous_scratch in previous_source_union_scratch.items():
                            if previous_scratch is missing_baseline:
                                self.__dict__.pop(name, None)
                            else:
                                setattr(self, name, previous_scratch)
                        if previous_source_union_required is missing_baseline:
                            delattr(self, '_active_group_source_union_required')
                        else:
                            self._active_group_source_union_required = previous_source_union_required
                        if previous_baseline is missing_baseline:
                            delattr(self, '_active_group_route_union_baseline')
                        else:
                            self._active_group_route_union_baseline = previous_baseline
                        if previous_local_iface is missing_baseline:
                            self.__dict__.pop('_active_group_route_local_interface', None)
                        else:
                            self._active_group_route_local_interface = previous_local_iface
                        self._active_group_route_local_closure = previous_local_closure
                        self._bounds = global_bounds
                        if prior_rail is not None:
                            self._rail_by_net[port.net] = prior_rail
                    except BaseException as ending:
                        if original is None:
                            raise
                        from clay.layout.route.group_endpoint_observation import record_observation_unwind
                        record_observation_unwind(original, ending)
            if local_route is None or not local_route.connected:
                message = (
                    f"group-route port {port.group!r}:{port.net!r} "
                    "cannot close inside its PDK-derived local envelope; "
                    f"edge_fail={getattr(self, '_edge_fail', None)!r}; "
                    f"details={getattr(self, '_route_failure_details', ())!r}"
                )
                edge_fail = getattr(self, "_edge_fail", None)
                full_budget = (type(edge_fail) is tuple and len(edge_fail) in (8, 9)
                               and edge_fail[0] in {"budget", "solve_budget"}
                               and all(type(value) is int and value >= 0 for value in edge_fail[1:8]))
                exhausted_solve = (type(edge_fail) is tuple and len(edge_fail) == 6
                                   and edge_fail[0] == "solve_budget"
                                   and all(type(value) is int and value >= 0 for value in edge_fail[1:]))
                region_budget = (type(edge_fail) is tuple and len(edge_fail) == 6
                                 and edge_fail[0] in {"budget", "solve_budget"}
                                 and all(type(value) is int and value >= 0 for value in edge_fail[1:5])
                                 and edge_fail[5] == "planar_region_geometry")
                if full_budget or exhausted_solve or region_budget:
                    is_graph = len(edge_fail) == 9 and edge_fail[8] == "causal_frontier_graph"
                    reason = ("SOLVE_BUDGET_EXCEEDED" if edge_fail[0] == "solve_budget"
                              else "RESOURCE_BUDGET_EXCEEDED" if is_graph or region_budget
                              else "EDGE_EXPANSION_BUDGET")
                    diagnostic = {
                        "schema": "clay.group-route-budget-stop.v1",
                        "reason": reason,
                        "group": str(port.group),
                        "net": str(port.net),
                        "phase": "group_route_local_closure",
                        "failure_kind": edge_fail[0],
                    }
                    if full_budget:
                        diagnostic.update({
                            "edge_expansions": edge_fail[1],
                            "edge_budget": edge_fail[2],
                            "solve_work_before_edge": edge_fail[3],
                            "solve_work_budget": edge_fail[4],
                            "graph_axes": [edge_fail[5], edge_fail[6]],
                            "allowed_layer_count": edge_fail[7],
                        })
                    elif region_budget:
                        diagnostic.update({"resource_kind": "planar_region_geometry",
                                           "region_work": edge_fail[1],
                                           "region_work_budget": edge_fail[2],
                                           "solve_work_budget": edge_fail[4]})
                    else:
                        diagnostic.update({
                            "solve_work_before_edge": edge_fail[1],
                            "solve_work_budget": edge_fail[2],
                            "graph_axes": [edge_fail[3], edge_fail[4]],
                            "allowed_layer_count": edge_fail[5],
                        })
                    actual_work = getattr(self, "_solve_expansions_total", None)
                    if type(actual_work) is int and actual_work >= 0:
                        diagnostic["solve_work_actual"] = actual_work
                    if is_graph:
                        diagnostic["resource_kind"] = "causal_frontier_graph"
                        diagnostic["estimated_graph_states"] = edge_fail[5] * edge_fail[6] * max(1, edge_fail[7]) * 3
                        diagnostic["graph_state_cap"] = edge_fail[2]
                    if len(edge_fail) == 9 and edge_fail[8] == "planar_region_events":
                        diagnostic["resource_kind"] = "planar_region_events"
                        event_work = dict(getattr(self, "_route_failure_details", ())).get("planar_region_event_work")
                        if type(event_work) is int and event_work >= 0:
                            diagnostic["event_work"] = event_work
                            diagnostic["total_edge_work"] = edge_fail[1] + event_work
                            breakdown = dict(getattr(self, "_route_failure_details", ())).get("planar_region_event_work_by_kind")
                            if (type(breakdown) is tuple and all(type(row) is tuple and len(row) == 2
                                    and type(row[0]) is str and type(row[1]) is int and row[1] > 0 for row in breakdown)
                                    and sum(row[1] for row in breakdown) == event_work):
                                diagnostic["event_work_by_kind"] = dict(breakdown)
                    raise GroupRouteBudgetStop(message, diagnostic)
                if (type(edge_fail) is tuple and len(edge_fail) == 6
                    and edge_fail[0] == "exhausted"):
                    raise LocalExhaustion(str(port.group), str(port.net), edge_fail,
                        getattr(self, "_route_failure_details", ()), ValueError(message),
                        self._hard_rejection_sample_evidence())
                if (type(edge_fail) is tuple and len(edge_fail) == 3
                    and edge_fail[0] == "symbolic_domain_incomplete"):
                    raise LocalInconclusive(str(port.group), str(port.net), edge_fail,
                        getattr(self, "_route_failure_details", ()), ValueError(message),
                        self._hard_rejection_sample_evidence())
                raise ValueError(message)

            # Some local routing paths construct raw segments.  Freeze their
            # scoped width before publishing occupancy for the next port.
            active_widths = getattr(self, "_active_half_width_overrides", {})
            port_segs = [
                _WireSeg(seg.gi, seg.ax, seg.ay, seg.bx, seg.by,
                         self._segment_half_width(port.net, seg))
                if not seg.half_width_nm and (str(port.net), int(seg.gi)) in active_widths
                else seg
                for seg in local_route.segs
            ]
            port_vias = list(local_route.vias)
            if via is not None and not any(
                (item.x, item.y, item.lo, item.n) == (via.x, via.y, via.lo, via.n)
                for item in port_vias
            ):
                port_vias.append(via)
            local_port_net = _RoutedNet(
                name=port.net,
                segs=port_segs,
                vias=port_vias,
                connected=True,
            )
            self._legalize_group_port_baseline_with_interface(
                port, local_port_net, portal_node
            )
            baseline.segs.extend(local_port_net.segs)
            seen_vias = {
                (item.x, item.y, item.lo, item.n) for item in baseline.vias
            }
            for local_via in local_port_net.vias:
                local_key = (
                    local_via.x,
                    local_via.y,
                    local_via.lo,
                    local_via.n,
                )
                if local_key in seen_vias:
                    continue
                seen_vias.add(local_key)
                baseline.vias.append(local_via)
            baseline.connected = True
            metal_by_gi, cuts = self._net_metal(baseline)
            book.set_persistent(
                port.net,
                metal_by_gi,
                cuts,
                self._net_via_pads(baseline),
            )
            self._book = book
            self._rebuild_metal_index()
            portals_by_net.setdefault(port.net, []).append(
                (portal_node, portal_pin)
            )
            actual_count = (
                0
                if via is None
                else self.via_redundancy if via.n == 0 else via.n
            )
            landings.append(GroupRouteLanding(
                group=port.group,
                net=port.net,
                source_access=access,
                xy_nm=(state[0], state[1]),
                portal_xy_nm=(portal_node[0], portal_node[1]),
                lower_layer_key=(
                    access.layer_key
                    if via is None
                    else physical_layer_key(via.lo)
                ),
                upper_layer_key=physical_layer_key(state[2]),
                portal_layer_key=(
                    physical_layer_key(portal_node[2])
                    if portal_node[2] != state[2]
                    else None
                ),
                interface_kind=interface_kind,
                via_count=actual_count,
            ))

        local = {"required_states": required_states, "baselines": baselines,
            "landings": landings, "grouped_pin_keys": grouped_pin_keys,
            "portals_by_net": portals_by_net, "net_nodes": net_nodes,
            "net_node_pins": net_node_pins, "committed_groups": set(),
            "group_contributions": {}}
        adapter = adapter_factory(self, book, local, regenerate, close_port)
        initial = adapter.snapshot()
        prior_atom_memo = getattr(self, "_transactional_atom_escape_memo", None)
        prior_tx_promoted = getattr(self, "_narrow_corridor_portal_gi", None)
        if prior_tx_promoted is None:
            self._narrow_corridor_portal_gi = (
                self._compute_narrow_corridor_portal_promotions(contract, net_layers)
            )
        self._transactional_atom_escape_memo = OrderedDict()
        try:
            allocate(ordered_ports, adapter, full_prefix_feedback=getattr(
                self, '_experimental_group_full_feedback', False))
            return self._finalize_group_route_landing_certificate(
                contract=contract,
                landings=landings,
                portals_by_net=portals_by_net,
                net_nodes=net_nodes,
                net_node_pins=net_node_pins,
                grouped_pin_keys=grouped_pin_keys,
                required_states=required_states,
                baselines=baselines,
                physical_layer_key=physical_layer_key,
            )
        except BaseException as original:
            if getattr(self, '_post_selection_observation', None) is not None:
                try:
                    adapter.restore(initial)
                except BaseException as ending:
                    from clay.layout.route.group_endpoint_observation import record_observation_unwind
                    record_observation_unwind(original, ending)
                raise
            adapter.restore(initial)
            raise
        finally:
            if prior_tx_promoted is None:
                self.__dict__.pop("_narrow_corridor_portal_gi", None)
            else:
                self._narrow_corridor_portal_gi = prior_tx_promoted
            if prior_atom_memo is None:
                self.__dict__.pop("_transactional_atom_escape_memo", None)
            else:
                self._transactional_atom_escape_memo = prior_atom_memo


    def _with_group_route_landing_baseline(self, routed: _RoutedNet) -> _RoutedNet:
        """Attach each pre-global emitted landing to its owning net result."""

        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        baseline = getattr(self, "_group_route_landing_baselines", {}).get(
            routed.name
        )
        if baseline is None:
            return routed
        if routed.name in getattr(self, "_tuned_1pin_baseline_nets", ()):
            return routed
        if (
            baseline.segs
            and len(routed.segs) >= len(baseline.segs)
            and all(a == b for a, b in zip(routed.segs, baseline.segs))
        ):
            segments = list(routed.segs)
        else:
            baseline_seg_keys = {
                (
                    int(seg.gi),
                    int(seg.ax),
                    int(seg.ay),
                    int(seg.bx),
                    int(seg.by),
                    int(getattr(seg, "half_width_nm", 0)),
                )
                for seg in baseline.segs
            }
            segments = list(baseline.segs) + [
                seg
                for seg in routed.segs
                if (
                    int(seg.gi),
                    int(seg.ax),
                    int(seg.ay),
                    int(seg.bx),
                    int(seg.by),
                    int(getattr(seg, "half_width_nm", 0)),
                )
                not in baseline_seg_keys
            ]
        vias = []
        seen = set()
        for via in (*baseline.vias, *routed.vias):
            key = (via.x, via.y, via.lo, via.n)
            if key in seen:
                continue
            seen.add(key)
            vias.append(via)
        return _RoutedNet(
            name=routed.name,
            segs=segments,
            vias=vias,
            connected=bool(routed.connected),
        )

