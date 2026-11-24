"""Source-selected ordered-channel topology on the canonical exact router.

Only topology selection is new. Endpoint ownership, via geometry, occupancy,
fixed artwork, emission and deck isolation remain the Gridless implementation.
"""
from __future__ import annotations

from .channel_geometry import ChannelRoute, channel_clearance_nm
from .constraints import Mirror, Width
from .engine_gridless import GridlessEngine, _RoutedNet, _ViaSite, _WireSeg


class ChannelRouteEngine(GridlessEngine):
    name = "channel"
    consumes_resource_plan = False
    consumes_diagnostic_resource_program = False
    consumes_rail_placement_plan = True

    def satisfies(self):
        return super().satisfies() | frozenset({"channel_route"})

    def route(self, pins_by_net, obstacles, constraints, pdk, **kwargs):
        channels = tuple(c for c in constraints if isinstance(c, ChannelRoute))
        if not channels:
            raise ValueError("channel strategy requires a source-declared channel_route")
        if kwargs.get("resource_plan") is not None:
            raise ValueError("channel topology cannot replace an allocated resource plan")
        self._channel_by_net = {}
        for channel in channels:
            declaration = channel.declaration
            declaration.assert_authoritative()
            clearance = channel_clearance_nm(declaration, pdk)
            required = (len(declaration.nets) - 1) * declaration.pitch_nm
            if required + 2 * clearance > channel.y_high_nm - channel.y_low_nm:
                raise ValueError("channel track bank exceeds its actual placed corridor")
            y0 = channel.y_low_nm + clearance
            for index, net in enumerate(declaration.nets):
                if net in self._channel_by_net or net not in pins_by_net:
                    raise ValueError("channel route has a duplicate or absent physical net")
                self._channel_by_net[net] = (channel, y0 + index * declaration.pitch_nm)
        self._channel_records = {}
        for constraint in constraints:
            if isinstance(constraint, Mirror) and (
                    constraint.net_a in self._channel_by_net
                    or constraint.net_b in self._channel_by_net):
                raise ValueError("exact vertical net Mirror conflicts with distinct channel tracks")
        remaining = [c for c in constraints if not isinstance(c, ChannelRoute)]
        # The declaration specifies both role widths. Its hard Width enters the
        # existing width model; the per-role table below preserves branch width.
        remaining.extend(Width(net, channel.declaration.width_um)
                         for net, (channel, _y) in self._channel_by_net.items())
        result = super().route(pins_by_net, obstacles, remaining, pdk, **kwargs)
        result.channel_routes = channels
        result.channel_records = tuple(self._channel_records.get(net, {"net": net,
            "pass": False, "reason": "no_complete_channel_route"})
            for channel in channels for net in channel.declaration.nets)
        for channel in channels:
            if any(not self._channel_records.get(net, {}).get("pass", False)
                   or not result.nets.get(net) or not result.nets[net].connected
                   for net in channel.declaration.nets):
                result.unsatisfied.append(channel)
                result.success = False
        return result

    def _requires_singleton_route(self, net):
        return net in self._channel_by_net

    def _route_net_to_rail(self, net, nodes, lg, rn, rail, node_pins=None,
                           endpoint_members=None):
        # The rail width is already materialized by @trunk(edge=...). The
        # independent Width declaration is the source's local-drop width.
        return self._route_net_to_rail_impl(net, nodes, lg, rn, rail,
            node_pins=node_pins, endpoint_members=endpoint_members)

    def _route_net(self, net, nodes, lg, node_pins=None):
        bound = self._channel_by_net.get(net)
        if bound is None:
            return super()._route_net(net, nodes, lg, node_pins)
        channel, track_y = bound
        declaration = channel.declaration
        trunk = self._tech.resolve_layer_to_gi(declaration.trunk_layer)
        branch = self._tech.resolve_layer_to_gi(
            dict(declaration.branch_layers).get(net, declaration.branch_layer))
        if trunk is None or branch is None or trunk not in lg or branch not in lg:
            self._edge_fail = ("channel_layer_not_allowed", net)
            return None
        if branch <= trunk:
            raise ValueError("channel branch layer must lie above its trunk layer")
        grid = self._manufacturing_grid_nm
        if track_y % grid or declaration.width_nm % (2 * grid):
            raise ValueError("channel track/width cannot be emitted exactly on the PDK grid")
        self._net_half_w[net] = [max(half, declaration.width_nm // 2)
                                   for half in self._half_w]
        self._rebuild_metal_index()
        self._attempt_cuts = {}
        self._attempt_metal = {}
        self._attempt_idx = {}
        nodes, pins, _members = self._contract_owned_component_endpoints(
            net, list(nodes), list(node_pins or [None] * len(nodes)))
        route = _RoutedNet(name=net)
        xs = []
        known_vias = set()
        for node, pin in zip(nodes, pins):
            px, py, gi = node
            self._begin_route_failure_context(phase="channel_branch",
                source_pin=pin, source=node, goal_pin=None,
                goal=(px, track_y, trunk), window=self._bounds,
                allowed_layers=lg, trunk_gi=trunk)
            # Canonical component access includes exact fixed-metal witness and
            # verifies all foreign artwork/cut keepouts before exporting a site.
            access = self._pin_access(net, px, py, gi)
            selected = None
            for state, escape in access:
                x, y, start = state
                if start > branch or any(level not in lg
                                         for level in range(start, branch + 1)):
                    continue
                sites = ([] if escape is None else [escape])
                sites += [_ViaSite(x, y, level) for level in range(start, branch)]
                sites += [_ViaSite(x, track_y, level)
                          for level in range(trunk, branch)]
                candidate = []
                valid = True
                for site in sites:
                    key = (site.x, site.y, site.lo)
                    if key in known_vias or any((v.x, v.y, v.lo) == key for v in candidate):
                        continue
                    if not self._via_legal(site.x, site.y, site.lo, net,
                                           n=self._site_n(site)):
                        valid = False
                        break
                    candidate.append(site)
                if not valid or not self._wire_legal(branch, x, y, x, track_y, net):
                    continue
                selected = (x, y, candidate)
                break
            if selected is None:
                self._edge_fail = ("channel_branch_no_legal_access", net, tuple(node))
                return None
            x, y, vias = selected
            xs.append(x)
            route.segs.append(_WireSeg(branch, x, y, x, track_y,
                                       self._hw(net, branch)))
            route.vias.extend(vias)
            for via in vias:
                known_vias.add((via.x, via.y, via.lo))
                self._add_attempt_cut(via)
        if not xs:
            self._edge_fail = ("channel_has_no_terminals", net)
            return None
        x0, x1 = min(xs), max(xs)
        trunk_half = self._hw(net, trunk)
        # The span is actual metal, including the declared reliability arrays.
        # Extend the wire endpoint when a larger via landing would otherwise
        # overhang it; the subsequent centering operates on the full footprint.
        for via in route.vias:
            if via.y != track_y or trunk not in (via.lo, via.lo + 1):
                continue
            below, above = self._via_pad_half_for(via)
            pad_half = below if trunk == via.lo else above
            x0 = min(x0, via.x - pad_half + trunk_half)
            x1 = max(x1, via.x + pad_half - trunk_half)
        if net in declaration.centered_nets:
            axis2 = channel.axis_x2_nm
            x0, x1 = min(x0, axis2 - x1), max(x1, axis2 - x0)
        if not self._wire_legal(trunk, x0, track_y, x1, track_y, net):
            self._edge_fail = ("channel_trunk_blocked", net)
            return None
        route.segs.append(_WireSeg(trunk, x0, track_y, x1, track_y,
                                   self._hw(net, trunk)))
        branch_spans = tuple((s.ax, s.ay, s.by) for s in route.segs
                             if s.gi == branch and s.ax == s.bx)
        self._legalize_same_net(net, route)
        route.connected = True
        self._channel_records[net] = {"net": net, "pass": True,
            "trunk_layer": declaration.trunk_layer,
            "branch_layer": dict(declaration.branch_layers).get(net, declaration.branch_layer),
            "track_y_nm": track_y, "trunk_span_nm": (x0, x1),
            "width_nm": declaration.width_nm,
            "branch_spans_nm": branch_spans,
            "source_channel_sha256": declaration.sha256,
            "axis_x2_nm": channel.axis_x2_nm if net in declaration.centered_nets else None}
        return route
