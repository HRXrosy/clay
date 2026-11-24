"""Independent, reopened-GDS checks of source-bound channel constructions."""
from __future__ import annotations

from .channel_geometry import channel_clearance_nm


def _require_terminal_branch_coverage(layout, cell, stack, view, net, branch,
                                      trunk_key, trunk_polygon, spans, half,
                                      terminal_probes):
    """Prove each real terminal component has a declared branch of its own.

    Full-net connectivity alone can hide a missing branch record: a conductor
    on another layer may reach the common trunk through an unrecorded path.
    Open that trunk in a separate extraction view, retain only real local
    landing metal at declared branch starts, and require a branch start on
    every terminal component. Multiple accesses of one native finger strap
    may share a branch, including an access displaced along the same strap.
    No producer count or producer component label is used as evidence.
    """
    import klayout.db as kdb
    from .deck_isolation import NetProbe, check_deck_net_isolation

    terminals = tuple(sorted({(tuple(probe.layer), probe.x, probe.y)
        for probe in terminal_probes if str(probe.net).upper() == net.upper()}))
    if not terminals:
        raise ValueError(f"channel {net!r} has no canonical terminal probes")
    branch_gi = view.resolve_layer_to_gi(branch)
    branch_key = tuple(view.gi_to_metal_key(branch_gi))
    # Native terminal metal may itself lie above the branch role. Preserve its
    # real stack; an unrelated upper routing layer cannot supply a missing
    # connector when no canonical terminal requires that layer.
    terminal_levels = [view.resolve_layer_to_pdk(layer) for layer, _x, _y in terminals]
    if any(level is None for level in terminal_levels):
        raise ValueError("channel terminal layer is outside the PDK metal stack")
    ceiling = max(view._gi_to_pdk[branch_gi], *terminal_levels)
    excluded = {key for key in stack.conductors
                if view.resolve_layer_to_pdk(key) > ceiling}
    projected = kdb.Layout()
    projected.dbu = layout.dbu
    projected_cell = projected.create_cell("CHANNEL_BRANCH_COVERAGE")
    removed_trunk = kdb.Region(trunk_polygon)
    local_starts = kdb.Region()
    for x, start_y, _end_y in spans:
        local_starts.insert(kdb.Box(x-half, start_y-half, x+half, start_y+half))
    layers = set(stack.conductors) | {connection.cut for connection in stack.connections}
    layers.update(carve.top_marker for carve in stack.mim_carves)
    for key in sorted(layers):
        if key in excluded:
            continue
        index = layout.find_layer(*key)
        if index is None or int(index) < 0:
            continue
        region = kdb.Region(cell.begin_shapes_rec(index)).merged()
        if key == trunk_key:
            # Intersect with existing metal: this observer never creates an
            # absent landing or makes a disconnected source component whole.
            region = (region - removed_trunk) | (region & removed_trunk & local_starts)
        projected_cell.shapes(projected.layer(*key)).insert(region)
    probes = [NetProbe(f"TERMINAL_{i}", layer, x, y)
              for i, (layer, x, y) in enumerate(terminals)]
    probes.extend(NetProbe(f"BRANCH_{i}", branch_key, x, start_y)
                  for i, (x, start_y, _end_y) in enumerate(spans))
    connectivity = check_deck_net_isolation(layout=projected, cell=projected_cell,
                                           stack=stack, probes=tuple(probes))
    if connectivity.unresolved_probes or connectivity.ambiguous_nets:
        raise ValueError(f"channel {net!r} terminal/branch coverage is unresolved")
    # Unique logical labels deliberately expose physical component membership
    # through the existing deck oracle's merged groups. A singleton has no
    # partner group and therefore cannot certify a terminal-to-branch path.
    covered = set()
    for group in connectivity.merged_net_groups:
        if (any(name.startswith("TERMINAL_") for name in group)
                and any(name.startswith("BRANCH_") for name in group)):
            covered.update(group)
    if covered != {probe.net for probe in probes}:
        raise ValueError(f"channel {net!r} branch witnesses do not cover canonical terminal components")


def require_emitted_channels(path, routed, tech, *, top_cell=None,
                             terminal_probes=(), source_channels=None):
    import klayout.db as kdb
    from .deck_isolation import (DeckConnectStack, NetProbe,
                                  check_deck_net_isolation)
    from .engine_maze import _Tech

    channels = tuple(getattr(routed, "channel_routes", ()))
    if source_channels is not None:
        if tuple(bound.declaration for bound in channels) != tuple(source_channels):
            raise ValueError("final channel inventory differs from source declarations")
    if not channels:
        return ()
    layout = kdb.Layout()
    layout.read(str(path))
    if abs(layout.dbu - 0.001) > 1e-12:
        raise ValueError("channel verification requires exact nanometre GDS DBU")
    cell = layout.cell(top_cell) if top_cell else layout.top_cell()
    if cell is None:
        raise ValueError("channel verification cannot find the emitted top cell")
    view = _Tech(tech)
    records = tuple(getattr(routed, "channel_records", ()))
    record_map = {row["net"]: row for row in records}
    expected_nets = [net for c in channels for net in c.declaration.nets]
    if len(record_map) != len(records) or set(record_map) != set(expected_nets):
        raise ValueError("final channel witness has missing or duplicate nets")
    regions = {}
    probes = list(terminal_probes)
    report = []
    stack = DeckConnectStack.from_tech(tech)

    def require_rectangle(net, layer, box):
        gi = view.resolve_layer_to_gi(layer)
        if gi is None:
            raise ValueError("channel checker cannot resolve its declared layer")
        key = tuple(view.gi_to_metal_key(gi))
        if key not in regions:
            regions[key] = kdb.Region(cell.begin_shapes_rec(layout.layer(*key))).merged()
        if not (kdb.Region(kdb.Box(*box)) - regions[key]).is_empty():
            raise ValueError(f"final GDS does not cover declared channel {net!r} {layer} {box}")
        probes.append(NetProbe(net.upper(), key, (box[0] + box[2]) // 2,
                               (box[1] + box[3]) // 2))

    for bound in channels:
        declaration = bound.declaration
        declaration.assert_authoritative()
        y0 = bound.y_low_nm + channel_clearance_nm(declaration, tech)
        for index, net in enumerate(declaration.nets):
            row = record_map[net]
            y = y0 + index * declaration.pitch_nm
            branch = dict(declaration.branch_layers).get(net, declaration.branch_layer)
            if (row.get("source_channel_sha256") != declaration.sha256
                    or row.get("track_y_nm") != y
                    or row.get("trunk_layer") != declaration.trunk_layer
                    or row.get("branch_layer") != branch
                    or row.get("width_nm") != declaration.width_nm):
                raise ValueError("channel witness differs from bound source semantics")
            x0, x1 = row["trunk_span_nm"]
            if x0 > x1:
                raise ValueError("channel span endpoints are reversed")
            if net in declaration.centered_nets and x0 + x1 != bound.axis_x2_nm:
                raise ValueError("channel span is not centered on the bound placement axis")
            half = declaration.width_nm // 2
            require_rectangle(net, declaration.trunk_layer,
                              (x0-half, y-half, x1+half, y+half))
            spans = tuple(row.get("branch_spans_nm", ()))
            if not spans:
                raise ValueError("channel has no physical branch witness")
            for x, start_y, end_y in spans:
                if end_y != y or not x0 <= x <= x1:
                    raise ValueError("channel branch does not meet its declared trunk")
                require_rectangle(net, branch,
                    (x-half, min(start_y, end_y)-half,
                     x+half, max(start_y, end_y)+half))
            # Measure the complete connected M2 conductor, not just its centre
            # line. A one-sided extension attached above or below the centre
            # still changes the physical span and its placement-axis symmetry.
            # Via pads may legally be wider than the wire: derive their possible
            # overhang from the same PDK model, rather than mistake them for an
            # extra trunk segment or accept arbitrary extra metal.
            from .via_model import via_landing
            trunk_gi = view.resolve_layer_to_gi(declaration.trunk_layer)
            branch_gi = view.resolve_layer_to_gi(branch)
            key = tuple(view.gi_to_metal_key(trunk_gi))
            expected_lo, expected_hi = x0-half, x1+half
            for x, _start_y, _end_y in spans:
                for lower in range(trunk_gi, branch_gi):
                    landing = via_landing(tech, f"via{view._gi_to_pdk[lower]}", x, y)
                    if landing is None:
                        raise ValueError("channel span lacks a PDK via rule")
                    if lower == trunk_gi:
                        expected_lo = min(expected_lo, landing.below_metal_rect[0])
                        expected_hi = max(expected_hi, landing.below_metal_rect[2])
            trunk_region = kdb.Region(kdb.Box(x0-half, y-half, x1+half, y+half))
            observed_polygons = [polygon for polygon in regions[key].each()
                                 if (trunk_region - kdb.Region(polygon)).is_empty()]
            observed = [polygon.bbox() for polygon in observed_polygons]
            if (len(observed) != 1 or observed[0].left != expected_lo
                    or observed[0].right != expected_hi):
                raise ValueError("final GDS channel trunk has an undeclared span extension")
            if net in declaration.centered_nets and (
                    observed[0].left + observed[0].right != bound.axis_x2_nm):
                raise ValueError("actual final GDS trunk is not centered on its placement axis")
            _require_terminal_branch_coverage(layout, cell, stack, view, net,
                branch, key, observed_polygons[0], spans, half, terminal_probes)
            report.append({"net": net, "track_y_nm": y, "span_nm": (x0, x1),
                           "branch_layer": branch, "actual_gds_coverage": True,
                           "canonical_terminal_branch_coverage": True})
    # Real deck connectivity joins each measured trunk/branch to the canonical
    # terminal probes. Coverage by a disconnected or foreign polygon is rejected.
    isolation = check_deck_net_isolation(layout=layout, cell=cell,
        stack=stack, probes=tuple(probes))
    if not isolation.ok:
        raise ValueError(f"final GDS channel connectivity failed: {isolation}")
    return tuple(report)
