"""PDK-derived inter-route-group corridor and native-row M1 escape floors.

When a cell selects multi-cut via redundancy (``via_redundancy >= 2``) or declares
``@route_group(..., ports=external)`` across floorplan columns, area-minimizing
placement must not crush inter-column corridors or non-abutting native-row channels
below the PDK landing pad + spacing width required for legal escape routing.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Set, Tuple


def _ceil_to_grid(value_nm: int, grid_nm: int) -> int:
    grid = max(1, int(grid_nm))
    val = max(0, int(value_nm))
    return ((val + grid - 1) // grid) * grid


def resolve_cell_via_redundancy(cell_def: Any) -> int:
    """Resolve the active route engine's ``via_redundancy`` via canonical policy."""
    if cell_def is None:
        return 1
    try:
        from clay.layout.policy import policy_for_cell
        from clay.layout.route.engine_registry import route_engine_from_policy

        source_file = getattr(cell_def, "_clay_file", None)
        policy = policy_for_cell(source_file, cell_def)
        engine = route_engine_from_policy(policy)
        raw = getattr(engine, "via_redundancy", 1)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return max(1, int(raw))
    except Exception:
        return 1
    return 1


def _via_pad_extent_nm(
    via_rule: Any,
    *,
    cuts: int,
    above: bool,
) -> Tuple[int, int]:
    """Return ``(short_side_nm, long_side_nm)`` of a ``1 x cuts`` via pad."""
    n = max(1, int(cuts))
    cut = int(getattr(via_rule.size, "value", 0))
    space = int(getattr(via_rule.spacing, "value", 0)) if n > 1 else 0
    enc_attr = "enclosure_above" if above else "enclosure_below"
    enc = int(getattr(getattr(via_rule, enc_attr), "value", 0))
    short_nm = cut + 2 * enc
    long_nm = n * cut + (n - 1) * space + 2 * enc
    return short_nm, long_nm


def pdk_m2_route_group_corridor_floor_nm(
    tech: Any,
    *,
    via_redundancy: int,
    multi_port: bool,
    grid_nm: int,
) -> int:
    """Minimum horizontal clearance between adjacent external ``@route_group`` columns.

    Derives ``W_via_pad(M2, N) + 2 * S_min(M2)`` directly from ``tech`` rules,
    adding one M2 track pitch ``W_min(M2) + S_min(M2)`` when an adjacent boundary
    carries multi-port group landings under ``N >= 2``.
    """
    n = max(1, int(via_redundancy))
    m2_w = int(tech.min_width("metal2"))
    m2_s = int(tech.min_space("metal2"))
    pad_m2 = m2_w
    try:
        v1 = tech.via_rule("via1")
        _, v1_m2 = _via_pad_extent_nm(v1, cuts=n, above=True)
        pad_m2 = max(pad_m2, v1_m2)
    except Exception:
        pass
    try:
        v2 = tech.via_rule("via2")
        _, v2_m2 = _via_pad_extent_nm(v2, cuts=n, above=False)
        pad_m2 = max(pad_m2, v2_m2)
    except Exception:
        pass
    extra_pitch = (m2_w + m2_s) if (multi_port and n >= 2) else 0
    return _ceil_to_grid(pad_m2 + 2 * m2_s + extra_pitch, grid_nm)


def pdk_m1_escape_channel_floor_nm(
    tech: Any,
    *,
    via_redundancy: int,
) -> Tuple[int, int, int]:
    """Return ``(v1_m1_short_nm, v1_m1_long_nm, m1_channel_floor_nm)`` from PDK rules."""
    n = max(1, int(via_redundancy))
    m1_w = int(tech.min_width("metal1"))
    m1_s = int(tech.min_space("metal1"))
    v1 = tech.via_rule("via1")
    short_nm, long_nm = _via_pad_extent_nm(v1, cuts=n, above=False)
    channel_nm = max(m1_w, long_nm) + 2 * m1_s
    return short_nm, long_nm, channel_nm


def route_group_corridor_orders_from_intent(
    intent: Any,
    block_member_devices: Mapping[str, Sequence[str]],
    *,
    known_device_names: Sequence[str],
    existing_orders: Sequence[Any] = (),
    tech: Any = None,
    cell_def: Any = None,
    base_gap_nm: int = 0,
) -> Tuple[Any, ...]:
    """Synthesize PDK-backed horizontal corridor separations between external route groups."""
    if tech is None or intent is None:
        return ()
    raw_route_groups = tuple(getattr(intent, "route_groups", ()) or ())
    if not raw_route_groups:
        return ()
    external_rgs = tuple(
        rg
        for rg in raw_route_groups
        if str(getattr(rg, "ports", "") or "") == "external"
    )
    if not external_rgs:
        return ()

    from clay.layout.intent import place_group_floorplan_row
    from clay.layout.place.group_separation_strength import (
        separation_gap_nm,
        separation_strength,
    )
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

    place_groups = tuple(getattr(intent, "place_groups", ()) or ())
    pg_by_name = {str(pg.group): pg for pg in place_groups}
    # Map each place_group to its owning external route_group (if any).
    pg_to_rg: Dict[str, str] = {}
    for rg in raw_route_groups:
        rg_name = str(rg.group)
        members = (
            tuple(getattr(rg, "place_groups", ()) or ())
            or tuple(getattr(rg, "groups", ()) or ())
            or (rg_name,)
        )
        for pg_name in members:
            pg_to_rg[str(pg_name)] = rg_name
    external_rg_names = {str(rg.group) for rg in external_rgs}

    # Count external ports per route_group when cell_def is available.
    ports_by_rg: Dict[str, int] = {}
    if cell_def is not None:
        try:
            from clay.layout.route.group_ports import build_group_route_port_inventory

            inv = build_group_route_port_inventory(cell_def, intent)
            for port in inv.ports:
                ports_by_rg[port.group] = ports_by_rg.get(port.group, 0) + 1
        except Exception:
            ports_by_rg = {}

    # Group floorplan place_groups by (row, col).
    cells_by_row: Dict[int, Dict[int, list[Any]]] = {}
    for pg in place_groups:
        col = getattr(pg, "col", None)
        row = place_group_floorplan_row(pg)
        if col is None or row is None:
            continue
        cells_by_row.setdefault(int(row), {}).setdefault(int(col), []).append(pg)

    if not cells_by_row:
        return ()

    known = {str(d) for d in known_device_names}
    via_redundancy = resolve_cell_via_redundancy(cell_def)
    grid_nm = pdk_manufacturing_grid_nm(tech)

    # Index existing horizontal gap separations by (before_masters, after_masters).
    existing_horiz_gaps: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], int] = {}
    for row in existing_orders:
        if len(row) == 7 and row[3] == "horizontal":
            gap_val = separation_gap_nm(row[4])
            if gap_val is not None:
                key = (tuple(row[5]), tuple(row[6]))
                existing_horiz_gaps[key] = max(existing_horiz_gaps.get(key, 0), int(gap_val))

    synthesized = []
    for row_idx in sorted(cells_by_row):
        col_map = cells_by_row[row_idx]
        cols = sorted(col_map)
        if len(cols) < 2:
            continue
        for idx in range(len(cols) - 1):
            c_left, c_right = cols[idx], cols[idx + 1]
            left_pgs = col_map[c_left]
            right_pgs = col_map[c_right]
            left_rgs = {pg_to_rg.get(str(pg.group)) for pg in left_pgs} - {None}
            right_rgs = {pg_to_rg.get(str(pg.group)) for pg in right_pgs} - {None}
            # Require at least one external route_group on this column boundary,
            # and skip intra-route-group column pairs.
            if not ((left_rgs | right_rgs) & external_rg_names):
                continue
            if left_rgs and right_rgs and left_rgs == right_rgs:
                continue

            left_devs = {
                str(d)
                for pg in left_pgs
                for d in (getattr(pg, "devices", ()) or ())
            } & known
            right_devs = {
                str(d)
                for pg in right_pgs
                for d in (getattr(pg, "devices", ()) or ())
            } & known
            if not left_devs or not right_devs or (left_devs & right_devs):
                continue

            before_masters = tuple(
                sorted(
                    master
                    for master, members in block_member_devices.items()
                    if ({str(m) for m in members} & left_devs)
                )
            )
            after_masters = tuple(
                sorted(
                    master
                    for master, members in block_member_devices.items()
                    if ({str(m) for m in members} & right_devs)
                )
            )
            if (
                not before_masters
                or not after_masters
                or (set(before_masters) & set(after_masters))
            ):
                continue
            # Ensure no master straddles foreign devices outside its column cell.
            if any(
                ({str(m) for m in block_member_devices[master]} & known) & right_devs
                for master in before_masters
            ) or any(
                ({str(m) for m in block_member_devices[master]} & known) & left_devs
                for master in after_masters
            ):
                continue

            max_ports = max(
                *(ports_by_rg.get(rg_name, 0) for rg_name in (left_rgs | right_rgs) if rg_name),
                0,
            )
            multi_port = max_ports >= 3
            floor_nm = pdk_m2_route_group_corridor_floor_nm(
                tech,
                via_redundancy=via_redundancy,
                multi_port=multi_port,
                grid_nm=grid_nm,
            )
            required_nm = max(int(base_gap_nm), int(floor_nm))
            if existing_horiz_gaps.get((before_masters, after_masters), -1) >= required_nm:
                continue
            scope = f"route_group_corridor:row={row_idx}:col={c_left}->{c_right}"
            synthesized.append(
                (
                    scope,
                    f"route_col_{row_idx}_{c_left}",
                    f"route_col_{row_idx}_{c_right}",
                    "horizontal",
                    separation_strength(required_nm),
                    before_masters,
                    after_masters,
                )
            )
    return tuple(synthesized)


def _stamp_requires_external_m1_via_escape(
    stamp: Any,
    *,
    m1_key: Tuple[int, int],
    via_redundancy: int,
    v1_m1_short_nm: int,
    v1_m1_long_nm: int,
) -> bool:
    """Return True if any M1 terminal polygon on ``stamp`` cannot host an in-place via1 pad."""
    polys = (getattr(stamp, "polys_by_layer", {}) or {}).get(m1_key, ()) or ()
    for ring in polys:
        xs = [round(float(pt[0]) * 1000) for pt in ring]
        ys = [round(float(pt[1]) * 1000) for pt in ring]
        if not xs or not ys:
            continue
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        if w <= 0 or h <= 0:
            continue
        # If a polygon is narrower than the single-cut pad width on its minor axis,
        # multi-cut pads cannot sit across fingers without bridging adjacent bars
        # unless the bar itself is wide enough on both axes.
        if min(w, h) < v1_m1_short_nm and (
            via_redundancy >= 2 or max(w, h) < v1_m1_long_nm
        ):
            return True
    return False


def native_row_escape_pair_clearances_nm(
    *,
    intent: Any,
    tech: Any,
    cell_def: Any,
    row_program: Any,
    solver_blocks: Sequence[Tuple[str, Tuple[int, int, int, int]]],
    materialized_blocks: Sequence[Tuple[str, Tuple[int, int, int, int]]],
    master_members: Mapping[str, Sequence[str]],
    member_offsets_nm: Mapping[str, Tuple[str, Tuple[int, int]]],
    block_stamps: Mapping[str, Sequence[str]],
    block_gring: Mapping[str, str],
    block_member_devices: Mapping[str, Set[str]],
    geometry: Mapping[str, Any],
    placement_um: Mapping[str, Tuple[float, float]],
    collision_excluded_layers: Sequence[Tuple[int, int]],
    refined_collision_owners: Set[str],
    base_gap_nm: int,
    adjacency_limits_nm: Sequence[Tuple[str, str, int]] = (),
    guest_masters: Set[str] = frozenset(),
) -> Dict[Tuple[str, str], int]:
    """Derive solver-master clearances guaranteeing legal M1 escape/landing channels
    around native diffusion rows without violating ``@abut`` or ``@adjacent`` caps.
    """
    if (
        tech is None
        or row_program is None
        or not getattr(row_program, "entries", ())
    ):
        return {}
    via_redundancy = resolve_cell_via_redundancy(cell_def)
    if via_redundancy < 2:
        return {}
    try:
        m1_key = tuple(tech.metal_key(1))
        v1_short_nm, v1_long_nm, m1_channel_floor_nm = pdk_m1_escape_channel_floor_nm(
            tech, via_redundancy=via_redundancy
        )
    except Exception:
        return {}

    from clay.layout.placed_geometry import placed_point_nm
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

    grid_nm = pdk_manufacturing_grid_nm(tech)
    excluded = {tuple(lk) for lk in (collision_excluded_layers or ())}
    row_stamp_names = {str(entry.group) for entry in row_program.entries}
    escape_row_stamps = {
        sname
        for sname in row_stamp_names
        if sname in geometry
        and _stamp_requires_external_m1_via_escape(
            geometry[sname],
            m1_key=m1_key,
            via_redundancy=via_redundancy,
            v1_m1_short_nm=v1_short_nm,
            v1_m1_long_nm=v1_long_nm,
        )
    }
    if not escape_row_stamps:
        return {}

    solver_map = dict(solver_blocks)
    mat_map = dict(materialized_blocks)

    # Compute each master's conservative M1 edge inset relative to its collision bbox,
    # and identify masters owning at least one escape-requiring native row stamp.
    escape_masters: Set[str] = set()
    min_m1_inset_by_master: Dict[str, int] = {}

    for master, master_box in sorted(solver_map.items()):
        members = tuple(master_members.get(master, (master,)))
        targets: Dict[str, Tuple[int, int, int, int]] = {}
        for member in members:
            if member not in mat_map:
                continue
            src = mat_map[member]
            if member in member_offsets_nm:
                _owner, off = member_offsets_nm[member]
                targets[member] = (
                    off[0],
                    off[1],
                    off[0] + src[2] - src[0],
                    off[1] + src[3] - src[1],
                )
            else:
                targets[member] = src
        if not targets:
            min_m1_inset_by_master[master] = 0
            continue
        union = (
            min(b[0] for b in targets.values()),
            min(b[1] for b in targets.values()),
            max(b[2] for b in targets.values()),
            max(b[3] for b in targets.values()),
        )
        m1_xs: list[int] = []
        m1_ys: list[int] = []
        col_xs: list[int] = []
        col_ys: list[int] = []
        for member, target in targets.items():
            src = mat_map[member]
            dx = target[0] - src[0] - union[0]
            dy = target[1] - src[1] - union[1]
            snames = tuple(block_stamps.get(member, ()))
            if block_gring.get(member) is not None:
                snames = snames + (block_gring[member],)
            for sname in snames:
                if sname in escape_row_stamps:
                    escape_masters.add(master)
                stamp = geometry.get(sname)
                pos = placement_um.get(sname)
                if stamp is None or pos is None:
                    continue
                for lk, rings in (getattr(stamp, "polys_by_layer", {}) or {}).items():
                    tlk = tuple(lk)
                    is_m1 = tlk == m1_key
                    is_col = tlk not in excluded
                    if not is_m1 and not is_col:
                        continue
                    for ring in rings:
                        for x_um, y_um in ring:
                            px, py = placed_point_nm(float(x_um), float(y_um), pos)
                            sx, sy = px + dx, py + dy
                            if is_m1:
                                m1_xs.append(sx)
                                m1_ys.append(sy)
                            if is_col:
                                col_xs.append(sx)
                                col_ys.append(sy)

        mw = master_box[2] - master_box[0]
        mh = master_box[3] - master_box[1]
        if master in refined_collision_owners and col_xs and col_ys:
            cx0, cy0, cx1, cy1 = min(col_xs), min(col_ys), max(col_xs), max(col_ys)
        else:
            cx0, cy0, cx1, cy1 = 0, 0, mw, mh
        if m1_xs and m1_ys:
            mx0, my0, mx1, my1 = min(m1_xs), min(m1_ys), max(m1_xs), max(m1_ys)
            inset = max(0, min(mx0 - cx0, cx1 - mx1, my0 - cy0, cy1 - my1))
        else:
            inset = 0
        min_m1_inset_by_master[master] = inset

    if not escape_masters:
        return {}

    # Build @abut master-pair exclusion set and @adjacent max_gap ceiling map.
    abut_dev_pairs = {
        frozenset((str(a), str(b)))
        for a, b in (getattr(intent, "abut_pairs", ()) or ())
    }
    abut_master_pairs: Set[Tuple[str, str]] = set()
    if abut_dev_pairs:
        dev_to_master: Dict[str, str] = {}
        for master in solver_map:
            for dev in block_member_devices.get(master, ()):
                dev_to_master[str(dev)] = master
        for dp in abut_dev_pairs:
            if len(dp) == 2:
                d1, d2 = tuple(dp)
                m1 = dev_to_master.get(d1)
                m2 = dev_to_master.get(d2)
                if m1 is not None and m2 is not None and m1 != m2:
                    abut_master_pairs.add(tuple(sorted((m1, m2))))

    adj_caps: Dict[Tuple[str, str], int] = {}
    for first, second, maximum in adjacency_limits_nm:
        pair = tuple(sorted((str(first), str(second))))
        adj_caps[pair] = (
            min(adj_caps[pair], int(maximum))
            if pair in adj_caps
            else int(maximum)
        )

    master_ids = sorted(solver_map)
    clearances: Dict[Tuple[str, str], int] = {}
    for i, a in enumerate(master_ids):
        if a in guest_masters:
            continue
        for b in master_ids[i + 1 :]:
            if b in guest_masters:
                continue
            if a not in escape_masters and b not in escape_masters:
                continue
            pair = (a, b)
            if pair in abut_master_pairs:
                continue
            raw_req = m1_channel_floor_nm - min_m1_inset_by_master.get(a, 0) - min_m1_inset_by_master.get(b, 0)
            req = _ceil_to_grid(max(int(base_gap_nm), raw_req), grid_nm)
            if pair in adj_caps:
                req = min(req, adj_caps[pair])
            if req > int(base_gap_nm):
                clearances[pair] = req
    return clearances
