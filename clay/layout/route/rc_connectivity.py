"""In-memory union-find cross-net short and net-connectivity advisory checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from clay.layout.placed_geometry import placed_point_nm, placed_ring_nm
from clay.layout.route.engine import NetRoute
from clay.layout.route.rc_devices import _NM_PER_UM
from clay.layout.tech import LayoutTech


@dataclass
class ShortsCheck:
    """A cross-net short verdict (same shape the generic-router union-find returns).

    ``merged_pairs`` is the sorted list of ``(net_a, net_b)`` that share a single
    physical conductor (a real cross-net short). ``distinct`` is True iff none do."""

    distinct: bool
    merged_pairs: List[Tuple[str, str]]


def _via_keys(tech: LayoutTech) -> set:
    """The PDK's full cut-layer set (via1..via6 + contact): the only legal cross-layer
    metal<->metal join (extractor physics), PDK-agnostic."""
    keys: set = set()
    for n in range(1, 7):
        try:
            keys.add(tuple(tech.via_key(n)))
        except Exception:
            break
    try:
        keys.add(tuple(tech.contact_key()))
    except Exception:
        pass
    return keys


def _metal_keys(tech: LayoutTech) -> set:
    keys: set = set()
    for n in range(1, 8):
        try:
            keys.add(tuple(tech.metal_key(n)))
        except Exception:
            break
    return keys


def _seeded_geometry_metal_regions(
    geometry: Optional[Dict[str, Any]],
    placement: Optional[Dict[str, _XY]],
    pins_by_net: Optional[Dict[str, List[Any]]],
    metal_keys: set,
    *,
    cap_names: Optional[set] = None,
) -> Dict[str, Dict[Tuple[int, int], Any]]:
    """Classify exact same-layer geometry components from their pin seeds.

    Device and ring stamps may spell one physical conductor as several touching
    polygons.  Ownership therefore belongs to the merged physical component, not
    independently to each source polygon: zero pin-net seeds leave a component
    unowned, one unique seed net owns all of it, and multiple unique seed nets own
    the same component concurrently so the short oracle reports the collision.
    """

    import klayout.db as kdb

    if geometry is None or placement is None or pins_by_net is None:
        return {}
    seeds_by_layer: Dict[Tuple[int, int], List[Tuple[int, int, str]]] = {}
    wild_seeds: List[Tuple[int, int, str]] = []
    # ``cap_names`` is retained for API compatibility.  A device/layer pair is
    # not itself an electrical conductor: modern MIM cells may bring both plate
    # terminals onto the same routing layer through two disjoint escape islands.
    # Ownership must therefore remain component-local for capacitors too.
    _ = cap_names
    for net, pins in sorted((pins_by_net or {}).items(), key=lambda item: str(item[0])):
        net_str = str(net)
        for pin in pins or ():
            layer = getattr(pin, "layer", None)
            px_nm = round(float(pin.xy_um[0]) * _NM_PER_UM)
            py_nm = round(float(pin.xy_um[1]) * _NM_PER_UM)
            if layer is not None:
                layer_key = (int(layer[0]), int(layer[1]))
                if layer_key in metal_keys:
                    seeds_by_layer.setdefault(layer_key, []).append(
                        (px_nm, py_nm, net_str)
                    )
            else:
                wild_seeds.append((px_nm, py_nm, net_str))

    if not seeds_by_layer and not wild_seeds:
        return {}

    _Point = kdb.Point
    _Polygon = kdb.Polygon
    _Region = kdb.Region
    _Box = kdb.Box

    classified: Dict[str, Dict[Tuple[int, int], "kdb.Region"]] = {}
    for name, geom in sorted((geometry or {}).items(), key=lambda item: str(item[0])):
        pos = (placement or {}).get(name)
        if pos is None:
            continue
        layer_rows = getattr(geom, "polys_by_layer", {}) or {}
        for key, polygons in sorted(
            layer_rows.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))
        ):
            layer_key = (int(key[0]), int(key[1]))
            if layer_key not in metal_keys:
                continue
            l_seeds = seeds_by_layer.get(layer_key)
            if not l_seeds and not wild_seeds:
                continue
            physical = _Region()
            for polygon in polygons:
                if len(polygon) < 3:
                    continue
                pts = placed_ring_nm(polygon, pos)
                physical.insert(_Polygon([_Point(x, y) for x, y in pts]))
            physical.merge()
            if physical.is_empty():
                continue
            pbb = physical.bbox()
            pl = pbb.left - 1
            pb = pbb.bottom - 1
            pr = pbb.right + 1
            pt = pbb.top + 1
            stamp_seeds: List[Tuple[int, int, str]] = []
            if l_seeds:
                for px_nm, py_nm, net_str in l_seeds:
                    if pl <= px_nm <= pr and pb <= py_nm <= pt:
                        stamp_seeds.append((px_nm, py_nm, net_str))
            if wild_seeds:
                for px_nm, py_nm, net_str in wild_seeds:
                    if pl <= px_nm <= pr and pb <= py_nm <= pt:
                        stamp_seeds.append((px_nm, py_nm, net_str))
            if not stamp_seeds:
                continue
            for component in physical.each():
                bb = component.bbox()
                cl = bb.left
                cb = bb.bottom
                cr = bb.right
                ct = bb.top
                is_box = component.is_box()
                component_region = None
                owners: set[str] = set()
                for px_nm, py_nm, net_str in stamp_seeds:
                    if net_str in owners:
                        continue
                    if cl < px_nm + 1 and cr > px_nm - 1 and cb < py_nm + 1 and ct > py_nm - 1:
                        if is_box:
                            owners.add(net_str)
                        else:
                            if component_region is None:
                                component_region = _Region(component)
                            probe = _Region(_Box(
                                px_nm - 1, py_nm - 1, px_nm + 1, py_nm + 1
                            ))
                            if not (component_region & probe).is_empty():
                                owners.add(net_str)
                for owner in sorted(owners):
                    classified.setdefault(owner, {}).setdefault(
                        layer_key, _Region()
                    ).insert(component)
    for layer_regions in classified.values():
        for region in layer_regions.values():
            region.merge()
    return classified


def check_cross_net_shorts(
    routed: Any,
    obstacles: Sequence[Obstacle],
    tech: LayoutTech,
    *,
    geometry: Optional[Dict[str, Any]] = None,
    placement: Optional[Dict[str, _XY]] = None,
    pins_by_net: Optional[Dict[str, List[Any]]] = None,
    cap_names: Optional[set] = None,
) -> ShortsCheck:
    """NATIVE (ADVISORY) cross-net short oracle: extractor-physics union-find over the
    REAL routed metal + net-owned device pads.

    NOT AUTHORITATIVE -- it is SAME-LAYER-METAL ONLY and DECK-UNAWARE. It catches a
    cross-net short only where two nets' metal OVERLAP on the SAME routing layer; it
    does NOT model the foundry deck's device-cell connectivity, so it MISSES every
    short that closes THROUGH a device-cell via stack. The proven counter-example is
    the LDO ``CC1`` MIM cap: the gf180 deck's ``connect(metal_ncap, via2)`` +
    ``connect(via2, metal3)`` ties the cap's Metal2 bottom-plate net (``NET034``) to a
    foreign Metal3 over it (``GNDA``) through the cap's via2 sea -- a genuine cross-net
    short that this oracle reports CLEAN (``merged_pairs == []``) AND the geometric
    ``net_complete`` reports connected. ONLY the independent KLayout LVS
    (:func:`clay.layout.route.lvs_gate.lvs_gate_route_cell`) catches it. So: gate
    SIGNOFF on the KLayout LVS, never on ``merged_pairs`` / ``net_complete`` alone.
    See memory ``project_route_cell_ota_fold_closed`` + ``feedback_separation_check_is_not_a_short_oracle``.

    This is the geometrically faithful sibling of
    :func:`clay.layout.generic_router.check_shorts` -- but it unions on real
    POLYGON-area overlap (KLayout regions), NOT bounding-box edge-touch, because the
    engine emits hundreds of per-cell unit rectangles whose bounding boxes abut
    across nets even with NO real metal overlap (a bbox check_shorts over raw
    fragments fabricates shorts). It uses the SAME short model the engine's own
    occupancy check uses (``engine_pathfinder``: occupancy>=2 on the SAME drawn-metal
    cell) and the foundry LVS extractor: a cross-net short is two DIFFERENT nets'
    metal overlapping on the SAME metal layer. A via cut belongs to a SINGLE net and
    can only join that net's own metals across layers, so a ROUTING via never shorts
    two different nets -- via geometry is therefore excluded from the cross-net test (a
    2D bbox overlap of a net's via with a foreign net's metal is not an electrical
    join; including it fabricates shorts, as the tail-via<->vdd-metal probe showed).
    CAVEAT (the blind spot above): this holds for SIMPLE routing vias, NOT for a
    DEVICE-CELL via stack whose deck connectivity ties two DIFFERENT plate nets (the
    MIM cap ``metal_ncap/via2/metal3`` chain) -- excluding vias makes this oracle
    structurally unable to see that class of short. The KLayout LVS, not this, is the
    via/deck-aware authority.

    DEVICE M1 -- REAL POLYGON, not bbox. When ``geometry`` + ``placement`` +
    ``pins_by_net`` are given, the device-pad contribution is the device cell's ACTUAL
    M1 ring polygons (each tagged with the net of the pin landing inside it), NOT the
    rectangular obstacle bbox. This matters for a FOLDED device: its source and drain
    M1 are interdigitated COMBS whose bounding boxes overlap fully but whose real
    polygons interleave with ZERO intersection -- collapsing each comb to its bbox
    (the ``obstacles`` fallback) fabricates a whole-device source<->drain short
    (the verified MM28 VDDA<->VOUT 6361um2 phantom). The emitted GDS draws the real
    combs, so the real-deck LVS sees no such short; this oracle must agree. Without
    ``geometry`` the function falls back to the net-owned obstacle bboxes (back-compat
    for callers that have only the router obstacles).

    NOT separation_check: a deck merges a same-layer cross-net overlap and so cannot
    SEE an overlap short; this union-find can."""
    import klayout.db as kdb

    metal_keys = _metal_keys(tech)
    via_keys = _via_keys(tech)

    # per-(net, layer) merged METAL regions only (a via is single-net -> never a
    # cross-net join), in nm.
    metal: Dict[str, Dict[Tuple[int, int], "kdb.Region"]] = {}

    def _add_rect(net: str, key: Tuple[int, int], x0, y0, x1, y1) -> None:
        if key in via_keys or key not in metal_keys:
            return
        poly = kdb.Polygon(kdb.Box(int(x0), int(y0), int(x1), int(y1)))
        metal.setdefault(net, {}).setdefault(key, kdb.Region()).insert(poly)

    def _add_poly(net: str, key: Tuple[int, int], pts_nm: List[Tuple[int, int]]) -> None:
        if key in via_keys or key not in metal_keys or len(pts_nm) < 3:
            return
        poly = kdb.Polygon([kdb.Point(int(x), int(y)) for x, y in pts_nm])
        metal.setdefault(net, {}).setdefault(key, kdb.Region()).insert(poly)

    # routed wires (nm already).
    for net, nr in (getattr(routed, "nets", {}) or {}).items():
        for lp in getattr(nr, "wires", []) or []:
            xs = [int(v.x) for v in lp.polygon.vertices]
            ys = [int(v.y) for v in lp.polygon.vertices]
            if xs:
                _add_rect(str(net), (int(lp.layer), int(lp.datatype)),
                          min(xs), min(ys), max(xs), max(ys))

    use_real_geom = (
        geometry is not None and placement is not None and pins_by_net is not None
    )
    if use_real_geom:
        owned_regions = _seeded_geometry_metal_regions(
            geometry,
            placement,
            pins_by_net,
            metal_keys,
            cap_names=cap_names,
        )
        for owner, layer_regions in owned_regions.items():
            for key, region in layer_regions.items():
                target = metal.setdefault(owner, {}).setdefault(key, kdb.Region())
                for polygon in region.each():
                    target.insert(polygon)
    else:
        # FALLBACK: net-owned device pad bboxes (obstacles are um -> nm).
        for ob in obstacles or []:
            owner = getattr(ob, "owner_net", None)
            layer = getattr(ob, "layer", None)
            if owner is None or layer is None:
                continue
            _add_rect(str(owner), (int(layer[0]), int(layer[1])),
                      round(float(ob.x0) * _NM_PER_UM), round(float(ob.y0) * _NM_PER_UM),
                      round(float(ob.x1) * _NM_PER_UM), round(float(ob.y1) * _NM_PER_UM))

    nets = sorted(metal)
    metal_bbs: Dict[str, Dict[Tuple[int, int], Any]] = {}
    for n in nets:
        bbs_n: Dict[Tuple[int, int], Any] = {}
        for key, reg in metal[n].items():
            reg.merge()
            bbs_n[key] = reg.bbox()
        metal_bbs[n] = bbs_n

    merged: set = set()
    for i in range(len(nets)):
        a = nets[i]
        ma = metal[a]
        ba = metal_bbs[a]
        for j in range(i + 1, len(nets)):
            b = nets[j]
            mb = metal[b]
            bb = metal_bbs[b]
            # same-layer metal overlap = one physical conductor on that layer = short.
            for key, ra in ma.items():
                rb = mb.get(key)
                if rb is not None and ba[key].overlaps(bb[key]) and (ra & rb).count() > 0:
                    merged.add((a, b))
                    break

    return ShortsCheck(distinct=not merged, merged_pairs=sorted(merged))


def check_net_connectivity(
    routed: Any,
    tech: LayoutTech,
    *,
    geometry: Optional[Dict[str, Any]] = None,
    placement: Optional[Dict[str, _XY]] = None,
    pins_by_net: Optional[Dict[str, List[Any]]] = None,
) -> Dict[str, bool]:
    """GEOMETRIC per-net connectivity oracle: ``{net -> connected}`` where a net is
    connected iff ALL its pins land in ONE via-joined component of its OWN EMITTED metal
    (routed wires + the device M1 pad each pin sits on, joined across layers only by a
    real via cut).

    This is the DUAL of :func:`check_cross_net_shorts` and the GEOMETRIC truth the
    engine's per-net ``connected`` flag misses: the flag is set from the GRID route
    (``engine_pathfinder`` sets ``nr.connected = True`` after routing the cells), BEFORE
    the emit-time VIA-ENVELOPE OBLIGATION (``engine_maze._emit_wires``) may SUPPRESS a
    via whose landing pad cannot hold its PDK enclosure clear of a foreign net -- the
    source/bulk via-landing deadlock that opens a net the grid thought connected (the
    LDO MM10 source). Reading the EMITTED geometry catches it; ``nr.connected`` cannot.
    Same metal model as the short oracle (real polygon overlap, not bbox), so the native
    completeness verdict agrees with the KLayout LVS oracle instead of over-reporting."""
    import klayout.db as kdb

    metal_keys = _metal_keys(tech)
    via_keys = _via_keys(tech)
    # via cut key -> (lower metal key, upper metal key) it electrically joins.
    via_join: Dict[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]] = {}
    for n in range(1, 7):
        try:
            vk = (int(tech.via_key(n)[0]), int(tech.via_key(n)[1]))
            via_join[vk] = (
                (int(tech.metal_key(n)[0]), int(tech.metal_key(n)[1])),
                (int(tech.metal_key(n + 1)[0]), int(tech.metal_key(n + 1)[1])),
            )
        except Exception:
            break

    out: Dict[str, bool] = {}
    pins_by_net = pins_by_net or {}
    owned_geometry = _seeded_geometry_metal_regions(
        geometry, placement, pins_by_net, metal_keys
    )
    owned_upper_geometry: Dict[
        str, Dict[Tuple[int, int], "kdb.Region"]
    ] = {}
    if geometry is not None and placement is not None:
        from clay.layout.model import device_upper_conductors

        for conductor in device_upper_conductors(
            None,
            placement,
            geometry,
            tech=tech,
            pins_by_net=pins_by_net,
        ):
            if conductor.net is None:
                continue
            key = (int(conductor.layer[0]), int(conductor.layer[1]))
            if key not in metal_keys and key not in via_keys:
                continue
            owned_upper_geometry.setdefault(str(conductor.net), {}).setdefault(
                key, kdb.Region()
            ).insert(kdb.Box(
                round(float(conductor.x0) * _NM_PER_UM),
                round(float(conductor.y0) * _NM_PER_UM),
                round(float(conductor.x1) * _NM_PER_UM),
                round(float(conductor.y1) * _NM_PER_UM),
            ))
    for net, nr in (getattr(routed, "nets", {}) or {}).items():
        net = str(net)
        by_metal: Dict[Tuple[int, int], "kdb.Region"] = {}
        by_via: Dict[Tuple[int, int], "kdb.Region"] = {}
        # routed wires (nm) -- metal layers and via cuts.
        for lp in getattr(nr, "wires", []) or []:
            key = (int(lp.layer), int(lp.datatype))
            poly = kdb.Polygon([kdb.Point(int(v.x), int(v.y))
                                for v in lp.polygon.vertices])
            if key in metal_keys:
                by_metal.setdefault(key, kdb.Region()).insert(poly)
            elif key in via_keys:
                by_via.setdefault(key, kdb.Region()).insert(poly)
        # the device M1 pad each of THIS net's pins sits on (so a pin lands on its net's
        # metal even where the routed wire only abuts the device pad), net-attributed by
        # pin containment -- the same ownership rule the short oracle uses.
        pins = pins_by_net.get(net, []) or []
        for key, region in owned_geometry.get(net, {}).items():
            target = by_metal.setdefault(key, kdb.Region())
            for polygon in region.each():
                target.insert(polygon)
        for key, region in owned_upper_geometry.get(net, {}).items():
            target = (
                by_metal.setdefault(key, kdb.Region())
                if key in metal_keys
                else by_via.setdefault(key, kdb.Region())
            )
            for polygon in region.each():
                target.insert(polygon)
        for r in by_metal.values():
            r.merge()

        # islands: one node per connected metal polygon per layer; vias union cross-layer.
        islands_poly: List[Any] = []
        islands_reg: List[Any] = []
        islands_by_layer: Dict[Tuple[int, int], List[Tuple[int, Any, bool]]] = {}
        for key, reg in by_metal.items():
            layer_list = islands_by_layer.setdefault(key, [])
            for poly in reg.each():
                idx = len(islands_poly)
                islands_poly.append(poly)
                islands_reg.append(None)
                layer_list.append((idx, poly.bbox(), poly.is_box()))
        parent = list(range(len(islands_poly)))

        def _island_reg(idx: int) -> "kdb.Region":
            r = islands_reg[idx]
            if r is None:
                r = kdb.Region(islands_poly[idx])
                islands_reg[idx] = r
            return r

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for vkey, vreg in by_via.items():
            join = via_join.get(vkey)
            if join is None:
                continue
            lo_k, hi_k = join
            lo_list = islands_by_layer.get(lo_k, ())
            hi_list = islands_by_layer.get(hi_k, ())
            if not lo_list or not hi_list:
                continue
            for vp in vreg.each():
                vbb = vp.bbox()
                v_is_box = vp.is_box()
                vr = None
                los = []
                for idx, bb, is_box in lo_list:
                    if bb.overlaps(vbb):
                        if is_box and v_is_box:
                            los.append(idx)
                        else:
                            if vr is None:
                                vr = kdb.Region(vp)
                            if not (_island_reg(idx) & vr).is_empty():
                                los.append(idx)
                if not los:
                    continue
                his = []
                for idx, bb, is_box in hi_list:
                    if bb.overlaps(vbb):
                        if is_box and v_is_box:
                            his.append(idx)
                        else:
                            if vr is None:
                                vr = kdb.Region(vp)
                            if not (_island_reg(idx) & vr).is_empty():
                                his.append(idx)
                for i in los:
                    for j in his:
                        ri, rj = _find(i), _find(j)
                        if ri != rj:
                            parent[rj] = ri

        # attribute each pin to the island (on its layer) containing it -> component.
        comps: List[int] = []
        ok = True
        for p in pins:
            pl = getattr(p, "layer", None)
            pk = (int(pl[0]), int(pl[1])) if pl is not None else None
            px = round(float(p.xy_um[0]) * _NM_PER_UM)
            py = round(float(p.xy_um[1]) * _NM_PER_UM)
            pbox = kdb.Box(px - 1, py - 1, px + 1, py + 1)
            probe = None
            comp = None
            if pk is not None:
                cand_lists = (islands_by_layer.get(pk, ()),)
            else:
                cand_lists = tuple(islands_by_layer.values())
            for cand_list in cand_lists:
                for idx, bb, is_box in cand_list:
                    if bb.overlaps(pbox):
                        if is_box:
                            comp = _find(idx)
                            break
                        if probe is None:
                            probe = kdb.Region(pbox)
                        if not (_island_reg(idx) & probe).is_empty():
                            comp = _find(idx)
                            break
                if comp is not None:
                    break
            if comp is None:
                ok = False  # pin lands on NO emitted metal of its net -> open
                break
            comps.append(comp)
        if ok and comps:
            ok = all(c == comps[0] for c in comps)
        out[net] = ok
    return out
