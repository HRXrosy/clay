"""COMMITTED, leakage-free external_pdk OTA routing anchor.

This is the committed replacement for the throwaway uncommitted harness that drove the
OLD anchor (:func:`clay.layout.extract.route_migrate.run_ota_route_tsmc`). The OLD anchor
fed REFERENCE per-net widths (``extract_reference_net_widths``) into the router input --
the routing analogue of memorising reference device coordinates (LEAKAGE). This anchor
derives every routing-intent value from the netlist + the PDK (DESIGN-sourced), asserts
the resulting :class:`~clay.layout.route.intent.RoutingIntent` is leakage-free, routes the
self-consistent external_pdk placement, and judges the result against the reference for COMPARISON
ONLY (the reference is read on the OUTPUT side, never fed into the router).

THE LEAKAGE BOUNDARY (kept crisp)
=================================
  * INPUT  to the router: pins (from the placement + device-cell geometry, leakage-free),
    obstacles (device M1 artwork, leakage-free), and a RoutingIntent whose widths come
    from :func:`clay.layout.route.width_model.derive_width_intent` -- DESIGN-sourced.
    ``routing_intent.is_leakage_free()`` is HARD-ASSERTED True before lowering.
  * OUTPUT side: :func:`clay.layout.route.verify.verify_consistency` reads the reference
    GDS (per-net widths/layers) to SCORE how close the router got. We ALSO call
    ``extract_reference_net_widths`` for COMPARISON ONLY -- to report a reference
    width_class match count. Neither ever touches the router input.

VERIFICATION DISCIPLINE
=======================
Connectivity, shorts, and cross-net spacing are measured on the MERGED per-net geometry
(the verifier's ``merged-netaware`` DRC path), NOT logical ``connected`` flags -- a
``connected=True`` flag once hid a geometric open (min-width pads not touching). If the
route is not clean the returned metrics report the HONEST partial state with real measured
numbers; nothing is spun as success.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

from clay.layout.route.as_built import AsBuiltBundle
from clay.layout.route.engine import Obstacle
from clay.layout.route.intent import RoutingIntent
from clay.layout.route.lower import lower_intent
from clay.layout.route.pins import extract_pins
from clay.layout.route.route_cell import route_prepared_cell
from clay.layout.route.verify import (
    verify_consistency,
    _check_connectivity,
    _check_drc,
    _width_class,
)
from clay.layout.route.width_model import (
    WidthPolicy,
    classify_net_role,
    derive_width_intent,
)
from clay.layout.tech import LayoutTech

_NM_PER_UM = 1000.0
_XY = Tuple[float, float]

# GDS layer of device active diffusion (OD) -- the production layermap. The
# planar-upper-metals route penalises upper metal over THIS region.
_GDS_OD = 3
# The lowest metal LEVEL that is upper (planar-channel) routing: M3+. M1/M2 are
# local interconnect and pay no over-active penalty. Config-shaped scalar.
_UPPER_METAL_MIN_LEVEL = 3
# The soft per-cell over-active cost for an upper metal over device active. Sized
# so the planar-channel detour (a few cells around the cluster) is cheaper than a
# long over-cell run, while staying SOFT (an over-active cell remains routable
# when no planar channel exists -- connectivity is never traded). Derived from the
# obligation (a planarity preference), not a per-net constant.
_OVER_ACTIVE_PENALTY = 4.0


def _device_od_rects_um(
    intent: Any,
    placement: Dict[str, _XY],
    geometry: Dict[str, Any],
) -> List[Tuple[float, float, float, float]]:
    """Per-device active-diffusion (OD, GDS layer 3) bounding rects in the placed
    frame the emitter stamps -- Clay's OWN geometry, never a reference coordinate.

    Excludes MIM caps (a cap footprint is not active diffusion). One rect per
    device's merged OD extent (the over-active soft-cost region the router avoids
    on upper metals)."""
    cap_names = {str(p.name) for p in (getattr(intent, "passives", []) or [])
                 if str(getattr(p, "kind", "")).lower().startswith("c")}
    rects: List[Tuple[float, float, float, float]] = []
    for name, g in (geometry or {}).items():
        if name in cap_names:
            continue
        pos = placement.get(name)
        if pos is None:
            continue
        x0 = y0 = 1e18
        x1 = y1 = -1e18
        for lay, rings in (getattr(g, "polys_by_layer", {}) or {}).items():
            if int(lay[0]) != _GDS_OD:
                continue
            for ring in rings:
                for (x, y) in ring:
                    ax, ay = pos[0] + x, pos[1] + y
                    x0 = min(x0, ax); x1 = max(x1, ax)
                    y0 = min(y0, ay); y1 = max(y1, ay)
        if x1 > x0 and y1 > y0:
            rects.append((x0, y0, x1, y1))
    return rects


def _reference_gds_path() -> str:
    """Absolute path to the read-only external_pdk OTA reference GDS (ground truth)."""
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(
        repo, "refCircuit", "twoStageAmp", "0208_amp", "project_tsmc18rf",
        "prj_gzc", "CMD", "GDS2", "layout.gds",
    )


# The PDK the reference layout above was drawn in -- resolves the reference's
# metal (layer, datatype) keys when MEASURING it (output-side / style reads).
# This describes the reference artifact, not the routing target.
_REF_TECH = "gf180mcu"


def _device_m1_obstacles(
    intent: Any,
    placement: Dict[str, _XY],
    geometry: Dict[str, Any],
    *,
    tech: LayoutTech,
    pins_by_net: Dict[str, List[Any]],
) -> List[Obstacle]:
    """Build the router's keep-out obstacles from the placed device M1 artwork.

    For every device's M1 polygon (centroid-relative um in its ``CellGeometry`` stamp),
    translate it to the device's placement centroid and emit it as an :class:`Obstacle`.
    Each obstacle is tagged ``owner_net`` = the net of the pin (terminal access point)
    that lands inside it, so a net may connect on its OWN terminal pad while every other
    net treats it as a keep-out (the engine's net-aware obstacle model). An M1 pad
    covering no signal pin (a body/well-tie pad on a power net) gets ``owner_net=None``
    -> a hard keep-out for all signal nets.

    Leakage-free: the artwork is the DEVICE CELL's M1 (vendor PCell capture or
    build_device), never a routed reference wire; the placement is the leakage-free
    non-overlap solve. This reads no reference ROUTING.

    COMPATIBILITY ADAPTER (dev-discipline rule 6): the net-tagged device conductors now
    live in the unified :class:`clay.layout.model.LayoutModel`; this function is a thin
    projection of that model to the engine's :class:`Obstacle` list. Long-term the router
    queries the model directly and this adapter is deleted.
    """
    from clay.layout.model import device_m1_conductors

    return [c.as_obstacle() for c in device_m1_conductors(
        intent, placement, geometry, tech=tech, pins_by_net=pins_by_net)]


def _edge_anchor_obligation_report(
    intent: Any,
    placement: Dict[str, _XY],
    geometry: Dict[str, Any],
    tech: LayoutTech,
    routed: Any = None,
) -> Optional[Dict[str, Any]]:
    """The @edge_anchor V-side verdict on the canonical AS-BUILT bundle, or None
    when the intent declares no edge-anchor device (the byte-stable default).

    When ``routed`` (the router's own result) is supplied the obligation reads
    the AS-BUILT geometry -- device diffusion + signal metal 1..4 + the routed
    wires -- in the absolute frame, the SAME layer algebra the independent oracle
    (tests/test_cap_not_surrounded.py) uses, so native and oracle agree. (The
    prior report passed device stamps only; it measured the placement-stamp frame
    and reported a routed-surrounded cap as PASS -- the native<->oracle bug this
    closes.) The cap footprint comes from the canonical
    ``under_cap_intent.cap_footprints_um`` inside the checker. The area budget is
    NOT supplied here (the caller compares the emitted GDS extent to the signed
    canonical) -- this report records the open-side state and the measured outline
    so the upper layer adjudicates the die-growth half."""
    if not (getattr(intent, "edge_anchor_devices", None) or []):
        return None
    from clay.layout.extract.edge_anchor import edge_anchor_obligation

    sizes: Dict[str, _XY] = {}
    for n, g in (geometry or {}).items():
        xs = [pt[0] for rings in getattr(g, "polys_by_layer", {}).values()
              for ring in rings for pt in ring]
        ys = [pt[1] for rings in getattr(g, "polys_by_layer", {}).values()
              for ring in rings for pt in ring]
        if xs:
            sizes[n] = (max(xs) - min(xs), max(ys) - min(ys))
    try:
        return edge_anchor_obligation(intent, placement, sizes,
                                      geometry=geometry, tech=tech,
                                      routed=routed)
    except Exception as exc:  # honest degrade -- never crash the route
        return {"verdict": "unmeasurable", "caps": {}, "n_caps": 0,
                "reason": f"obligation measurement raised: {exc!r}"}


# Frozen compatibility aliases.  Canonical route/emit code owns these helpers in
# ``emit_gds``; the retired router keeps its historical names without remaining a
# dependency of the canonical spine.
from clay.layout.route.emit_gds import (  # noqa: E402
    emit_net_labels,
    emit_routed_gds as _generic_emit_routed_gds,
    net_label_points,
)


def _emit_routed_gds(
    routed: Any,
    out_gds: str,
    *,
    cell_name: str = "ota_route",
    dbu: float = 0.001,
    merge: bool = True,
) -> str:
    """Legacy spelling for the generic emission helper."""
    return _generic_emit_routed_gds(
        routed,
        out_gds,
        cell_name=cell_name,
        dbu=dbu,
        merge=merge,
    )


def _routed_trunk_width_um(net_route: Any, layers: Any) -> float:
    """Widest routed-wire SHORT edge (um) of a net over the given metal layer(s)
    -- its routed trunk width. ``layers`` is a single (layer, datatype) key OR a
    SET of such keys: the power trunk may ride an upper metal (trunk_layer_class,
    e.g. M2), so measuring M1 alone misses it (the net10 power-trunk-on-M2 vs
    M1-comb-only-on-GND/VDD mismatch). Scanning all the net's metal layers takes
    the trunk wherever the router drew it."""
    keys = {tuple(layers)} if (isinstance(layers, tuple)
                               and len(layers) == 2
                               and all(isinstance(v, int) for v in layers)) \
        else {tuple(k) for k in layers}
    best = 0.0
    for lp in getattr(net_route, "wires", []) or []:
        if (int(lp.layer), int(lp.datatype)) not in keys:
            continue
        xs = [v.x for v in lp.polygon.vertices]
        ys = [v.y for v in lp.polygon.vertices]
        w = (max(xs) - min(xs)) / _NM_PER_UM
        h = (max(ys) - min(ys)) / _NM_PER_UM
        best = max(best, min(w, h))
    return best


def _via_layer_set(tech: LayoutTech) -> set:
    """The PDK's via/contact cut (layer, datatype) keys -- excluded from metal-width DRC."""
    s = set()
    for n in range(1, 7):
        try:
            s.add(tuple(tech.via_key(n)))
        except Exception:
            break
    try:
        s.add(tuple(tech.contact_key()))
    except Exception:
        pass
    return s




def _verify_cap_plates(
    intent: Any,
    placement: Dict[str, _XY],
    routed: Any,
    *,
    tech: LayoutTech,
) -> List[Dict[str, Any]]:
    """The ELECTRICAL bar for MIM caps: each cap's two plates connect to the RIGHT nets
    and are NOT merged into one node by the route.

    For every passive cap, this rebuilds the cap's device cell (the SAME parametric
    ``passive_device_cell.build_capacitor`` the pin extractor uses -- leakage-free),
    places it at its solved centroid, ADDS the routed wire artwork, and runs the
    layer-aware connectivity primitive (:func:`clay.layout.connectivity.extract_nodes`)
    with the PDK's MIM device-recognition model (:func:`device_terminal._mim_spec_for`).
    It then locates the electrical NODE of each plate's contact tab and checks the two
    tabs land on DISTINCT nodes (the cap is not shorted). The distinct-plate model keeps
    the two plates two nodes; if the route grazed both plates onto one node this would
    show ``distinct=False``. Returns one row per cap with the measured node indices.

    This reads ONLY the cap device-cell geometry + the routed output (the router's own
    result) + the PDK MIM model DATA -- never a reference routed layout.
    """
    import klayout.db as kdb

    from clay.layout import passive_device_cell as _pdc
    from clay.layout.connectivity import ConnectModel, extract_nodes, node_index_at
    from clay.layout.route.pins import _geom_extent_center_nm, _mim_spec_for

    rows: List[Dict[str, Any]] = []
    spec = _mim_spec_for(tech)
    if spec is None:
        return rows

    from clay.layout.device_terminal import extract_terminal_access
    from clay.layout.route.pins import _mim_spec_for_passive

    m1k = tuple(tech.metal_key(1))
    m2k = tuple(tech.metal_key(2))
    try:
        via1 = tuple(tech.via_key(1))
    except Exception:
        via1 = None

    def _model_for(pspec):
        # the carved routable-Metal2 synthetic key (gf180) where cap tabs and routed
        # Metal2 wires both live -- located generically from the spec's metals; PDKs
        # whose plates live on upper metals (external_pdk) simply have no layer-18 member and
        # keep the plain m2k join. The MIM derived split + the routing M1 plane joined
        # to carved-Metal2 by Via1. We deliberately do NOT add a raw
        # (Via2, Metal2, Metal3) routing rule -- that would bypass the plate carve and
        # falsely short the two plates.
        ncap_l = next((k for k in pspec.metals if k[0] == m2k[0] and k != m2k), m2k)
        extra = set(pspec.vias)
        if via1 is not None:
            extra.add((via1, m1k, ncap_l))
        return ConnectModel(
            metals=tuple(set(pspec.metals) | {m1k}),
            vias=tuple(extra),
            derived=pspec.derived,
            bridges=pspec.bridges,
        )

    for p in getattr(intent, "passives", []) or []:
        kind = str(getattr(p, "kind", "")).lower()
        if not (kind.startswith("cap") or kind == "c"):
            continue
        name = str(getattr(p, "name", ""))
        pos = placement.get(name)
        val = getattr(p, "value", None)
        nets = getattr(p, "nets", {}) or {}
        if pos is None or val is None:
            continue
        try:
            cell = _pdc.build_capacitor_for_tech(
                tech, float(val),
                nets={"p": str(nets.get("p", "")), "n": str(nets.get("n", ""))},
                m=int(getattr(p, "M", 1) or 1),
                model=str(getattr(p, "model", "") or ""),
            )
        except Exception:
            continue
        cx, cy = _geom_extent_center_nm(cell)
        px = int(round(pos[0] * _NM_PER_UM))
        py = int(round(pos[1] * _NM_PER_UM))

        # the spec + connect model at THIS passive's level (the $.MODEL annotation;
        # a level-5 bank checked against the level-4 default reported its plates
        # unmapped -- the spec must follow the cell the same passive data built).
        pspec = _mim_spec_for_passive(tech, p)
        if pspec is None:
            continue
        model = _model_for(pspec)

        shapes: Dict[Tuple[int, int], List[Any]] = {}
        for lp in cell.polygons:
            shapes.setdefault((lp.layer, lp.datatype), []).append(
                kdb.Polygon([kdb.Point(int(v.x - cx + px), int(v.y - cy + py))
                             for v in lp.polygon.vertices])
            )
        # add ALL routed wires (the full route -- so any grazing short would show), and
        # keep a per-net region so we can confirm each plate connects to the RIGHT net.
        net_regions: Dict[str, Any] = {}
        for net, nr in (getattr(routed, "nets", {}) or {}).items():
            rr = net_regions.setdefault(net, kdb.Region())
            for lp in getattr(nr, "wires", []) or []:
                poly = kdb.Polygon([kdb.Point(int(v.x), int(v.y))
                                    for v in lp.polygon.vertices])
                shapes.setdefault((int(lp.layer), int(lp.datatype)), []).append(poly)
                rr.insert(poly)
        for rr in net_regions.values():
            rr.merge()
        nodes = extract_nodes(shapes, model)

        # each terminal's node, looked up at the terminal's OWN routing access
        # (layer + point) from the canonical access extractor -- never a single
        # assumed tab layer (the gf180-era carved-Metal2 assumption).
        try:
            access, _rep = extract_terminal_access(cell, tech=tech, mim_spec=pspec)
        except Exception:
            access = {}
        tab_node: Dict[str, Any] = {}
        tab_net_ok: Dict[str, bool] = {}
        for term in ("p", "n"):
            a = access.get(term)
            if a is None:
                tab_node[term] = None
                tab_net_ok[term] = False
                continue
            ax = int(round(float(a.xy_um[0]) * _NM_PER_UM)) - cx + px
            ay = int(round(float(a.xy_um[1]) * _NM_PER_UM)) - cy + py
            idx = node_index_at(nodes, tuple(a.layer), int(ax), int(ay))
            if idx is None:
                # the access extractor reports the tab on the RAW metal key,
                # but the plate connect model CARVES that metal into DERIVED
                # keys (gf180: Metal2 (36,0) -> routable (36,901) plus the
                # bottom-plate region named (75,901) -- the carve's name layer
                # is arbitrary, so layer-number matching is NOT enough). An
                # exact-key miss is a key alias, not an unmapped plate: retry
                # every derived key whose ops consume the access's raw layer,
                # at the same point. (native<->oracle fix 2026-06-11: gf180
                # LVS matched while this reported p_plate_node=None on every
                # gf180 cap.)
                raw = tuple(a.layer)
                for d in getattr(model, "derived", ()) or ():
                    refs = set()
                    for op in getattr(d, "ops", ()) or ():
                        for x in op:
                            if (isinstance(x, tuple) and len(x) == 2
                                    and all(isinstance(v, int) for v in x)):
                                refs.add(x)
                    if raw not in refs:
                        continue
                    idx = node_index_at(nodes, tuple(d.name), int(ax), int(ay))
                    if idx is not None:
                        break
            tab_node[term] = idx
            # does this plate's node carry the routed wire of the terminal's DECLARED net?
            want = str(nets.get(term, ""))
            ok = False
            if idx is not None and want in net_regions:
                allreg = kdb.Region()
                for reg in nodes[idx].values():
                    allreg += reg
                allreg.merge()
                ok = not (allreg & net_regions[want]).is_empty()
            tab_net_ok[term] = ok
        ip, inn = tab_node["p"], tab_node["n"]
        distinct = ip is not None and inn is not None and ip != inn
        rows.append({
            "cap": name,
            "p_net": str(nets.get("p", "")),
            "n_net": str(nets.get("n", "")),
            "p_plate_node": ip,
            "n_plate_node": inn,
            "n_nodes": len(nodes),
            "plates_distinct": distinct,
            "p_connects_right_net": tab_net_ok["p"],
            "n_connects_right_net": tab_net_ok["n"],
        })
    return rows


def _recenter_cap_placement(intent, placement, *, tech) -> Dict[str, _XY]:
    """RETIRED to an IDENTITY (2026-06-11): returns ``placement`` unchanged.

    The shift this used to apply was a DOUBLE correction. Its premise was that
    route consumers do raw ``cell_coord + placement`` (lower-left cell origin);
    in reality every consumer (lvs_gate emit, _verify_cap_plates, the layout
    model, pins) places ``cell_coord - extent_centre + placement`` -- so the
    placement already IS the extent centre, and subtracting the bbox centre
    AGAIN slid each cap by half its own extent. On the test4 m=2 bank that was
    a (-29.5, -14.6) um shift that parked the 62.9 um cap ON TOP of MM21/MM19
    (the user-spotted "cap overlaps devices" anomaly; placement-integrity
    invariant test in tests/test_under_cap_intent.py), silently destroying the
    non-overlap solve's guarantee. THE convention is single: placement ==
    extent centre, enforced by the invariant test -- never a post-solve shift.
    The function is kept as an identity (not deleted) so the call site
    documents the history and the cap-cap overlap metric below keeps watching
    the failure the old shift claimed to fix.
    """
    return dict(placement)


def _cap_cap_overlap_um2(intent, placement, *, tech) -> float:
    """Max area (um^2) where two DIFFERENT-net cap plate/cover regions overlap on a shared
    layer = a real cap-cap short. 0 = caps physically clear of each other."""
    import klayout.db as kdb
    from clay.layout.device_terminal import extract_terminal_access
    from clay.layout.route.pins import (
        _geom_extent_center_nm, _mim_spec_for, _mim_spec_for_passive)
    import clay.layout.passive_device_cell as _pdc

    spec = _mim_spec_for(tech)
    if spec is None:
        return 0.0
    layers = [tuple(tech.metal_key(2)), tuple(tech.metal_key(3))]
    caps = []
    for p in getattr(intent, "passives", []) or []:
        if getattr(p, "kind", "") not in ("cap", "c"):
            continue
        pos = placement.get(getattr(p, "name", ""))
        if pos is None:
            continue
        try:
            g = _pdc.build_capacitor_for_tech(
                tech, float(p.value), nets=getattr(p, "nets", {}),
                m=int(getattr(p, "M", 1) or 1),
                model=str(getattr(p, "model", "") or ""),
            )
            acc, _ = extract_terminal_access(
                g, tech=tech, mim_spec=_mim_spec_for_passive(tech, p))
        except Exception:
            continue
        owner = {tuple(ta.layer): ta.net for ta in acc.values()}
        regs = {}
        cx, cy = float(pos[0]), float(pos[1])
        # THE production placement frame: placement == artwork extent CENTRE
        # (the single convention since the v0.4.21 recenter retirement; same
        # transform as model.py cap_plate_conductors / lvs_gate / pins). This
        # metric measured in the RETIRED raw cell+placement frame for three
        # versions and reported a 206.7um2 PHANTOM overlap (caps' local origin
        # is the lower-left corner) -- bisected to v0.4.21, refuted by the
        # gf180 LVS matched oracle; frame fixed 2026-06-12.
        ccx, ccy = _geom_extent_center_nm(g)
        for lay in layers:
            r = kdb.Region()
            for lp in g.polygons:
                if (int(lp.layer), int(lp.datatype)) == lay:
                    r.insert(kdb.Polygon([kdb.Point(round(v.x - ccx + cx * 1000),
                                                    round(v.y - ccy + cy * 1000))
                                          for v in lp.polygon.vertices]))
            r.merge()
            regs[lay] = (r, owner.get(lay))
        caps.append(regs)
    worst = 0.0
    for i in range(len(caps)):
        for j in range(i + 1, len(caps)):
            for lay in layers:
                ra, na = caps[i][lay]
                rb, nb = caps[j][lay]
                if na is not None and nb is not None and na == nb:
                    continue
                a = (ra & rb).area() * 1e-6
                if a > worst:
                    worst = a
    return worst


def _cap_plate_obstacles(
    intent: Any,
    placement: Dict[str, _XY],
    *,
    tech: LayoutTech,
) -> List[Obstacle]:
    """Net-aware keep-out obstacles for every MIM cap's M2/M3 PLATE.

    A cap plate is a big conductor; a foreign net routed across it shorts to the plate
    (real LVS short the per-net connectivity check misses). For each cap with a device-cell
    builder (gf180) and a MIM spec, this emits the plate polygons as :class:`Obstacle`s whose
    ``owner_net`` is that plate's net -- so foreign signals route AROUND while the plate's own
    net may land on it. Each obstacle is bloated by the layer min-spacing so foreign metal
    keeps clearance. Leakage-free: only the cap device cell + placement, no reference routing.

    COMPATIBILITY ADAPTER (dev-discipline rule 6): the net-tagged cap-plate conductors now
    live in the unified :class:`clay.layout.model.LayoutModel`; this function projects that
    model to the engine's :class:`Obstacle` list. Long-term the router queries the model.
    """
    from clay.layout.model import cap_plate_conductors

    return [c.as_obstacle() for c in cap_plate_conductors(intent, placement, tech=tech)]


def _foreign_metal_over_cap_plates(intent, placement, routed, *, tech) -> float:
    """Max area (um^2) of any FOREIGN net's metal overlapping a cap plate, on the same layer.

    The LVS-found short metric: a foreign net's metal over a cap plate = electrical short to
    that plate. Measured on the ROUTED artwork + the cap device cell. 0 = no cap-plate short.
    """
    import klayout.db as kdb
    from clay.layout.device_terminal import extract_device_nodes, extract_terminal_access
    from clay.layout.route.pins import _geom_extent_center_nm, _mim_spec_for_passive
    import clay.layout.passive_device_cell as _pdc

    worst = 0.0
    for p in getattr(intent, "passives", []) or []:
        if getattr(p, "kind", "") not in ("cap", "c"):
            continue
        pos = placement.get(getattr(p, "name", ""))
        if pos is None:
            continue
        cx, cy = float(pos[0]), float(pos[1])
        spec = _mim_spec_for_passive(tech, p)
        if spec is None:
            continue
        try:
            cell = _pdc.build_capacitor_for_tech(
                tech, float(p.value), nets=getattr(p, "nets", {}),
                m=int(getattr(p, "M", 1) or 1),
                model=str(getattr(p, "model", "") or ""),
            )
            access, report = extract_terminal_access(cell, tech=tech, mim_spec=spec)
        except Exception:
            continue
        if report.unmapped or report.collapsed:
            raise ValueError(f"cap plate overlap is not measurable for {p.name}: {report}")
        nodes = extract_device_nodes(cell, tech=tech, mim_spec=spec)
        owners = {}
        for terminal in access.values():
            old = owners.setdefault(terminal.node_id, terminal.net)
            if old != terminal.net:
                raise ValueError(f"cap plate node has conflicting owners for {p.name}")
        routing_layers = {tuple(key) for key in spec.metal_order}
        cut_layers = {tuple(via[0]) for via in spec.vias}
        derived_layers = {tuple(item.name) for item in spec.derived}
        marker_layers = {tuple(key) for key in spec.metals
                         if tuple(key) not in routing_layers | cut_layers | derived_layers}
        for item in spec.derived:
            for operation in item.ops:
                marker_layers.update(tuple(arg) for arg in operation[1:]
                                     if isinstance(arg, tuple) and len(arg) == 2
                                     and tuple(arg) not in routing_layers | cut_layers | derived_layers)
        marker = kdb.Region()
        for polygon in cell.polygons:
            if (int(polygon.layer), int(polygon.datatype)) in marker_layers:
                marker.insert(kdb.Polygon([kdb.Point(int(v.x), int(v.y))
                                           for v in polygon.polygon.vertices]))
        marker.merge()
        if marker.is_empty():
            raise ValueError(f"cap plate overlap has no actual MIM marker for {p.name}")
        # SAME centroid recentre the emitter uses (see _cap_plate_obstacles): the cap
        # plate is stamped centroid-relative, so the metric must measure the plate at its
        # REAL stamped location, not the raw cell origin -- otherwise it checks empty space
        # ~(centroid) um off the plate and falsely reports 0 foreign overlap.
        ccx_nm, ccy_nm = _geom_extent_center_nm(cell)
        transform = kdb.Trans(int(round(cx * 1000)) - ccx_nm,
                              int(round(cy * 1000)) - ccy_nm)
        plate_layers = {tuple(key) for key in spec.metal_order[-2:]}
        owned_plates = {}
        for index, node in enumerate(nodes):
            for key, region in node.items():
                player = tuple(spec.synthetic_to_real.get(key, key))
                if player not in plate_layers:
                    continue
                actual_plate = region & marker
                if actual_plate.is_empty():
                    continue
                if index not in owners:
                    raise ValueError(f"cap plate has no actual terminal owner for {p.name}")
                pair = (player, owners[index])
                owned_plates.setdefault(pair, kdb.Region()).insert(actual_plate.transformed(transform))
        for (player, ownnet), plate in owned_plates.items():
            plate.merge()
            for net, nr in (getattr(routed, "nets", {}) or {}).items():
                if net == ownnet:
                    continue
                r = kdb.Region()
                for lp in getattr(nr, "wires", []) or []:
                    if (int(lp.layer), int(lp.datatype)) == player:
                        r.insert(kdb.Polygon([kdb.Point(int(v.x), int(v.y)) for v in lp.polygon.vertices]))
                r.merge()
                a = (r & plate).area() * 1e-6
                if a > worst:
                    worst = a
    return worst


# --------------------------------------------------------------------------- #
# T2 composite matched-group stamps (opt-in consumption face of
# clay.layout.extract.interdigitate.composite_matched_cell on the style path)
# --------------------------------------------------------------------------- #
def _composite_groups_for_style(
    intent: Any,
    style_rows: Any,
    tech: LayoutTech,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Topology-derived matched groups realisable as ONE composite stamp.

    Selection is DESIGN data only: :func:`clay.layout.extract.topo_sketch.
    mirror_groups` (netlist mirror pairs) gives the candidate groups; a group
    composes iff its 2-4 members share an EXACT (kind, W, L) (the composite
    builder's matched contract) AND all sit in the SAME style row
    (``style_rows`` is the row partition the placement will enforce; a
    cross-row group cannot be one stamp). Every refusal is an honest note,
    never a silent squeeze. The composite cell itself is built AND verified by
    :func:`clay.layout.extract.interdigitate.composite_matched_cell`
    (build-time node oracle: complete, net-pure per-member terminal access).

    No reference geometry is read: the inputs are the netlist intent, the
    style ROW PARTITION (names only) and the PDK surface.
    """
    from clay.layout.extract.interdigitate import composite_matched_cell
    from clay.layout.extract.topo_sketch import mirror_groups

    role_map = {"drain": "d", "gate": "g", "source": "s", "body": "b"}
    row_of: Dict[str, int] = {}
    for ri, row in enumerate(style_rows or ()):
        for n in row:
            row_of.setdefault(str(n), ri)
    devs = {str(d.name): d for d in getattr(intent, "devices", []) or []}
    taken = set(devs) | {
        str(p.name) for p in getattr(intent, "passives", []) or []
    }
    groups: List[Dict[str, Any]] = []
    notes: List[str] = []
    for g in mirror_groups(intent):
        label = "/".join(g.members)
        members = [m for m in g.members if m in devs]
        if len(members) != len(g.members) or not (2 <= len(members) <= 4):
            notes.append(
                f"composite skip {label}: needs 2-4 netlist MOS members "
                f"(got {len(members)} usable)"
            )
            continue
        d0 = devs[members[0]]
        sig0 = (str(d0.kind), float(d0.W), float(d0.L))
        if any(
            (str(devs[m].kind), float(devs[m].W), float(devs[m].L)) != sig0
            for m in members
        ):
            notes.append(
                f"composite skip {label}: members not matched on (kind, W, L)"
            )
            continue
        rows_hit = {row_of.get(m) for m in members}
        if None in rows_hit or len(rows_hit) != 1:
            notes.append(
                f"composite skip {label}: members span style rows "
                f"{sorted(str(r) for r in rows_hit)} (same-style-row "
                f"membership contract)"
            )
            continue
        member_dicts = [
            {
                "name": m,
                "kind": str(devs[m].kind),
                # intent W/L are um; composite_matched_cell takes METRES
                # (the build_device .cir convention).
                "W": float(devs[m].W) * 1e-6,
                "L": float(devs[m].L) * 1e-6,
                "M": int(getattr(devs[m], "M", 1) or 1),
                "nets": {
                    role_map[r]: str(n)
                    for r, n in (devs[m].nets or {}).items()
                    if r in role_map
                },
            }
            for m in members
        ]
        try:
            cell = composite_matched_cell(tech, member_dicts)
        except Exception as exc:
            notes.append(f"composite skip {label}: build refused ({exc})")
            continue
        key = "CMP_" + "_".join(members)
        if key in taken:
            notes.append(
                f"composite skip {label}: stamp key {key} collides with an "
                f"existing design name"
            )
            continue
        groups.append({
            "key": key,
            "members": tuple(members),
            "row": rows_hit.pop(),
            "cell": cell,
            "geom": cell.to_cell_geometry(name=key),
            "devices": [devs[m] for m in members],
        })
    return groups, notes


def _composite_sketch_inputs(
    intent: Any,
    style_order: Any,
    sizes: Dict[str, _XY],
    groups: List[Dict[str, Any]],
) -> Tuple[Any, Any, Dict[str, _XY]]:
    """Project (intent view, style order, sizes) onto the composited universe.

    Each composited group becomes ONE pseudo device for the placement solve:
    its stamp extent enters ``sizes`` under the group key, its merged member
    nets carry the group's connectivity, and the member names disappear; pair
    lists are renamed onto the group key (pairs internal to one composite
    drop). The view feeds ONLY the sketch -> hierarchical-graph -> solve
    chain -- the real intent keeps driving widths / pins / caps /
    verification.
    """
    from clay.layout.extract.style_order import rows_to_relative_order
    from clay.layout.extract.types import ExtractedDevice

    member_key: Dict[str, str] = {}
    for g in groups:
        for m in g["members"]:
            member_key[str(m)] = g["key"]

    out_sizes: Dict[str, _XY] = {
        n: s for n, s in sizes.items() if n not in member_key
    }
    pseudos: List[Any] = []
    for g in groups:
        e = g["cell"].drawn_extent()
        out_sizes[g["key"]] = (
            (e[2] - e[0]) / _NM_PER_UM,
            (e[3] - e[1]) / _NM_PER_UM,
        )
        merged_nets: Dict[str, str] = {}
        for d in g["devices"]:
            for r, n in (d.nets or {}).items():
                merged_nets[f"{r}:{d.name}"] = str(n)
        d0 = g["devices"][0]
        pseudos.append(ExtractedDevice(
            name=g["key"], kind=str(d0.kind),
            W=float(d0.W), L=float(d0.L),
            M=sum(int(getattr(d, "M", 1) or 1) for d in g["devices"]),
            nets=merged_nets,
        ))

    def _ren_pairs(pairs: Any) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        seen: set = set()
        for ab in pairs or ():
            if len(ab) != 2:
                continue
            a = member_key.get(str(ab[0]), str(ab[0]))
            b = member_key.get(str(ab[1]), str(ab[1]))
            if a == b:
                continue  # the pair is internal to one composite stamp
            if (a, b) in seen or (b, a) in seen:
                continue
            seen.add((a, b))
            out.append((a, b))
        return out

    class _View:
        pass

    view = _View()
    view.cell = getattr(intent, "cell", "")
    view.devices = [
        d for d in (getattr(intent, "devices", []) or [])
        if str(d.name) not in member_key
    ] + pseudos
    view.passives = list(getattr(intent, "passives", []) or [])
    view.matched_pairs = _ren_pairs(getattr(intent, "matched_pairs", ()))
    view.mirror_pairs = _ren_pairs(getattr(intent, "mirror_pairs", ()))
    view.diff_pairs = _ren_pairs(getattr(intent, "diff_pairs", ()))
    view.device_bboxes = {}
    view.symmetry_axis = None
    # Carry the edge-anchor intent SOURCES onto the projected view so the
    # place_relax lowering (which reads edge_anchor_members(view, seed)) sees
    # them in the composite-stamps path too. Both are symbolic/zero-coordinate:
    # the DECLARED entries (back-compat) and the P2-EXTRACTED RelationNetwork
    # (P4 closed loop). device_bboxes is deliberately blanked above; neither of
    # these carries a coordinate, so the leakage boundary is preserved.
    view.edge_anchor_devices = list(
        getattr(intent, "edge_anchor_devices", None) or [])
    view.placement_relations = getattr(intent, "placement_relations", None)

    rows_out: List[List[str]] = []
    for row in getattr(style_order, "rows", ()) or ():
        new_row: List[str] = []
        for n in row:
            n = str(n)
            k = member_key.get(n)
            if k is None:
                new_row.append(n)
            elif k not in new_row:
                new_row.append(k)  # the group key sits at its first member
        if new_row:
            rows_out.append(new_row)
    order = rows_to_relative_order(
        str(getattr(style_order, "cell", "")) or "style", rows_out
    )
    return view, order, out_sizes


class _IntentSansDevices:
    """``extract_pins`` view: the intent minus the composited member devices
    (their pins come from the composite cell's own terminal access instead --
    snapping each member's PARAMETRIC cell terminals onto composite metal
    could land a pin on a foreign net's comb column)."""

    def __init__(self, intent: Any, drop: set) -> None:
        self._intent = intent
        self.devices = [
            d for d in (getattr(intent, "devices", []) or [])
            if str(d.name) not in drop
        ]
        self.passives = list(getattr(intent, "passives", []) or [])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._intent, name)


def _composite_member_pins(
    groups: List[Dict[str, Any]],
    placement: Dict[str, _XY],
    pins_by_net: Dict[str, List[Any]],
    report: Any,
    *,
    tech: LayoutTech,
) -> None:
    """Append every composited member's terminal pins from the composite
    cell's OWN per-device terminal access (:func:`clay.layout.device_terminal.
    extract_terminal_access` -- the same source path every device cell uses).

    Frame convention: ``TerminalAccess.xy_um`` is in the raw cell frame; the
    placed stamp is extent-centre-relative (the ``to_cell_geometry`` /
    placement contract), so the absolute pin is
    ``placement + (xy_um - extent_centre)``. Honest gaps (unmapped terminals,
    the array's absent per-finger substrate taps) go to ``report.uncovered``
    -- never a fabricated pin.
    """
    from clay.layout.device_terminal import extract_terminal_access
    from clay.layout.route.engine import Pin as _Pin

    for g in groups:
        cell = g["cell"]
        pos = placement.get(g["key"])
        if pos is None:
            for tkey, term in sorted(cell.terminals.items()):
                dev, _, role = str(tkey).rpartition(".")
                report.uncovered.append((
                    dev or str(tkey), role, str(getattr(term, "net", "")),
                    "composite stamp not placed",
                ))
            continue
        e = cell.drawn_extent()
        cx_um = (e[0] + e[2]) / 2.0 / _NM_PER_UM
        cy_um = (e[1] + e[3]) / 2.0 / _NM_PER_UM
        access, rep = extract_terminal_access(cell, tech=tech)
        for tkey in sorted(access):
            a = access[tkey]
            dev, _, role = str(tkey).rpartition(".")
            net = str(getattr(a, "net", "") or "")
            if not net:
                continue
            pins_by_net.setdefault(net, []).append(_Pin(
                net=net, device=dev or str(tkey), terminal=role,
                xy_um=(
                    float(pos[0]) + float(a.xy_um[0]) - cx_um,
                    float(pos[1]) + float(a.xy_um[1]) - cy_um,
                ),
                layer=tuple(a.layer),
            ))
        for tkey in getattr(rep, "unmapped", ()) or ():
            dev, _, role = str(tkey).rpartition(".")
            term = cell.terminals.get(tkey)
            report.uncovered.append((
                dev or str(tkey), role, str(getattr(term, "net", "")),
                "composite terminal maps to no electrical node",
            ))
        if str(cell.kind) == "nmos":
            # the composite array draws NO per-finger substrate tap (its
            # module contract: tap rows are the assembly's concern); a member
            # body net is therefore not a routable terminal here -- reported
            # as an honest gap, never a fabricated pin.
            for d in g["devices"]:
                bnet = str((d.nets or {}).get("body", "") or "")
                if bnet:
                    report.uncovered.append((
                        str(d.name), "b", bnet,
                        "composite array draws no per-finger substrate tap",
                    ))


def _composite_trunk_obstacles(
    groups: List[Dict[str, Any]],
    placement: Dict[str, _XY],
) -> List[Obstacle]:
    """Net-aware keep-outs for each composite stamp's terminal-comb TRUNK bars.

    The trunks are upper-metal (metal2) conductors that the M1-only
    device-pad projection of the layout model does not carry; a foreign net
    routed across one would short to the comb (the cap-plate precedent). Each
    trunk bar becomes an :class:`Obstacle` owned by its net, in the stamp's
    placed extent-centre frame. Composite-cell data + placement only -- no
    reference read.
    """
    out: List[Obstacle] = []
    for g in groups:
        pos = placement.get(g["key"])
        if pos is None:
            continue
        cell = g["cell"]
        e = cell.drawn_extent()
        cx_nm = (e[0] + e[2]) / 2.0
        cy_nm = (e[1] + e[3]) / 2.0
        for tr in cell.trunks:
            lkey = tuple(tr.layer)
            bar = None
            for lp in cell.polygons:
                if (int(lp.layer), int(lp.datatype)) != lkey:
                    continue
                xs = [v.x for v in lp.polygon.vertices]
                ys = [v.y for v in lp.polygon.vertices]
                if (min(xs) <= tr.x0_nm + 1 and max(xs) >= tr.x1_nm - 1
                        and min(ys) <= tr.y_nm <= max(ys)):
                    bar = (min(xs), min(ys), max(xs), max(ys))
                    break
            if bar is None:
                continue
            out.append(Obstacle(
                (bar[0] - cx_nm) / _NM_PER_UM + float(pos[0]),
                (bar[1] - cy_nm) / _NM_PER_UM + float(pos[1]),
                (bar[2] - cx_nm) / _NM_PER_UM + float(pos[0]),
                (bar[3] - cy_nm) / _NM_PER_UM + float(pos[1]),
                lkey,
                owner_net=tr.net,
            ))
    return out


def _mirror_route_constraints(
    intent: Any,
    placement: Dict[str, _XY],
    pins_by_net: Dict[str, List[Any]],
    sizes: Dict[str, _XY],
    solved_axes: Optional[Mapping[Tuple[str, str], float]] = None,
) -> Tuple[List[Any], List[str]]:
    """Mirror routing constraints from DESIGN symmetry + the SOLVED placement.

    Returns ``(constraints, notes)``. Pairs come from netlist TOPOLOGY only:
    2-member mirror groups (:func:`clay.layout.extract.topo_sketch.
    mirror_groups`) plus the intent's ``diff_pairs``. The AXIS is the placed
    midpoint x of the two paired devices: a SOLVE OUTPUT (design data), never
    a reference coordinate, so the leakage account is untouched.

    AXIS-SUPPLY CONTRACT (the anaroute premise: the axis comes from the
    placement's symmetry output): an axis exists only where the solve actually
    placed the pair as a mirror CANDIDATE -- both members on ONE row (equal y
    within a quarter of the smaller stamp height) and laterally adjacent
    (centre distance within twice their combined width). A pair the solve
    scattered has NO meaningful mirror axis; emitting a Mirror for it would
    hand the engine a constraint unsatisfiable BY PLACEMENT, so the pair is
    honestly SKIPPED with a note -- routing never pretends to mirror what
    placement did not. (Measured on t4, 2026-06-12: the style rows put the
    diff-input pair MM4/MM18 66 um apart on different rows -- no axis; and the
    pre-contract first-wins ordering let the scattered MM29/MM30 pair claim
    net7/net15's axis while the truly co-row MM5/MM6 pair -- the one mirror
    candidate the solve produced -- got nothing.)

    SOLVED-AXIS SUPPLY (P1b extension, 2026-06-13): ``solved_axes`` maps a
    name-sorted pair key to the axis the PLACEMENT solved for it -- the
    realized mirror-rigid pairs of ``RelaxReport.mirror_pairs``
    (``place_relax`` posed the pair as one mirror body, so the posture is
    held by construction). Such a pair takes the supplied axis directly and
    bypasses the geometric re-derivation gate above (the placement already
    adjudicated candidacy); a note records the supply provenance. Pairs
    without an entry keep the measured-candidacy contract verbatim --
    ``solved_axes=None`` is byte-identical.

    Per qualifying pair, each terminal ROLE (gate first -- the diff inputs --
    then drain, then source) whose nets DIFFER between the members and both
    carry routable pins yields one :class:`~clay.layout.route.constraints.
    Mirror`. A net enters at most one Mirror (first wins among QUALIFYING
    pairs, deterministic order).
    """
    from clay.layout.extract.topo_sketch import mirror_groups
    from clay.layout.route import constraints as _C

    devs = {str(d.name): d for d in getattr(intent, "devices", []) or []}
    pairs: List[Tuple[str, str]] = []
    seen_pairs: set = set()
    for g in mirror_groups(intent):
        if len(g.members) == 2:
            pairs.append((str(g.members[0]), str(g.members[1])))
    for ab in getattr(intent, "diff_pairs", []) or []:
        if len(ab) == 2:
            pairs.append((str(ab[0]), str(ab[1])))

    out: List[Any] = []
    notes: List[str] = []
    used_nets: set = set()
    for (da, db) in pairs:
        key = tuple(sorted((da, db)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        label = f"{da}/{db}"
        dev_a, dev_b = devs.get(da), devs.get(db)
        pos_a, pos_b = placement.get(da), placement.get(db)
        if dev_a is None or dev_b is None or pos_a is None or pos_b is None:
            notes.append(
                f"mirror skip {label}: member unplaced/composited -- no "
                "per-device axis to derive"
            )
            continue
        # P1b solved-axis supply: a realized mirror-rigid pair's axis comes
        # straight from the placement report (the solve adjudicated
        # candidacy by posing the pair as one mirror body).
        solved = (solved_axes or {}).get(key)
        if solved is not None:
            axis_x = float(solved)
            notes.append(
                f"mirror axis {label}: supplied by the placement solve "
                f"(mirror-rigid pair, axis_x={axis_x:.3f} um)"
            )
        else:
            # axis-supply contract: the solve must have placed the pair as a
            # mirror candidate (one row + laterally adjacent), else no axis.
            sz_a = sizes.get(da) or (0.0, 0.0)
            sz_b = sizes.get(db) or (0.0, 0.0)
            dy = abs(float(pos_a[1]) - float(pos_b[1]))
            dx = abs(float(pos_a[0]) - float(pos_b[0]))
            h_tol = 0.25 * min(float(sz_a[1]) or 1.0, float(sz_b[1]) or 1.0)
            x_max = 2.0 * (float(sz_a[0]) + float(sz_b[0]))
            if dy > h_tol or (x_max > 0 and dx > x_max):
                notes.append(
                    f"mirror skip {label}: placement is not a mirror candidate "
                    f"(dy={dy:.2f}um vs row tol {h_tol:.2f}um, dx={dx:.2f}um vs "
                    f"adjacency bound {x_max:.2f}um) -- no axis supplied"
                )
                continue
            axis_x = (float(pos_a[0]) + float(pos_b[0])) / 2.0
        for role in ("gate", "drain", "source"):
            na = str((dev_a.nets or {}).get(role, "") or "")
            nb = str((dev_b.nets or {}).get(role, "") or "")
            if not na or not nb or na == nb:
                continue  # shared net (tail source / common gate): nothing to mirror
            if na in used_nets or nb in used_nets:
                continue  # first Mirror per net wins (engine contract)
            if na not in pins_by_net or nb not in pins_by_net:
                continue  # no routable pins -> the engine would reject it anyway
            out.append(_C.Mirror(net_a=na, net_b=nb, axis_x_um=axis_x))
            used_nets.update((na, nb))
    return out, notes


def route_amp_leakfree(
    tech_id: str,
    out_gds: Optional[str] = None,
    *,
    policy: Optional[WidthPolicy] = None,
    trunk_layers: bool = False,
    intent: Any = None,
    style_from_reference: bool = False,
    direction_from_reference: bool = False,
    direction_weight: float = 0.0,
    rail_trunk_anchors: bool = False,
    composite_stamps: bool = False,
    guard_rings: bool = False,
    mirror_routing: bool = False,
    placement_engine: str = "bands",
    mirror_pairs_rigid: bool = False,
    mirror_axis_mode: str = "shared_hard",
    row_height_target_um: Optional[float] = None,
    ring_granularity: str = "island",
    dense_contacts: bool = False,
    planar_upper_metals: bool = False,
) -> dict:
    """Route the two-stage amplifier LEAKAGE-FREE on ANY target PDK (cross-PDK validation).

    Same design, migrated to ``tech_id`` (e.g. ``'gf180mcu'`` / ``'open_pdk'``): the
    geometry is the target-PDK device artwork (:func:`real_device_geometry` with
    ``ref_gds=None`` -- ``build_device`` + native passives), and the WIDTHS are
    DESIGN-derived and re-floored on the target PDK's ``min_width`` (the leakage-free part).
    The result is judged by the self-contained clean-route contract (connectivity, no
    shorts, 0 merged cross-net spacing violations, 0 geometric opens, rails at power class)
    plus the independent KLayout LVS gate (gf180 = the strong oracle).

    PLACEMENT LEAKAGE STATUS (remediated 2026-06-10): the device positions come from
    :func:`open_pdk_ota_positions`, whose placement STRUCTURE (band membership, within-band
    order, cross-band X) is DESIGN-sourced -- derived by
    :mod:`clay.layout.extract.topo_sketch` from netlist topology (mirror groups / diff
    pairs / net connectivity), never from the reference centroids in
    ``intent.device_bboxes`` (the SEV-1 incident this remedied). Evidence:
    ``tests/test_placement_leakage.py::test_placement_independent_of_reference_positions``
    (perturbing the reference centroids moves 0/17 devices; the size FALLBACK for
    cell-less passives is the one accounted reference read) + the gf180 KLayout LVS
    keystone + the external_pdk KLayout DRC signoff, all green at the design-sourced placement.

    MIM caps are wired by the ROOT distinct-plate mechanism (each plate on its OWN routing
    layer via clay.layout.device_terminal.extract_terminal_access + the PDK MIM spec DATA)
    where the PDK has a cap DEVICE CELL (gf180): ``cap_plate_rows`` reports the measured
    per-cap plate nodes and ``caps_all_plates_distinct`` / ``caps_all_plates_right_net``
    are the ELECTRICAL bar (two plates stay two nodes, each on the right net). A PDK with
    no cap device cell (external_pdk/external_pdk) leaves its caps in ``uncovered_terminals`` (honest gap)
    -- never the reverted Metal2-both-plates hack that shorted the cap.

    Honest gap: a passive whose device cell exposes no routable terminal on the target PDK
    is reported in ``uncovered_terminals`` -- that passive is not wired. The transistor
    core + power rails + (on gf180) the MIM caps and resistors are.

    ``style_from_reference`` (plan S1, opt-in): the placement is STYLE-CONDITIONED --
    the reference layout's RELATIVE order (row partition / left-of, names only, never
    coordinates; see :mod:`clay.layout.extract.style_order`) is extracted from the
    intent's reference bboxes, mapped onto the design's devices, and fed to the SAME
    sketch->hier->solve machinery ``open_pdk_ota_positions`` runs (this is the style
    parameter face of that one machine, NOT a second placement path). The result dict
    declares ``placement_provenance='style-transfer'`` and carries a ``placement_note``;
    a style-conditioned placement is NEVER claimed leakage-free (the reference relative
    order is consumed by design). Default False keeps the design-only placement and
    ``placement_provenance='design-only'``.

    ``direction_from_reference`` (plan S3 style face, opt-in; REQUIRES
    ``style_from_reference=True`` -- it is a style-channel input and may only ride the
    self-declaring style path): the reference's per-layer H/V composition is measured
    as RELATIVE ratios (:func:`clay.layout.extract.direction_intent.
    extract_direction_intent`; never a coordinate) and each >=0.6-purity layer's
    majority axis becomes an explicit :class:`~clay.layout.route.constraints.Direction`
    soft preference appended AFTER the lowered list (the documented override face --
    a later Direction per layer wins). The report lands in the result under
    ``direction_intent`` with ``provenance='style-transfer'``. Default False appends
    nothing (``direction_intent`` is None) -- byte-stable.

    ``direction_weight`` (plan S3 declared face): the design-side HVH discipline knob,
    stored on ``RoutingIntent.direction_weight`` and wired to
    ``lower_intent(direction_weight=...)`` (which emits the CONVENTIONAL_AXES table at
    that weight). A cost knob, never a measured reference value. Default 0.0 lowers to
    no Direction constraint -- byte-stable.

    ``composite_stamps`` (T2, opt-in; REQUIRES ``style_from_reference=True``):
    each topology-derived matched group (``topo_sketch.mirror_groups`` --
    DESIGN data) whose 2-4 members share an exact (kind, W, L) and sit in
    ONE style row is realised as a single interdigitated composite stamp
    (``interdigitate.composite_matched_cell``, build-time node oracle). The
    placement solves the group as ONE unit (members leave placement/geometry,
    the stamp enters both), member pins come from the composite cell's
    per-device terminal access, and the comb trunks become net-aware
    keep-outs. Groups that cannot compose (cross-row, unmatched, builder
    refusal) are skipped with notes in ``composite_notes``. Default False
    composes nothing -- byte-stable.

    external_pdk SIGNOFF STATE (measured 2026-06-12, test4 intent; the oracle tests
    live in tests/test_composite_in_anchor.py): composite-only is KLayout-DRC
    CLEAN on the synthesized external_pdk deck (0 violations; was 335 before the
    via1-enclosure + heal-containment contract fixes), and its LVS residue is
    exactly the MIM account (CC1+NET15/VOUT, deck mim=None) plus the external_pdk
    RESISTOR family (MM15+GND/NET048/NET07/NET08): external_pdk emits no resistor
    artwork and the deck has res=None while the reference collapses
    resistors, and the composite -- drawing no per-finger substrate tap --
    honestly stops net08 from merging into GND through the SUBSTRATE the way
    the old per-device tap stamp accidentally did. composite x trunk adds
    ZERO LVS damage over composite-only; its single DRC residue (1x
    VIA1.EN.1 on a router-emitted net7 via) is an ENGINE emit gap (via cut
    not clipped with its landing pad), named in the strict xfail there.

    ``mirror_pairs_rigid`` (P1b, opt-in; REQUIRES ``placement_engine='relax'``):
    the relax placement stage rigidifies the design's declared diff/mirror
    pairs as mirror bodies about a common vertical axis (diff pairs converge
    across style rows under the soft band face, deviation reported; 2-member
    mirror groups re-pose same-row only -- see
    :class:`clay.layout.extract.place_relax.RelaxKnobs`). Each REALIZED
    pair's solved axis (``relax_report.mirror_pairs``) feeds
    ``_mirror_route_constraints`` directly when ``mirror_routing=True`` --
    the axis-supply contract's placement-side completion (the t4 MM4/MM18
    refusal removed at its root). Default False is byte-stable.
    """
    from clay.layout.extract.migrate import build_ota_intent
    from clay.layout.extract.realize_render import real_device_geometry
    from clay.layout.extract.route_migrate import open_pdk_ota_positions

    if direction_from_reference and not style_from_reference:
        raise ValueError(
            "direction_from_reference is a style-channel input (reference "
            "direction statistics); it requires style_from_reference=True so "
            "the result self-declares its style conditioning"
        )
    if composite_stamps and not style_from_reference:
        raise ValueError(
            "composite_stamps consumes the style ROW PARTITION (the "
            "same-style-row membership contract for a matched group); it "
            "requires style_from_reference=True"
        )
    if (str(placement_engine) not in ("bands", "cp")
            and not style_from_reference):
        # 'relax' / 'seqpair' refine the STYLE sketch's bands (their input
        # STRUCTURE -- band membership + order -- is the reference-relative
        # style order), so they require the self-declaring style path. 'cp' is
        # the exception: cp_group_placement_pos is leakage-free (it reads
        # compact_matched_groups + mirror_pairs + design footprints only -- no
        # reference order, no reference coordinate), so the CP engine runs on
        # ANY PDK with style_from_reference OFF. Its placement preamble (sizes /
        # rails / sketch rail rows) is then sourced from the PDK + topology
        # below, not from a reference.
        raise ValueError(
            "placement_engine is consumed by the style-path sketch/solve "
            "chain (the default path's open_pdk_ota_positions has no engine "
            "face); placement_engine='relax'/'seqpair' requires "
            "style_from_reference=True. placement_engine='cp' is leakage-free "
            "and does NOT require it (it runs on any PDK without a reference)."
        )
    if mirror_pairs_rigid and str(placement_engine) not in ("relax", "cp"):
        raise ValueError(
            "mirror_pairs_rigid re-poses declared diff/mirror pairs inside "
            "the relax placement stage (the placement-side mirror-axis "
            "supply); it requires placement_engine='relax' (which itself "
            "requires style_from_reference=True) or placement_engine='cp' "
            "(the CP envelope-min placer enforces declared pair mirror "
            "symmetry as a HARD constraint by construction)"
        )

    # ENGINE FACES (defined early so the leakage-free CP path can self-declare
    # before the style gates): 'cp' is the ONLY placement engine that is
    # leakage-free by construction (cp_group_placement_pos reads
    # compact_matched_groups + mirror_pairs + design footprints; no reference
    # order/coordinate), so it -- and the topology-derived rings it carries --
    # may run with style_from_reference OFF on any PDK.
    _cp_envelope = str(placement_engine) == "cp"
    _seqpair = str(placement_engine) == "seqpair"
    # the placement preamble (sketch/rails/tight sizes the CP branch consumes)
    # runs whenever a style placement is requested OR the CP engine is selected.
    # On the CP path it is sourced from PDK + topology only.
    _run_placement_preamble = bool(style_from_reference) or _cp_envelope

    pol = policy or WidthPolicy()
    tech = LayoutTech.resolve(tech_id)
    m1: Tuple[int, int] = tuple(tech.metal_key(1))  # type: ignore[assignment]

    # ``intent``: any ExtractedIntent (netlist-extracted; the default is the
    # OTA_CS_Q amp). The whole pipeline below is GENERIC over the intent's
    # devices/passives -- passing e.g. the SIGNED test4 netlist's intent routes
    # that design through the identical machinery (no per-circuit branch).
    if intent is None:
        intent = build_ota_intent()

    # UNIFIED ALL-ELEMENT PLACEMENT (the cap is not special): the comp cap is a
    # first-class placeable TILE co-placed in the SAME envelope-min pack as the
    # MOS groups (cp_group_placement_pos), pulled into the device mosaic by its
    # net wirelength. The retired cap_over_nodes / cap_planar_edge intent
    # derivations + their post-placement lowering modes no longer exist -- there
    # is no per-cap placement target step. Any circuit's passives are co-placed by
    # the one general mechanism, with no cap/per-circuit branch.

    # SERPENTINE WIDE-FLAT FOLD (opt-in; the device-footprint contract capability
    # threaded consistently through the OTA). ``row_height_target_um`` folds every
    # device stamp to the DESIGN wide-flat band height as a serpentine
    # shared-diffusion 2D-compact fold (aspect<=6 by construction) so the realized
    # devices go wide-flat and the CONT density rises toward the reference. The
    # target threads into BOTH the SIZE source (open_pdk_ota_positions) and the GEOMETRY
    # pass (real_device_geometry) as ONE consistent fold.
    #
    # NOT auto-engaged in the ring path (DELIBERATELY default-off): integrating the
    # serpentine fold into the rings_group OTA introduces a router GND/net08 short on
    # the MM15 substrate-tap family (the serpentine NMOS draws a per-device P+
    # substrate tap whose GND pin the maze routes into net08's source corridor -- the
    # same net08<->GND family the composite cell avoids by drawing NO per-device
    # tap). That is a HARD-GATE regression (shorts must be 0), so the OTA does NOT
    # auto-apply the fold; a caller may opt in explicitly (the capability + the
    # device footprint contract are GREEN in isolation). None keeps the canonical /
    # rings_group path byte-identical (the no-target build).
    if row_height_target_um is not None:
        _fold_target: Optional[float] = float(row_height_target_um)
    else:
        _fold_target = None

    placement, _sizes = open_pdk_ota_positions(
        intent, tech_id=tech_id, row_height_target_um=_fold_target
    )
    # GUARD RINGS (opt-in): the granularity face is normalised by the SAME
    # realize_render._guard_ring_mode the geometry pipeline uses (False ->
    # 'off'; True -> 'device'; 'device'/'group' pass through; anything else
    # raises). Never a silent granularity fallback -- the pre-2026-06-13
    # anchor coerced 'group' to device rings, exactly the downgrade that
    # normaliser forbids at the geometry face. Style-path-only like the
    # other style faces.
    #   * 'device' (v0.4.34 signed): ring every MOS stamp; the ringed stamp
    #     extents REPLACE the solve sizes so the placement reserves the ring
    #     room (single sizes pipeline, no second path).
    #   * 'group' (option-matrix wiring 2026-06-13, oracle record in
    #     tests/test_option_matrix.py): stamps stay BARE; the solve sizes are
    #     PADDED with group-ring room (group_ring_padded_extents -- the same
    #     sizes pipeline), and the rings are built AFTER the final placement
    #     (group_guard_rings: design topology + the solved placement only),
    #     then materialised INTO the same geometry dict every consumer reads
    #     (build_layout_model obstacles / emit / LVS) -- no second
    #     materialization path.
    # COMBINATIONS (the composite x trunk lesson: unsigned combinations are
    # refused loudly, never silently allowed):
    #   * 'device' + composite_stamps: REFUSED permanently -- composited
    #     members leave the geometry dict, so per-member device rings would
    #     be a silent PARTIAL ring set (geometrically void for the fused
    #     members).
    #   * 'group' + composite_stamps: REFUSED below -- ADJUDICATED 2026-06-13
    #     by the option matrix (tests/test_option_matrix.py + docs/audits/
    #     option-matrix-2026-06-13.html): the lift precondition (BOTH single
    #     options green on the canonical base) FAILED, because rings='group'
    #     alone measured RED there -- the y-interval row merge collapses the
    #     canonical relax+2D packing into ONE row, the x-runs then cluster
    #     vertically distant devices, 5 ring-bbox pairs overlap, and the
    #     oracles verdict: FULL LVS merged GND,NET10,NET15,NET7,VOUT into one
    #     layout net + 54 deck-DRC violations. The refusal stays until the
    #     named missing contract lands (cluster derivation must consume the
    #     placement's band structure, or rings must enter the solve as
    #     first-class footprints).
    from clay.layout.extract.realize_render import _guard_ring_mode

    ring_mode = _guard_ring_mode(guard_rings)
    _ringed_geometry = None
    _group_ring_base = None
    if ring_mode != "off":
        if not style_from_reference and not _cp_envelope:
            # guard rings on the bands/relax path band by the STYLE-derived row
            # stack, so they require the self-declaring style path. On the CP
            # path the rings cluster from compact_matched_groups (topology) over
            # the leakage-free CP placement and the rail rows come from
            # island_rail_intents (topology) -- so 'group' rings run on any PDK
            # with style_from_reference OFF (no reference is read).
            raise ValueError(
                "guard_rings is a style-face option; it requires "
                "style_from_reference=True (or placement_engine='cp', the "
                "leakage-free path whose rings are topology-derived)")
        if ring_mode == "device" and composite_stamps:
            raise ValueError(
                "guard_rings='device' + composite_stamps is not signed off "
                "and never will be: composited members leave the geometry "
                "dict, so per-member device rings would be a silent partial "
                "ring set; use guard_rings='group' (ring-on-group-stamp)")
        if ring_mode == "group" and composite_stamps:
            raise ValueError(
                "guard_rings='group' + composite_stamps is not signed off: "
                "the option-matrix adjudication (2026-06-13) measured "
                "rings='group' alone RED on the canonical base (ring-ring "
                "overlaps -> LVS merge + 54 DRC), so the combination's lift "
                "precondition failed; see tests/test_option_matrix.py")
        from clay.layout.extract.realize_render import (
            group_ring_padded_extents,
            stamp_extents,
        )
        try:
            _base_geom = real_device_geometry(
                intent, ref_gds=None, tech_id=tech_id, native_passive=True,
                guard_rings=ring_mode, row_height_target_um=_fold_target,
                contact_rows=dense_contacts)
        except Exception:
            _base_geom = real_device_geometry(
                intent, ref_gds=None, tech_id=tech_id, guard_rings=ring_mode,
                row_height_target_um=_fold_target,
                contact_rows=dense_contacts)
        if ring_mode == "device":
            _ringed_geometry = _base_geom
            _sizes.update(stamp_extents(_ringed_geometry))
        else:
            # bare stamps (byte-identical to off); reused as the route's
            # geometry below -- the rings consume the FINAL placement.
            _group_ring_base = _base_geom
            _sizes.update(group_ring_padded_extents(
                _group_ring_base, intent, tech_id=tech_id))
    placement_provenance = "design-only"
    placement_note = ""
    relax_report: Optional[Any] = None
    rail_intents: List[Dict[str, Any]] = []
    rail_corridor_y: Dict[str, float] = {}
    # T2 composite matched-group stamps (opt-in; populated on the style path
    # only -- the same-style-row membership constraint needs the style rows).
    comp_groups: List[Dict[str, Any]] = []
    comp_notes: List[str] = []
    if _run_placement_preamble:
        # PLACEMENT PREAMBLE (sketch -> hierarchical-graph -> solve). Two entry
        # modes share ONE machine (no second placement path):
        #   * style_from_reference=True: STYLE-CONDITIONED -- the reference's
        #     RELATIVE order seeds sk_order (self-declares 'style-transfer').
        #   * placement_engine='cp' with style OFF: LEAKAGE-FREE -- sk_order /
        #     rails / sizes are TOPOLOGY + PDK derived (no reference read), the
        #     CP placer owns the pos solve. This is the any-PDK CP path.
        from clay.layout.extract import relgraph as _rg
        from clay.layout.extract import topo_sketch as _ts
        from clay.layout.extract.style_order import (
            extract_rail_intent,
            extract_reference_order,
            map_order_to_design,
        )

        if style_from_reference:
            # the ONLY reference reads in this block (order, names only -- the
            # style channel's leakage boundary). Skipped on the leakage-free CP
            # path: there styled stays None and sk_order comes from topology.
            ref_order = extract_reference_order(intent)
            styled = map_order_to_design(ref_order, intent, intent)
        else:
            ref_order = None
            styled = None
        # @edge_anchor IO-EDGE EXCLUSION (priority 1): when an edge-anchor cap is
        # declared, the SOLVED clear side must avoid a die wall an IO terminal
        # already needs. The IO-edge intent (VIP/VIN left, VOUT right) is a
        # CATEGORICAL fact -- a side token per labelled signal net -- derived
        # here from the SAME extract_io_edge_intent the routing stub consumes,
        # and stashed on the intent so the relax anchor (via the sketch view)
        # reads it. Inert when no edge-anchor device is declared (the set stays
        # empty -> the anchor picks the most-room axis with no exclusion, the
        # prior byte-identical path).
        if getattr(intent, "edge_anchor_devices", None):
            from clay.layout.extract.style_order import extract_io_edge_intent
            try:
                _io = extract_io_edge_intent(intent)
            except ValueError:
                _io = []
            intent.edge_anchor_io_edges = sorted({
                str(f["edge"]) for f in _io if f.get("edge")})
        # T2 COMPOSITE STAMPS (opt-in): topology-derived matched groups whose
        # members share (kind, W, L) AND one style row become ONE
        # interdigitated stamp; the solve then places the group as one unit
        # (members leave sizes/intent-view, the stamp enters them). Default
        # False leaves sk_* as the existing inputs -- byte-stable.
        sk_intent: Any = intent
        # style path: the reference RELATIVE order. CP/leakage-free path: None
        # here -- the ring branch below overwrites it with the topology order
        # (compact_matched_group_order / well_island_order). A non-ring CP path
        # leaves sk_order None, and the bands sketch then seriates from topology.
        sk_order: Any = styled.order if styled is not None else None
        sk_sizes: Dict[str, _XY] = _sizes
        if ring_mode != "off":
            # RING-AWARE PLACEMENT (root-cure): when guard rings are requested the
            # placement bands by WELL-ISLAND (the intent-layer grouping: same kind
            # + same body net), not the reference style rows -- so each island is a
            # contiguous, separated band and a band-level ring per island is clean
            # by construction (the signed reference's topology). This consumes the
            # well-island intent (clay.layout.extract.well_islands, pure topology)
            # through the SAME style-row machinery; it self-declares as a ring-aware
            # placement, never a reference-coordinate read.
            if str(ring_granularity) == "compact_group":
                # F2 ring-granularity (geometric-fidelity): band by COMPACT-
                # MATCHED-GROUP (~9 tight per-group rings, the reference
                # granularity, enclosure ~85%) instead of the coarse 3 well-
                # island bands. Each compact group is kept X-contiguous in its
                # row so group_guard_rings wraps one non-overlapping frame per
                # group. Same style-row machinery; pure topology, no coordinate.
                from clay.layout.extract.well_islands import (
                    compact_matched_group_order,
                )
                # P-B 2D BAND-GRID: pack each kind's compact groups into rows
                # against a DESIGN-derived target row width chosen so the cell
                # aspect approaches the reference-calibrated aspect band (the
                # config scalar -- the band CENTER). After P-A's intra-group abut
                # the groups are tight enough that this wider band-grid no longer
                # fuses adjacent groups' padded ring boxes (the F2-era blocker;
                # re-measured route-clean at the wider packings). The packer reads
                # the group footprint widths (_sizes, the padded extents already
                # built above) + the band center; pure topology + footprint width
                # + a config scalar, no coordinate. The ceil(sqrt) default lands
                # the cell portrait for this design -- the width target drives it
                # landscape into [1.4, 1.9].
                from clay.config import fidelity_thresholds as _ft
                _th = _ft()
                _aspect_band_center = (_th.aspect_lo + _th.aspect_hi) / 2.0
                # The packer estimates each group's placed footprint as RAW
                # member WIDTH (P-A abuts intra-group members at raw spacing, so
                # the group's placed width ~ sum of raw member widths -- the
                # PADDED extents over-count intra-group width and mis-pick the
                # per-kind row split) but PADDED member HEIGHT (a row's height
                # carries the group ring's vertical room). Combined per member;
                # pure drawn-stamp footprint + ring padding, no coordinate.
                _aspect_sizes = dict(_sizes)
                if _group_ring_base is not None:
                    from clay.layout.extract.realize_render import (
                        stamp_extents as _stamp_extents_aspect,
                    )
                    _raw_ext = _stamp_extents_aspect(_group_ring_base)
                    _aspect_sizes = {
                        m: (float(_raw_ext.get(m, _sizes.get(m, (1.0, 1.0)))[0]),
                            float(_sizes.get(m, _raw_ext.get(m, (1.0, 1.0)))[1]))
                        for m in set(_raw_ext) | set(_sizes)
                    }
                sk_order = compact_matched_group_order(
                    intent, sizes=_aspect_sizes,
                    aspect_target=_aspect_band_center)
            else:
                from clay.layout.extract.well_islands import well_island_order
                sk_order = well_island_order(intent)
        # The ROW PARTITION the composite stamping + the placement solve must
        # band by. With rings ENGAGED this is the WELL-ISLAND partition
        # (sk_order, set above) so each island is a contiguous, spatially
        # SEPARABLE band and the per-island group rings are DISJOINT by
        # construction (the measured RED was 3 rings overlapping ~2000-2800um^2
        # because the composite path rebuilt the order from the interleaved
        # reference STYLE rows, discarding the island banding -> ring-ring +
        # ring-device contact/OD merges = CO.S.1 + member layout=NONE). With
        # rings OFF this is the reference style order -- byte-identical to the
        # prior composite path. Mirror groups are within-island by construction
        # (same kind + body net), so same-row composability is preserved.
        _band_order = sk_order
        if composite_stamps:
            comp_groups, comp_notes = _composite_groups_for_style(
                intent, getattr(_band_order, "rows", ()) or (), tech
            )
            if comp_groups:
                sk_intent, sk_order, sk_sizes = _composite_sketch_inputs(
                    intent, _band_order, _sizes, comp_groups
                )
                if ring_mode == "group":
                    # group-ring room around each composite stamp: the same
                    # per-side pad its MEMBERS' well polarity gets (delta of
                    # padded vs raw member extent -- the public padding face,
                    # never a second padding rule).
                    from clay.layout.extract.realize_render import (
                        group_ring_padded_extents as _grpe,
                        stamp_extents as _se,
                    )
                    _raw_ext = _se(_group_ring_base)
                    _pad_ext = _grpe(_group_ring_base, intent,
                                     tech_id=tech_id)
                    for _g in comp_groups:
                        _m = next((m for m in _g["members"]
                                   if m in _raw_ext and m in _pad_ext), None)
                        if _m is None or _g["key"] not in sk_sizes:
                            continue
                        _dw = _pad_ext[_m][0] - _raw_ext[_m][0]
                        _dh = _pad_ext[_m][1] - _raw_ext[_m][1]
                        _w0, _h0 = sk_sizes[_g["key"]]
                        sk_sizes[_g["key"]] = (_w0 + _dw, _h0 + _dh)
        # S4 RAIL INTENT (declared or style-inherited; RELATIVE facts only --
        # net + side / row-order gap, never a coordinate). Lowering attaches
        # each positioned rail's DESIGN-derived width class as the corridor
        # height (derive_width_intent, the same policy face the routing widths
        # use); the sketch then reserves a device-free rail row there so the
        # router's power trunk has a through-going corridor. The ENGINE is
        # untouched: the rail intent acts purely through the placement input.
        if ring_mode != "off":
            # Ring-aware placement bands by well-island, so the rails ride their
            # island bands (design-derived, shift-invariant) instead of the
            # reference rail extractor (whose top/bottom decision flipped under a
            # reference-bbox shift -- the 4.46um leakage).
            from clay.layout.extract.well_islands import island_rail_intents
            # F2: the rails must ride the SAME row stack the placement banded by
            # (compact-group order on the F2 path, else the well-island order).
            rail_intents = island_rail_intents(
                intent,
                order=(sk_order if str(ring_granularity) == "compact_group"
                       else None),
            )
        elif style_from_reference:
            rail_intents = extract_rail_intent(intent, tech_id=tech_id)
        else:
            # leakage-free CP path with rings OFF: extract_rail_intent reads the
            # reference GDS (rail side/trunk evidence), so it is NOT permitted
            # here. Use the topology rail intent (island_rail_intents); declared
            # intent.rails still take priority inside it. No reference read.
            from clay.layout.extract.well_islands import island_rail_intents
            rail_intents = island_rail_intents(intent, order=None)
        _w_by_net = {
            str(k).strip().rstrip("!").lower(): d
            for k, d in derive_width_intent(
                intent, tech, layer="metal1", policy=pol).items()
        }
        # corridor height = trunk width + clearance to BOTH walls (the PDK's
        # metal1 min spacing) -- a corridor that fits only the bare trunk
        # leaves zero slack for grid snap, and the widen pass clips. PDK DATA,
        # uniform for every rail net (no per-net special case).
        _m1_sp_nm = tech.min_space("metal1")
        _clear_um = 2.0 * (float(_m1_sp_nm) / _NM_PER_UM if _m1_sp_nm else 0.0)
        rails_for_sketch: List[Dict[str, Any]] = []
        for ri in rail_intents:
            if ri.get("side") in ("top", "bottom") or \
                    ri.get("between_rows") is not None:
                d = _w_by_net.get(str(ri["net"]).strip().rstrip("!").lower())
                entry = dict(ri)
                entry["width_um"] = (float(d.value) if d is not None
                                     else float(pol.power_min_um)) + _clear_um
                rails_for_sketch.append(entry)
        # R7 ASPECT (style face): the reference's device-region width/height
        # RATIO (a scale-free proportion, not a coordinate) sets the row-packing
        # target width. Consumed only on the self-declaring style path -- on the
        # leakage-free CP path device_bboxes carries the reference proportion, so
        # it is NOT read (aspect_target stays None; the CP placer uses the config
        # aspect band, never device_bboxes).
        aspect_target = None
        if style_from_reference:
            _abb = getattr(intent, "device_bboxes", None) or {}
            if _abb:
                _aw = max(b[2] for b in _abb.values()) - min(b[0] for b in _abb.values())
                _ah = max(b[3] for b in _abb.values()) - min(b[1] for b in _abb.values())
                if _ah > 0:
                    aspect_target = float(_aw) / float(_ah)
        # balance_rows: P3's objective-driven row-width balancing (design-only
        # objective -- minimise the hull estimate; matched groups protected;
        # with style_order present the row MEMBERSHIP is a hard constraint and
        # balance degrades to within-row work -- R3).
        #
        # CONTRACT matched_body_group.O-ringnonoverlap-fold: the 2D embed stage
        # (compact='2d' -> compaction.embed_search) re-slots partner-rigid groups
        # ACROSS rows to shrink the device outline. On the ring path the rows ARE
        # the WELL-ISLAND bands (sk_order = well_island_order); the embed search
        # is BLIND to the island partition, so under the device FOLD (every device
        # folded wide-flat, no composite stamp keeping a matched group rigid in
        # one row) it scatters an island's members across non-contiguous y-bands
        # -- e.g. an NMOS-GND member up at y~30 and a PMOS-VDD member down at
        # y~18. The per-island group ring then encloses that scattered y-extent,
        # and the GND substrate ring fuses with the VDD nwell ring (measured
        # 3226 um^2 M1 overlap -> FULL-extraction LVS merges GND+VDD). The
        # by-construction island banding (the same well-island partition P2 gives
        # the non-fold composite unit) is broken precisely because the area-driven
        # cross-row repack does not respect it.
        #
        # FIX: under the fold AND on the ring path, DISABLE the outline
        # compaction (compact=None) so the placement keeps the FULL-WIDTH
        # DISJOINT well-island band stack solve_hierarchical_nonoverlap /
        # _stack_bands produces by construction (row k+1's bottom sits a channel
        # above row k's PADDED bbox top, across the full row width). Both the 2D
        # embed AND the x-overlap-aware press are wrong for a BAND-granularity
        # ring: the ring is a full-island-width frame, but the compaction graph
        # constrains two stamps in y ONLY where they x-OVERLAP -- so it lets an
        # x-disjoint NMOS-GND member and PMOS-VDD member's PADDED boxes overlap
        # in y (the "embedding freedom"), and the two full-width island rings
        # then fuse even though no two stamps overlap. The group-ring-padded
        # extents (already in solve_sizes) reserve each band's ring room, and the
        # disjoint band stack keeps whole islands y-separated, so the per-island
        # rings are non-overlapping by construction. Gated on the fold+ring
        # combination only: canonical / composite / non-fold ring paths keep
        # compact='2d' byte-identically.
        _compact_mode: Any = "2d"
        if _fold_target is not None and ring_mode != "off":
            _compact_mode = None
        # F2 compact-group ring path: the 2D embed re-slots matched-group members
        # across rows (the F2 r1 contiguity blocker), fusing the tight per-group
        # rings. Disable the outline compaction so the band stack keeps each
        # compact group in its own row-band (X-contiguous, y-disjoint) -> per-
        # group rings non-overlapping by construction (same reasoning as the
        # fold+ring path above).
        if str(ring_granularity) == "compact_group" and ring_mode != "off":
            _compact_mode = None
        # S2 SEQPAIR OPT-IN: the sequence-pair longest-path engine replaces the
        # band-stack solve at THIS swap point (the SAME hier + solve_sizes input,
        # the SAME {name->(cx,cy)} output). The sketch itself only supplies the
        # design-sourced STRUCTURE (band membership + within-band order); the
        # seqpair engine OWNS the solve, so the sketch runs with engine='bands'
        # (its 'relax' refinement stage is what seqpair stands in for). Default
        # paths ('bands' / 'relax') are byte-unchanged (the seqpair branch below
        # never fires). _seqpair / _cp_envelope hoisted to the top of the fn.
        _sketch_engine = ("bands" if (_seqpair or _cp_envelope)
                          else str(placement_engine))
        # mirror_pairs_rigid is a RELAX-stage refinement face on the sketch; on
        # the cp path the CP envelope-min placer owns the rigid symmetry press
        # (cp_group_placement_pos below), so the sketch (engine='bands') must NOT
        # receive it (the bands sketch would reject it). Pass it only when the
        # sketch actually runs the relax stage.
        _sketch_rigid = bool(mirror_pairs_rigid) and _sketch_engine == "relax"
        sketch = _ts.derive_design_sketch(sk_intent, sk_sizes,
                                          style_order=sk_order,
                                          rails=rails_for_sketch or None,
                                          balance_rows=True,
                                          aspect_target=aspect_target,
                                          compact=_compact_mode,  # type: ignore[arg-type]
                                          engine=_sketch_engine,
                                          mirror_pairs_rigid=_sketch_rigid)
        # the rail pseudo rows must survive the band solve: give the solver
        # their sketch extents as sizes, then strip them from the placement.
        solve_sizes = dict(sk_sizes)
        for pseudo, _net, _pos_label in sketch.rail_reservations:
            bb = sketch.bboxes[pseudo]
            solve_sizes[pseudo] = (float(bb[2]) - float(bb[0]),
                                   float(bb[3]) - float(bb[1]))
        hier = _rg.build_hierarchical_graph(
            _ts.sketch_intent_view(sk_intent, sketch))
        # P-A INTRA-GROUP ABUT: on the compact-group ring path the per-device
        # group-ring padding (group_ring_padded_extents -> solve_sizes) is
        # applied symmetrically to EVERY member, so two members of the SAME
        # compact group (which share ONE ring -- the pad belongs at the group's
        # outer boundary, not between members) sit a full ring-pad apart and the
        # per-group ring window is loose (enclosure ceiling 0.562). Feed the band
        # solve the RAW (unpadded) stamp extents + the same-group pair set so
        # intra-group members separate by their true device spacing (abut), while
        # inter-group/inter-kind pairs keep the padded separation so the per-group
        # rings stay disjoint. Pure topology (compact_matched_groups) + the raw
        # drawn-stamp extents; no reference coordinate, byte-identical off-path.
        _tight_sizes = None
        _tight_pairs = None
        if (str(ring_granularity) == "compact_group" and ring_mode != "off"
                and _group_ring_base is not None):
            from clay.layout.extract.realize_render import (
                stamp_extents as _stamp_extents,
            )
            from clay.layout.extract.well_islands import (
                compact_matched_groups as _cmg,
            )
            _tight_sizes = dict(_stamp_extents(_group_ring_base))
            _tight_pairs = set()
            for _k, _b, _members in _cmg(intent):
                for _ii in range(len(_members)):
                    for _jj in range(_ii + 1, len(_members)):
                        _tight_pairs.add(frozenset((_members[_ii],
                                                    _members[_jj])))
        if _seqpair:
            # S2: pack the SAME design-sourced hier with the sequence-pair
            # longest-path engine. Mirror pairs / abut come from the DESIGN
            # topology (intent.mirror_pairs / @abut), NOT a reference coordinate
            # -- the leakage boundary is the sketch's (the hier already derives
            # band membership + order from the sketch intent view). Output is the
            # SAME {name->(cx,cy)} shape, consumed unchanged below.
            #
            # S4 STATUS (honest, measured 2026-06-17 on the EXACT oracle
            # om._run_cell('rings_group', placement_engine='seqpair')):
            # the ring-aware GROUP-BLOCK area search (solve_hier_seqpair with
            # compact_groups + tight_sizes) IS landed + its NAMED primary contract
            # HOLDS on the emitted GDS -- the per-group ring bboxes are DISJOINT
            # (ring_overlaps == 0, vs the S3 band fold that tiled them into a strip
            # and fused them), leakage 0, rings 9, route connected 13/13. BUT it is
            # DELIBERATELY NOT ENGAGED on this oracle path: the 2D group-block pack
            # is ring-disjoint yet NOT routable -- the FULL LVS goes UNMATCHED (many
            # devices extract layout=NONE; GND merges with NET048/NET07/NET08 and
            # NET10 with NET4/VDD) and deck DRC rises to 15, a REGRESSION past the S2
            # seed (which is LVS-matched with a 1-short bring-up). Ring-disjoint
            # holds; ROUTABILITY under the 2D block arrangement does not. The named
            # blocker: the 2D-packed group blocks place cross-group nets so the
            # mirror-routing + trunk router cannot form the correct nets (the block
            # adjacency the band-stack's full-width stack gives the router is lost).
            # Per the checker-first revert discipline (no hard-gate regression on the
            # production path), this oracle path keeps the S2 vertical-stack seed
            # order (area_search OFF). The ring-aware mechanism + its by-construction
            # ring-disjoint / symmetry / determinism / leakage unit contracts stay
            # (tests/test_seqpair_ring_aware.py) -- the disjointness is proven; the
            # routability of a 2D-packed analog OTA is the next phase's work
            # (router/block-adjacency aware order search), NOT a band-stack S5 swap.
            #
            # S5-PROBE (honest, measured 2026-06-17 on the EXACT oracle): wiring the
            # SAME single-source intra-group abut (tight_sizes + tight_pairs from
            # well_islands.compact_matched_groups + raw stamp extents -- byte-identical
            # to the band-stack tight path) into the seqpair SEED collapsed the
            # per-group double-ring-pad: the seqpair seed went area 20710 -> 18162
            # (-12%), DRC 1 -> 0, M1 spacing 1 -> 0, LVS still matched, leakage 0,
            # rings 9. STILL NOT S5-viable: area 18162 > band-stack 16555 (the
            # portrait vertical-stack order is intrinsically less compact than the
            # band-stack's balanced rows) AND the documented VOUT/net15 1-short
            # persists. S5 (swap + band-stack delete) requires the seqpair path to
            # be all-hard-green AND area-improved vs 16555; neither holds. The
            # seqpair path is improved + leakage-clean but remains a bring-up
            # diagnostic, NOT the production default. NO S5.
            from clay.layout.seqpair_placement import solve_hier_seqpair
            _sp_mirror = tuple(
                (str(p[0]), str(p[1])) for p in (
                    getattr(intent, "mirror_pairs", ()) or ())
                if len(p) == 2
            )
            _sp_abut = tuple(
                (str(a), str(b)) for (a, b) in (
                    getattr(intent, "abut_pairs", ()) or ())
            )
            _sp_order = tuple(
                (str(a), str(b), str(rel)) for (a, b, rel) in (
                    getattr(intent, "order_constraints", ()) or ())
            )
            pos = solve_hier_seqpair(
                hier, solve_sizes, tech=tech,
                mirror_pairs=_sp_mirror, abut_pairs=_sp_abut,
                order_constraints=_sp_order,
                tight_sizes=_tight_sizes, tight_pairs=_tight_pairs)
            _resid = 0.0
        elif _cp_envelope:
            # P2 CP ENVELOPE-MIN GROUP PLACEMENT (opt-in, placement_engine='cp'):
            # the discrete CP-SAT placer (clay.layout.synth.placement) places the
            # COMPACT MATCHED GROUPS to minimize the bounding-rectangle ENVELOPE,
            # N and P groups INTERLEAVED in a 2D pack (not the segregated band
            # stack). The group placement is expanded to the SAME device-level
            # {name->(cx,cy)} pos dict the band-stack/seqpair return, consumed
            # unchanged below -> SAME canonical materialization + emitted GDS, no
            # second path. Member footprints are the RAW (tight) stamp extents
            # (_tight_sizes), the leakage-free design footprint; no reference
            # coordinate is read (compact_matched_groups + mirror_pairs only).
            from clay.layout.synth.placement import cp_group_placement_pos
            _cp_sizes = dict(_tight_sizes or {})
            for _n, _wh in solve_sizes.items():
                _cp_sizes.setdefault(_n, _wh)
            # GROUP-RING RESERVATION: when the rings materialise at compact-
            # matched-group granularity below, the CP group footprint must
            # reserve the per-group ring pad so the rings stay DISJOINT (the
            # raw-stamp footprint dropped this obligation -> 15 ring overlaps).
            # Pass tech_id only on the ring path; the bare path keeps the
            # legacy raw footprint byte-identically.
            _cp_tech = (tech_id
                        if (ring_mode == "group"
                            and str(ring_granularity) == "compact_group")
                        else None)
            # E2 SYMMETRY-RIGID PRESS (opt-in, mirror_pairs_rigid on the cp
            # path): the CP placer ALWAYS binds CROSS-group declared mirror
            # pairs to one shared axis on equal rows (hard symmetry). The rigid
            # press additionally enforces SAME-group 2-member mirror pairs to be
            # exactly mirror-posed about their group centre (so every declared
            # diff/mirror pair -- not only the cross-group axis pairs -- is
            # placement-symmetric by construction). Pure topology (mirror_pairs);
            # no reference coordinate. Default off is byte-identical.
            # UNIFIED ALL-ELEMENT PLACEMENT: cp_group_placement_pos now packs the
            # MOS groups AND the passives (caps/resistors) as first-class tiles in
            # ONE envelope-min pack (net wirelength pulls each passive into the
            # device mosaic). No cap_over_cell / cap_planar_edge target step -- the
            # cap is just another element, co-placed, not bolted on.
            # ASPECT-BAND target: rotate the 2D pack toward the landscape floorplan
            # band CENTER (a config scalar -- clay.config.fidelity_thresholds, the
            # reference only calibrates the band edges, never a coordinate). General
            # over any circuit; None-safe (no term added when unset).
            from clay.config import fidelity_thresholds as _fth
            _fth = _fth()
            _cp_aspect = (float(_fth.aspect_lo) + float(_fth.aspect_hi)) / 2.0
            pos, _cp_result = cp_group_placement_pos(
                intent, _cp_sizes, tech_id=_cp_tech,
                mirror_pairs_rigid=bool(mirror_pairs_rigid),
                mirror_axis_mode=str(mirror_axis_mode),
                aspect_target=_cp_aspect)
            _resid = 0.0
        elif False:
            _sp_mirror = tuple(
                (str(p[0]), str(p[1])) for p in (
                    getattr(intent, "mirror_pairs", ()) or ())
                if len(p) == 2
            )
            _sp_abut = tuple(
                (str(a), str(b)) for (a, b) in (
                    getattr(intent, "abut_pairs", ()) or ())
            )
            _sp_order = tuple(
                (str(a), str(b), str(rel)) for (a, b, rel) in (
                    getattr(intent, "order_constraints", ()) or ())
            )
            pos = solve_hier_seqpair(
                hier, solve_sizes, tech=tech,
                mirror_pairs=_sp_mirror, abut_pairs=_sp_abut,
                order_constraints=_sp_order,
                tight_sizes=_tight_sizes, tight_pairs=_tight_pairs)
            _resid = 0.0
        else:
            pos, _resid = _rg.solve_hierarchical_nonoverlap(
                hier, solve_sizes, (0.0, 0.0),
                tight_sizes=_tight_sizes, tight_pairs=_tight_pairs)
        placement = {
            n: (float(p[0]), float(p[1])) for n, p in pos.items()
            if not str(n).startswith(_ts.RAIL_ROW_PREFIX)
        }
        # RAIL TRUNK CORRIDOR Y (design output): the solved centre of each
        # rail pseudo row. The reservation alone is placement-side only -- the
        # router has no reason to PUT the trunk there (measured: the side=top
        # VDD corridor sat empty while the trunk formed among its pins two
        # rows down). The lowering below turns each corridor into a routing
        # TARGET via two same-net anchor stubs at the placement extent ends,
        # so the trunk runs THROUGH the corridor full-width -- the reference's
        # edge-trunk + distribution-comb structure, from the side fact alone.
        rail_corridor_y = {
            net: (float(pos[pseudo][0]), float(pos[pseudo][1]),
                  float(solve_sizes[pseudo][0]))
            for pseudo, net, _pos_label in sketch.rail_reservations
            if pseudo in pos
        }
        if styled is not None:
            placement_provenance = sketch.provenance  # 'style-transfer'
            placement_note = (
                "style-conditioned placement: reference RELATIVE order (names "
                f"only) consumed, alignment coverage {styled.coverage:.2f}; "
                "this placement makes NO design-only placement claim"
            )
        else:
            # leakage-free CP path: no reference order was read; the placement
            # is design-only (CP envelope-min over topology + PDK footprints).
            placement_provenance = "design-only"
            placement_note = (
                "leakage-free CP envelope-min placement: no reference read "
                "(groups+passives placed by size+nets; symmetry from declared "
                "mirror pairs); style_from_reference OFF"
            )
        # P1 RELAX honest self-report passthrough: the engine='relax' stage
        # softens the style bands; every device it pulled off its band is
        # surfaced here (the user/upper layer decides the trade), and the
        # full RelaxReport rides the result under 'relax_report'.
        relax_report = getattr(hier, "relax_report", None)
        if relax_report is not None:
            placement_note += (
                f"; relax engine: {relax_report.selected}, "
                f"{len(relax_report.band_deviations)} band deviation note(s)"
            )
            # P1b mirror-rigid self-declaration: how many declared pairs
            # were rigidified (and whether each realized its posture).
            _mrp = tuple(getattr(relax_report, "mirror_pairs", ()) or ())
            if _mrp:
                _n_real = sum(1 for d in _mrp if d.get("realized"))
                placement_note += (
                    f", {len(_mrp)} mirror-rigid pair(s) "
                    f"({_n_real} realized)"
                )

    # S3 DIRECTION INTENT (style face): measure the reference's per-layer H/V
    # composition (relative ratios, zero coordinates) and turn each >=0.6-purity
    # layer's majority axis into an explicit Direction soft preference. Style-
    # channel input -> only reachable on the self-declaring style path (the
    # ValueError gate above). Default OFF appends nothing: byte-stable.
    direction_report: Optional[Dict[str, Any]] = None
    direction_constraints: List[Any] = []
    if direction_from_reference:
        from clay.layout.extract.direction_intent import extract_direction_intent
        from clay.layout.route import constraints as _C

        direction_report = extract_direction_intent(
            _reference_gds_path(), _REF_TECH)
        direction_constraints = [
            _C.Direction(layer_n=int(row["layer_n"]), axis=str(row["axis"]),
                         weight=float(row["weight"]))
            for row in direction_report["directions"]
        ]
    # correct the cap placement so each cap's bbox centre lands at the placer's intended
    # centroid (caps built by build_capacitor are not bbox-centre-relative) -- prevents the
    # cap-cap M3-cover overlap short LVS caught.
    placement = _recenter_cap_placement(intent, placement, tech=tech)

    widths = derive_width_intent(intent, tech, layer="metal1", policy=pol)
    routing_intent = RoutingIntent(cell="OTA_CS_Q", widths=widths,
                                   direction_weight=float(direction_weight))
    if not routing_intent.is_leakage_free():
        raise AssertionError(
            f"routing intent not leakage-free on {tech_id}: "
            f"{routing_intent.leakage_count()} reference-sourced value(s)"
        )

    if _ringed_geometry is not None:
        geometry = _ringed_geometry
    elif _group_ring_base is not None:
        # 'group' mode: the bare stamps built at the sizes stage (byte-equal
        # to the default build); the rings materialise post-placement below.
        geometry = _group_ring_base
    else:
        try:
            geometry = real_device_geometry(intent, ref_gds=None,
                                            tech_id=tech_id,
                                            native_passive=True,
                                            row_height_target_um=_fold_target,
                                            contact_rows=dense_contacts)
        except Exception:
            geometry = real_device_geometry(intent, ref_gds=None,
                                            tech_id=tech_id,
                                            row_height_target_um=_fold_target,
                                            contact_rows=dense_contacts)

    # T2: a composited group is ONE stamp -- its members leave the geometry
    # dict (they have no placement of their own) and the group stamp enters
    # under the group key the placement solved.
    if comp_groups:
        for g in comp_groups:
            for m in g["members"]:
                geometry.pop(m, None)
            geometry[g["key"]] = g["geom"]

    # GROUP GUARD RINGS ('group' mode): built from the FINAL placement
    # (recentred caps included) on the final geometry dict (composite stamps
    # participate through their members' design data), then materialised INTO
    # that geometry dict as placement-absolute ring stamps. Everything
    # downstream consumes them through the EXISTING single paths: the router
    # sees the ring frames as unowned hard keep-outs via build_layout_model's
    # projection, the emit boundary writes them through the same device loop,
    # and the LVS/DRC oracles judge the same artifact -- no second
    # materialization path, no router-obstacle blindness.
    group_ring_rows: List[Dict[str, Any]] = []
    group_ring_notes: List[str] = []
    if ring_mode == "group":
        from clay.layout.extract.library_device import CellGeometry as _CG
        from clay.layout.extract.realize_render import group_guard_rings
        from clay.layout.extract.well_islands import well_islands

        # Ring clustering consumes the WELL-ISLAND membership the placement was
        # banded by (topology, kind+body net) -- not a re-derivation from placed
        # bboxes -- so the cluster set is FOLD-INVARIANT (P1 device folding can
        # change device extents without flipping the ring count/overlap).
        #
        # F2 NOTE (geometric-fidelity ring-granularity, 2026-06-16, r2 LANDED):
        # ring_granularity='compact_group' drives BOTH the placement banding
        # (compact_matched_group_order above) AND these ring clusters from the
        # SAME compact_matched_groups partition (~9 tight per-group rings, the
        # reference's granularity). The F2 r1 contiguity blocker is SOLVED: the
        # placement bands by compact group (each X-contiguous) and the per-group
        # ring path disables the 2D outline compaction (so groups stay y-disjoint)
        # -- measured on the FULL oracle: ring_count 3 -> 9 (disjoint), enclosure
        # 0.337 -> 0.562, ALL hard gates green (FULL LVS 0/0, deck DRC 0, route
        # 13/13/0/0/0). ring_granularity='island' (the default) keeps the coarse
        # (kind,body) band rings (3 rings) byte-identically. Honest residual: the
        # compact_group path is RED on enclosure 0.562 < ~0.85 and portrait aspect
        # (the F0 checker reports it -- the remaining blocker is intra-group abut +
        # landscape y-disjoint same-row packing, named in tests/test_option_matrix.
        if str(ring_granularity) == "compact_group":
            # F2: cluster the rings from the SAME compact-matched-group
            # partition the placement banded by (one source -- no second path,
            # no drift). Each compact group -> one tight ring around its
            # members' placed bboxes; the placement kept each group X-contiguous
            # so the per-group rings are disjoint by construction.
            from clay.layout.extract.well_islands import compact_matched_groups
            _islands = [members for (_k, _b, members)
                        in compact_matched_groups(intent)]
        else:
            _islands = [members for (_k, _b, members) in well_islands(intent)]
        _rings, group_ring_notes = group_guard_rings(
            geometry, placement, intent, tech_id,
            island_clusters=_islands or None)
        for _i, _gr in enumerate(_rings):
            _key = f"GRING{_i}"
            _polys: Dict[Tuple[int, int], List[List[Tuple[float, float]]]] = {}
            for _lp in _gr.ring.polygons:
                _polys.setdefault(
                    (int(_lp.layer), int(_lp.datatype)), []
                ).append([(float(v.x) / _NM_PER_UM, float(v.y) / _NM_PER_UM)
                          for v in _lp.polygon.vertices])
            _cg = _CG(name=_key, kind="guard_ring", ref_centroid=(0.0, 0.0),
                      polys_by_layer=_polys)
            _cg.source = "group_guard_ring"
            geometry[_key] = _cg
            placement[_key] = (0.0, 0.0)
            group_ring_rows.append({
                "ring": _key,
                "members": list(_gr.members),
                "kind": _gr.kind,
                "well": _gr.well,
                "tap_net": _gr.tap_net,
                "row": int(_gr.row),
                "ring_bbox_nm": tuple(int(v) for v in _gr.ring.ring_bbox),
            })

    # pass geometry= so a passive with no parametric cell still gets its p/n pins from its
    # OWN placed device-stamp contacts (leakage-free device-cell artwork, not a routed wire).
    # T2: composited members are excluded here -- their pins come from the
    # composite cell's own per-device terminal access right below.
    pin_intent: Any = intent
    if comp_groups:
        pin_intent = _IntentSansDevices(
            intent, {m for g in comp_groups for m in g["members"]}
        )
    pin_report = extract_pins(pin_intent, placement, tech=tech,
                              return_report=True,
                              include_power=True, geometry=geometry,
                              row_height_target_um=_fold_target)
    pins_by_net = pin_report.pins_by_net
    if comp_groups:
        _composite_member_pins(comp_groups, placement, pins_by_net,
                               pin_report, tech=tech)

    # R4 IO EDGE STUBS (style face): the reference brings every LABELLED IO
    # net to a block edge (VIP+VIN enter together on the left, VOUT exits
    # right). The consumed fact is CATEGORICAL (net -> edge side + adjacency
    # group; extract_io_edge_intent emits zero numeric leaves); the stub's
    # POSITION is DESIGN-derived: x at the design placement extent on the
    # declared side, y of the net's own nearest existing pin (its device
    # terminal row). The stub is one extra same-net Pin -- the router wires
    # it like any terminal, no engine change.
    io_edge_facts: List[Dict[str, Any]] = []
    if style_from_reference and pins_by_net:
        from clay.layout.extract.style_order import extract_io_edge_intent
        from clay.layout.route.engine import Pin as _Pin
        from clay.layout.route.wire_endpoints import declared_wire_endpoint_pin

        try:
            io_edge_facts = extract_io_edge_intent(intent)
        except ValueError:
            io_edge_facts = []
        if io_edge_facts:
            _px0 = min(p.xy_um[0] for ps in pins_by_net.values() for p in ps)
            _px1 = max(p.xy_um[0] for ps in pins_by_net.values() for p in ps)
            _py0 = min(p.xy_um[1] for ps in pins_by_net.values() for p in ps)
            _py1 = max(p.xy_um[1] for ps in pins_by_net.values() for p in ps)
            # group-stub pitch: 8x min-width. 3x (0.84um on external_pdk M1) put the two
            # foreign stub pads inside each other's emit-clip halo and left
            # 7-22nm slivers (the 4xM1.S.1+2xM1.W.1 cluster at the stub site);
            # 8x keeps the pads outside the clip interaction while still
            # ADJACENT (reference labels sit 1.3um apart -- the consumed fact
            # is the grouping, not a distance).
            _pitch_um = 8.0 * float(tech.min_width("metal1") or 280) / _NM_PER_UM

            def _design_net(name: str) -> Optional[str]:
                want = str(name).strip().rstrip("!").upper()
                return next((n for n in pins_by_net
                             if n.strip().rstrip("!").upper() == want), None)

            # STUB KEEPOUT CONTRACT (named 2026-06-12 by the 2D x-capacity
            # round): the stub lowering was KEEPOUT-BLIND -- it placed the
            # M1 stub at the bare pin-extent edge, and a legal x-press that
            # narrows the extent onto a cap's @no_route_under footprint put
            # the stub INSIDE an all-net hard keep-out = an unreachable pin
            # (VOUT 12/13, cured at +3.06um in the agent's control
            # experiment). The stub must land on ROUTABLE space: when the
            # candidate point falls inside a forbidden-metal cap footprint,
            # slide ALONG the edge (deterministic min-pitch steps, outward
            # first) to the nearest clear point; no clear point within the
            # extent -> skip the stub with an honest note (the edge fact
            # stays declared, unrealised).
            _keepout_boxes: List[Tuple[float, float, float, float]] = []
            if getattr(intent, "no_route_under", None):
                try:
                    from clay.layout.extract.under_cap_intent import (
                        cap_footprints_um,
                    )
                    _fp = cap_footprints_um(intent, placement, geometry,
                                            tech=tech)
                    _keepout_boxes = [tuple(map(float, bb))
                                      for bb in _fp.values()]
                except Exception:
                    _keepout_boxes = []

            def _clear_of_keepouts(x: float, y: float) -> bool:
                return not any(bx0 <= x <= bx1 and by0 <= y <= by1
                               for (bx0, by0, bx1, by1) in _keepout_boxes)

            def _slide_clear(x: float, y: float, axis: str,
                             lo: float, hi: float) -> Optional[Tuple[float, float]]:
                if _clear_of_keepouts(x, y):
                    return (x, y)
                step = _pitch_um
                for k in range(1, 256):
                    for sgn in (1.0, -1.0):
                        if axis == "y":
                            cy = y + sgn * k * step
                            if lo <= cy <= hi and _clear_of_keepouts(x, cy):
                                return (x, cy)
                        else:
                            cx = x + sgn * k * step
                            if lo <= cx <= hi and _clear_of_keepouts(cx, y):
                                return (cx, y)
                return None

            # the ADJACENCY GROUP is a grouping fact: the group's stubs sit
            # TOGETHER. The shared coordinate is DESIGN-derived -- the y (or x)
            # of the member pin that already sits closest to the declared edge
            # (the cheapest entry); other members stack at min-pitch offsets.
            placed_groups: set = set()
            for fact in io_edge_facts:
                edge = str(fact["edge"])
                group = tuple(sorted(fact.get("adjacent_group") or [fact["net"]]))
                if (edge, group) in placed_groups:
                    continue
                placed_groups.add((edge, group))
                members = [(_design_net(g), g) for g in group]
                members = [(n, g) for n, g in members if n and pins_by_net[n]]
                if not members:
                    continue
                if edge in ("left", "right"):
                    ex = _px0 if edge == "left" else _px1
                    key = (min if edge == "left" else max)
                    anchor_net = key(
                        members,
                        key=lambda t: key(p.xy_um[0] for p in pins_by_net[t[0]]))[0]
                    ay = float(key(pins_by_net[anchor_net],
                                   key=lambda p: p.xy_um[0]).xy_um[1])
                    for k, (n, _g) in enumerate(sorted(members)):
                        spt = _slide_clear(ex, ay + k * _pitch_um, "y",
                                           _py0, _py1)
                        if spt is None:
                            io_edge_facts = list(io_edge_facts)
                            continue  # honest skip: no routable stub point
                        pins_by_net[n] = list(pins_by_net[n]) + [
                            declared_wire_endpoint_pin(n, "__io_edge__", edge, spt, m1, tech)]
                else:
                    ey = _py0 if edge == "bottom" else _py1
                    key = (min if edge == "bottom" else max)
                    anchor_net = key(
                        members,
                        key=lambda t: key(p.xy_um[1] for p in pins_by_net[t[0]]))[0]
                    ax = float(key(pins_by_net[anchor_net],
                                   key=lambda p: p.xy_um[1]).xy_um[0])
                    for k, (n, _g) in enumerate(sorted(members)):
                        spt = _slide_clear(ax + k * _pitch_um, ey, "x",
                                           _px0, _px1)
                        if spt is None:
                            continue  # honest skip: no routable stub point
                        pins_by_net[n] = list(pins_by_net[n]) + [
                            declared_wire_endpoint_pin(n, "__io_edge__", edge, spt, m1, tech)]

    # RAIL TRUNK ANCHORS (EXPERIMENT, default OFF -- measured 2026-06-12): a
    # LINE of same-net stubs along the corridor's solved y, meant to pull the
    # trunk THROUGH the reserved corridor (the reference's edge-trunk shape).
    # Measured failures: 2 endpoint stubs -> the Steiner tree drops each
    # endpoint to the comb and the corridor stays empty; densified stubs
    # (every ~12um) -> the stub/drop interfaces add 4x M1.S.1 + 2x M1.W.1
    # (the terminal-landing emit-clip family). Forcing trunks with pin lines
    # FIGHTS the router; the honest mechanism is an engine PREROUTE contract
    # (same-net seed geometry the router connects to -- what production
    # routers do for power; same gap blocks consuming trunk_layer_class=M2).
    # Opt-in for experiments only; the corridor reservation + side fact stay
    # live on the default style path.
    if rail_trunk_anchors and rail_corridor_y and pins_by_net:
        from clay.layout.route.engine import Pin as _PinR
        _sp_um = float(tech.min_space("metal1") or 230) / _NM_PER_UM
        for net, (ccx, ccy, cw) in sorted(rail_corridor_y.items()):
            dn = next((n for n in pins_by_net
                       if n.strip().rstrip("!").upper()
                       == str(net).strip().rstrip("!").upper()), None)
            if dn is None or not pins_by_net[dn]:
                continue
            half = max(0.0, cw / 2.0 - _sp_um)
            n_anchor = max(2, int((2 * half) / 12.0) + 1)
            step = (2 * half) / (n_anchor - 1) if n_anchor > 1 else 0.0
            extra = [
                _PinR(dn, "__rail_anchor__", f"a{k}",
                      (ccx - half + k * step, float(ccy)), m1)
                for k in range(n_anchor)
            ]
            pins_by_net[dn] = list(pins_by_net[dn]) + extra
    # NOTE: the production route does NOT collapse resistor nets (that would physically short
    # the poly resistor and rename real nets). The poly resistors RR0/RR5 keep their two real
    # nets (net08!=gnd!, net25!=net15). They appear UNMATCHED in LVS only because the
    # synthesized deck does not yet EXTRACT poly resistors -- the principled fix is deck
    # resistor extraction (then layout net08-R-gnd! pairs the schematic R device), NOT a
    # routing-side collapse hack. Tracked as the resistor root fix.

    # MIM caps are wired by the ROOT mechanism (NOT the reverted per-case Metal2 hack
    # that shorted both plates onto one node): clay.layout.route.pins.extract_pins lands
    # each plate's terminal on its OWN DISTINCT plate node + routing layer via
    # clay.layout.device_terminal.extract_terminal_access + the PDK MIM spec DATA. A cap
    # whose plates the extractor reports COLLAPSED/unmapped (or a PDK with no cap device
    # cell -- external_pdk/external_pdk) is left in uncovered_terminals (honest gap), never faked. The
    # electrical bar (the two plates stay two nodes, on the right nets) is measured below
    # by _verify_cap_plates on the routed artwork.
    # Build the UNIFIED net-tagged LayoutModel ONCE (rule 6: the router's physical world-view
    # is a single model, not ad-hoc obstacle builders). It carries the device M1 pads AND the
    # MIM cap M2/M3 PLATES as net-tagged conductors. Cap plates are conductors: a foreign net
    # routed across a plate shorts to it -- a real KLayout-LVS short the per-net connectivity
    # check never sees -- so each plate is a net-aware keep-out (foreign nets route around; the
    # plate's own net may land on it). The router consumes the model's obstacle projection.
    from clay.layout.model import build_layout_model

    # Native MIM terminals can both exit on M2 while the actual cover is on
    # M3 (or higher). Access-tip obstacles alone do not protect that body.
    # Carry its literal per-layer conductors and cuts through the same model;
    # the router must not discover those already drawn shapes only at emit.
    layout_model = build_layout_model(intent, placement, geometry, tech=tech,
                                      pins_by_net=pins_by_net,
                                      include_upper_artwork=True)
    obstacles = layout_model.router_obstacles()
    if comp_groups:
        # the composite terminal-comb TRUNKS are metal2 conductors the M1
        # device-pad projection does not carry -- net-aware keep-outs, same
        # contract as the cap plates.
        obstacles = obstacles + _composite_trunk_obstacles(comp_groups,
                                                           placement)

    # declared face: RoutingIntent.direction_weight -> lower_intent (the
    # conventional HVH table; 0.0 emits nothing -> the lowered list is
    # byte-unchanged). Style face: the extracted per-layer Direction soft
    # preferences are appended AFTER the lowered list -- a later Direction per
    # layer wins (constraints.Direction contract), so the measured reference
    # axes override the convention where both are present.
    #
    # The style face carries the conventional base so no layer is a penalty-
    # free sink: a command on the measured layers only would REPEL long runs
    # onto the unmeasured ones. HONEST LIMIT (measured 2026-06-11): the base
    # closes the cost-model hole but does NOT yet yield a purity improvement
    # on the congested test4 block -- PathFinder's escalating negotiation
    # costs dominate the static step penalty at every weight tried (off=0.768
    # vs w=0.25/0.5/1.0 -> 0.746/0.609/0.295; M4 absorbed the overflow even
    # while V-commanded). Engine gap named in test_direction_intent.py's
    # strict xfail; the uncongested obedience contract holds (probe tests).
    base_weight = float(routing_intent.direction_weight)
    if direction_constraints and base_weight <= 0.0:
        base_weight = max(float(c.weight) for c in direction_constraints)
    constraints = lower_intent(routing_intent, direction_weight=base_weight)
    if direction_constraints:
        constraints = list(constraints) + direction_constraints
    # RAIL TRUNK PREROUTES (the engine preroute contract, v0.4.29): each rail
    # with a reserved corridor gets its trunk as SAME-NET SEED GEOMETRY -- a
    # full-corridor-width bar at the corridor's solved y, on the layer the
    # reference evidence names (trunk_layer_class, the M2 style fact), at the
    # rail's DESIGN width class. The router connects every pin to the trunk
    # (the reference's edge-trunk + distribution-comb structure); foreign nets
    # treat it as owned metal. This replaces the falsified pin-line forcing
    # (rail_trunk_anchors, kept opt-in as the experiment record).
    preroutes: Dict[str, List[Any]] = {}
    if rail_corridor_y and pins_by_net:
        from clay.layout.backend import LayerPolygon as _LP
        from clay.layout.backend import Polygon as _Poly

        _layer_class_by_net = {
            str(ri.get("net", "")).strip().rstrip("!").upper():
                ri.get("trunk_layer_class")
            for ri in rail_intents
        }
        _w_by_norm = {
            str(k).strip().rstrip("!").lower(): d
            for k, d in widths.items()
        }
        _sp_um = float(tech.min_space("metal1") or 230) / _NM_PER_UM
        for net, (ccx, ccy, cw) in sorted(rail_corridor_y.items()):
            dn = next((n for n in pins_by_net
                       if n.strip().rstrip("!").upper()
                       == str(net).strip().rstrip("!").upper()), None)
            if dn is None or not pins_by_net[dn]:
                continue
            lvl = _layer_class_by_net.get(
                str(net).strip().rstrip("!").upper()) or 1
            try:
                lkey = tuple(tech.metal_key(int(lvl)))
            except Exception:
                lkey = m1
            d = _w_by_norm.get(str(net).strip().rstrip("!").lower())
            tw_um = float(d.value) if d is not None else float(pol.power_min_um)
            # ADAPTIVE trunk length (design-derived, like the reference's
            # unequal rails): the trunk spans THIS net's own pin x-extent plus
            # one trunk-width of margin per end, clamped to the corridor. The
            # reference's GND (66.8um, devices-only span) vs VDD (94.2um,
            # near-full) asymmetry falls out of the same rule applied to its
            # own pins -- nothing is read from the reference geometry.
            _pin_xs = [float(p.xy_um[0]) for p in pins_by_net[dn]]
            half_w = max(0.0, cw / 2.0 - _sp_um)
            lo = max(ccx - half_w, min(_pin_xs) - tw_um)
            hi = min(ccx + half_w, max(_pin_xs) + tw_um)
            x0 = int(round(lo * _NM_PER_UM))
            x1 = int(round(hi * _NM_PER_UM))
            y0 = int(round((ccy - tw_um / 2.0) * _NM_PER_UM))
            y1 = int(round((ccy + tw_um / 2.0) * _NM_PER_UM))
            if x1 > x0 and y1 > y0:
                preroutes[dn] = [_LP(layer=int(lkey[0]), datatype=int(lkey[1]),
                                     polygon=_Poly.rectangle(x0, y0, x1, y1))]

    # MIRROR ROUTING (P1, opt-in): symmetric-net pair constraints from the
    # design's mirror/diff pairs (netlist topology) + the SOLVED placement axis
    # (design output -- never a reference coordinate). The engine routes the
    # primary and mirrors/guides the partner; any residual asymmetry is
    # QUANTIFIED in mirror_reports (an unsatisfied Mirror keeps route_success
    # False -- honest, never a claimed symmetry). Default False appends
    # nothing: byte-stable.
    mirror_cons: List[Any] = []
    mirror_notes: List[str] = []
    if mirror_routing:
        # P1b solved-axis supply: every REALIZED mirror-rigid pair of the
        # relax placement report hands its solved axis to the constraint
        # builder (design output; an unrealized pair supplies nothing and
        # falls back to the measured-candidacy gate).
        solved_axes: Dict[Tuple[str, str], float] = {}
        if relax_report is not None:
            for mp in getattr(relax_report, "mirror_pairs", ()) or ():
                if mp.get("realized"):
                    solved_axes[tuple(sorted((str(mp["a"]), str(mp["b"]))))] \
                        = float(mp["axis_x_um"])
        mirror_cons, mirror_notes = _mirror_route_constraints(
            intent, placement, pins_by_net, _sizes,
            solved_axes=solved_axes or None)
        constraints = list(constraints) + mirror_cons

    # PLANAR UPPER METALS (opt-in, AL2): the upper-metal-not-over-active
    # obligation lowered into a routing PREFERENCE. The device active-diffusion
    # rects (Clay's own OD geometry) become a SOFT over-active cost on M3+ so the
    # router prefers PLANAR channels for its long upper-metal runs (the reference's
    # planar style), falling back to over-active only when no planar channel exists
    # (soft -> connectivity never traded). M1/M2 (local interconnect) pay nothing.
    # The penalty is derived from the obligation, NOT a per-net constant.
    # Default off -> byte-identical route.
    _oa_rects: List[Tuple[float, float, float, float]] = []
    _oa_penalty = 0.0
    if planar_upper_metals:
        _oa_rects = _device_od_rects_um(intent, placement, geometry)
        _oa_penalty = _OVER_ACTIVE_PENALTY
    from clay.layout.route.wire_endpoints import bind_native_endpoint_access

    pins_by_net = bind_native_endpoint_access(pins_by_net, obstacles)
    routed = route_prepared_cell(
        pins_by_net,
        obstacles,
        constraints,
        tech,
        trunk_layers=trunk_layers,
        over_active_penalty=_oa_penalty,
        over_active_min_level=_UPPER_METAL_MIN_LEVEL,
        over_active_rects_um=_oa_rects or None,
        preroutes=preroutes or None,
    )

    # reference-free verification: connectivity + merged-geometry DRC + via-aware opens.
    conn = _check_connectivity(routed, list(pins_by_net))
    drc = _check_drc(routed, obstacles, tech, _via_layer_set(tech), 0.001)
    geometric_open_nets = _geometric_open_nets(routed, tech)

    # the cap ELECTRICAL bar: each cap's two plates connect distinct (not shorted).
    cap_plate_rows = _verify_cap_plates(intent, placement, routed, tech=tech)
    # the anti-short bar (LVS-found): NO foreign net's metal may overlap a cap plate.
    max_foreign_over_cap_plate_um2 = _foreign_metal_over_cap_plates(
        intent, placement, routed, tech=tech)
    # cap-cap overlap (LVS-found bug #4): two different-net cap covers must not overlap.
    max_cap_cap_overlap_um2 = _cap_cap_overlap_um2(intent, placement, tech=tech)

    # the power trunk may ride an upper metal (trunk_layer_class); measure the
    # trunk over ALL metal layers, not M1 alone (else an M2-trunk power net with
    # no wide M1 comb reads as signal width -- the net10 case).
    _metal_keys = set()
    for _n in range(1, 7):
        try:
            _metal_keys.add(tuple(tech.metal_key(_n)))
        except Exception:
            break
    rail_rows: List[Dict[str, Any]] = []
    for net, nr in routed.nets.items():
        if classify_net_role(net, intent) != "power":
            continue
        tw = _routed_trunk_width_um(nr, _metal_keys or {m1})
        rail_rows.append({"net": net, "routed_width_um": round(tw, 4),
                          "class": _width_class(tw, is_power=False)})
    rails_power_class = bool(rail_rows) and all(
        r["class"] == "power" for r in rail_rows if r["routed_width_um"] > 0
    )

    written = None
    if out_gds:
        written = _emit_routed_gds(routed, out_gds, cell_name=f"amp_{tech_id}")

    result = {
        "tech_id": tech_id,
        # how the PLACEMENT was sourced: 'design-only' (leakage-free path) or
        # 'style-transfer' (reference relative order consumed -- self-declared,
        # never presented as a leakage-free placement).
        "placement_provenance": placement_provenance,
        "placement_note": placement_note,
        # P1 relax engine self-report (None on engine='bands'): the honest
        # off-band deviation notes + selection account of the relax stage.
        "relax_report": relax_report,
        # leakage_count below is the ROUTING-WIDTH leakage account (widths stay
        # design-derived on both placement paths); it is not a placement claim.
        "leakage_count": routing_intent.leakage_count(),
        "n_required_nets": conn.detail.get("n_required_nets", len(pins_by_net)),
        "n_connected": conn.detail.get("n_connected", 0),
        "n_shorts": conn.detail.get("n_shorts", 0),
        "n_spacing_violations": drc.detail.get("n_spacing_violations", 0),
        # R4 net-agnostic pre-emit spacing diagnostic (same-net + cross-net merged
        # space_check; -1 = not measured). Catches same-net non-abutting sub-min gaps
        # the per-net-pair count is blind to; the post-emit authority is
        # metal_min_spacing_check on the emitted GDS.
        "n_spacing_violations_netagnostic": drc.detail.get(
            "n_spacing_violations_netagnostic", -1),
        "worst_spacing_pair": drc.detail.get("worst_spacing_pair"),
        "wire_over_device_overlap_um2": drc.detail.get("wire_over_device_overlap_um2", 0.0),
        "drc_method": drc.detail.get("method"),
        "geometric_open_nets": geometric_open_nets,
        "rail_rows": rail_rows,
        "rails_power_class": rails_power_class,
        # S4 structural rail intents (RELATIVE facts only; [] on the default
        # path -- rail consumption is the style path's parameter face).
        "rail_intents": rail_intents,
        # R4 IO edge facts consumed (categorical only; [] off the style path).
        "io_edge_intents": io_edge_facts,
        # T2 composite matched-group stamps consumed ([] unless
        # composite_stamps=True on the style path) + the honest skip notes.
        "composite_stamps": [
            {"group": g["key"], "members": list(g["members"]),
             "row": g["row"], "sequence": g["cell"].sequence,
             "note": g["cell"].note}
            for g in comp_groups
        ],
        "composite_notes": comp_notes,
        # guard-ring face: the normalised granularity mode + the built GROUP
        # ring rows ('device' rings live inside each stamp's own geometry and
        # report nothing here; 'off' reports mode='off' + []).
        "guard_ring_mode": ring_mode,
        "guard_rings": group_ring_rows,
        "guard_ring_notes": group_ring_notes,
        # S3 direction intent report (RELATIVE ratios only; None unless
        # direction_from_reference -- self-declares provenance='style-transfer').
        "direction_intent": direction_report,
        # P1 mirror routing (opt-in): the produced Mirror constraints (design
        # pairs + solved-placement axis) and the engine's honest per-pair
        # asymmetry reports ([] when mirror_routing=False).
        "mirror_routing": bool(mirror_routing),
        "mirror_constraints": [
            (c.net_a, c.net_b, round(float(c.axis_x_um), 4))
            for c in mirror_cons
        ],
        "mirror_notes": mirror_notes,
        "mirror_reports": list(getattr(routed, "mirror_reports", []) or []),
        "route_success": bool(routed.success),
        "uncovered_terminals": [(d, t, n) for (d, t, n, _r) in pin_report.uncovered],
        "wired_passive_cap_terminals": [
            (p.device, p.terminal, p.net)
            for pins in pins_by_net.values() for p in pins
            if str(p.device).upper().startswith(("CC", "CR"))
        ],
        "cap_plate_rows": cap_plate_rows,
        "max_foreign_over_cap_plate_um2": max_foreign_over_cap_plate_um2,
        "max_cap_cap_overlap_um2": max_cap_cap_overlap_um2,
        "caps_all_plates_distinct": bool(cap_plate_rows) and all(
            r["plates_distinct"] for r in cap_plate_rows
        ),
        "caps_all_plates_right_net": bool(cap_plate_rows) and all(
            r["plates_distinct"] and r["p_connects_right_net"]
            and r["n_connects_right_net"]
            for r in cap_plate_rows
        ),
        "design_widths_um": {n: round(float(d.value), 4) for n, d in widths.items()},
        # @edge_anchor obligation (None unless the intent declares
        # edge_anchor_devices): the V-side verdict measured on THIS route's
        # canonical artwork (5-state -- the cap's open side carries no active).
        # The placement-side L (the edge-anchor potential) acts only on the
        # relax engine; this report records the resulting obligation state.
        "edge_anchor_obligation": _edge_anchor_obligation_report(
            intent, placement, geometry, tech, routed=routed),
        "out_gds": written,
        # CANONICAL AS-BUILT BUNDLE: the live route objects, so every independent checker
        # (lvs_gate, the electrical gate, the P6 reconcile) verifies the EXACT layout this
        # production route built -- never a re-derived/divergent copy. Codex review: the gate
        # had been re-building a non-recentered, differently-collapsed route, i.e. checking a
        # DIFFERENT layout. This eliminates that path drift.
        "_bundle": AsBuiltBundle(
            routed=routed,
            geometry=geometry,
            placement=placement,
            pins_by_net=pins_by_net,
            obstacles=obstacles,
            tech=tech,
            intent=intent,
            layout_model=layout_model,
        ),
    }
    # rail OBLIGATION verdicts from the measured evidence above (never a
    # silent pass: unrouted rail intents report 'unmeasurable').
    result["rail_obligations"] = rail_obligation_report(result)
    return result


def rail_obligation_report(r: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Per-power-net rail OBLIGATION verdicts from a route result dict.

    Obligation: every rail net (routed power net or declared/extracted rail
    intent) must be geometrically CONTINUOUS (one connected component in the
    via-aware ``_geometric_open_nets`` check) at the power width class.
    Verdicts are honest 3-state -- ``'pass'`` / ``'fail'`` / ``'unmeasurable'``
    (a rail intent with no routed evidence is unmeasurable, never a silent
    pass). NOTE on independence: the continuity evidence is the route
    subsystem's own geometric check, NOT a signoff oracle -- KLayout LVS/DRC
    remain the independent oracles; this report never substitutes for them.
    """
    opens = set(r.get("geometric_open_nets") or [])
    rows = {str(row.get("net")): row for row in (r.get("rail_rows") or [])}
    intents = {
        str(ri.get("net")): ri for ri in (r.get("rail_intents") or [])
        if isinstance(ri, dict) and ri.get("net")
    }
    out: Dict[str, Dict[str, Any]] = {}
    for net in sorted(set(rows) | set(intents)):
        row = rows.get(net)
        ri = intents.get(net)
        routed_flag = row is not None
        continuous = routed_flag and net not in opens
        power_class = bool(row and row.get("class") == "power")
        if not routed_flag:
            verdict = "unmeasurable"
        elif continuous and power_class:
            verdict = "pass"
        else:
            verdict = "fail"
        out[net] = {
            "routed": routed_flag,
            "continuous": continuous,
            "width_class": row.get("class") if row else None,
            "power_class": power_class,
            "side": ri.get("side") if ri else None,
            "between_rows": ri.get("between_rows") if ri else None,
            "intent_source": ri.get("source") if ri else None,
            "verdict": verdict,
        }
    return out


def _geometric_open_nets(routed: Any, tech: LayoutTech) -> List[str]:
    """Nets that are NOT one connected component in REAL geometry (via-aware union-find).

    The hardest-won lesson on this subsystem: a ``connected=True`` flag once hid a geometric
    open (min-width pads not touching). This re-checks the routed artwork directly.

    NATIVE<->ORACLE FIDELITY (2026-06-18 fix): the earlier version merged EVERY metal
    layer into ONE 2D region before decomposing into islands. That made a trunk riser
    on M2 sitting OVER two disjoint M1 strips read as ONE island (2D projections
    overlap) EVEN WITH NO via1 cut between them -- a SILENT PASS the real KLayout
    full-extraction LVS contradicts (extraction connects M1<->M2 ONLY through an
    actual via cut; the missing-bottom-via net07 open in the centered-symmetric cp
    column was clean here but UNMATCHED at LVS). The check now keeps each metal
    layer's islands SEPARATE and unions two islands ACROSS layers ONLY through a via
    cut that overlaps both (same physics as the extractor). Same-layer islands still
    union on direct overlap. A correctly routed net forms exactly one component; any
    net with >1 component is geometrically OPEN and is returned.
    """
    import klayout.db as kdb

    metal_keys = set()
    for n in range(1, 8):
        try:
            metal_keys.add(tuple(tech.metal_key(n)))
        except Exception:
            break
    via_keys = set()
    for n in range(1, 7):
        try:
            via_keys.add(tuple(tech.via_key(n)))
        except Exception:
            break
    try:
        via_keys.add(tuple(tech.contact_key()))
    except Exception:
        pass

    def _poly(lp):
        return kdb.Polygon([kdb.Point(int(v.x), int(v.y)) for v in lp.polygon.vertices])

    opens: List[str] = []
    for net, nr in (getattr(routed, "nets", {}) or {}).items():
        # PER-LAYER metal regions (NOT one flattened 2D union) so an M2-over-M1
        # overlap is NOT counted as connected without a via cut -- the extractor's
        # physics. Islands carry their layer key so cross-layer union is gated on
        # a via; same-layer union is direct overlap.
        metal_by_key: Dict[Tuple[int, int], "kdb.Region"] = {}
        vias = kdb.Region()
        for lp in getattr(nr, "wires", []) or []:
            key = (int(lp.layer), int(lp.datatype))
            if key in via_keys:
                vias.insert(_poly(lp))
            elif key in metal_keys:
                metal_by_key.setdefault(key, kdb.Region()).insert(_poly(lp))
        vias.merge()
        islands: List["kdb.Region"] = []
        island_key: List[Tuple[int, int]] = []
        for key, reg in metal_by_key.items():
            reg.merge()
            for p in reg.each():
                islands.append(kdb.Region(p))
                island_key.append(key)
        k = len(islands)
        if k <= 1:
            continue  # 0 or 1 island -> trivially one component
        parent = list(range(k))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # cross-layer: a via cut that overlaps two islands stitches them (the only
        # legal M(n)<->M(n+1) join in the extractor).
        for vp in vias.each():
            vr = kdb.Region(vp)
            touch = [i for i in range(k) if (islands[i] & vr).count() > 0]
            for t in touch[1:]:
                union(touch[0], t)
        # same-layer: two islands on the SAME metal layer that directly overlap are
        # one physical conductor (cross-layer overlap WITHOUT a via does NOT join).
        for i in range(k):
            for j in range(i + 1, k):
                if island_key[i] != island_key[j]:
                    continue
                if (islands[i] & islands[j]).count() > 0:
                    union(i, j)
        if len({find(i) for i in range(k)}) > 1:
            opens.append(net)
    return opens


def route_ota_leakfree(
    out_gds: str = "_review_gds/external_pdk.gds",
    *,
    tech_id: str = "gf180mcu",
    policy: Optional[WidthPolicy] = None,
) -> dict:
    """Route the external_pdk OTA from a LEAKAGE-FREE routing intent and judge it vs the reference.

    Steps (see the module docstring for the leakage boundary):
      1. Build the real OTA intent (``build_ota_intent``).
      2. Solve the self-consistent leakage-free placement (``open_pdk_ota_positions``).
      3. Build a :class:`RoutingIntent` whose ``widths`` are DESIGN-sourced
         (``derive_width_intent``); HARD-ASSERT ``is_leakage_free()``.
      4. Extract M1 pins (``extract_pins``), build device-M1 obstacles, lower the intent,
         route with :class:`PathFinderEngine` (``widen=True``).
      5. Verify with the merged-geometry consistency verifier; compute the reference
         width_class match COUNT using ``extract_reference_net_widths`` (comparison only).
      6. Emit the routed GDS.

    Returns a dict of the measured metrics (all REAL numbers from the tools).
    """
    from clay.layout.extract.migrate import build_ota_intent
    from clay.layout.extract.realize_render import real_device_geometry
    from clay.layout.extract.route_migrate import (
        extract_reference_net_widths,
        open_pdk_ota_positions,
    )

    pol = policy or WidthPolicy()
    tech = LayoutTech.resolve(tech_id)
    m1: Tuple[int, int] = tuple(tech.metal_key(1))  # type: ignore[assignment]
    ref_gds = _reference_gds_path()

    # 1. real OTA intent ----------------------------------------------------------------
    intent = build_ota_intent()

    # 2. leakage-free self-consistent placement -----------------------------------------
    placement, _sizes = open_pdk_ota_positions(intent, tech_id=tech_id)

    # 3. DESIGN-sourced routing intent (NO reference widths) -> hard leakage gate --------
    widths = derive_width_intent(intent, tech, layer="metal1", policy=pol)
    routing_intent = RoutingIntent(cell="OTA_CS_Q", widths=widths)
    if not routing_intent.is_leakage_free():
        raise AssertionError(
            "routing intent is NOT leakage-free: "
            f"{routing_intent.leakage_count()} reference-sourced value(s) -- "
            "the anchor must derive widths from the netlist/PDK, never the reference."
        )
    leakage_count = routing_intent.leakage_count()

    # 4a. pins (leakage-free) -----------------------------------------------------------
    geometry = real_device_geometry(intent, ref_gds=ref_gds, tech_id=tech_id)
    # include_power=True: route the supply rails themselves (at their DESIGN-derived power
    # width), not just the signal nets -- so width-class is verified on real routed power
    # geometry, and the rails are connected nets rather than mere keep-outs.
    # NOTE: the external_pdk anchor does NOT pass geometry= here on purpose -- on external_pdk the
    # parametric passive cell already covers the resistors, and snapping its terminals onto
    # the reference passive stamp perturbs the route enough to flip one signal's width class
    # (the reference-comparison metric). The cross-PDK path (route_amp_leakfree) DOES pass
    # geometry to recover the external_pdk resistor contacts the parametric cell cannot supply.
    pin_report = extract_pins(intent, placement, tech=tech, return_report=True,
                              include_power=True)
    pins_by_net = pin_report.pins_by_net

    # 4b. obstacles from placed device M1 artwork (leakage-free) ------------------------
    obstacles = _device_m1_obstacles(
        intent, placement, geometry, tech=tech, pins_by_net=pins_by_net
    )

    # 4c. lower the (leakage-free) intent + route --------------------------------------
    constraints = lower_intent(routing_intent)
    routed = route_prepared_cell(pins_by_net, obstacles, constraints, tech)

    # 5. verify against the reference (OUTPUT side -- comparison only) ------------------
    report = verify_consistency(
        routed,
        intent=intent,
        ref_gds=ref_gds,
        tech=tech,
        pins_by_net=pins_by_net,
        obstacles=obstacles,
    )
    conn = report.by_name("connectivity")
    drc = report.by_name("drc_no_wire_over_device")

    n_required = conn.detail.get("n_required_nets", len(pins_by_net))
    n_connected = conn.detail.get("n_connected", 0)
    n_shorts = conn.detail.get("n_shorts", 0)
    n_spacing_viol = drc.detail.get("n_spacing_violations", 0)
    wire_over_device_um2 = drc.detail.get("wire_over_device_overlap_um2", 0.0)

    # width_class match COUNT vs the reference (comparison ONLY -- the reference per-net
    # widths are read here purely to SCORE, never fed into routing).
    ref = extract_reference_net_widths(ref_gds, intent=intent)
    # normalise net keys (strip the auCdl global '!' mark, upper-case) so a routed 'vdd!'
    # matches a reference 'VDD'.
    ref_per_net = {k.upper().rstrip("!"): float(v) for k, v in ref.per_net.items()}
    width_class_match = 0
    width_class_compared = 0
    width_class_rows: List[Dict[str, Any]] = []
    for net, nr in routed.nets.items():
        ref_w = ref_per_net.get(net.upper().rstrip("!"))
        if ref_w is None:
            continue
        is_power = net.strip().lower().rstrip("!") in {"vdd", "vss", "gnd"}
        ref_cls = _width_class(ref_w, is_power=is_power)
        routed_w = _routed_trunk_width_um(nr, m1)
        routed_cls = _width_class(routed_w, is_power=False)
        width_class_compared += 1
        if ref_cls == routed_cls:
            width_class_match += 1
        width_class_rows.append({
            "net": net, "ref_width_um": round(ref_w, 4),
            "routed_width_um": round(routed_w, 4),
            "ref_class": ref_cls, "routed_class": routed_cls,
            "ok": ref_cls == routed_cls,
        })

    # independent via-aware geometric connectivity (corroborates the connectivity flag) -----
    geometric_open_nets = _geometric_open_nets(routed, tech)

    # 6. emit GDS -----------------------------------------------------------------------
    written = _emit_routed_gds(routed, out_gds)

    metrics = {
        "leakage_count": leakage_count,
        "is_leakage_free": routing_intent.is_leakage_free(),
        "n_required_nets": n_required,
        "n_connected": n_connected,
        "geometric_open_nets": geometric_open_nets,
        "all_connected": n_connected == n_required and not conn.detail.get("missing_nets")
        and not conn.detail.get("disconnected_nets"),
        "n_shorts": n_shorts,
        "n_spacing_violations": n_spacing_viol,
        "n_spacing_violations_netagnostic": drc.detail.get(
            "n_spacing_violations_netagnostic", -1),
        "wire_over_device_overlap_um2": wire_over_device_um2,
        "drc_method": drc.detail.get("method"),
        "width_class_compared": width_class_compared,
        "width_class_match": width_class_match,
        "width_class_rows": width_class_rows,
        "route_success": bool(routed.success),
        "verify_ok": bool(report.ok),
        "out_gds": written,
        "design_widths_um": {n: round(float(d.value), 4) for n, d in widths.items()},
        "uncovered_terminals": list(pin_report.uncovered),
        "consistency_summary": report.summary(),
    }
    return metrics


__all__ = ["route_ota_leakfree", "route_amp_leakfree", "rail_obligation_report"]
