"""Emit-time merge hygiene for routed-wire GDS emission -- the ONE shared helper.

CONTRACT (P1, 2026-06-11)
=========================
Every GDS write boundary that emits routed wires merges them PER NET x LAYER
(klayout ``Region.merged`` on the integer-nm ``LayerPolygon`` currency) before
writing:

* SAME net, SAME (layer, datatype): fragments union into merged islands. The
  emitters historically stacked every pad-per-cell, bridge, widen fragment and
  riser landing pad verbatim -- the canonical style as-built carried M1 = 2599
  raw polygons at a raw/merged area ratio of 5.24x (the reference layout: 443 /
  1.52x). Via cuts merge like any other layer.
* NEVER across nets: two different nets' polygons on one layer stay separate
  polygons. A cross-net union would fabricate a short in the artifact that the
  router never drew and the shorts oracle never saw.
* EMISSION HYGIENE ONLY: the in-memory ``NetRoute.wires`` fragments are never
  mutated -- the verifier / shorts oracle / electrical checks keep consuming
  the original route structure. Device artwork is not touched either (device /
  cap stamps are emitted verbatim by their own emitters).

Consumers (the single source rule -- every routed-wire GDS write goes through
:func:`merge_net_wires`):

* :func:`emit_routed_gds` (the canonical writer; legacy callers retain a thin alias),
* :func:`clay.layout.route.lvs_gate.emit_labelled_gds` (the LVS gate's full
  as-built emission; its same-net M1 heal material joins the owning net's
  merge),
* :func:`clay.layout.extract.route_migrate.emit_route_gds` (the migration
  trunk router's emission).
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import os
import re
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from clay.layout.backend import LayerPolygon, Point, Polygon
from clay.layout.placed_geometry import placed_point_nm, placed_ring_nm

_LayerKey = Tuple[int, int]
_NM_PER_UM = 1000

# the manufacturing-grid quantum (um) every drawn layer is checked against by a
# foundry DRC OFFGRID rule, e.g. gf180mcu.drc: ``metal1.ongrid(0.005)``.
_ONGRID_RE = re.compile(r"\bongrid\(\s*([0-9]*\.?[0-9]+)\s*\)")

# Bounded LRU cache of parsed DRC deck manufacturing grids keyed by
# (abspath, st_mtime_ns, st_size, optional_recent_sha256). For freshly modified
# files (within 100ms, e.g., rapid same-size rewrites on Windows NTFS timer
# ticks), raw-byte SHA-256 is included so content edits are detected immediately.
_DECK_GRID_CACHE_MAX = 64
_DECK_GRID_RECENT_NS = 100_000_000
_DECK_GRID_CACHE: OrderedDict[Tuple[Any, ...], frozenset[int]] = OrderedDict()
_DECK_GRID_CACHE_LOCK = threading.Lock()


def clear_manufacturing_grid_cache() -> None:
    """Clear the parsed DRC deck manufacturing-grid cache (for tests)."""
    with _DECK_GRID_CACHE_LOCK:
        _DECK_GRID_CACHE.clear()


def _parse_deck_grids_text(txt: str) -> frozenset[int]:
    grids: set[int] = set()
    for m in _ONGRID_RE.finditer(txt):
        um = float(m.group(1))
        if um > 0:
            grids.add(int(round(um * 1000.0)))
    return frozenset(grids)


def _cached_deck_grids_nm(deck: Any) -> frozenset[int]:
    deck_str = str(deck)
    try:
        abs_deck = os.path.abspath(deck_str)
        st = os.stat(abs_deck)
    except OSError:
        # Preserve fallback for non-stat'able paths / mocked open() in tests.
        try:
            with open(deck_str, encoding="utf-8", errors="replace") as fh:
                return _parse_deck_grids_text(fh.read())
        except Exception:
            return frozenset()
    now_ns = time.time_ns()
    recent = abs(now_ns - st.st_mtime_ns) <= _DECK_GRID_RECENT_NS
    raw_bytes: Optional[bytes] = None
    if recent:
        try:
            with open(abs_deck, "rb") as fh:
                raw_bytes = fh.read()
        except Exception:
            return frozenset()
        key: Tuple[Any, ...] = (
            abs_deck,
            st.st_mtime_ns,
            st.st_size,
            hashlib.sha256(raw_bytes).digest(),
        )
    else:
        key = (abs_deck, st.st_mtime_ns, st.st_size)
    with _DECK_GRID_CACHE_LOCK:
        cached = _DECK_GRID_CACHE.get(key)
        if cached is not None:
            _DECK_GRID_CACHE.move_to_end(key)
            return cached
    try:
        if raw_bytes is not None:
            txt = raw_bytes.decode("utf-8", errors="replace")
        else:
            with open(abs_deck, encoding="utf-8", errors="replace") as fh:
                txt = fh.read()
        grids = _parse_deck_grids_text(txt)
    except Exception:
        return frozenset()
    with _DECK_GRID_CACHE_LOCK:
        _DECK_GRID_CACHE[key] = grids
        _DECK_GRID_CACHE.move_to_end(key)
        while len(_DECK_GRID_CACHE) > _DECK_GRID_CACHE_MAX:
            _DECK_GRID_CACHE.popitem(last=False)
    return grids


def emit_routed_gds(
    routed: Any,
    out_gds: str,
    *,
    cell_name: str = "routed_cell",
    dbu: float = 0.001,
    merge: bool = True,
) -> str:
    """Write router-owned wires to a flat GDS without entering a legacy router."""
    import pathlib

    import klayout.db as kdb

    nets = getattr(routed, "nets", None)
    wires = merge_net_wires(nets) if merge and nets else routed.wires
    layout = kdb.Layout()
    layout.dbu = dbu
    top = layout.create_cell(cell_name)
    for polygon in wires:
        layer_index = layout.layer(int(polygon.layer), int(polygon.datatype))
        top.shapes(layer_index).insert(
            kdb.Polygon(
                [
                    kdb.Point(int(vertex.x), int(vertex.y))
                    for vertex in polygon.polygon.vertices
                ]
            )
        )
    output = pathlib.Path(out_gds)
    output.parent.mkdir(parents=True, exist_ok=True)
    options = kdb.SaveLayoutOptions()
    options.gds2_write_timestamps = False
    layout.write(str(output), options)
    return str(output)


def net_label_points(
    routed: Any,
    pins_by_net: Dict[str, List[Any]],
    *,
    tech: Any,
    geometry: Optional[Dict[str, Any]] = None,
    placement: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Tuple[int, int]]:
    """Choose one deterministic on-M1 label point for every routed net."""
    import klayout.db as kdb

    metal1 = tuple(tech.metal_key(1))

    def routed_polygon(layer_polygon: Any) -> Any:
        return kdb.Polygon(
            [
                kdb.Point(int(vertex.x), int(vertex.y))
                for vertex in layer_polygon.polygon.vertices
            ]
        )

    pin_points: Dict[str, List[Tuple[int, int]]] = {}
    for net, pins in (pins_by_net or {}).items():
        for pin in pins:
            pin_points.setdefault(net, []).append(
                (
                    int(round(float(pin.xy_um[0]) * _NM_PER_UM)),
                    int(round(float(pin.xy_um[1]) * _NM_PER_UM)),
                )
            )

    device_metal1: List[Tuple[Any, Any]] = []
    if geometry is not None and placement is not None:
        for name, device_geometry in geometry.items():
            position = placement.get(name)
            if position is None:
                continue
            rings = (getattr(device_geometry, "polys_by_layer", {}) or {}).get(
                metal1, []
            )
            for ring in rings:
                if len(ring) < 3:
                    continue
                points = [
                    kdb.Point(x, y)
                    for x, y in placed_ring_nm(ring, position)
                ]
                poly = kdb.Polygon(points)
                device_metal1.append((poly.bbox(), poly))

    def inside_point(rectangles: List[Any]) -> Optional[Tuple[int, int]]:
        if not rectangles:
            return None
        center = max(rectangles, key=lambda polygon: polygon.area()).bbox().center()
        return int(center.x), int(center.y)

    output: Dict[str, Tuple[int, int]] = {}
    for net in sorted(pins_by_net or {}):
        rectangles: List[Any] = []
        net_route = (getattr(routed, "nets", {}) or {}).get(net)
        if net_route is not None:
            for layer_polygon in getattr(net_route, "wires", []) or []:
                if (
                    int(layer_polygon.layer),
                    int(layer_polygon.datatype),
                ) == metal1:
                    rectangles.append(routed_polygon(layer_polygon))
        point = inside_point(rectangles)
        if point is None:
            # A one-pin upper-metal net may have no router-owned M1.  Its device
            # pin can still sit vertically over the device's own M1 conductor.
            # Use that exact, already-proven access point for the M1 label.  Do
            # not use the covering polygon's bbox centre: ring/U-shaped device
            # metal can have its bbox centre in a hole occupied by another net
            # (the multi-unit BJT B-ring/E-bar regression).
            for px, py in pin_points.get(net, []):
                probe = None
                for bb, polygon in device_metal1:
                    if (
                        bb.left <= px + 1
                        and bb.right >= px - 1
                        and bb.bottom <= py + 1
                        and bb.top >= py - 1
                    ):
                        if probe is None:
                            probe = kdb.Region(kdb.Box(px - 1, py - 1, px + 1, py + 1))
                        if not (kdb.Region(polygon) & probe).is_empty():
                            point = (px, py)
                            break
                if point is not None:
                    break
        if point is not None:
            output[net] = point
    return output


def emit_net_labels(
    top: Any,
    layout: Any,
    label_points: Dict[str, Tuple[int, int]],
    *,
    tech: Any,
    pin_nets: Optional[Sequence[str]] = None,
) -> int:
    """Insert declared M1 net labels into an already assembled cell.

    TEXT belongs on the PDK's label purpose, which can differ from the metal
    drawing purpose (gf180 M1 is 34/0 while its LVS labels are 34/10).  Preserve
    the declared spelling: deck-global names such as gf180 ``vss`` are joined by
    exact name.  The case-folded collision check remains fail-closed so two
    distinct declared nets cannot collapse in a case-insensitive reader.
    Only explicitly supplied source ports receive the distinct PDK port purpose.
    Keep their ordinary text as well: geometric/via and synthesized-LVS readers
    consume that purpose. Both texts use the same proven metal anchor; ordinary
    internal net names must not become extra subcircuit ports.
    """
    import klayout.db as kdb

    label_key = tuple(tech.label_key(1))
    layer_index = layout.layer(int(label_key[0]), int(label_key[1]))
    declared_ports = {str(net).upper() for net in pin_nets or ()}
    port_key = tech.pin_label_key(1) if declared_ports else None
    port_layer_index = (
        layout.layer(int(port_key[0]), int(port_key[1]))
        if port_key is not None else layer_index
    )
    seen: Dict[str, str] = {}
    for net in label_points:
        normalized = str(net).upper()
        if normalized in seen and seen[normalized] != str(net):
            raise ValueError(
                "net-name case collision under upper(): "
                f"{seen[normalized]!r} vs {net!r} -> {normalized!r}; "
                "label case-normalisation would merge two distinct nets."
            )
        seen[normalized] = str(net)
    if port_key is not None:
        missing = declared_ports - set(seen)
        if missing:
            raise ValueError(
                "declared source ports have no on-metal label anchor: "
                + ", ".join(sorted(missing))
            )
    for net, (x_nm, y_nm) in label_points.items():
        text = kdb.Text(
            str(net),
            kdb.Trans(kdb.Point(int(x_nm), int(y_nm))),
        )
        top.shapes(layer_index).insert(text)
        if str(net).upper() in declared_ports and port_layer_index != layer_index:
            top.shapes(port_layer_index).insert(text)
    return len(label_points)


def pdk_manufacturing_grid_nm(tech: Any) -> int:
    """The PDK manufacturing grid (nm) every emitted vertex must land on.

    SINGLE SOURCE = the foundry DRC deck. Every drawn layer in a foundry deck is
    checked with ``<layer>.ongrid(<g>um)`` (the OFFGRID section); ``<g>`` is the
    manufacturing grid, identical across layers (gf180mcu = 0.005um = 5nm). We
    parse it from a host-local deck, or consume the adapter's explicit foundry
    fact when the bound signoff deck lives in a separate runtime. If both are
    available, they must agree; a caller cannot override the foundry grid.

    When the PDK exposes no foundry DRC deck (open_pdk / open_pdk ship encrypted
    / .oa decks that are not present), there is no parsed grid to honour, so the
    snap degrades to the database unit (``dbu_nm``, normally 1nm) -- a no-op for the
    already integer-nm currency, leaving those PDKs' emit bytes byte-identical. The
    coarsest declared grid is used (``min`` over distinct values is the finest, so we
    take the value the deck actually checks -- a single value in practice)."""
    declared = getattr(tech, "manufacturing_grid_nm", None)
    if declared is not None:
        if isinstance(declared, bool) or not isinstance(declared, int) or declared <= 0:
            raise ValueError("PDK manufacturing grid must be a positive integer in nm")
    decks = {}
    try:
        decks = tech.klayout_decks() or {}
    except Exception:
        decks = {}
    deck = decks.get("drc")
    grids: frozenset[int] = _cached_deck_grids_nm(deck) if deck is not None else frozenset()
    if grids:
        if declared is not None and grids != {declared}:
            raise ValueError(
                f"PDK manufacturing grid {declared}nm conflicts with DRC deck grids {sorted(grids)}nm"
            )
        # a foundry deck checks ONE manufacturing grid; if several appear, the
        # smallest is the finest grid every layer must satisfy.
        return min(grids)
    if declared is not None:
        return declared
    # no parsed grid -> the database unit (integer-nm => snap is a no-op).
    try:
        dbu = int(getattr(tech.rules._data, "dbu_nm", 1) or 1)
    except Exception:
        dbu = 1
    return max(1, dbu)


def _nearest_on_grid(value_nm: int, grid_nm: int) -> int:
    """Round an integer-nm coordinate to the NEAREST grid multiple (ties up).

    Identical semantics to :func:`clay.layout.grid_snap.snap_polygon_nearest`
    (the DRC-safe nearest snap): a uniform off-grid nudge moves both edges of a
    gap the same way, so widths / spacings are preserved while every vertex lands
    on grid."""
    floor = (value_nm // grid_nm) * grid_nm  # floors toward -inf for either sign
    rem = value_nm - floor
    return floor if rem * 2 < grid_nm else floor + grid_nm


def snap_kdb_cell_to_grid(cell: Any, ly: Any, grid_nm: int) -> int:
    """Snap every POLYGON shape in ``cell`` (all layers) to ``grid_nm``, in place.

    The manufacturing-grid contract at a klayout write boundary: each polygon's
    vertices are rounded to the nearest grid line INDEPENDENTLY PER POLYGON --
    never via a ``Region`` merge, so two different nets on one layer are never
    unioned (a cross-net union would fabricate a short the router never drew).
    A uniform off-grid offset moves both edges of a wire/gap the same way, so the
    snap preserves widths and spacings (it only quantises the absolute position).

    Texts / labels are left untouched (an OFFGRID rule checks drawn polygons, not
    label anchors). Returns the number of polygons whose vertices moved.

    ``grid_nm <= 1`` is a no-op (the currency is already integer nm)."""
    import klayout.db as kdb

    g = int(grid_nm)
    if g <= 1:
        return 0
    moved = 0
    for li in ly.layer_indexes():
        shapes = cell.shapes(li)
        repl: List[Tuple[Any, Any]] = []
        for s in shapes.each():
            poly = s.polygon
            if poly is None:  # texts / paths / boxes-as-text -> leave as-is
                continue
            hull = [(p.x, p.y) for p in poly.each_point_hull()]
            snapped_hull = [(_nearest_on_grid(x, g), _nearest_on_grid(y, g))
                            for (x, y) in hull]
            holes = []
            changed = snapped_hull != hull
            for hi in range(poly.holes()):
                h = [(p.x, p.y) for p in poly.each_point_hole(hi)]
                sh = [(_nearest_on_grid(x, g), _nearest_on_grid(y, g)) for (x, y) in h]
                if sh != h:
                    changed = True
                holes.append(sh)
            if not changed:
                continue
            np_ = kdb.Polygon([kdb.Point(x, y) for (x, y) in snapped_hull])
            for sh in holes:
                np_.insert_hole([kdb.Point(x, y) for (x, y) in sh])
            repl.append((s, np_))
        for old, np_ in repl:
            shapes.erase(old)
            shapes.insert(np_)
            moved += 1
    return moved


def _wires_of(net_route: Any) -> list:
    """The ``LayerPolygon`` list of one net's route (tolerates a bare list)."""
    w = getattr(net_route, "wires", None)
    return list(w) if w is not None else list(net_route)


def _kdb_poly(lp: Any):
    import klayout.db as kdb

    return kdb.Polygon(
        [kdb.Point(int(v.x), int(v.y)) for v in lp.polygon.vertices]
    )


def _region_to_layer_polygons(reg, key: _LayerKey) -> List[LayerPolygon]:
    """Decompose a merged region into hole-free ``LayerPolygon``s (a merged
    region may carry holes; GDS-safe single contours via ``resolve_holes``)."""
    out: List[LayerPolygon] = []
    for poly in reg.each():
        p = poly.dup()
        p.resolve_holes()
        verts = [Point(int(pt.x), int(pt.y)) for pt in p.each_point_hull()]
        if len(verts) >= 3:
            out.append(LayerPolygon(layer=int(key[0]), datatype=int(key[1]),
                                    polygon=Polygon(vertices=verts)))
    return out


def net_layer_regions(nets: Mapping[str, Any]) -> Dict[str, Dict[_LayerKey, Any]]:
    """Per-net, per-(layer, datatype) MERGED ``kdb.Region`` of the routed wires.

    The region view of the route used both by :func:`merge_net_wires` and by
    emit-boundary same-net materializations (the LVS gate's M1 heal) that must
    join the owning net's merge instead of stacking extra fragments."""
    import klayout.db as kdb

    out: Dict[str, Dict[_LayerKey, Any]] = {}
    for net in sorted(nets):
        by_layer: Dict[_LayerKey, Any] = {}
        for lp in _wires_of(nets[net]):
            key = (int(lp.layer), int(lp.datatype))
            by_layer.setdefault(key, kdb.Region()).insert(_kdb_poly(lp))
        for key in by_layer:
            by_layer[key].merge()
        out[net] = by_layer
    return out


def merge_net_wires(
    nets: Mapping[str, Any],
    *,
    extra_m1: Optional[Mapping[str, Any]] = None,
    m1_key: Optional[_LayerKey] = None,
    extra_by_layer: Optional[Mapping[str, Mapping[_LayerKey, Any]]] = None,
) -> List[LayerPolygon]:
    """Merge every net's routed wires per (layer, datatype) for GDS emission.

    Parameters
    ----------
    nets:
        ``{net: NetRoute-like}`` (anything with a ``.wires`` list of
        ``LayerPolygon``; a bare polygon list also works).
    extra_m1 / m1_key:
        Optional per-net SAME-NET M1 material (``{net: kdb.Region}``) produced
        at the emit boundary (the LVS gate's heal bridges / min-area grows),
        unioned into the owning net's ``m1_key`` merge -- never another net's.

    Returns
    -------
    list[LayerPolygon]
        Deterministic order: nets sorted by name, layers sorted by key, region
        polygon order as produced by klayout's merge (stable for equal input).
        Cross-net polygons are never unioned -- by construction the merge runs
        inside one net at a time.
    """
    import klayout.db as kdb

    regions = net_layer_regions(nets)
    out: List[LayerPolygon] = []
    for net in sorted(regions):
        by_layer = regions[net]
        if extra_m1 is not None and m1_key is not None:
            extra = extra_m1.get(net)
            if extra is not None and not extra.is_empty():
                base = by_layer.setdefault(tuple(m1_key), kdb.Region())
                base += extra
                base.merge()
                by_layer[tuple(m1_key)] = base
        # SAME-NET upper-metal closing (M2+): each net's own same-layer notch/gap
        # fill, already clipped clear of every FOREIGN net's metal, unioned into the
        # OWNING net's merge for that layer -- never another net's (a same-net merge
        # cannot short). The general dual of ``extra_m1`` across all routing layers.
        if extra_by_layer is not None:
            for key, reg in (extra_by_layer.get(net) or {}).items():
                if reg is None or reg.is_empty():
                    continue
                base = by_layer.setdefault(tuple(key), kdb.Region())
                base += reg
                base.merge()
                by_layer[tuple(key)] = base
        for key in sorted(by_layer):
            out.extend(_region_to_layer_polygons(by_layer[key], key))
    return out


def emit_routed_gds_cpp(
    routed: Any,
    out_gds: str,
    *,
    cell_name: str = "routed_cell",
    dbu: float = 0.001,
    merge: bool = True,
) -> str:
    """Write router-owned orthogonal wires to a flat binary GDSII using pure C++20 `_engine_cpp`.

    Zero KLayout dependency: merges per-net x per-layer via `clay_pnr::layout::merge_net_boxes_exact`
    and writes binary GDSII v6.0 via `clay_pnr::gds::write_gds_file` (deterministic, zero timestamps).
    """
    import pathlib
    from clay._native import _engine_cpp

    nets = getattr(routed, "nets", None)
    polys_out: List[Tuple[int, int, List[Tuple[int, int]]]] = []
    if merge and nets:
        net_items: List[Tuple[str, int, int, List[Tuple[int, int, int, int]]]] = []
        for net_name in sorted(nets):
            by_layer: Dict[_LayerKey, List[Tuple[int, int, int, int]]] = {}
            for lp in _wires_of(nets[net_name]):
                verts = [(int(v.x), int(v.y)) for v in lp.polygon.vertices]
                xs = [p[0] for p in verts]
                ys = [p[1] for p in verts]
                box = (min(xs), min(ys), max(xs), max(ys))
                if box[2] > box[0] and box[3] > box[1]:
                    by_layer.setdefault((int(lp.layer), int(lp.datatype)), []).append(box)
            for (layer, dt), boxes in sorted(by_layer.items()):
                net_items.append((str(net_name), layer, dt, boxes))
        polys_out = _engine_cpp.merge_net_boxes_cpp(net_items)
    else:
        for lp in getattr(routed, "wires", ()) or ():
            verts = [(int(v.x), int(v.y)) for v in lp.polygon.vertices]
            if len(verts) >= 3:
                polys_out.append((int(lp.layer), int(lp.datatype), verts))

    output = pathlib.Path(out_gds)
    output.parent.mkdir(parents=True, exist_ok=True)
    _engine_cpp.gds_write_polygons_cpp(
        str(output),
        "CLAY",
        str(cell_name),
        polys_out,
        None,
        float(dbu),
        float(dbu) * 1e-6,
    )
    return str(output)


__all__ = [
    "emit_routed_gds_cpp",
    "merge_net_wires",
    "net_layer_regions",
    "pdk_manufacturing_grid_nm",
    "snap_kdb_cell_to_grid",
]
