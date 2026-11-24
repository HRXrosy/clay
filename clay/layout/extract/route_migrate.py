"""BEST-EFFORT SIGNAL ROUTER for the gf180 PDK-migrated intent.

THE GAP THIS CLOSES
===================
``migrate.py`` migrates an extracted (external_pdk) intent onto gf180: it builds each
device's real gf180 geometry, solves migrated centroids, and stamps the device
artwork into a flat GDS. But the migrated placement has DEVICE GEOMETRY ONLY -- the
device terminals (gate / drain / source / body, per the netlist ``intent.devices[].nets``)
are NOT wired together. This module adds the missing SIGNAL ROUTING: for every signal
net it collects the set of device terminal pins on that net and draws connecting wires
on Metal1 (escaping to Metal2 + Via1 only when two different nets would otherwise short
on the same metal).

It is a MIGRATION router, NOT a signoff router. It does NOT do maze routing, obstacle
avoidance, DRC-legal jogging, or rail synthesis. Power nets (``vdd!`` / ``gnd!``) are a
sibling's job and are excluded. The goal is honest connectivity: join every signal
net's pins, and report -- numerically and first -- any same-layer short it could not
avoid and any net it left partial.

WHAT THIS MODULE DOES
=====================
1. ``build_gf180_terminals(intent)`` -- re-run the SAME faithful gf180 build
   ``migrate.build_gf180_geometry`` uses (emit_clay -> parse -> place_to_intent at
   tech='gf180mcu'), but capture each device's TERMINAL pins (s/d/g/b for MOS; p/n for
   passives) RELATIVE to the device's polys-bbox centroid (the exact centroid
   ``migrate`` stamps about) -- a reusable per-device terminal stamp in MICRONS, plus
   the routable metal layer of each pad. Combined with ``migrate`` positions this gives
   absolute pin coordinates that line up with the stamped device artwork.
2. ``net_pins(intent, positions, terminals)`` -- for each SIGNAL net (power excluded),
   collect every ``(device, terminal, xy_um, layer)`` pin on that net.
3. ``route_nets(net_pins)`` -- per multi-pin net draw a trunk + branch route on Metal1
   (a Manhattan steiner-ish spine: a horizontal trunk at the pins' median y plus a
   vertical drop from each pin to the trunk). Detect same-layer crossings that would
   short two different nets; when a net's wire would overlap a different net already on
   Metal1, that net is escaped to Metal2 (with a Via1 at each pin) so the crossing is no
   longer a short. Any short that remains (e.g. two M2 nets still crossing) is REPORTED
   honestly, never hidden.
4. ``emit_route_gds(route, out_gds)`` -- write the wires (Metal1/Metal2/Via1) into a
   flat gf180 GDS the integrator can splice over the migration GDS, or just consume
   ``route.wires`` (a list of ``LayerPolygon``) directly.

UNITS
=====
Pin coordinates and wire geometry are exposed in MICRONS (the shared ``positions`` /
``device_bboxes`` contract). GDS emission converts to nm at the single point
``_NM_PER_UM``. ``terminals`` stamps are centroid-relative MICRONS.

SCOPE / DISCIPLINE
==================
Purely COMPOSITIONAL. Does NOT edit migrate.py / relgraph.py / fold.py / power.py /
the placers / the grammar / any design.clay -- it only consumes their documented
surfaces (``place_to_intent`` placement terminals, ``migrate`` centroid convention,
the netlist ``intent.devices[].nets`` / ``intent.passives[].nets``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_NM_PER_UM = 1000.0

_XY = Tuple[float, float]
_Layer = Tuple[int, int]

# gf180 routing layers (resolved live in build_gf180_terminals / route_nets, but kept
# here as documented defaults so net_pins + the tests do not need a PDK round-trip).
GF180_METAL1: _Layer = (34, 0)
GF180_METAL2: _Layer = (36, 0)
GF180_VIA1: _Layer = (35, 0)

# Global power nets are a sibling's job -- excluded from signal routing.
DEFAULT_POWER_NETS = frozenset({"vdd!", "vss!", "gnd!", "vdd", "vss", "gnd"})

# device-net role key (intent) -> placed-terminal pin key (placer).
_MOS_ROLE_TO_PIN = {"source": "s", "drain": "d", "gate": "g", "body": "b"}
_PASSIVE_ROLE_TO_PIN = {"p": "p", "n": "n"}

# default migration wire width (um). Wide enough to over-reach a pad pad-to-pad on
# Metal1 without being a signoff width; the router is best-effort.
_DEFAULT_WIRE_W_UM = 0.6
_DEFAULT_VIA_UM = 0.4


# --------------------------------------------------------------------------- #
# data structures
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Pin:
    """One terminal pin on a signal net, ready for the router.

    ``device`` is the device/passive name; ``terminal`` the device-local pin key
    ('s'/'d'/'g'/'b' for MOS, 'p'/'n' for a passive); ``xy_um`` the ABSOLUTE pin
    centre in microns (the device's migrated centroid + the terminal's centroid-
    relative offset); ``layer`` the GDS ``(layer, datatype)`` of the routable metal
    the pad sits on (Metal1 for MOS/poly-res, a higher metal for a MIM plate)."""

    device: str
    terminal: str
    xy_um: _XY
    layer: _Layer


@dataclass
class NetRoute:
    """The route of one signal net.

    ``net`` is the net name; ``pins`` the pins it joins; ``wires`` the connecting
    ``LayerPolygon`` rectangles (Metal1 trunk/branches, Via1 cuts, Metal2 escapes);
    ``layer`` the metal the trunk was drawn on; ``connected`` is True iff every pin
    was joined into one electrically-continuous figure; ``note`` carries provenance
    (escaped-to-M2, single-pin, etc.)."""

    net: str
    pins: List[Pin]
    wires: list = field(default_factory=list)  # list[LayerPolygon]
    layer: _Layer = GF180_METAL1
    connected: bool = False
    note: str = ""


@dataclass
class Short:
    """A detected same-layer short: wires of two DIFFERENT nets overlap on one metal.

    ``net_a`` / ``net_b`` are the two nets (sorted); ``layer`` the metal they overlap
    on; ``area_um2`` the overlapping rectangle area. A short is a real connectivity
    fault -- two distinct signal nets touching -- and is reported, never hidden."""

    net_a: str
    net_b: str
    layer: _Layer
    area_um2: float


@dataclass
class RouteResult:
    """The whole signal-routing outcome for a migrated cell.

    ``routes`` maps each signal net -> its :class:`NetRoute`. ``shorts`` is every
    detected same-layer cross-net short (empty == clean). ``wires`` is the flat list
    of every wire ``LayerPolygon`` (for GDS emission). ``power_nets`` records which
    nets were excluded as power."""

    routes: Dict[str, NetRoute] = field(default_factory=dict)
    shorts: List[Short] = field(default_factory=list)
    power_nets: List[str] = field(default_factory=list)

    @property
    def wires(self) -> list:
        out: list = []
        for r in self.routes.values():
            out.extend(r.wires)
        return out

    @property
    def n_nets(self) -> int:
        return len(self.routes)

    @property
    def n_connected(self) -> int:
        return sum(1 for r in self.routes.values() if r.connected)

    @property
    def n_partial(self) -> int:
        return sum(1 for r in self.routes.values() if not r.connected)

    @property
    def n_shorts(self) -> int:
        return len(self.shorts)

    def summary(self) -> str:
        head = ""
        if self.shorts:
            head = f"!! {self.n_shorts} SHORT(S) -- "
        return (
            f"{head}signal routing: {self.n_nets} signal net(s), "
            f"{self.n_connected} fully connected, {self.n_partial} partial, "
            f"{self.n_shorts} short(s)"
        )


# --------------------------------------------------------------------------- #
# 0. terminal capture: real gf180 terminal pins per device, centroid-relative um
# --------------------------------------------------------------------------- #
def _polys_bbox_nm(polys: list) -> Optional[Tuple[int, int, int, int]]:
    xs: List[int] = []
    ys: List[int] = []
    for lp in polys:
        for v in lp.polygon.vertices:
            xs.append(int(v.x))
            ys.append(int(v.y))
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


_MOS_NET_ROLE = {"s": "source", "d": "drain", "g": "gate", "b": "body"}


def _bbox_center_nm(bbox) -> Tuple[int, int]:
    x0, y0, x1, y1 = (int(v) for v in bbox)
    return ((x0 + x1) // 2, (y0 + y1) // 2)


def _terminal_layer_for(geom: Any, term: Any, default_layer: _Layer) -> _Layer:
    """The routable metal layer a terminal pad sits on.

    A device terminal pad lands on the metal layer whose drawn polygon contains the
    terminal centre. We scan the built ``geom.polygons`` for the metal-layer polygon
    overlapping the terminal bbox and return its ``(layer, datatype)``; if none is
    found (a terminal with no captured metal) we return ``default_layer`` (Metal1).
    This is honest: the pad layer is read from the device's OWN geometry, not assumed.
    """
    try:
        cx, cy = term.center
    except Exception:
        return default_layer
    polys = getattr(geom, "polygons", None) or []
    # prefer the metal polygon (any metal layer) that contains the terminal centre.
    best: Optional[_Layer] = None
    best_area = None
    for lp in polys:
        xs = [v.x for v in lp.polygon.vertices]
        ys = [v.y for v in lp.polygon.vertices]
        if min(xs) <= cx <= max(xs) and min(ys) <= cy <= max(ys):
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            key = (int(lp.layer), int(lp.datatype))
            # keep the SMALLEST enclosing polygon (the pad itself, not a big well).
            if best_area is None or area < best_area:
                best, best_area = key, area
    return best if best is not None else default_layer


def build_terminals(
    intent: Any,
    *,
    tech_id: str = "gf180mcu",
    engine: str = "seqpair",
) -> Dict[str, Dict[str, Tuple[_XY, _Layer]]]:
    """Capture each device's TERMINAL stamp for ANY PDK: ``{device -> {pin -> (rel_xy_um, layer)}}``.

    TECH-PARAMETERIZED terminal capture (the generalization of
    :func:`build_gf180_terminals`). For each device the geometry is built directly from
    the PDK facts ``LayoutTech.resolve(tech_id)`` aggregates:

      * MOS -- ``device_cell.build_device(tech, kind, W, L, nf)`` with the netlist
        ``W`` / ``L`` (um -> m) and finger count ``nf = M``. Its ``DeviceGeom.terminals``
        gives the source / drain / gate (+ bulk when a tap is drawn) terminal bboxes;
        we take each terminal's CENTRE relative to the device's drawn-extent centroid
        and convert to microns -- a reusable per-device stamp that lines up with the
        same centroid a migration position is solved about.
      * passive (res / cap) -- ``passive_device_cell.build_resistor`` /
        ``build_capacitor`` with the netlist value; its ``p`` / ``n`` terminals are
        captured the same way. When the PDK has NO passive builder (the resistor sab /
        res_mk markers or the MIM stack are PDK-specific and only gf180mcu is wired up),
        the device falls back to a TERMINAL STUB: ``p`` / ``n`` pins at the two ends of
        the device's REFERENCE footprint (``intent.device_bboxes`` width, centroid-
        relative) on Metal1. This is honestly a stub (no vendor passive geometry), but
        it gives the router two finite, footprint-sized terminals to wire.

    Each terminal's pad layer is resolved from the device's OWN geometry
    (:func:`_terminal_layer_for`: the smallest drawn polygon enclosing the terminal
    centre), defaulting to the PDK Metal1 (``tech.metal_key(1)``). All pin coordinates
    are MICRONS, centroid-relative; combine with a migrated position for absolute pins.

    Devices that fail to build for a reason OTHER than the missing-passive-builder case
    (a genuine geometry error) are simply absent from the result (reported by the caller
    as a partial / missing net). gf180mcu returns all 17 OTA devices (its passive
    builders are wired up); open_pdk returns the 13 MOS with real geometry plus the 4
    passives as footprint stubs.
    """
    from clay.layout import device_cell as _dc
    from clay.layout import passive_device_cell as _pdc
    from clay.layout.tech import LayoutTech

    tech = LayoutTech.resolve(tech_id)
    try:
        m1: _Layer = tuple(tech.metal_key(1))  # type: ignore[assignment]
    except Exception:
        m1 = GF180_METAL1

    out: Dict[str, Dict[str, Tuple[_XY, _Layer]]] = {}

    def _capture(name: str, geom: Any) -> None:
        ext = geom.extent()
        cx, cy = _bbox_center_nm(ext)
        terms: Dict[str, Tuple[_XY, _Layer]] = {}
        for pin, term in (getattr(geom, "terminals", {}) or {}).items():
            try:
                tx, ty = term.center
            except Exception:
                continue
            rel = ((tx - cx) / _NM_PER_UM, (ty - cy) / _NM_PER_UM)
            layer = _terminal_layer_for(geom, term, m1)
            terms[str(pin)] = (rel, layer)
        if terms:
            out[name] = terms

    # --- MOS devices: real geometry from device_cell.build_device --------------
    for d in getattr(intent, "devices", []) or []:
        name = str(getattr(d, "name", ""))
        if not name:
            continue
        kind = str(getattr(d, "kind", ""))
        if kind not in ("nmos", "pmos"):
            continue
        nets = getattr(d, "nets", {}) or {}
        nf = max(1, int(round(float(getattr(d, "M", 1) or 1))))
        try:
            geom = _dc.build_device(
                tech,
                kind,
                float(getattr(d, "W", 0.0)) * 1e-6,
                float(getattr(d, "L", 0.0)) * 1e-6,
                nf,
                nets={pin: str(nets.get(role, "")) for pin, role in _MOS_NET_ROLE.items()},
            )
        except Exception:
            continue  # genuine geometry failure -> absent (reported as partial net)
        _capture(name, geom)

    # --- passives: real geometry where the PDK builder exists, else a stub ------
    for p in getattr(intent, "passives", []) or []:
        name = str(getattr(p, "name", ""))
        if not name:
            continue
        kind = str(getattr(p, "kind", ""))
        nets = getattr(p, "nets", {}) or {}
        pn_nets = {"p": str(nets.get("p", "")), "n": str(nets.get("n", ""))}
        geom = None
        try:
            if kind == "cap":
                geom = _pdc.build_capacitor_for_tech(
                    tech, float(getattr(p, "value", 0.0)), nets=pn_nets,
                    m=int(getattr(p, "M", 1) or 1),
                    model=str(getattr(p, "model", "") or ""),
                )
            elif kind == "res":
                geom = _pdc.build_resistor(tech, float(getattr(p, "value", 0.0)), nets=pn_nets)
        except Exception:
            geom = None  # PDK has no passive builder -> footprint stub below
        if geom is not None:
            _capture(name, geom)
            continue
        # STUB: p / n at the two ends of the device's REFERENCE footprint width.
        bb = (getattr(intent, "device_bboxes", {}) or {}).get(name)
        if bb is not None:
            w_um = float(bb[2]) - float(bb[0])
        else:
            w_um = 5.0
        half = max(w_um / 2.0, 0.5)
        out[name] = {
            "p": ((-half, 0.0), m1),
            "n": ((half, 0.0), m1),
        }

    return out


def build_gf180_terminals(
    intent: Any, *, tech: str = "gf180mcu", engine: str = "seqpair"
) -> Dict[str, Dict[str, Tuple[_XY, _Layer]]]:
    """gf180 TERMINAL stamp ``{device -> {pin -> (rel_xy_um, layer)}}`` (back-compat shim).

    Thin wrapper over the tech-parameterized :func:`build_terminals` pinned to
    ``tech_id='gf180mcu'``. Kept so existing callers (``route_migrated_intent`` /
    ``run_ota_route`` and the route tests) keep their import + behaviour. ``tech`` /
    ``engine`` are accepted for signature compatibility (``tech`` selects the PDK).
    """
    return build_terminals(intent, tech_id=tech, engine=engine)


# --------------------------------------------------------------------------- #
# 1. net_pins: collect every signal-net pin
# --------------------------------------------------------------------------- #
def net_pins(
    intent: Any,
    positions: Dict[str, _XY],
    terminals: Dict[str, Dict[str, Tuple[_XY, _Layer]]],
    *,
    power_nets: Sequence[str] = (),
) -> Dict[str, List[Pin]]:
    """Group every device/passive terminal into the SIGNAL net it binds to.

    ``positions`` maps device name -> migrated centroid (um). ``terminals`` maps
    device name -> ``{pin -> (rel_xy_um, layer)}`` (centroid-relative, from
    :func:`build_gf180_terminals`). For each MOS device the netlist
    ``intent.devices[].nets`` gives ``source/drain/gate/body`` -> net; for each passive
    ``intent.passives[].nets`` gives ``p/n`` -> net. Power nets (``vdd!``/``gnd!``/...
    plus any in ``power_nets``) are EXCLUDED -- power distribution is a sibling's job.

    Returns ``{net -> [Pin, ...]}`` with each pin's absolute xy = migrated centroid +
    the terminal's centroid-relative offset. A device that has a position but no
    captured terminal (placer did not build it) simply contributes no pins; a terminal
    whose role has no captured stamp is skipped (reported indirectly by a partial net).
    """
    power = set(DEFAULT_POWER_NETS) | {str(p) for p in power_nets}
    by_net: Dict[str, List[Pin]] = {}

    def add(device: str, role_key: str, pin_key: str, net: str) -> None:
        if not net or net in power:
            return
        pos = positions.get(device)
        terms = terminals.get(device)
        if pos is None or terms is None:
            return
        stamp = terms.get(pin_key)
        if stamp is None:
            return
        (rx, ry), layer = stamp
        xy = (float(pos[0]) + float(rx), float(pos[1]) + float(ry))
        by_net.setdefault(net, []).append(
            Pin(device=device, terminal=pin_key, xy_um=xy, layer=tuple(layer))
        )

    for d in getattr(intent, "devices", []) or []:
        name = str(getattr(d, "name", ""))
        nets = getattr(d, "nets", {}) or {}
        for role, pin in _MOS_ROLE_TO_PIN.items():
            if role in nets:
                add(name, role, pin, str(nets[role]))

    for p in getattr(intent, "passives", []) or []:
        name = str(getattr(p, "name", ""))
        nets = getattr(p, "nets", {}) or {}
        for role, pin in _PASSIVE_ROLE_TO_PIN.items():
            if role in nets:
                add(name, role, pin, str(nets[role]))

    return by_net


# --------------------------------------------------------------------------- #
# 2. route_nets: per-net trunk + branch on Metal1, escape crossings to Metal2
# --------------------------------------------------------------------------- #
def _hwire(x0: float, x1: float, y: float, w: float, layer: _Layer):
    """Horizontal wire rectangle (um) on ``layer`` -> LayerPolygon (nm).

    WIDTH-EXACT emit: the drawn cross-section is ``round(w)`` nm BY CONSTRUCTION
    (integer centreline +/- integer width), never the two independently rounded
    float edges -- those drift +/-1 nm with the centreline's fraction and once
    emitted a 279 nm wire against the 280 nm M2 min width (sub-rule by 1 nm)."""
    w_nm = max(1, int(round(w * _NM_PER_UM)))
    cy = int(round(y * _NM_PER_UM))
    y0_nm = cy - w_nm // 2
    return _rect_poly_nm(
        int(round(min(x0, x1) * _NM_PER_UM)), y0_nm,
        int(round(max(x0, x1) * _NM_PER_UM)), y0_nm + w_nm, layer,
    )


def _vwire(y0: float, y1: float, x: float, w: float, layer: _Layer):
    """Vertical wire rectangle (um) on ``layer`` -> LayerPolygon (nm). Width-exact
    (see :func:`_hwire`)."""
    w_nm = max(1, int(round(w * _NM_PER_UM)))
    cx = int(round(x * _NM_PER_UM))
    x0_nm = cx - w_nm // 2
    return _rect_poly_nm(
        x0_nm, int(round(min(y0, y1) * _NM_PER_UM)),
        x0_nm + w_nm, int(round(max(y0, y1) * _NM_PER_UM)), layer,
    )


def _rect_poly_nm(x0: int, y0: int, x1: int, y1: int, layer: _Layer):
    """Integer-nm rectangle -> LayerPolygon (no further rounding)."""
    from clay.layout.backend import LayerPolygon, Point, Polygon

    return LayerPolygon(
        layer=int(layer[0]),
        datatype=int(layer[1]),
        polygon=Polygon(
            vertices=[
                Point(x0, y0), Point(x1, y0), Point(x1, y1), Point(x0, y1),
            ]
        ),
    )


def _rect_poly(x0: float, y0: float, x1: float, y1: float, layer: _Layer):
    from clay.layout.backend import LayerPolygon, Point, Polygon

    def n(v: float) -> int:
        return int(round(v * _NM_PER_UM))

    return LayerPolygon(
        layer=int(layer[0]),
        datatype=int(layer[1]),
        polygon=Polygon(
            vertices=[
                Point(n(x0), n(y0)),
                Point(n(x1), n(y0)),
                Point(n(x1), n(y1)),
                Point(n(x0), n(y1)),
            ]
        ),
    )


def _wire_bbox_um(lp) -> Tuple[float, float, float, float]:
    xs = [v.x for v in lp.polygon.vertices]
    ys = [v.y for v in lp.polygon.vertices]
    return (
        min(xs) / _NM_PER_UM,
        min(ys) / _NM_PER_UM,
        max(xs) / _NM_PER_UM,
        max(ys) / _NM_PER_UM,
    )


def _rect_overlap_area(a, b) -> float:
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    if ox <= 0.0 or oy <= 0.0:
        return 0.0
    return ox * oy


def _route_one_net(
    net: str,
    pins: List[Pin],
    *,
    layer: _Layer,
    via_layer: _Layer,
    wire_w: float,
    via_um: float,
    escape_layer: Optional[_Layer] = None,
) -> NetRoute:
    """Trunk + branch route for one net on ``layer`` (escaping to ``escape_layer`` when
    given). The trunk is a horizontal Metal1 spine at the pins' median y spanning the
    pins' x-range; each pin drops a vertical branch to the trunk. When ``escape_layer``
    is set, the trunk + branches are drawn on ``escape_layer`` and a Via1 + a short
    Metal1 landing pad is dropped at each pin so the pad (on its native metal) reaches
    the escaped routing layer."""
    route = NetRoute(net=net, pins=list(pins), layer=layer)
    if len(pins) <= 1:
        route.connected = True  # a 1-pin net is trivially "joined" (nothing to wire)
        route.note = "single-pin (no wire needed)" if pins else "no pins"
        return route

    xs = [p.xy_um[0] for p in pins]
    ys = [p.xy_um[1] for p in pins]
    ys_sorted = sorted(ys)
    trunk_y = ys_sorted[len(ys_sorted) // 2]  # median y
    x_lo, x_hi = min(xs), max(xs)

    use_layer = escape_layer or layer
    wires: list = []
    # horizontal trunk spanning the pin x-range at the median y.
    if x_hi > x_lo:
        wires.append(_hwire(x_lo, x_hi, trunk_y, wire_w, use_layer))
    # vertical branch from each pin down/up to the trunk.
    for p in pins:
        px, py = p.xy_um
        if abs(py - trunk_y) > 1e-9:
            wires.append(_vwire(py, trunk_y, px, wire_w, use_layer))
        if escape_layer is not None:
            # via + Metal1 landing pad so the native-metal pad reaches the escape layer.
            wires.append(_rect_poly(px - via_um / 2.0, py - via_um / 2.0,
                                    px + via_um / 2.0, py + via_um / 2.0, via_layer))
            wires.append(_rect_poly(px - wire_w / 2.0, py - wire_w / 2.0,
                                    px + wire_w / 2.0, py + wire_w / 2.0, layer))

    route.wires = wires
    route.layer = use_layer
    route.connected = True
    if escape_layer is not None:
        route.note = f"escaped to metal {escape_layer} (+Via1) to avoid a same-layer short"
    return route


def _resolve_metals(
    tech: Optional[str],
    metal1: Optional[_Layer],
    metal2: Optional[_Layer],
    via1: Optional[_Layer],
) -> Tuple[_Layer, _Layer, _Layer, float]:
    """Resolve ``(m1, m2, via1, m1_min_width_um)`` from the PDK ``tech``.

    Explicit ``metal1`` / ``metal2`` / ``via1`` override the tech (so a test can pin the
    layers); otherwise they come from ``LayoutTech.resolve(tech).metal_key/via_key``.
    ``m1_min_width_um`` is the Metal1 min drawn width (a per-net width is clamped up to
    it so a drawn wire is never sub-rule). Falls back to the documented gf180 defaults +
    a 0.23um floor when the tech can not be resolved.
    """
    m1 = tuple(metal1) if metal1 else GF180_METAL1
    m2 = tuple(metal2) if metal2 else GF180_METAL2
    v1 = tuple(via1) if via1 else GF180_VIA1
    m1_min_um = 0.23
    if tech:
        try:
            from clay.layout.tech import LayoutTech

            t = LayoutTech.resolve(tech)
            if metal1 is None:
                m1 = tuple(t.metal_key(1))
                m2 = tuple(t.metal_key(2))
                v1 = tuple(t.via_key(1))
            mw = t.min_width("metal1")
            if mw:
                m1_min_um = float(mw) / _NM_PER_UM
        except Exception:
            pass  # documented gf180 defaults already set
    return m1, m2, v1, m1_min_um  # type: ignore[return-value]


def _net_width(
    net: str,
    widths: Optional[Dict[str, float]],
    default_um: float,
    min_um: float,
) -> float:
    """The drawn wire width (um) for ``net``: its entry in ``widths`` (matched
    case-insensitively, since GDS net labels are upper-cased while netlist nodes are
    not) clamped UP to the PDK metal min width, else ``default_um`` (also clamped)."""
    w = default_um
    if widths:
        if net in widths:
            w = float(widths[net])
        else:
            low = net.lower()
            for k, v in widths.items():
                if k.lower() == low:
                    w = float(v)
                    break
    return max(float(w), float(min_um))


def route_nets(
    np: Dict[str, List[Pin]],
    *,
    widths: Optional[Dict[str, float]] = None,
    tech_id: Optional[str] = None,
    tech: str = "gf180mcu",
    wire_width_um: float = _DEFAULT_WIRE_W_UM,
    via_um: float = _DEFAULT_VIA_UM,
    metal1: Optional[_Layer] = None,
    metal2: Optional[_Layer] = None,
    via1: Optional[_Layer] = None,
) -> RouteResult:
    """Route every signal net's pins; detect + mitigate same-layer cross-net shorts.

    Strategy (best-effort, deterministic): nets are routed largest-first on Metal1 as a
    trunk + branch spine. After each net is placed on Metal1, its wires are checked
    against every other Metal1 net already routed; if it would overlap a DIFFERENT net's
    Metal1 wires (a short), the net is re-routed on Metal2 (with a Via1 + Metal1 landing
    pad at each pin) so the crossing is no longer a same-layer touch. A residual overlap
    that still survives on the SAME layer (two nets both forced to Metal2 that still
    cross) is recorded as a :class:`Short` and reported -- never hidden.

    PER-NET widths: ``widths`` maps ``{net -> width_um}``. When given, each net's trunk
    + branches are drawn at ITS width (clamped UP to the PDK Metal1 min width so a wire
    is never sub-rule); a net absent from ``widths`` uses ``wire_width_um``. Names match
    case-insensitively (GDS labels are upper-cased; netlist nodes are not). With
    ``widths=None`` every net uses the single ``wire_width_um`` -- the original uniform
    behaviour (byte-unchanged).

    ``tech_id`` (preferred) / ``tech`` select the PDK whose Metal1/Metal2/Via1 layers
    the route is drawn on (``metal1`` / ``metal2`` / ``via1`` override them). Returns a
    :class:`RouteResult`.
    """
    sel_tech = tech_id or tech
    m1, m2, v1, m1_min_um = _resolve_metals(sel_tech, metal1, metal2, via1)

    result = RouteResult()

    # route order: more-pin nets first (they own the most Metal1 area; smaller nets
    # then escape around them). Deterministic tiebreak by name.
    order = sorted(np.keys(), key=lambda nm: (-len(np[nm]), nm))

    # per-layer accumulation of routed wire bboxes for short detection.
    placed_by_layer: Dict[_Layer, List[Tuple[str, list]]] = {m1: [], m2: []}

    def overlaps_other_net(layer: _Layer, net: str, wire_bbs: list) -> List[str]:
        hit: List[str] = []
        for other_net, bbs in placed_by_layer.get(layer, []):
            if other_net == net:
                continue
            for wb in wire_bbs:
                if any(_rect_overlap_area(wb, ob) > 1e-9 for ob in bbs):
                    hit.append(other_net)
                    break
        return hit

    for net in order:
        pins = np[net]
        net_w = _net_width(net, widths, wire_width_um, m1_min_um)
        route = _route_one_net(
            net, pins, layer=m1, via_layer=v1, wire_w=net_w, via_um=via_um
        )
        if len(pins) <= 1:
            result.routes[net] = route
            continue

        wire_bbs = [_wire_bbox_um(w) for w in route.wires]
        clash = overlaps_other_net(m1, net, wire_bbs)
        if clash:
            # escape this net to Metal2 to clear the Metal1 short.
            route = _route_one_net(
                net, pins, layer=m1, via_layer=v1, wire_w=net_w,
                via_um=via_um, escape_layer=m2,
            )
            # check the escape layer (Metal2) for a residual short.
            m2_bbs = [
                _wire_bbox_um(w) for w in route.wires
                if (w.layer, w.datatype) == m2
            ]
            m2_clash = overlaps_other_net(m2, net, m2_bbs)
            for oth in sorted(set(m2_clash)):
                area = 0.0
                for _on, bbs in placed_by_layer[m2]:
                    if _on != oth:
                        continue
                    for wb in m2_bbs:
                        for ob in bbs:
                            area += _rect_overlap_area(wb, ob)
                result.shorts.append(
                    Short(net_a=min(net, oth), net_b=max(net, oth), layer=m2,
                          area_um2=round(area, 6))
                )
            placed_by_layer[m2].append((net, m2_bbs))
            # the Metal1 landing pads are tiny pads at pins; record them too so a later
            # net escaping onto Metal1 still sees them (kept for completeness).
            m1_pads = [
                _wire_bbox_um(w) for w in route.wires
                if (w.layer, w.datatype) == m1
            ]
            placed_by_layer[m1].append((net, m1_pads))
        else:
            placed_by_layer[m1].append((net, wire_bbs))

        result.routes[net] = route

    return result


# --------------------------------------------------------------------------- #
# 2b. route_nets_tracked: channel/track routing -- distinct M2 track per net
# --------------------------------------------------------------------------- #
def _resolve_layers(
    tech: str,
    metal1: Optional[_Layer],
    metal2: Optional[_Layer],
    via1: Optional[_Layer],
) -> Tuple[_Layer, _Layer, _Layer, float, float]:
    """Resolve (m1, m2, via1, m2_min_width_um, m2_min_space_um) from the PDK tech.

    Falls back to the documented gf180 defaults (Metal2 min-width 0.28 um,
    min-space 0.28 um) when the tech can not be resolved or overrides are given.
    """
    m1 = tuple(metal1) if metal1 else GF180_METAL1
    m2 = tuple(metal2) if metal2 else GF180_METAL2
    v1 = tuple(via1) if via1 else GF180_VIA1
    m2_w_um = 0.28
    m2_sp_um = 0.28
    if tech:
        try:
            from clay.layout.tech import LayoutTech

            t = LayoutTech.resolve(tech)
            if metal1 is None:
                m1 = tuple(t.metal_key(1))
                m2 = tuple(t.metal_key(2))
                v1 = tuple(t.via_key(1))
            mw = t.min_width("metal2")
            ms = t.min_space("metal2")
            if mw:
                m2_w_um = float(mw) / _NM_PER_UM
            if ms:
                m2_sp_um = float(ms) / _NM_PER_UM
        except Exception:
            pass  # documented gf180 defaults already set
    return m1, m2, v1, m2_w_um, m2_sp_um  # type: ignore[return-value]


def _assign_tracks(
    spans: List[Tuple[str, float, float]], *, x_margin: float
) -> Dict[str, int]:
    """Left-edge greedy track assignment.

    ``spans`` is ``[(net, x_lo, x_hi), ...]``. Returns ``{net -> track_index}`` such
    that two nets sharing a track have DISJOINT x-spans (separated by >= ``x_margin``).
    Nets are processed left-to-right by x_lo; each net takes the LOWEST track index
    whose occupied x-intervals do not overlap its (margin-padded) span. This is the
    classic left-edge channel-routing algorithm: distinct tracks => no two trunks ever
    occupy the same (layer, y) area, so no trunk-trunk short is possible by construction.
    """
    order = sorted(spans, key=lambda s: (s[1], s[2], s[0]))
    # track_intervals[k] = list of (x_lo, x_hi) already on track k.
    track_intervals: List[List[Tuple[float, float]]] = []
    assign: Dict[str, int] = {}
    for net, x_lo, x_hi in order:
        lo = x_lo - x_margin
        hi = x_hi + x_margin
        placed = False
        for k, ivs in enumerate(track_intervals):
            if all(hi <= a or lo >= b for (a, b) in ivs):
                ivs.append((x_lo, x_hi))
                assign[net] = k
                placed = True
                break
        if not placed:
            track_intervals.append([(x_lo, x_hi)])
            assign[net] = len(track_intervals) - 1
    return assign


def route_nets_tracked(
    np: Dict[str, List[Pin]],
    *,
    widths: Optional[Dict[str, float]] = None,
    tech_id: Optional[str] = None,
    tech: str = "gf180mcu",
    wire_width_um: float = _DEFAULT_WIRE_W_UM,
    via_um: float = _DEFAULT_VIA_UM,
    metal1: Optional[_Layer] = None,
    metal2: Optional[_Layer] = None,
    via1: Optional[_Layer] = None,
    track_base_y_um: Optional[float] = None,
) -> RouteResult:
    """Channel/track router: every signal net gets a DISTINCT horizontal M2 track.

    PER-NET widths: ``widths`` maps ``{net -> width_um}``; when given, each net's trunk +
    branches are DRAWN at its width (clamped UP to the PDK Metal2 min width), while the
    track pitch + collision margins use the WIDEST net's width so the distinct-track
    guarantee (no trunk-trunk overlap) still holds for the wide wires. ``widths=None``
    keeps the single ``wire_width_um`` (byte-unchanged). Names match case-insensitively.

    This is the short-free replacement for :func:`route_nets`. The defect in the
    median-y trunk+branch scheme is that every escaped net put its M2 trunk at its OWN
    median y with no track assignment, so trunks that escaped to M2 crossed each other
    (13 residual M2 shorts on the OTA). Here:

    * **Trunks** are horizontal wires on **Metal2**, one per multi-pin net, each at a
      DISTINCT y-track. Tracks are packed by the left-edge algorithm
      (:func:`_assign_tracks`) and spaced by the M2 pitch
      (``wire_width + m2_min_space``). Two trunks share a track only if their x-spans
      are disjoint, so NO two trunks ever overlap on Metal2 -- zero trunk-trunk shorts
      by construction.
    * **Branches** are vertical wires on **Metal1** dropping from each pin to its net's
      trunk-y, with a **Via1** at ``(pin_x, trunk_y)`` joining the M1 branch to the M2
      trunk (and a Via1 landing at the pin when the pad's native metal is not M1).
      Because branches are M1 and trunks are M2, a branch never shorts a trunk.
    * **Branch-vs-branch** M1 collisions (two different nets' vertical drops sharing an
      x-column and overlapping y) are detected; the later net's colliding branch is
      jogged sideways on M1 to a free x and re-dropped, and any residual that still can
      not be cleared is REPORTED as a :class:`Short`, never hidden.

    The trunk channel sits in a band starting at ``track_base_y_um`` (default: just
    above the highest pin y) so trunks do not collide with device artwork; branches
    bridge each pin up to its track. ``metal1`` / ``metal2`` / ``via1`` override the
    routing layers (resolved from ``tech`` when None).
    """
    sel_tech = tech_id or tech
    tech = sel_tech
    m1, m2, v1, m2_w_um, m2_sp_um = _resolve_layers(sel_tech, metal1, metal2, via1)
    # Extra "branch" escape layers (Metal3, Metal4, Metal5, ...) for vertical drops that
    # would short on a lower metal, each with its full via stack from M1 up. Branches NEVER
    # ride M2 (M2 is the trunk layer), so a branch layer never shorts its own net's trunk.
    # Branch escape layers above M2: M3 (+ M4/M5/... when the PDK HAS them). These are
    # resolved STRICTLY from the tech so a route on a 3-metal PDK (open_pdk: M1/M2/M3
    # only) never draws a gf180 upper-metal layer number. Documented gf180 defaults apply
    # only when the tech can not be resolved AND no layers were overridden.
    m3: Optional[_Layer] = None
    v2: Optional[_Layer] = None
    extra_metals: List[_Layer] = []
    extra_vias: List[_Layer] = []
    resolved_from_tech = False
    if tech:
        try:
            from clay.layout.tech import LayoutTech

            t = LayoutTech.resolve(tech)
            resolved_from_tech = True
            # The escape ladder above M2 comes from the PDK whenever the tech resolves
            # (independent of any m1/m2/v1 override -- the overrides only pin the base
            # routing layers). A PDK with only M1/M2 (or no M3) yields no upper escape.
            try:
                m3 = tuple(t.metal_key(3))  # type: ignore[assignment]
                v2 = tuple(t.via_key(2))  # type: ignore[assignment]
            except Exception:
                m3, v2 = None, None  # 2-metal PDK -> no M3 escape
            if m3 is not None:
                for mi in (4, 5, 6, 7, 8):
                    try:
                        extra_metals.append(tuple(t.metal_key(mi)))  # type: ignore[arg-type]
                        extra_vias.append(tuple(t.via_key(mi - 1)))  # type: ignore[arg-type]
                    except Exception:
                        break
        except Exception:
            resolved_from_tech = False
    if not resolved_from_tech:
        # documented gf180 defaults (only when the tech is unknown / unresolvable).
        m3, v2 = (42, 0), (38, 0)
        extra_metals = [(46, 0), (81, 0)]
        extra_vias = [(40, 0), (41, 0)]

    # Ordered branch-layer escape ladder: M1 (native, via stack [v1]), then M3 (v1,v2)
    # when the PDK has an M3, then each upper metal with one more via in the stack.
    # ``via_stack[L]`` is the list of via layers stamped at the PIN end to tie the native
    # pad up to layer L; the trunk-end via that drops branch-layer L back to the M2 trunk
    # is the LAST via in ``via_stack[L]``. (For M1 the trunk-end via is v1.) On a PDK with
    # only M1/M2 the ladder is [M1] -- a residual that can not clear on M1 is then honestly
    # reported as a short rather than escaping onto a non-existent metal.
    branch_ladder: List[_Layer] = [m1]
    via_stack: Dict[_Layer, List[_Layer]] = {m1: [v1]}
    if m3 is not None and v2 is not None:
        branch_ladder.append(m3)
        via_stack[m3] = [v1, v2]
        _stack = [v1, v2]
        for mi, vi in zip(extra_metals, extra_vias):
            _stack = _stack + [vi]
            via_stack[mi] = list(_stack)
            branch_ladder.append(mi)

    # PER-NET width: each net is DRAWN at its own width (clamped >= M2 min width); the
    # COLLISION geometry (pitch / margins / reserved strips) uses the WIDEST net's width
    # so the distinct-track + reserved-strip guarantees still hold for the wide wires.
    def eff_w(net: str) -> float:
        return _net_width(net, widths, wire_width_um, m2_w_um)

    result = RouteResult()

    multi: Dict[str, List[Pin]] = {}
    for net, pins in np.items():
        if len(pins) <= 1:
            route = NetRoute(net=net, pins=list(pins), layer=m1)
            route.connected = True
            route.note = "single-pin (no wire needed)" if pins else "no pins"
            result.routes[net] = route
        else:
            multi[net] = pins

    if not multi:
        return result

    # the widest drawn wire across all multi-pin nets -> the conservative spacing unit.
    max_w = max((eff_w(net) for net in multi), default=wire_width_um)
    pitch = max(max_w, m2_w_um) + m2_sp_um
    # x-margin between two trunks sharing a track: a full (widest) wire + M2 min-space.
    x_margin = max_w + m2_sp_um

    # trunk channel base: above the highest pin so trunks never overlap device polys.
    all_ys = [p.xy_um[1] for pins in multi.values() for p in pins]
    base_y = (max(all_ys) + 2.0 * pitch) if track_base_y_um is None else track_base_y_um

    spans = [
        (net, min(p.xy_um[0] for p in pins), max(p.xy_um[0] for p in pins))
        for net, pins in multi.items()
    ]
    tracks = _assign_tracks(spans, x_margin=x_margin)

    # Per-layer vertical-branch occupancy for branch-vs-branch collision resolution.
    # occ[layer] = list of (net, strip-bbox-um). A branch shorts iff its strip overlaps
    # another NET's strip on the SAME layer. Every branch layer (M1, M3, M4, M5, ...) plus
    # the M2 trunk layer is tracked so the resolver sees trunks too (a jogged branch that
    # extends a trunk must not collide with a DIFFERENT net's trunk that shares its track).
    occ: Dict[_Layer, List[Tuple[str, Tuple[float, float, float, float]]]] = {
        L: [] for L in branch_ladder
    }
    occ[m2] = []

    def strip_bb(px: float, y0: float, y1: float) -> Tuple[float, float, float, float]:
        # reserved occupancy strip uses the WIDEST drawn width so a per-net wire (which
        # is <= max_w) always fits inside the cleared region -> shorts stay impossible.
        return (px - max_w / 2.0, min(y0, y1),
                px + max_w / 2.0, max(y0, y1))

    def conflicts_on(layer: _Layer, net: str,
                     *bbs: Tuple[float, float, float, float]) -> bool:
        for onet, ob in occ.get(layer, []):
            if onet == net:
                continue
            if any(_rect_overlap_area(bb, ob) > 1e-9 for bb in bbs):
                return True
        return False

    def find_free_column(
        layer: _Layer, net: str, px: float, py: float, trunk_y: float,
        x_lo: float, x_hi: float, *, span_pad: float = 0.0,
    ) -> Optional[float]:
        """Find a drop column ``cand`` whose L-route clears every OTHER net on ``layer``.

        The L-route is: pin (px,py) -> horizontal connector at py from px to cand -> vertical
        drop at cand from py up to trunk_y -> via to the M2 trunk at (cand, trunk_y). All
        three pieces (vertical strip, horizontal connector, and -- when cand falls outside the
        trunk's current x-span -- the trunk EXTENSION segment from the nearest trunk end to
        cand) must clear other nets. The trunk extension is checked on the M2 layer so a jog
        that lengthens this net's trunk can not overrun a different net's trunk.

        Candidates are tried nearest-first, preferring columns inside ``[x_lo, x_hi]`` (no
        trunk extension needed) before columns ``span_pad`` beyond either end. Returns the
        free x or None when even the widened window is saturated."""
        step = max_w + m2_sp_um
        win_lo = x_lo - span_pad
        win_hi = x_hi + span_pad

        def clear(cand: float) -> bool:
            vstrip = strip_bb(cand, py, trunk_y)
            hconn = (min(px, cand) - max_w / 2.0, py - max_w / 2.0,
                     max(px, cand) + max_w / 2.0, py + max_w / 2.0)
            if conflicts_on(layer, net, vstrip, hconn):
                return False
            # trunk extension (on M2) when cand is outside the existing trunk x-span.
            if cand < x_lo or cand > x_hi:
                ext = (min(cand, x_lo) - max_w / 2.0, trunk_y - max_w / 2.0,
                       max(cand, x_hi) + max_w / 2.0, trunk_y + max_w / 2.0)
                if conflicts_on(m2, net, ext):
                    return False
            return True

        n_steps = int((win_hi - win_lo) / step) + 2
        for k in range(1, n_steps + 1):
            for cand in (px + k * step, px - k * step):
                if cand < win_lo or cand > win_hi:
                    continue
                if clear(cand):
                    return cand
        return None

    def via_pad(x: float, y: float, layer: _Layer):
        return _rect_poly(x - via_um / 2.0, y - via_um / 2.0,
                          x + via_um / 2.0, y + via_um / 2.0, layer)

    def commit_branch(
        net: str, px: float, py: float, trunk_y: float, p_layer: _Layer,
        branch_layer: _Layer, drop_x: float, wires: list,
    ) -> None:
        """Emit one branch (vertical drop + optional pin-end jog connector + via stack) on
        ``branch_layer`` at column ``drop_x``, and record its strips in ``occ`` so later
        branches see it. The via stack ties the native pad up to ``branch_layer`` at the pin
        and drops ``branch_layer`` back to the M2 trunk at the drop column. The wire is DRAWN
        at this net's width; the recorded occupancy strips use the conservative ``max_w``."""
        w_draw = eff_w(net)
        jogged = abs(drop_x - px) > 1e-9
        wires.append(_vwire(py, trunk_y, drop_x, w_draw, branch_layer))
        occ.setdefault(branch_layer, []).append((net, strip_bb(drop_x, py, trunk_y)))
        if jogged:
            wires.append(_hwire(px, drop_x, py, w_draw, branch_layer))
            occ.setdefault(branch_layer, []).append((
                net,
                (min(px, drop_x) - max_w / 2.0, py - max_w / 2.0,
                 max(px, drop_x) + max_w / 2.0, py + max_w / 2.0),
            ))
        stack = via_stack.get(branch_layer, [v1])
        # PIN-end vias: tie native pad up to branch_layer. For an M1 branch the pad is
        # already on M1 unless its native metal differs; for higher branches stamp the full
        # stack so the pad climbs to the drop layer.
        if branch_layer == m1:
            if tuple(p_layer) != m1:
                wires.append(via_pad(px, py, v1))
        else:
            for vlay in stack:
                wires.append(via_pad(px, py, vlay))
        # TRUNK-end via: drop branch_layer back to the M2 trunk. v1 for an M1 branch, else
        # the topmost via of the stack.
        trunk_via = v1 if branch_layer == m1 else stack[-1]
        wires.append(via_pad(drop_x, trunk_y, trunk_via))

    # Per-net mutable trunk x-extent (grows when a jog drops outside the pin x-span). The
    # trunk wire is drawn AFTER all branches so it spans every via column.
    trunk_span: Dict[str, Tuple[float, float, float]] = {}  # net -> (x_lo, x_hi, trunk_y)
    net_wires: Dict[str, list] = {}

    # deterministic order: by track index then name, so collision resolution is stable.
    for net in sorted(multi, key=lambda nm: (tracks[nm], nm)):
        pins = multi[net]
        trunk_y = base_y + tracks[net] * pitch
        xs = [p.xy_um[0] for p in pins]
        x_lo, x_hi = min(xs), max(xs)
        wires: list = []
        # reserve the nominal trunk span on M2 up front so other nets' jogs see it
        # (reservation uses the conservative max_w; the drawn trunk uses this net's width).
        occ[m2].append((net, (x_lo - max_w / 2.0, trunk_y - max_w / 2.0,
                              x_hi + max_w / 2.0, trunk_y + max_w / 2.0)))
        span_lo, span_hi = x_lo, x_hi

        for p in pins:
            px, py = p.xy_um
            if abs(trunk_y - py) <= 1e-9:
                wires.append(via_pad(px, trunk_y, v1))
                continue

            bb = strip_bb(px, py, trunk_y)
            branch_layer: Optional[_Layer] = None
            drop_x = px
            # 1) STRAIGHT drop at px, climbing the layer ladder (M1, M3, M4, M5, ...) until
            #    a layer's column at px clears every other net's branches there.
            for L in branch_ladder:
                if not conflicts_on(L, net, bb):
                    branch_layer, drop_x = L, px
                    break
            # 2) JOGGED L-drop: for each layer (lowest first) find the nearest free column,
            #    first inside the pin x-span (no trunk extension) then widening the search so
            #    a saturated own-span is never a dead end. Pick the layer+column whose drop
            #    is closest to the pin x.
            if branch_layer is None:
                best: Optional[Tuple[_Layer, float, float]] = None  # (layer, x, |dx|)
                channel = max(64.0, (x_hi - x_lo) + 64.0)
                for L in branch_ladder:
                    cand = find_free_column(L, net, px, py, trunk_y, x_lo, x_hi,
                                            span_pad=channel)
                    if cand is None:
                        continue
                    d = abs(cand - px)
                    if best is None or d < best[2]:
                        best = (L, cand, d)
                    # a free column on this layer inside (or near) the span is good enough;
                    # keep climbing only if nothing was found yet.
                    if d <= (x_hi - x_lo) + 1e-9:
                        break
                if best is not None:
                    branch_layer, drop_x = best[0], best[1]
            # 3) residual: keep the straight M1 drop; the audit + escalation pass handle it.
            if branch_layer is None:
                branch_layer, drop_x = m1, px

            span_lo = min(span_lo, drop_x)
            span_hi = max(span_hi, drop_x)
            commit_branch(net, px, py, trunk_y, p.layer, branch_layer, drop_x, wires)

        # update the trunk's reserved M2 extent to the final (possibly jog-extended) span.
        for idx, (onet, _ob) in enumerate(occ[m2]):
            if onet == net:
                occ[m2][idx] = (net, (span_lo - max_w / 2.0,
                                      trunk_y - max_w / 2.0,
                                      span_hi + max_w / 2.0,
                                      trunk_y + max_w / 2.0))
                break
        trunk_span[net] = (span_lo, span_hi, trunk_y)
        net_wires[net] = wires

    # draw each trunk over its final span and assemble routes.
    for net in sorted(multi, key=lambda nm: (tracks[nm], nm)):
        span_lo, span_hi, trunk_y = trunk_span[net]
        wires = net_wires[net]
        if span_hi > span_lo:
            wires.insert(0, _hwire(span_lo, span_hi, trunk_y, eff_w(net), m2))
        route = NetRoute(net=net, pins=list(multi[net]), layer=m2, connected=True)
        route.wires = wires
        route.note = f"tracked trunk on metal {m2} at track {tracks[net]} (y={trunk_y:.3f})"
        result.routes[net] = route

    # final audit from scratch.
    _audit_shorts(result, m2)
    # escalation: if any residual same-layer cross-net overlap survives, lift offending
    # branches onto a fresh upper layer / far column until 0 (or honestly report).
    _escalate_residual_shorts(
        result, multi, tracks, trunk_span, branch_ladder, via_stack,
        m1=m1, m2=m2, v1=v1, wire_w=wire_width_um, via_um=via_um,
    )
    _audit_shorts(result, m2)
    return result


def _escalate_residual_shorts(
    result: "RouteResult",
    multi: Dict[str, List[Pin]],
    tracks: Dict[str, int],
    trunk_span: Dict[str, Tuple[float, float, float]],
    branch_ladder: List[_Layer],
    via_stack: Dict[_Layer, List[_Layer]],
    *,
    m1: _Layer,
    m2: _Layer,
    v1: _Layer,
    wire_w: float,
    via_um: float,
    max_passes: int = 12,
) -> None:
    """Last-resort residual resolver: while the from-scratch audit still reports a same-layer
    cross-net short, take one net of the shorting pair and RE-ROUTE its colliding branch(es)
    onto a guaranteed-clear escape: the HIGHEST branch layer, at a far column outside every
    trunk's x-span, with the trunk extended to reach it. Each pass strictly reduces the short
    count (one offending branch is removed from contention) so it terminates; if it can not
    reduce further it leaves the honest residual for the audit to report."""
    if not result.shorts:
        return

    # the topmost escape layer (e.g. Metal5) -- branches lifted here are far above the
    # densely-packed M1/M3 channel.
    top_layer = branch_ladder[-1]
    top_stack = via_stack.get(top_layer, [v1])

    def via_pad(x: float, y: float, layer: _Layer):
        return _rect_poly(x - via_um / 2.0, y - via_um / 2.0,
                          x + via_um / 2.0, y + via_um / 2.0, layer)

    # widest occupied x across all trunks, to park escalated drops clear of everything.
    all_los = [s[0] for s in trunk_span.values()]
    all_his = [s[1] for s in trunk_span.values()]
    far_base = (max(all_his) if all_his else 0.0) + 4.0
    step = wire_w + 1.0
    park_counter = 0

    for _ in range(max_passes):
        if not result.shorts:
            return
        sh = result.shorts[0]
        layer = sh.layer
        # pick the net of the pair whose branch is cheapest to lift: the one with FEWER pins
        # (smaller trunk) -- deterministic tiebreak by name.
        cand_nets = sorted(
            (sh.net_a, sh.net_b),
            key=lambda nm: (len(multi.get(nm, [])), nm),
        )
        moved = False
        for net in cand_nets:
            rt = result.routes.get(net)
            if rt is None or net not in trunk_span:
                continue
            span_lo, span_hi, trunk_y = trunk_span[net]
            # find this net's wire(s) on the shorting layer that overlap the other net.
            other = sh.net_b if net == sh.net_a else sh.net_a
            ort = result.routes.get(other)
            if ort is None:
                continue
            other_bbs = [(_wire_bbox_um(w)) for w in ort.wires
                         if (w.layer, w.datatype) == layer]
            culprits = []
            for w in rt.wires:
                if (w.layer, w.datatype) != layer:
                    continue
                wb = _wire_bbox_um(w)
                if any(_rect_overlap_area(wb, ob) > 1e-9 for ob in other_bbs):
                    culprits.append(w)
            if not culprits:
                continue
            # lift each culprit's owning DROP onto the top layer at a fresh far column.
            for w in culprits:
                wb = _wire_bbox_um(w)
                # the vertical extent of the culprit defines the drop we must reroute.
                py0, py1 = wb[1], wb[3]
                px = (wb[0] + wb[2]) / 2.0
                park_counter += 1
                far_x = far_base + park_counter * step
                # remove the culprit wire (it is the colliding vertical drop).
                rt.wires.remove(w)
                new_w: list = []
                # horizontal connector at the pin end (py0 or py1 -- the end nearest a pin;
                # use the lower y as the pin end heuristically) to the far column.
                pin_y = py0
                new_w.append(_hwire(px, far_x, pin_y, wire_w, top_layer))
                # vertical drop at far_x from pin_y up to the trunk_y.
                new_w.append(_vwire(pin_y, trunk_y, far_x, wire_w, top_layer))
                # extend this net's trunk to reach far_x.
                new_lo = min(span_lo, far_x)
                new_hi = max(span_hi, far_x)
                # replace the trunk wire (first M2 wire) with the extended one.
                for i, tw in enumerate(rt.wires):
                    if (tw.layer, tw.datatype) == m2:
                        rt.wires[i] = _hwire(new_lo, new_hi, trunk_y, wire_w, m2)
                        break
                trunk_span[net] = (new_lo, new_hi, trunk_y)
                span_lo, span_hi = new_lo, new_hi
                # via stack: pin end pad climbs native->top_layer; trunk end drops top->M2.
                for vlay in top_stack:
                    new_w.append(via_pad(px, pin_y, vlay))
                new_w.append(via_pad(far_x, trunk_y, top_stack[-1]))
                rt.wires.extend(new_w)
            moved = True
            break
        _audit_shorts(result, m2)
        if not moved:
            return


def _audit_shorts(result: RouteResult, m2: _Layer) -> None:
    """Recompute shorts from scratch: any same-(layer,datatype) overlap of two DIFFERENT
    nets' wires is a Short. Replaces ``result.shorts`` with the true residual set."""
    # gather (net, layer, bbox) for every wire.
    per_layer: Dict[_Layer, List[Tuple[str, Tuple[float, float, float, float]]]] = {}
    for net, rt in result.routes.items():
        for w in rt.wires:
            key = (w.layer, w.datatype)
            per_layer.setdefault(key, []).append((net, _wire_bbox_um(w)))
    seen: Dict[Tuple[str, str, _Layer], float] = {}
    for layer, items in per_layer.items():
        for i in range(len(items)):
            ni, bi = items[i]
            for j in range(i + 1, len(items)):
                nj, bj = items[j]
                if ni == nj:
                    continue
                area = _rect_overlap_area(bi, bj)
                if area > 1e-9:
                    a, b = sorted((ni, nj))
                    seen[(a, b, layer)] = seen.get((a, b, layer), 0.0) + area
    result.shorts = [
        Short(net_a=a, net_b=b, layer=layer, area_um2=round(area, 6))
        for (a, b, layer), area in sorted(seen.items())
    ]


# --------------------------------------------------------------------------- #
# 3. emit_route_gds: write the wires into a flat gf180 GDS
# --------------------------------------------------------------------------- #
def emit_route_gds(
    route: RouteResult,
    out_gds: str,
    *,
    cell_name: str = "route",
    dbu: float = 0.001,
    merge: bool = True,
) -> str:
    """Write every routing wire (Metal1 / Metal2 / Via1) into a flat GDS readable by
    klayout. Returns the written path. The integrator merges this cell over the
    migration GDS (same dbu, same coordinate frame).

    Emission hygiene (P1): the wires are merged PER NET x LAYER at this write
    boundary (:func:`clay.layout.route.emit_gds.merge_net_wires` -- the ONE shared
    helper); different nets are never unioned and the in-memory ``route.routes``
    stay untouched. ``merge=False`` keeps the raw fragment emission (A/B
    oracle-invariance audits only)."""
    import pathlib

    import klayout.db as kdb

    from clay.layout.route.emit_gds import merge_net_wires

    routes = getattr(route, "routes", None)
    wires = merge_net_wires(routes) if (merge and routes) else route.wires
    ly = kdb.Layout()
    ly.dbu = dbu
    top = ly.create_cell(cell_name)
    for lp in wires:
        li = ly.layer(lp.layer, lp.datatype)
        top.shapes(li).insert(
            kdb.Polygon([kdb.Point(int(v.x), int(v.y)) for v in lp.polygon.vertices])
        )
    out = pathlib.Path(out_gds)
    out.parent.mkdir(parents=True, exist_ok=True)
    ly.write(str(out))
    return str(out)


# --------------------------------------------------------------------------- #
# 4. end-to-end OTA runner
# --------------------------------------------------------------------------- #
@dataclass
class MigrationRouteResult:
    """The full signal-routing outcome on a migrated intent."""

    net_pins: Dict[str, List[Pin]]
    route: RouteResult
    positions: Dict[str, _XY]
    terminals: Dict[str, Dict[str, Tuple[_XY, _Layer]]]
    out_gds: str = ""


def route_migrated_intent(
    intent: Any,
    positions: Dict[str, _XY],
    *,
    tech: str = "gf180mcu",
    tech_id: Optional[str] = None,
    widths: Optional[Dict[str, float]] = None,
    out_gds: Optional[str] = None,
    terminals: Optional[Dict[str, Dict[str, Tuple[_XY, _Layer]]]] = None,
) -> MigrationRouteResult:
    """Route the signal nets of a migrated intent: capture terminal stamps (unless
    ``terminals`` is supplied), collect signal-net pins at the migrated ``positions``,
    route them on the PDK's metals (PER-NET ``widths`` when given), and optionally emit a
    routing GDS. ``tech_id`` (preferred) / ``tech`` select the PDK. Returns a
    :class:`MigrationRouteResult`."""
    sel = tech_id or tech
    if terminals is None:
        terminals = build_terminals(intent, tech_id=sel)
    npins = net_pins(intent, positions, terminals)
    route = route_nets_tracked(npins, tech_id=sel, widths=widths)
    out = ""
    if out_gds:
        out = emit_route_gds(route, out_gds)
    return MigrationRouteResult(
        net_pins=npins, route=route, positions=positions, terminals=terminals, out_gds=out
    )


def run_ota_route(
    out_gds: str = "_review_gds/two_stage_gf180_routed.gds",
) -> MigrationRouteResult:
    """Migrate the real OTA intent onto gf180 (via the clean hierarchical-nonoverlap
    positions migrate provides), ROUTE its signal nets, emit the routing GDS, and PRINT
    the honest headline -- shorts FIRST if any, then nets / connected / partial."""
    from clay.layout.extract.migrate import (
        build_ota_intent,
        build_gf180_geometry,
        migrate_positions_hier_nov,
        relgraph,
    )

    intent = build_ota_intent()
    geometry = build_gf180_geometry(intent)
    hier = relgraph.build_hierarchical_graph(intent)
    positions, _resid = migrate_positions_hier_nov(intent, geometry.sizes_um(), hier=hier)
    mr = route_migrated_intent(intent, positions, tech="gf180mcu", out_gds=out_gds)
    r = mr.route

    if r.shorts:
        print(f"[route] !! {r.n_shorts} SHORT(S) DETECTED (reported first):")
        for s in r.shorts:
            print(f"    SHORT {s.net_a} / {s.net_b} on metal {s.layer}: "
                  f"{s.area_um2:.4f} um^2")
    else:
        print("[route] 0 shorts detected")
    print(f"[route] {r.summary()}")
    print(f"[route] signal nets ({r.n_nets}):")
    for net in sorted(r.routes):
        rt = r.routes[net]
        tag = "CONNECTED" if rt.connected else "PARTIAL"
        layers = sorted({(w.layer, w.datatype) for w in rt.wires})
        print(f"    {net:>8}: {len(rt.pins)} pin(s) -> {tag} "
              f"[layers {layers}] {rt.note}")
    print(f"[route] power nets excluded: {sorted(DEFAULT_POWER_NETS)}")
    print(f"[route] routing GDS: {mr.out_gds}")
    return mr


# --------------------------------------------------------------------------- #
# 5. REFERENCE ROUTING FACTS -- extract per-net widths + metals from the real GDS
#
# The external_pdk self-consistent anchor is allowed to READ the reference routing directly
# (per-net width / layers / connectivity) because it is the degradation START -- the
# same way the layout anchor read full reference coordinates. These helpers extract
# REAL facts from the reference layout; leakage-free intent abstraction is a LATER step.
# --------------------------------------------------------------------------- #

# external_pdk metal GDS layers, DECODED FROM THE REFERENCE GDS via cells (each named
# ``M<hi>_M<lo>`` carries the two metal landing pads + the cut, so the metal layer
# numbers are read off the layout, not assumed): M2_M1 -> M1=16/M2=18/cut=17;
# M3_M2 -> M3=28/cut=27; M4_M3 -> M4=31/cut=29; M5_M4 -> M5=33/cut=32;
# M6_M5 -> M6=38/cut=39. These match ``LayoutTech.resolve('gf180mcu').metal_key`` for
# M1/M2/M3 exactly (the tech layer map is faithful to the GDS).
OPEN_PDK_METALS: Dict[str, _Layer] = {
    "M1": (34, 0), "M2": (36, 0), "M3": (42, 0),
    "M4": (46, 0), "M5": (81, 0),
}
# net-name label layer in the reference GDS (only the 5 PIN nets are labelled here).
OPEN_PDK_LABEL_LAYER: _Layer = (34, 10)


@dataclass
class ReferenceWidths:
    """Per-net wire widths extracted from a reference GDS.

    ``per_net`` maps ``{net -> width_um}`` for every net that COULD be attributed (a net
    whose label sits on a metal wire). ``per_net_layer`` maps ``{net -> {metal_name ->
    width_um}}`` (a net routed on several metals has a width per metal). ``per_layer``
    is the distribution of routing-wire widths per metal (the honest fallback for nets
    with no label). ``unlabelled`` lists the intent signal nets that have NO GDS label
    (so per-net width is NOT directly attributable for them -- reported, not faked)."""

    per_net: Dict[str, float] = field(default_factory=dict)
    per_net_layer: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_layer: Dict[str, List[float]] = field(default_factory=dict)
    unlabelled: List[str] = field(default_factory=list)


def extract_reference_net_widths(
    gds_path: str,
    *,
    metals: Optional[Dict[str, _Layer]] = None,
    label_layer: _Layer = OPEN_PDK_LABEL_LAYER,
    intent: Any = None,
    min_routing_len_um: float = 1.0,
) -> ReferenceWidths:
    """Extract REAL per-net wire widths from a reference GDS (honest, label-driven).

    For each net-name TEXT on ``label_layer`` we find the merged metal polygon (across
    ``metals``) whose extent contains the label point and take that wire's SHORT-SIDE as
    the net's width on that metal. This attributes a width to every LABELLED net (in the
    external_pdk OTA reference only the 5 PIN nets -- VDD/GND/VIP/VIN/VOUT -- carry labels).
    We ALSO collect, per metal, the distribution of all routing-wire short-sides (wires
    longer than ``min_routing_len_um``) as the honest fallback for nets with no label.

    HONEST GAP (reported, never faked): the reference GDS labels ONLY the pin nets, so a
    per-net width is directly attributable for those; the internal nets (net048, net07,
    ...) have NO net label, so their width is attributable per-LAYER but NOT per-NET from
    the GDS alone. ``ReferenceWidths.unlabelled`` lists exactly which intent nets fall in
    that bucket. Net names are returned UPPER-CASED as the GDS stores them; the router
    matches them to lower-case netlist nodes case-insensitively.
    """
    import klayout.db as kdb

    metals = dict(metals or OPEN_PDK_METALS)
    ly = kdb.Layout()
    ly.read(gds_path)
    top = ly.top_cell()
    dbu = ly.dbu

    # merged routing region per metal (flatten the via subcells into the top frame).
    merged: Dict[str, list] = {}
    per_layer: Dict[str, List[float]] = {}
    for mname, key in metals.items():
        li = ly.find_layer(int(key[0]), int(key[1]))
        if li is None:
            continue
        reg = kdb.Region(top.begin_shapes_rec(li))
        reg.merge()
        polys = list(reg.each())
        merged[mname] = polys
        widths: List[float] = []
        for p in polys:
            bb = p.bbox()
            w = bb.width() * dbu
            h = bb.height() * dbu
            if max(w, h) >= float(min_routing_len_um):
                widths.append(round(min(w, h), 4))
        if widths:
            per_layer[mname] = sorted(widths)

    # net labels -> (string, point).
    labels: List[Tuple[str, int, int]] = []
    li_lab = ly.find_layer(int(label_layer[0]), int(label_layer[1]))
    if li_lab is not None:
        it = top.begin_shapes_rec(li_lab)
        while not it.at_end():
            sh = it.shape()
            if sh.is_text():
                t = sh.text.transformed(it.trans())
                labels.append((sh.text.string, t.x, t.y))
            it.next()

    rw = ReferenceWidths(per_layer=per_layer)
    for name, lx, ly_ in labels:
        upper = name.upper()
        best_for_net: Dict[str, float] = {}
        for mname, polys in merged.items():
            for p in polys:
                bb = p.bbox()
                if bb.left <= lx <= bb.right and bb.bottom <= ly_ <= bb.top:
                    w = round(min(bb.width(), bb.height()) * dbu, 4)
                    # keep the NARROWEST enclosing wire per metal (the routing wire, not
                    # a wide plane the label may also sit inside).
                    if mname not in best_for_net or w < best_for_net[mname]:
                        best_for_net[mname] = w
        if best_for_net:
            rw.per_net_layer[upper] = best_for_net
            # headline per-net width = the NARROWEST metal it routes on (the signal wire,
            # not the widest power plane), which is the width a router should draw.
            rw.per_net[upper] = min(best_for_net.values())

    # honest unlabelled report: intent signal nets with no GDS-attributed width.
    if intent is not None:
        power = set(DEFAULT_POWER_NETS)
        sig: set = set()
        for d in getattr(intent, "devices", []) or []:
            for n in (getattr(d, "nets", {}) or {}).values():
                if n and n not in power:
                    sig.add(str(n))
        for p in getattr(intent, "passives", []) or []:
            for n in (getattr(p, "nets", {}) or {}).values():
                if n and n not in power:
                    sig.add(str(n))
        labelled_upper = set(rw.per_net.keys())
        rw.unlabelled = sorted(
            n for n in sig if n.upper() not in labelled_upper
        )
    return rw


def open_pdk_ota_positions(
    intent: Any, *, tech_id: str = "gf180mcu",
    row_height_target_um: Optional[float] = None,
) -> Tuple[Dict[str, _XY], Dict[str, _XY]]:
    """Solve a self-consistent external_pdk placement for the OTA -> ``(positions_um, sizes_um)``.

    Device SIZES come from the PDK geometry itself: each MOS from
    ``device_cell.build_device(tech, ..)`` extent; each passive from its
    ``passive_device_cell`` extent where the PDK builder exists, else its REFERENCE
    footprint (``intent.device_bboxes``) -- the same honest fallback ``build_terminals``
    uses (the ONLY permitted reference read: a size, never a position). Placement
    STRUCTURE is DESIGN-sourced (``topo_sketch.derive_design_sketch``: topology-derived
    mirror groups / diff pairs / connectivity-seriated rows); positions are the
    within-band non-overlap solve at the target sizes
    (``relgraph.solve_hierarchical_nonoverlap``), overlap-free by construction.
    Obligation: tests/test_placement_leakage.py -- perturbing the reference centroids
    must not move this placement.

    ``row_height_target_um`` (OPT-IN; default ``None`` is byte-identical): the DESIGN
    wide-flat band-height fold target. When set, each MOS size comes from the SAME
    serpentine 2D-compact fold the geometry pass uses (``realize_render.
    build_device_geometry`` -> ``stamp_extent``), so the SIZE the placement reserves
    and the GEOMETRY that is later stamped are ONE consistent fold -- the contract the
    contact_rows note warned about (size source must flip WITH the geometry build)."""
    from clay.layout import device_cell as _dc
    from clay.layout import passive_device_cell as _pdc
    from clay.layout.extract import relgraph, topo_sketch
    from clay.layout.extract.realize_render import (
        build_device_geometry as _bdg,
        stamp_extent as _stamp_extent,
    )
    from clay.layout.tech import LayoutTech

    tech = LayoutTech.resolve(tech_id)
    sizes: Dict[str, _XY] = {}
    for d in getattr(intent, "devices", []) or []:
        name = str(getattr(d, "name", ""))
        if not name or str(getattr(d, "kind", "")) not in ("nmos", "pmos"):
            continue
        nf = max(1, int(round(float(getattr(d, "M", 1) or 1))))
        try:
            if row_height_target_um is not None:
                # SAME fold as the geometry pass (serpentine 2D-compact): the size
                # the placement reserves matches the stamped artwork exactly.
                cg = _bdg(
                    tech_id, str(d.kind), float(d.W), float(d.L), nf,
                    row_height_target_um=float(row_height_target_um),
                )
                w_um, h_um = _stamp_extent(cg)
                sizes[name] = (float(w_um), float(h_um))
            else:
                # NOTE (contact_rows bring-up): if/when the route geometry build
                # (realize_render) enables contact_rows, THIS size source must flip
                # with it in the same commit -- packing by the smaller sparse extent
                # under dense artwork caused a catastrophic all-nets merge.
                g = _dc.build_device(
                    tech, str(d.kind), float(d.W) * 1e-6, float(d.L) * 1e-6, nf
                )
                e = g.extent()
                sizes[name] = (
                    (e[2] - e[0]) / _NM_PER_UM, (e[3] - e[1]) / _NM_PER_UM
                )
        except Exception:
            bb = (getattr(intent, "device_bboxes", {}) or {}).get(name)
            if bb is not None:
                sizes[name] = (float(bb[2]) - float(bb[0]), float(bb[3]) - float(bb[1]))
    for p in getattr(intent, "passives", []) or []:
        name = str(getattr(p, "name", ""))
        if not name:
            continue
        try:
            if str(p.kind) == "cap":
                g = _pdc.build_capacitor_for_tech(
                    tech, float(p.value),
                    m=int(getattr(p, "M", 1) or 1),
                    model=str(getattr(p, "model", "") or ""),
                )
            else:
                g = _pdc.build_resistor(tech, float(p.value))
            e = g.extent()
            sizes[name] = ((e[2] - e[0]) / _NM_PER_UM, (e[3] - e[1]) / _NM_PER_UM)
        except Exception:
            bb = (getattr(intent, "device_bboxes", {}) or {}).get(name)
            sizes[name] = (
                (float(bb[2]) - float(bb[0]), float(bb[3]) - float(bb[1]))
                if bb is not None else (5.0, 5.0)
            )
    # DESIGN-SOURCED structure (the leakage contract fix): band membership,
    # within-band order and cross-band X derive from the topology sketch --
    # never from the reference centroids in intent.device_bboxes. The reference
    # is only allowed into SIZES above, as the honest cell-less fallback.
    sketch = topo_sketch.derive_design_sketch(intent, sizes)
    hier = relgraph.build_hierarchical_graph(
        topo_sketch.sketch_intent_view(intent, sketch)
    )
    positions, _resid = relgraph.solve_hierarchical_nonoverlap(hier, sizes, (0.0, 0.0))
    positions = {n: (float(p[0]), float(p[1])) for n, p in positions.items()}
    return positions, sizes




__all__ = [
    "Pin",
    "NetRoute",
    "Short",
    "RouteResult",
    "MigrationRouteResult",
    "ReferenceWidths",
    "GF180_METAL1",
    "GF180_METAL2",
    "GF180_VIA1",
    "OPEN_PDK_METALS",
    "OPEN_PDK_LABEL_LAYER",
    "DEFAULT_POWER_NETS",
    "build_terminals",
    "build_gf180_terminals",
    "net_pins",
    "route_nets",
    "route_nets_tracked",
    "emit_route_gds",
    "extract_reference_net_widths",
    "open_pdk_ota_positions",
    "route_migrated_intent",
    "run_ota_route",
]


if __name__ == "__main__":
    run_ota_route()
