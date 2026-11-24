"""Clean, leakage-free PIN EXTRACTOR for the faithful router.

WHAT THIS CLOSES
================
The router engine (:mod:`clay.layout.route.engine`) consumes per-net PINS: for
every signal net, the set of device-terminal access points it must join, each on a
ROUTING-accessible metal layer. The legacy pin source
(:func:`clay.layout.extract.route_migrate.build_terminals` / ``net_pins``) is broken
for routing in two ways this module fixes:

  * It put GATE terminals on the device's smallest enclosing polygon -- which for a
    gate is the poly->M1 CONTACT CUT (open_pdk ``(15, 0)``), NOT a routing metal
    (routing metals are M1=``(16, 0)`` / M2=``(18, 0)`` / M3=``(28, 0)``). A router
    cannot land on a pin that is not on a routing layer.
  * Net capture leaned on a separate ``terminals`` stamp keyed off a different
    centroid than the placement, so some pins could be incomplete or off-device.

This module produces, for each SIGNAL net (power nets excluded), the device-terminal
pins at ABSOLUTE micron positions, each on M1, COMPLETE per net (every device
terminal the netlist puts on the net).

THE GATE -> M1 ACCESS SOLUTION (the crux)
========================================
A gate is poly; in a real cell it climbs poly -> contact -> M1. We do NOT route on
poly. The device cell (:func:`clay.layout.device_cell.build_device`) already draws an
M1 PAD at the gate contact (``_m1_pad(gate_cx, gate_cy)`` over the gate-contact cut):
the gate terminal's centre sits INSIDE that M1 pad. So the gate's M1 landing access
point is simply the gate terminal centre snapped onto the device cell's OWN M1 shape
that covers it. No M1 tap has to be synthesised on the standard cells here -- the
device cell exposes the poly->M1 landing already. (When -- on some other cell -- a
terminal's centre were NOT covered by any M1 shape, we snap to the NEAREST M1 shape
of the same cell and document that as a synthesized landing; on the external_pdk OTA every
s/d/g terminal centre is inside an M1 shape, dist 0.)

LEAKAGE DISCIPLINE (why this module exists)
==========================================
This whole router subproject exists because a past experiment leaked the reference
answer into the input. Pin positions + layers here are DERIVED ONLY from:
  * the device PLACEMENT (``placement`` -- solved centroids, leakage-free), and
  * the DEVICE-CELL terminal geometry (``build_device`` for MOS, ``passive_cell``
    for a passive where the PDK has a builder -- parametric from the netlist W/L/M
    or value + the PDK rules), and
  * the PDK (``tech`` -- the M1 layer key).
We NEVER read pin positions, widths, or tracks from the REFERENCE ROUTING (the
reference's metal wires) -- that routing is the answer the router must reproduce.
We may read the reference for device PLACEMENT/extent (the device cell), but never its
routing geometry. (The one place a placed-stamp M1 shape is consulted -- a passive on
a PDK with no parametric passive cell -- uses the passive's OWN terminal-contact M1
artwork from its device stamp, i.e. the device cell, not a routed wire; see
:func:`_passive_pins_from_stamp` and the honest-gap note it carries.)

GENERICITY
==========
No per-circuit branching: every device/passive is handled by the same parametric
path. The only DATA branch is device KIND (MOS vs cap vs res) -- the geometry source
differs by kind, never by device or cell name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from clay.layout.placed_geometry import placed_point_nm
from clay.layout.route import Pin

_NM_PER_UM = 1000.0

_XY = Tuple[float, float]
_Layer = Tuple[int, int]
_Bbox = Tuple[int, int, int, int]

# Global power / supply nets are a sibling's job (power distribution / rails) and are
# excluded from SIGNAL pin extraction. Matched case-insensitively, with or without a
# trailing '!' (auCdl global-net mark) and with the common vdd/vss/gnd spellings.
_POWER_BASES = frozenset(
    {"vdd", "vss", "gnd", "vcc", "vee", "vpwr", "vgnd", "vnb", "vpb", "avdd", "avss"}
)

# device-net ROLE key (the netlist ``ExtractedDevice.nets`` keys) -> the device-cell
# TERMINAL key (the ``DeviceGeom.terminals`` keys). MOS roles are full words in the
# netlist; the device cell uses single letters.
_MOS_ROLE_TO_TERM = {"source": "s", "drain": "d", "gate": "g", "body": "b"}
# passive ROLE key -> terminal key (already single letters in both).
_PASSIVE_ROLE_TO_TERM = {"p": "p", "n": "n"}
_BJT_ROLE_TO_TERM = {"c": "c", "b": "b", "e": "e"}


# --------------------------------------------------------------------------- #
# result container (honest provenance + gaps)
# --------------------------------------------------------------------------- #
@dataclass
class PinReport:
    """Provenance + honest gaps for an :func:`extract_pins` run.

    ``pins_by_net`` is the routable result. ``power_nets`` are the nets excluded as
    supply. ``uncovered`` lists ``(device, role, net, reason)`` for any device
    terminal that could NOT be given a real M1 access point (e.g. a cap whose device
    cell has no M1 on a PDK with no parametric cap cell) -- these are reported, never
    faked into a floating pin.

    ``snaps`` lists ``(device, terminal, dist_um)`` for EVERY pin whose parametric
    terminal position had to be SNAPPED onto the placed stamp's M1 (dist > a sub-grid
    epsilon). This is honest provenance for the vendor-stamp case: a device whose
    VENDOR PCell folds its width differently than the parametric ``build_device`` cell
    (e.g. a tall single-finger build_device vs a multi-finger vendor fold) has its
    terminal snapped onto the nearest placed M1 shape, so the pin still lands on REAL
    placed metal of THAT device, but at the nearest M1 edge rather than the exact
    parametric finger. ``max_snap_um`` summarises the worst snap. A small snap
    (sub-micron) is just edge-clamping; a large snap flags a build/vendor fold
    mismatch the caller may want to know about. ``synthesized_landings`` is the subset
    of ``snaps`` whose terminal centre was not inside ANY placed M1 shape (kept for
    back-compat as ``(device, terminal)`` pairs).
    """

    pins_by_net: Dict[str, List[Pin]] = field(default_factory=dict)
    power_nets: List[str] = field(default_factory=list)
    uncovered: List[Tuple[str, str, str, str]] = field(default_factory=list)
    snaps: List[Tuple[str, str, float]] = field(default_factory=list)
    synthesized_landings: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def n_pins(self) -> int:
        return sum(len(v) for v in self.pins_by_net.values())

    @property
    def max_snap_um(self) -> float:
        return max((d for (_dev, _t, d) in self.snaps), default=0.0)


# --------------------------------------------------------------------------- #
# net classification
# --------------------------------------------------------------------------- #
def _is_power_net(net: str, extra_power: Sequence[str]) -> bool:
    """True iff ``net`` is a global supply net (excluded from signal routing).

    Matches a small vocabulary (vdd/vss/gnd/...) case-insensitively, with or without a
    trailing '!' global mark, plus any caller-supplied ``extra_power`` names (exact,
    case-insensitive). A pure signal name like ``net048`` / ``vout`` / ``VIP`` is
    never matched."""
    if not net:
        return True  # an unbound terminal is not routable
    low = net.strip().lower()
    base = low[:-1] if low.endswith("!") else low
    if base in _POWER_BASES:
        return True
    extra = {str(p).strip().lower().rstrip("!") for p in extra_power}
    return base in extra


def _manufacturing_grid_access_on_stamp(
    stamp: Any,
    placement_um: _XY,
    layer: _Layer,
    access_nm: Tuple[int, int],
    tech: Any,
) -> Tuple[int, int]:
    """Return the nearest on-grid point on the access's original conductor.

    A terminal access is electrical ownership, not merely a coordinate.  Moving
    it independently to the nearest manufacturing-grid point can cross a narrow
    gap onto another conductor (notably the two plates of a MIM capacitor).  This
    helper reconstructs the exact placed stamp component containing the original
    access and searches only its one-grid interior.  If the component has no such
    point, retain the original access so the exact-artifact gate fails closed
    rather than fabricating connectivity.
    """

    from clay.layout.route.emit_gds import (
        _nearest_on_grid,
        pdk_manufacturing_grid_nm,
    )

    grid_nm = int(pdk_manufacturing_grid_nm(tech))
    source = (int(access_nm[0]), int(access_nm[1]))
    if grid_nm <= 1 or all(value % grid_nm == 0 for value in source):
        return source
    rings = (getattr(stamp, "polys_by_layer", {}) or {}).get(tuple(layer), ())
    if not rings:
        return source

    import klayout.db as kdb

    region = kdb.Region()
    for ring in rings:
        points = [
            kdb.Point(*placed_point_nm(float(x), float(y), placement_um))
            for x, y in ring
        ]
        if len(points) >= 3:
            region.insert(kdb.Polygon(points))
    region.merge()
    source_probe = kdb.Region(
        kdb.Box(source[0] - 1, source[1] - 1, source[0] + 1, source[1] + 1)
    )
    owned = next(
        (
            kdb.Region(component)
            for component in region.each_merged()
            if not (kdb.Region(component) & source_probe).is_empty()
        ),
        None,
    )
    if owned is None:
        return source

    interior = owned.sized(-max(1, grid_nm // 2))
    if interior.is_empty():
        return source
    base = (
        _nearest_on_grid(source[0], grid_nm),
        _nearest_on_grid(source[1], grid_nm),
    )
    candidates = []
    for radius in range(0, 9):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if radius and max(abs(dx), abs(dy)) != radius:
                    continue
                candidate = (base[0] + dx * grid_nm, base[1] + dy * grid_nm)
                candidates.append(candidate)
        candidates.sort(
            key=lambda point: (
                (point[0] - source[0]) ** 2 + (point[1] - source[1]) ** 2,
                point,
            )
        )
        for px, py in candidates:
            probe = kdb.Region(kdb.Box(px, py, px + 1, py + 1))
            if not (interior & probe).is_empty():
                return (int(px), int(py))
        candidates.clear()
    return source


# --------------------------------------------------------------------------- #
# geometry helpers (integer nm device-cell frame)
# --------------------------------------------------------------------------- #
def _poly_bbox_nm(lp) -> _Bbox:
    xs = [v.x for v in lp.polygon.vertices]
    ys = [v.y for v in lp.polygon.vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def _geom_extent_center_nm(geom: Any) -> Tuple[float, float]:
    """The centre of a ``DeviceGeom``'s drawn-polygon extent (nm, float).

    The terminal stamp is taken relative to THIS centre so it is centroid-relative,
    matching the placement contract (``placement`` gives the stamp's bbox centroid).
    We use the DRAWN extent (every polygon) so the stamp frame is the device's real
    footprint centre."""
    xs: List[int] = []
    ys: List[int] = []
    for lp in geom.polygons:
        b = _poly_bbox_nm(lp)
        xs.extend((b[0], b[2]))
        ys.extend((b[1], b[3]))
    if not xs:
        return (0.0, 0.0)
    return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)


def _m1_boxes_nm(geom: Any, m1: _Layer) -> List[_Bbox]:
    """Every M1 polygon bbox (nm) of a ``DeviceGeom``."""
    return [
        _poly_bbox_nm(lp)
        for lp in geom.polygons
        if (lp.layer, lp.datatype) == m1
    ]


def _snap_point_to_boxes(
    px: float, py: float, boxes: Sequence[_Bbox]
) -> Tuple[Optional[Tuple[float, float]], float, bool]:
    """Snap ``(px, py)`` to the nearest box: return ``(point, dist, inside)``.

    ``point`` is ``(px, py)`` CLAMPED into the nearest box (so it lands ON the M1
    shape); ``dist`` is the euclidean distance from the original point to that box (0
    when the point is already inside); ``inside`` is whether the original point was
    inside some box. Returns ``(None, inf, False)`` when there are no boxes."""
    best_pt: Optional[Tuple[float, float]] = None
    best_d = float("inf")
    inside = False
    for (x0, y0, x1, y1) in boxes:
        qx = min(max(px, x0), x1)
        qy = min(max(py, y0), y1)
        d = ((qx - px) ** 2 + (qy - py) ** 2) ** 0.5
        if d < best_d:
            best_d = d
            best_pt = (qx, qy)
        if x0 <= px <= x1 and y0 <= py <= y1:
            inside = True
    return best_pt, best_d, inside


# --------------------------------------------------------------------------- #
# per-device terminal -> M1 access stamp (centroid-relative um)
# --------------------------------------------------------------------------- #
def _terminal_m1_stamp(
    geom: Any,
    m1: _Layer,
    snap_boxes_um: Optional[Sequence[Tuple[float, float, float, float]]] = None,
) -> Dict[str, Tuple[_XY, float, bool]]:
    """For a built device cell, return ``{term_key -> (rel_xy_um, snap_dist_um, inside)}``.

    The terminal IDENTITY + its centroid-relative position come from the device cell's
    parametric terminal map (``geom.terminals`` from ``build_device`` / a passive
    cell). The ACCESS POINT is that terminal position snapped onto a real M1 shape:

      * When ``snap_boxes_um`` is given (the PLACED device stamp's OWN M1 polygons,
        centroid-relative um -- the artwork that is actually rendered), the terminal is
        snapped onto the nearest of THOSE shapes. This guarantees the pin lands on the
        metal that exists in the placed layout (vendor PCell M1 stripes or build_device
        M1), not merely on the parametric cell's notion of M1. Leakage-free: the placed
        stamp is the DEVICE CELL's artwork (vendor PCell or build_device), NOT a routed
        reference wire.
      * When ``snap_boxes_um`` is None, the terminal is snapped onto the nearest M1
        shape of the PARAMETRIC cell itself (the build_device M1: gate pad / rails).

    Expressed CENTROID-RELATIVE in microns (relative to the cell's drawn-extent
    centre, which is the placement contract's centroid). The gate terminal's parametric
    centre is inside the build_device M1 gate pad (the poly->M1 landing); snapped to the
    placed stamp it lands on the placed gate-side M1. Each entry carries the SNAP
    DISTANCE (um from the parametric terminal to the chosen M1 point; 0 when the
    parametric terminal is already inside a placed M1 box) and ``inside`` (whether the
    parametric terminal was inside ANY snap box) so the caller can report snap
    provenance honestly.

    A terminal with no M1 shape to snap to at all is omitted (no M1 access point here);
    the caller treats its net coverage as a gap.
    """
    cx, cy = _geom_extent_center_nm(geom)
    # Work entirely in CENTROID-RELATIVE microns so the parametric terminal positions
    # and the snap boxes share one frame (both are relative to the device centre):
    #   * the terminal centre is converted to centroid-relative um (minus the
    #     parametric cell's drawn-extent centre);
    #   * ``snap_boxes_um`` are the placed stamp's centroid-relative um M1 boxes
    #     (already in this frame); when absent, the parametric cell's own M1 boxes are
    #     converted to centroid-relative um too.
    # The returned offset is then directly centroid-relative (add the placement
    # centroid at the pin site). This is what makes a placed-stamp snap correct: both
    # the parametric centroid and the stamp centroid denote the SAME device centre.
    if snap_boxes_um is not None:
        boxes = [tuple(float(v) for v in b) for b in snap_boxes_um]
    else:
        boxes = [
            (
                (b[0] - cx) / _NM_PER_UM,
                (b[1] - cy) / _NM_PER_UM,
                (b[2] - cx) / _NM_PER_UM,
                (b[3] - cy) / _NM_PER_UM,
            )
            for b in _m1_boxes_nm(geom, m1)
        ]
    out: Dict[str, Tuple[_XY, float, bool]] = {}
    for key, term in (getattr(geom, "terminals", {}) or {}).items():
        try:
            tx, ty = term.center
        except Exception:
            continue
        rx = (float(tx) - cx) / _NM_PER_UM
        ry = (float(ty) - cy) / _NM_PER_UM
        pt, dist, inside = _snap_point_to_boxes(rx, ry, boxes)
        if pt is None:
            continue  # no M1 to snap to -> no M1 access point for this terminal
        out[str(key)] = ((pt[0], pt[1]), dist, inside)  # centroid-relative um
    return out


def _stamp_m1_boxes_um(stamp: Any, m1: _Layer) -> List[Tuple[float, float, float, float]]:
    """Every M1 polygon bbox (centroid-relative um) of a placed ``CellGeometry`` stamp,
    or ``[]`` when the stamp has no M1 (e.g. a MIM cap). The stamp polygons are stored
    centroid-relative, so the returned boxes are centroid-relative -- directly usable as
    ``snap_boxes_um`` (the placement centroid is added once, at the pin site)."""
    if stamp is None:
        return []
    out: List[Tuple[float, float, float, float]] = []
    for ring in (getattr(stamp, "polys_by_layer", {}) or {}).get(m1, []):
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        if xs:
            out.append((min(xs), min(ys), max(xs), max(ys)))
    return out


def _build_mos_cell(
    intent_device: Any,
    tech: Any,
    row_height_target_um: Optional[float] = None,
    substrate_rail_nets: Sequence[str] = (),
):
    """Build the device cell for one MOS ``ExtractedDevice`` (parametric, leakage-free).

    Geometry comes ONLY from the netlist ``(kind, W, L, M)`` + ``tech`` (the PDK) --
    never a reference coordinate. The netlist net bindings are stamped onto the
    terminal map (so a downstream consumer can see each terminal's net), but the PIN
    net binding below uses the netlist roles directly. Returns the ``DeviceGeom`` or
    ``None`` on a genuine build failure (reported as a gap).

    FOLD AWARENESS (the folded pin-probe contract): when ``row_height_target_um`` is
    set, the realized stamp is a SHARED-DIFFUSION SERPENTINE multi-row fold (finger
    split k + R bands; :func:`realize_render.folded_mos_cell`). The probe is then
    built with the SAME fold so its terminal s/d/g/b bboxes sit on the real per-band
    M1 landings of the folded artwork -- never single-row coordinates that would snap
    the bulk 'b' tap onto the source/drain M1 (the verified net08<->GND short). When
    ``row_height_target_um`` is ``None`` (default) the probe is the byte-identical
    single-row ``build_device`` build, so the non-folded path is unchanged."""
    from clay.layout import device_cell as _dc

    kind = str(getattr(intent_device, "kind", "")).lower()
    if kind not in ("nmos", "pmos"):
        return None
    nets = getattr(intent_device, "nets", {}) or {}
    nf = max(1, int(round(float(getattr(intent_device, "M", 1) or 1))))
    term_nets = {
        term: str(nets.get(role, ""))
        for role, term in _MOS_ROLE_TO_TERM.items()
    }
    W_um = float(getattr(intent_device, "W", 0.0))
    L_um = float(getattr(intent_device, "L", 0.0))
    if row_height_target_um is not None:
        try:
            from clay.layout.extract.realize_render import folded_mos_cell
            return folded_mos_cell(
                tech, kind, W_um, L_um, nf,
                float(row_height_target_um), nets=term_nets,
                substrate_rail_nets=substrate_rail_nets,
                oxide=str(getattr(intent_device, "oxide", "thin")),
            )
        except Exception:
            # fold build failed -> fall back to the single-row probe (a snap-based
            # access point on the placed stamp; reported as a snap, never faked).
            pass
    try:
        return _dc.build_device(
            tech,
            kind,
            W_um * 1e-6,
            L_um * 1e-6,
            nf,
            nets=term_nets,
            substrate_rail_nets=substrate_rail_nets,
            oxide=str(getattr(intent_device, "oxide", "thin")),
        )
    except Exception:
        return None


def _build_passive_cell(intent_passive: Any, tech: Any, mim_level: Optional[int] = None):
    """Build the device cell for one passive (cap/res) where the PDK has a builder.

    Uses :mod:`clay.layout.passive_device_cell` (parametric from the netlist VALUE +
    the PDK's passive stack). Returns the built geom or ``None`` when the PDK has no
    parametric passive cell (e.g. open_pdk -- only gf180mcu is wired up); the caller
    then falls back to the placed-stamp M1 contacts or reports an honest gap. NEVER
    fabricates geometry on layers the PDK does not define."""
    try:
        from clay.layout import passive_device_cell as _pdc
    except Exception:
        return None
    kind = str(getattr(intent_passive, "kind", "")).lower()
    nets = getattr(intent_passive, "nets", {}) or {}
    pn = {"p": str(nets.get("p", "")), "n": str(nets.get("n", ""))}
    val = float(getattr(intent_passive, "value", 0.0) or 0.0)
    orient = str(getattr(intent_passive, "orient", "R0") or "R0").upper()
    segment_intent = getattr(intent_passive, "resistor_segments", None)
    try:
        if kind.startswith("cap") or kind == "c":
            # mim_level: the stamp-detected CTM/METAL level (level-parameterised
            # PDKs only) -- the built cell must share the placed artwork's level
            # so pin positions/layers stay co-located with the stamp's metal.
            # m/model ride along from the passive itself (the signed netlist's
            # ``m=2 $.MODEL=M5`` bank form) so the pin cell IS the emitted cell.
            kw: Dict[str, Any] = {"level": mim_level} if mim_level is not None else {}
            if str(getattr(tech, "pdk_id", "")).lower() == "gf180mcu":
                mim_opt = getattr(intent_passive, "mim_option", None) or getattr(
                    intent_passive, "cap_option", None
                )
                m_lvl = getattr(intent_passive, "metal_level", None)
                pex_b = getattr(intent_passive, "pex_backend", None)
                if mim_opt is not None:
                    kw["mim_option"] = mim_opt
                if m_lvl is not None:
                    kw["metal_level"] = m_lvl
                if pex_b is not None:
                    kw["pex_backend"] = pex_b
            return _pdc.build_capacitor_for_tech(
                tech, val, nets=pn,
                m=int(getattr(intent_passive, "M", 1) or 1),
                model=str(getattr(intent_passive, "model", "") or ""),
                orient=orient,
                **kw,
            )
        if kind.startswith("res") or kind == "r":
            if segment_intent is not None:
                from clay.layout.annotations import ResistorSegments

                if type(segment_intent) is not ResistorSegments:
                    raise TypeError(
                        "resistor_segments route view must be exact ResistorSegments"
                    )
                if segment_intent.pdk != str(getattr(tech, "pdk_id", "")):
                    raise ValueError(
                        "resistor_segments route view does not match active PDK"
                    )
                return _pdc.build_segmented_resistor(
                    tech,
                    val,
                    count=segment_intent.count,
                    connection=segment_intent.connection,
                    segment_width_um=segment_intent.width_um,
                    segment_length_um=segment_intent.length_um,
                    segment_resistance_ohms=(
                        segment_intent.segment_resistance_ohms
                    ),
                    value_quantum_ohms=segment_intent.value_quantum_ohms,
                    segment_resistance_quantum_ohms=(
                        segment_intent.segment_resistance_quantum_ohms
                    ),
                    model=str(getattr(intent_passive, "model", "") or ""),
                    orient=orient,
                    nets=pn,
                )
            return _pdc.build_resistor(tech, val, nets=pn, orient=orient)
    except Exception:
        # A typed declaration is authoritative.  Returning ``None`` here would
        # turn a malformed or unsupported segmented PCell into a missing-pin
        # legacy fallback and sever routing from the geometry consumed by the
        # placer.  Preserve the historical best-effort behavior only for
        # passives that never declared this topology.
        if segment_intent is not None:
            raise
        return None
    return None


# --------------------------------------------------------------------------- #
# distinct-plate MIM cap terminal extraction (generic, DATA-driven by PDK)
# --------------------------------------------------------------------------- #
# Map a PDK id -> the MIM device-recognition SPEC builder (DATA lookup, NOT a code
# branch per circuit). The spec carries the per-PDK derived-layer recipe + connect
# rules that split a MIM cap's two plates into two DISTINCT electrical nodes (the
# foundry LVS device-recognition, reverse-engineered as DATA in device_terminal).
# Adding a PDK is one row here, not a code change in extract_pins.
def _mim_spec_for(tech: Any):
    """The :class:`MimConnectSpec` for ``tech`` (by PDK id), or ``None`` when the PDK
    has no tabulated MIM spec. Pure DATA dispatch -- no per-circuit logic."""
    from clay.layout import device_terminal as _dt

    builders = {
        "gf180mcu": _dt.gf180_mim_spec,
    }
    builder = builders.get(str(getattr(tech, "pdk_id", "")))
    if builder is None:
        return None
    try:
        return builder(tech)
    except Exception:
        return None


def _pdk_mim_is_level_parameterized(tech: Any) -> bool:
    """True iff this PDK's MIM spec builder accepts a ``level=`` parameter (its MIM
    stack is drawn at one of several CTMn/METALn levels -- external_pdk). False for a PDK
    whose MIM stack has a single fixed layer set (gf180 FuseTop / external_pdk 0.18 MIM10),
    where a ``$.MODEL`` level annotation is INERT. Probed generically by trying a
    ``level=`` call and catching ``TypeError`` -- the same mechanism
    :func:`_mim_spec_for_passive` uses; NO per-PDK branch."""
    from clay.layout import device_terminal as _dt

    builders = {
        "gf180mcu": _dt.gf180_mim_spec,
    }
    builder = builders.get(str(getattr(tech, "pdk_id", "")))
    if builder is None:
        return False
    try:
        builder(tech, level=2)
        return True
    except TypeError:
        return False
    except Exception:
        return False


def _gf180_mim_spec_from_attrs(tech: Any, passive: Any):
    """Resolve GF180 Option-A/B MimConnectSpec from passive/tech hints without model-name guessing."""
    from clay.layout import device_terminal as _dt
    from clay.pdk.gf180mcu import resolve_gf180_mim_config

    raw_opt = getattr(passive, "mim_option", None)
    if raw_opt is None:
        raw_cap_opt = getattr(passive, "cap_option", None)
        if str(raw_cap_opt or "").strip().lower() in ("a", "b", "option_a", "option_b", "opt_a", "opt_b"):
            raw_opt = raw_cap_opt
    opt, _, lvl = resolve_gf180_mim_config(
        getattr(tech, "adapter", None) or tech,
        mim_option=raw_opt,
        metal_level=getattr(passive, "metal_level", None),
        pex_backend=getattr(passive, "pex_backend", None),
    )
    if opt == "A":
        return _dt.gf180_mim_spec(tech)
    return _dt.gf180_mim_spec(tech, mim_option=opt, metal_level=lvl)


def _mim_spec_for_passive(tech: Any, passive: Any):
    """The MIM spec at the LEVEL the passive's own ``$.MODEL`` annotation names
    (design-sourced; the signed netlist's ``$.MODEL=M5`` -> CTM5), else the PDK
    default. Verification sites MUST use this (not the bare default spec) so the
    spec level always follows the cell the same passive data built -- a level-5
    bank checked against the level-4 default spec reports its plates unmapped."""
    from clay.layout import device_terminal as _dt

    if str(getattr(tech, "pdk_id", "")).lower() == "gf180mcu":
        return _gf180_mim_spec_from_attrs(tech, passive)

    spec = _mim_spec_for(tech)
    if spec is None:
        return None
    model = str(getattr(passive, "model", "") or "")
    if not model:
        return spec
    try:
        lv = None
    except Exception:
        lv = None
    if lv is None:
        return spec
    builders = {
        "gf180mcu": _dt.gf180_mim_spec,
    }
    builder = builders.get(str(getattr(tech, "pdk_id", "")))
    if builder is None:
        return spec
    try:
        return builder(tech, level=lv)
    except TypeError:
        return spec  # builder has no level concept -> annotation is inert here
    except Exception:
        return spec


def _mim_spec_and_level_for_stamp(
    tech: Any, stamp: Any
) -> Tuple[Optional[Any], Optional[int]]:
    """The MIM spec CONSISTENT WITH THE PLACED ARTWORK -> ``(spec, level)``."""
    from clay.layout import device_terminal as _dt

    builders = {
        "gf180mcu": _dt.gf180_mim_spec,
    }
    pdk_id = str(getattr(tech, "pdk_id", "")).lower()
    builder = builders.get(pdk_id)
    if builder is None:
        return None, None
    try:
        default = builder(tech)
    except Exception:
        default = None

    drawn = set(
        tuple(k) for k in (getattr(stamp, "polys_by_layer", {}) or {}).keys()
    )
    if not drawn and hasattr(stamp, "polygons"):
        drawn = {(int(lp.layer), int(lp.datatype)) for lp in (getattr(stamp, "polygons", ()) or ())}
    if not drawn:
        return default, None

    if pdk_id == "gf180mcu":
        gf_candidates = []
        if default is not None:
            gf_candidates.append(default)
        for lvl in ("5LM", "4LM"):
            try:
                gf_candidates.append(_dt.gf180_mim_spec(tech, mim_option="B", metal_level=lvl))
            except Exception:
                continue
        best_spec = None
        best_score = -1
        for spec in gf_candidates:
            real = set(map(tuple, (getattr(spec, "synthetic_to_real", {}) or {}).values()))
            if not real or not real <= drawn:
                continue
            via_keys = {tuple(v[0]) for v in (getattr(spec, "vias", ()) or ())}
            has_via = 1 if (via_keys & drawn) else 0
            order_hit = len(set(map(tuple, getattr(spec, "metal_order", ()) or ())) & drawn)
            score = has_via * 100 + order_hit * 10 + len(
                set(map(tuple, getattr(spec, "metals", ()) or ())) & drawn
            )
            if score > best_score:
                best_spec, best_score = spec, score
        return best_spec, None

    candidates: List[Tuple[Any, Optional[int]]] = []
    if default is not None:
        candidates.append((default, None))
    for lv in range(2, 8):
        try:
            candidates.append((builder(tech, level=lv), lv))
        except TypeError:
            break  # builder takes no level -> the default is the only candidate
        except Exception:
            continue

    best: Tuple[Optional[Any], Optional[int]] = (None, None)
    best_score = -1
    for spec, lv in candidates:
        real = set(map(tuple, (getattr(spec, "synthetic_to_real", {}) or {}).values()))
        if not real or not real <= drawn:
            continue
        score = len(set(map(tuple, getattr(spec, "metals", ()) or ())) & drawn)
        if score > best_score:
            best, best_score = (spec, lv), score
    return best


def _routing_metal_keys(tech: Any) -> List[_Layer]:
    """The PDK's routing-metal (layer, datatype) keys, highest index first.

    Used to pick, among the device-cell polygons covering a terminal tab, the
    routing metal the tab actually exits on (so the pin's layer matches the metal
    physically present at its xy)."""
    out: List[_Layer] = []
    for n in range(1, 8):
        try:
            out.append(tuple(tech.metal_key(n)))  # type: ignore[arg-type]
        except Exception:
            break
    return out


def _metal_layer_covering(
    cell: Any, tx: int, ty: int, tech: Any, bbox: Optional[_Bbox] = None
) -> Optional[_Layer]:
    """The routing-metal layer of a ``cell`` polygon that covers point ``(tx, ty)`` nm
    (the terminal tab centre), or that maximally overlaps the terminal ``bbox``.

    Returns the routing metal (``tech.metal_key``) physically present at the tab, so
    the pin's LAYER and XY are co-located on real device-cell metal. Prefers a
    polygon that strictly contains the point; falls back to the routing metal whose
    polygon overlaps the terminal bbox most. ``None`` if no routing metal is at the
    tab (the caller then keeps the node-landing layer)."""
    metal_keys = set(_routing_metal_keys(tech))
    # 1) a routing-metal polygon that CONTAINS the tab centre.
    for lp in getattr(cell, "polygons", []) or []:
        key = (int(lp.layer), int(lp.datatype))
        if key not in metal_keys:
            continue
        b = _poly_bbox_nm(lp)
        if b[0] <= tx <= b[2] and b[1] <= ty <= b[3]:
            return key
    # 2) else the routing metal with the largest overlap of the terminal bbox.
    if bbox is not None:
        best_key: Optional[_Layer] = None
        best_ov = 0
        for lp in getattr(cell, "polygons", []) or []:
            key = (int(lp.layer), int(lp.datatype))
            if key not in metal_keys:
                continue
            b = _poly_bbox_nm(lp)
            ox = max(0, min(b[2], bbox[2]) - max(b[0], bbox[0]))
            oy = max(0, min(b[3], bbox[3]) - max(b[1], bbox[1]))
            ov = ox * oy
            if ov > best_ov:
                best_ov = ov
                best_key = key
        if best_key is not None:
            return best_key
    return None


def _cover_overhang_landing(
    cell: Any, tech: Any, top_cover: _Layer
) -> Optional[Tuple[int, int]]:
    """A RISER-SAFE landing on the top-plate COVER metal for a cap whose cover
    spans the whole bottom-plate + via stack (the external_pdk over-cell MIM): the point
    on ``top_cover`` farthest OUTSIDE every LOWER plate-metal or via-cut shape
    of the cap's stack, so a via ladder dropped there punches NO lower-plate
    metal / via (no top<->bottom plate short).

    Pure cell geometry (cover minus stack) -- never a reference coordinate.
    ``None`` when the cover has no overhang outside the stack (no safe point;
    the caller keeps the prior landing and reports the honest geometric risk)."""
    cov = [lp for lp in (getattr(cell, "polygons", []) or [])
           if (int(lp.layer), int(lp.datatype)) == tuple(top_cover)]
    if not cov:
        return None
    metal_keys = set(_routing_metal_keys(tech))
    stack_keys = {k for k in metal_keys if tuple(k) != tuple(top_cover)}
    try:
        vias = {tuple(tech.via_key(n)) for n in range(0, 11)}
    except Exception:
        vias = set()
    blockers = []
    for lp in (getattr(cell, "polygons", []) or []):
        key = (int(lp.layer), int(lp.datatype))
        if key in stack_keys or key in vias:
            blockers.append(_poly_bbox_nm(lp))
    if not blockers:
        return None
    def _outside(px: int, py: int) -> bool:
        # This is a set-union test, not a union-*bbox* test.  The external_pdk MIM exit
        # tab deliberately has a legal cover-only throat between the main
        # bottom plate and the disconnected internal down-riser.  Collapsing
        # those shapes to one bbox erases that throat and incorrectly falls
        # back to an on-plate cover centroid.
        return not any(x0 <= px <= x1 and y0 <= py <= y1
                       for x0, y0, x1, y1 in blockers)

    def _clearance2(px: int, py: int) -> int:
        """Squared distance to the nearest blocker bbox (positive off-shape)."""
        ds = []
        for x0, y0, x1, y1 in blockers:
            dx = max(x0 - px, 0, px - x1)
            dy = max(y0 - py, 0, py - y1)
            ds.append(dx * dx + dy * dy)
        return min(ds) if ds else 0

    best: Optional[Tuple[float, Tuple[int, int]]] = None
    for lp in cov:
        cb = _poly_bbox_nm(lp)
        # candidate cover points: the cover bbox corners + edge midpoints.
        cands = [
            (cb[0], cb[1]), (cb[2], cb[1]), (cb[0], cb[3]), (cb[2], cb[3]),
            ((cb[0] + cb[2]) // 2, cb[1]), ((cb[0] + cb[2]) // 2, cb[3]),
            (cb[0], (cb[1] + cb[3]) // 2), (cb[2], (cb[1] + cb[3]) // 2),
        ]
        for (px, py) in cands:
            if not _outside(px, py):
                continue
            if _metal_layer_covering(cell, px, py, tech, None) != tuple(top_cover):
                continue
            # Prefer the deepest cover-only throat.  This also avoids choosing
            # a cover edge that merely touches a lower conductor bbox.
            d = float(_clearance2(px, py))
            if best is None or d > best[0]:
                best = (d, (int(px), int(py)))
    return best[1] if best is not None else None


def _tab_metal_center(
    cell: Any,
    tx: int,
    ty: int,
    layer: _Layer,
    terminal_bbox: Optional[_Bbox] = None,
) -> Optional[Tuple[int, int]]:
    """Centre (nm) of the cap-terminal routing TAB stub covering ``(tx, ty)`` on
    ``layer``.

    The SMALLEST metal polygon on that layer that contains the declared terminal centre
    is the routable STUB (vs the big plate). Landing the pin at the stub's centre -- not
    at the declared terminal box, which sits at the stub TIP -- gives a min-width routed
    wire margin on every side, so the wire does not overhang the stub edge into a
    sub-min-width ``M2.1`` sliver at the wire-to-cap junction (the two-stage OTA ncc
    trunk). Generic: the stub is the cap cell's own geometry, PDK-agnostic, no dimension
    constant. Returns ``None`` (keep the safe declared centre) when no covering stub is
    found, the covering polygon is a plate rather than a local tab, or the stub's bbox
    centre is not on its metal (a non-rectangular stub whose centre falls in a notch).

    A terminal marker may legitimately sit directly on a plate edge (external_pdk M5).  In
    that case the smallest covering polygon is still the whole plate; blindly taking
    its bbox centre erases the materializer's edge-access obligation and strands the
    route inside the MIM keep-out.  A true local tab is thin/local in at least one
    axis relative to the terminal bbox.  Reject only polygons that expand by more
    than 2x in BOTH axes, so the existing GF180/external_pdk strip tabs retain their centred
    landings while a two-dimensional plate keeps the declared terminal centre.
    """
    import klayout.db as kdb

    best = None  # (area, polygon, cx, cy, bbox)
    for lp in getattr(cell, "polygons", []) or []:
        if (int(lp.layer), int(lp.datatype)) != tuple(layer):
            continue
        x0, y0, x1, y1 = _poly_bbox_nm(lp)
        if x0 <= tx <= x1 and y0 <= ty <= y1:
            area = (x1 - x0) * (y1 - y0)
            if best is None or area < best[0]:
                best = (
                    area,
                    lp.polygon,
                    (x0 + x1) // 2,
                    (y0 + y1) // 2,
                    (x0, y0, x1, y1),
                )
    if best is None:
        return None
    _, poly, cxn, cyn, polygon_bbox = best
    if terminal_bbox is not None:
        tbx0, tby0, tbx1, tby1 = (int(value) for value in terminal_bbox)
        tw = max(1, tbx1 - tbx0)
        th = max(1, tby1 - tby0)
        px0, py0, px1, py1 = polygon_bbox
        if (px1 - px0) > 2 * tw and (py1 - py0) > 2 * th:
            return None
    kpoly = kdb.Polygon([kdb.Point(int(v.x), int(v.y)) for v in poly.vertices])
    probe = kdb.Region(kdb.Box(cxn - 1, cyn - 1, cxn + 1, cyn + 1))
    if (kdb.Region(kpoly) & probe).is_empty():
        return None  # bbox centre lies in a concave notch -> keep declared centre
    return (cxn, cyn)


def _passive_cap_pins(
    intent_passive: Any,
    tech: Any,
    stamp: Any = None,
) -> Tuple[Optional[Dict[str, Tuple[_XY, _Layer, str]]], List[Tuple[str, str, str, str]]]:
    """Distinct-plate routing access for a MIM cap, GENERICALLY (no per-PDK pin hack).

    Builds the cap's device cell (``passive_device_cell.build_capacitor`` -- parametric
    from the netlist VALUE + the PDK MIM stack; leakage-free), selects the PDK's MIM
    device-recognition SPEC by DATA (:func:`_mim_spec_for`), and runs
    :func:`clay.layout.device_terminal.extract_terminal_access`. That extractor returns
    each terminal's access on its OWN DISTINCT electrical plate node (the layer-aware
    connectivity primitive keeps the two plates on two nodes; the 2D-overlapping plates
    are NOT merged). Each terminal's access lands on REAL materialized routing metal.
    Two terminals may deliberately share one routing layer (for example, two distinct
    external_pdk M1 down-riser tips); in that case their separate connected components, not the
    layer name alone, preserve plate ownership and electrical isolation.

    Returns ``(per_term, gaps)`` where ``per_term`` maps each terminal role to
    ``(rel_xy_um, routing_layer, net)`` (centroid-relative um, on the terminal's
    distinct-plate routing layer). HONEST GAP: if the spec/cell is unavailable, or the
    extractor reports COLLAPSED (two terminals on one node -- a shorted cap) or UNMAPPED
    terminals, ``per_term`` is ``None`` and ``gaps`` lists ``(name, term, net, reason)``;
    nothing is faked. The two plates are emitted ONLY when they are proven distinct.
    """
    import klayout.db as kdb

    from clay.layout import device_terminal as _dt

    name = str(getattr(intent_passive, "name", ""))
    nets = getattr(intent_passive, "nets", {}) or {}
    pn = {"p": str(nets.get("p", "")), "n": str(nets.get("n", ""))}

    def _all_gaps(reason: str) -> List[Tuple[str, str, str, str]]:
        return [(name, t, pn[t], reason) for t in ("p", "n")]

    # spec/artwork single-source. Level priority:
    #   1. the placed STAMP's own drawn MIM stack (a vendor cap may sit at a
    #      different CTM/METAL level than the PDK default);
    #   2. the passive's $.MODEL annotation (the signed netlist's ``$.MODEL=M5``
    #      -- the same data the cell build below consumes, so spec and cell can
    #      never disagree on level);
    #   3. the PDK default.
    # An inconsistent spec either emits a floating off-device pin (stamp case)
    # or reports the bank's plates COLLAPSED (annotation case) -> honest gap /
    # aligned spec instead.
    spec, mim_level = _mim_spec_and_level_for_stamp(tech, stamp)
    if spec is None:
        return None, _all_gaps(
            "no MIM spec for PDK consistent with the placed stamp's drawn MIM "
            "stack (cap not wired)"
        )
    if mim_level is None and _pdk_mim_is_level_parameterized(tech):
        # The cap's ``$.MODEL`` annotation names a MIM LEVEL only for a PDK whose MIM
        # builder is LEVEL-PARAMETERIZED (external_pdk CTMn/METALn -- the two reference caps
        # sit at different levels). For a PDK whose MIM stack has NO level concept
        # (gf180 FuseTop, external_pdk 0.18 MIM10 -- one fixed MIM layer), the same ``M5``
        # token is just the bottom-plate METAL name, not a CTMn selector; reading it
        # as a level would force ``level=5`` into a builder that has no level
        # parameter (the external_pdk cap builder raised, leaving the cap unwired). Gate the
        # whole annotation-level adoption on the builder accepting a level -- generic
        # (no per-PDK branch): a non-level PDK keeps mim_level=None (annotation inert).
        try:
            from clay.layout.passive_device_cell_tsmc import mim_level_from_model
            lv_ann = mim_level_from_model(
                str(getattr(intent_passive, "model", "") or ""))
        except Exception:
            lv_ann = None
        if lv_ann is not None:
            annotated = _mim_spec_for_passive(tech, intent_passive)
            if annotated is not None:
                spec, mim_level = annotated, lv_ann

    build_passive = intent_passive
    if str(getattr(tech, "pdk_id", "")).lower() == "gf180mcu":
        from types import SimpleNamespace as _SNS

        bot_key = tuple(spec.metal_order[0]) if getattr(spec, "metal_order", ()) else None
        inferred_opt = None
        inferred_lvl = None
        try:
            if bot_key == tuple(tech.metal_key(4)):
                inferred_opt, inferred_lvl = "B", "5LM"
            elif bot_key == tuple(tech.metal_key(3)):
                inferred_opt, inferred_lvl = "B", "4LM"
            elif bot_key == tuple(tech.metal_key(2)):
                inferred_opt = "A"
        except Exception:
            inferred_opt = None
        if inferred_opt == "B" or getattr(intent_passive, "mim_option", None) is not None:
            build_passive = _SNS(
                name=getattr(intent_passive, "name", ""),
                kind=getattr(intent_passive, "kind", "cap"),
                value=getattr(intent_passive, "value", 0.0),
                M=getattr(intent_passive, "M", 1),
                model=getattr(intent_passive, "model", ""),
                orient=getattr(intent_passive, "orient", "R0"),
                nets=getattr(intent_passive, "nets", {}) or {},
                mim_option=getattr(intent_passive, "mim_option", None) or inferred_opt,
                metal_level=getattr(intent_passive, "metal_level", None) or inferred_lvl,
                pex_backend=getattr(intent_passive, "pex_backend", None),
            )
    cell = _build_passive_cell(build_passive, tech, mim_level=mim_level)
    if cell is None or not getattr(cell, "terminals", None):
        return None, _all_gaps("no cap device cell on PDK (cap not wired)")

    access, report = _dt.extract_terminal_access(cell, tech=tech, mim_spec=spec)
    device_nodes = _dt.extract_device_nodes(cell, tech=tech, mim_spec=spec)

    if report.collapsed:
        # two distinct declared terminals resolved to ONE node -> a SHORTED cap.
        # Honest correctness error: emit NOTHING (never a faked pin), report the gap.
        return None, _all_gaps(
            f"cap plates COLLAPSE to one node {report.collapsed} (shorted -- not emitted)"
        )

    cx, cy = _geom_extent_center_nm(cell)  # device-cell extent centre (nm)
    terms = getattr(cell, "terminals", {}) or {}
    per_term: Dict[str, Tuple[_XY, _Layer, str]] = {}
    gaps: List[Tuple[str, str, str, str]] = []
    plate_handles = {
        tuple(value) for value in
        (getattr(spec, "synthetic_to_real", {}) or {}).values()
    }

    def _tab_layer_on_access_node(td, access_fact, layer) -> bool:
        """Whether ``td`` intersects ``layer`` on its extracted plate node."""
        node_id = int(access_fact.node_id)
        if node_id < 0 or node_id >= len(device_nodes):
            return False
        region = device_nodes[node_id].get(tuple(layer))
        bbox = getattr(td, "bbox", None)
        if region is None or region.is_empty() or bbox is None:
            return False
        terminal_region = kdb.Region(kdb.Box(*(int(value) for value in bbox)))
        return not (region & terminal_region).is_empty()

    def _preconnected_upper_tab(td, access_fact, layer):
        """Return the adjacent pad already joined to a low terminal by its PCell.

        A materialized down-riser occupies the VIA cut at its M1 tip.  Publishing
        that tip asks the router to place a second cut on top of the first one.
        Prefer the immediately adjacent routing metal only when the extracted
        device node proves the PCell already connects it and real metal covers
        the selected point.
        """
        current_level = None
        for level in range(1, 33):
            try:
                if tuple(tech.metal_key(level)) == tuple(layer):
                    current_level = level
                    break
            except Exception:
                break
        if current_level is None:
            return None
        try:
            upper_layer = tuple(tech.metal_key(current_level + 1))
        except Exception:
            return None
        node_id = int(access_fact.node_id)
        if node_id < 0 or node_id >= len(device_nodes):
            return None
        region = device_nodes[node_id].get(upper_layer)
        if region is None or region.is_empty():
            return None
        tx, ty = (int(value) for value in td.center)
        candidates = []
        for component in region.each_merged():
            component_region = kdb.Region(component)
            bbox = component.bbox()
            candidate = (
                (int(bbox.left) + int(bbox.right)) // 2,
                (int(bbox.bottom) + int(bbox.top)) // 2,
            )
            probe = kdb.Region(kdb.Box(
                candidate[0] - 1,
                candidate[1] - 1,
                candidate[0] + 1,
                candidate[1] + 1,
            ))
            if (component_region & probe).is_empty():
                continue
            candidates.append((
                (candidate[0] - tx) ** 2 + (candidate[1] - ty) ** 2,
                (int(bbox.left), int(bbox.bottom), int(bbox.right), int(bbox.top)),
                candidate,
            ))
        if not candidates:
            return None
        _distance, _bbox, candidate = min(candidates)
        return candidate, upper_layer

    for term in ("p", "n"):
        a = access.get(term)
        if a is None:
            # terminal mapped to no routable node (honest gap).
            gaps.append((name, term, pn[term], "cap terminal has no routable plate landing"))
            continue
        # The pin LANDS on this terminal's OWN routing TAB (the device-cell terminal
        # artwork -- a metal stub off the plate, spatially separated from the other
        # plate's tab), NOT the node centroid (the node centroid is the big plate region
        # over the fusetop / bottom plate; routing a foreign wire to the centre crosses
        # the plate and risks a short). The tab is the terminal's own ``center`` + bbox.
        #
        # CRITICAL (the LVS open the gate caught): the pin's (xy, LAYER) MUST be
        # CONSISTENT -- a foundry extractor joins a routed wire to the plate node only if
        # the wire lands on metal of the SAME layer AT that point. The terminal TAB is on
        # a SPECIFIC layer (gf180 'p' bottom-plate Metal2 exit pad; 'n' top-plate handle);
        # the node's ``a.layer`` may differ (the node's lowest landing). Mixing the tab xy
        # with ``a.layer`` lands the router where that layer is absent -> the wire never
        # touches the plate -> the plate extracts FLOATING. Fix: land on the tab at the
        # layer of the device cell's OWN metal that COVERS that tab, so xy + layer are
        # co-located on real device metal; the deck MIM connect chain carries it into the
        # plate node. All from the device cell, never a reference wire (leakage-free).
        td = terms.get(term)
        rel: _XY
        layer: _Layer = tuple(a.layer)
        top_cover = tuple(spec.top_cover) if getattr(spec, "top_cover", None) else None
        is_top_plate = top_cover is not None and tuple(a.layer) == top_cover
        tab_layer = None
        if td is not None and getattr(td, "center", None) is not None:
            tx, ty = td.center
            tab_layer = _metal_layer_covering(
                cell, int(tx), int(ty), tech, getattr(td, "bbox", None)
            )

        # A declared terminal on a lower metal than its plate handle is a
        # materialized internal access riser.  ``extract_terminal_access`` already
        # proved that terminal bbox belongs to this terminal's distinct electrical
        # node; preserve that device-cell contract instead of replacing it with an
        # out-of-graph upper handle.  A tab that is itself on ``top_cover`` still
        # takes the riser-safe cover logic below.
        if (td is not None
                and getattr(td, "center", None) is not None
                and tab_layer is not None
                and tuple(tab_layer) not in plate_handles
                and _tab_layer_on_access_node(td, a, tab_layer)
                and (not is_top_plate or tuple(tab_layer) != top_cover)):
            tx, ty = td.center
            layer = tuple(tab_layer)
            upper_tab = _preconnected_upper_tab(td, a, layer)
            if upper_tab is not None:
                (tx, ty), layer = upper_tab
            else:
                cand = _tab_metal_center(
                    cell, int(tx), int(ty), layer, getattr(td, "bbox", None)
                )
                if cand is not None:
                    tx, ty = cand
            rel = ((tx - cx) / _NM_PER_UM, (ty - cy) / _NM_PER_UM)
        elif is_top_plate:
            # TOP plate: prefer the device cell's OWN 'n' terminal TAB when the cover
            # metal physically COVERS it (the external_pdk exit-tab contract: the tab tip is the
            # only RISER-SAFE landing -- an on-cover point over the bottom plate makes
            # the router's via ladder drop its METAL n pad + cut INSIDE the plate, the
            # canonical t4 net25/VOUT short). STRICT point containment on the cell's own
            # cover metal keeps xy + layer co-located on real device metal -- the same
            # cell whose terminal map extract_terminal_access resolved (single source).
            # When the tab is NOT on the cover (gf180: the FuseTop tab sits in a GAP
            # where the emitted M3 cover does not extend -- landing there left the top
            # plate FLOATING, an LVS open the gate caught), fall back to the
            # extract_terminal_access ON-COVER landing (a.xy_um/a.layer) exactly as
            # before. Leakage-free either way (device cell, never a reference wire).
            rel = (a.xy_um[0] - cx / _NM_PER_UM, a.xy_um[1] - cy / _NM_PER_UM)
            if td is not None and getattr(td, "center", None) is not None:
                tx, ty = td.center  # nm, device-cell absolute frame
                # RISER-SAFE discriminator: the exit-tab landing is taken only
                # when the tab tip carries the COVER metal and NO OTHER plate
                # metal of this cap's stack sits under the point -- i.e. the
                # tab is OUTSIDE the bottom-plate footprint, so the via ladder
                # cannot punch the plate (the canonical t4 net25/VOUT short).
                # Cover-coverage alone is NOT the property: on gf180 the
                # cover extends over bank regions whose Metal2 bottom plate is
                # right below, and landing there bypassed the on-cover
                # reprojection (guarded by tests/test_cap_top_cover_landing).
                # "no other plate metal" is tested on the RAW cell layers (the
                # spec's carved keys (36,901)... never appear on cell polygons):
                # any REAL routing metal other than the cover under the point
                # means the tab still sits over the plate stack (gf180 Metal2
                # bottom plate) -> unsafe. The external_pdk exit-tab tip carries the
                # cover metal alone.
                _lower_metals = set()
                for _n in range(1, 11):
                    try:
                        _k = tuple(tech.metal_key(_n))
                    except Exception:
                        continue
                    if _k != top_cover:
                        _lower_metals.add(_k)
                _under_plate = any(
                    (int(lp.layer), int(lp.datatype)) in _lower_metals
                    and _poly_bbox_nm(lp)[0] <= tx <= _poly_bbox_nm(lp)[2]
                    and _poly_bbox_nm(lp)[1] <= ty <= _poly_bbox_nm(lp)[3]
                    for lp in getattr(cell, "polygons", []) or []
                )
                if (not _under_plate
                        and _metal_layer_covering(cell, tx, ty, tech, None) == top_cover):
                    rel = ((tx - cx) / _NM_PER_UM, (ty - cy) / _NM_PER_UM)
                else:
                    # No riser-safe tab: the tab tip is over the plate stack
                    # (the external_pdk over-cell MIM -- the cover spans the whole bottom
                    # plate + VIA array). The via ladder MUST land where laddering
                    # up touches NO lower-plate metal AND no via cut of this cap's
                    # OWN stack (else the riser shorts the top plate to the bottom
                    # plate -- the t4 net25/VOUT merge). Search the COVER metal for
                    # the point FARTHEST OUTSIDE the plate-stack column: the cover
                    # OVERHANG beyond the bottom plate + via array. Pure cell
                    # geometry (cover minus stack), never a reference coordinate.
                    safe = _cover_overhang_landing(cell, tech, top_cover)
                    if safe is not None:
                        sx, sy = safe
                        rel = ((sx - cx) / _NM_PER_UM, (sy - cy) / _NM_PER_UM)
                        layer = top_cover
        elif td is not None and getattr(td, "center", None) is not None:
            # BOTTOM plate: land on the terminal's own routing TAB (a metal stub off the
            # big plate region), at the device-cell metal layer that covers the tab.
            tx, ty = td.center  # nm, device-cell absolute frame
            tb = getattr(td, "bbox", None)
            tab_layer = _metal_layer_covering(cell, tx, ty, tech, tb)
            if tab_layer is not None:
                layer = tab_layer  # co-locate xy + layer on real device metal
            # CENTRE the landing on the tab metal stub: the declared terminal box sits at
            # the stub TIP, so a min-width trunk landing there overhangs the stub edge ->
            # a sub-min-width M2.1 sliver at the wire-to-cap junction. The stub centre
            # gives margin on every side (generic, no dimension constant).
            cand = _tab_metal_center(
                cell, int(tx), int(ty), layer, getattr(td, "bbox", None)
            )
            if cand is not None:
                tx, ty = cand
            rel = ((tx - cx) / _NM_PER_UM, (ty - cy) / _NM_PER_UM)
        else:
            rel = (a.xy_um[0] - cx / _NM_PER_UM, a.xy_um[1] - cy / _NM_PER_UM)
        per_term[term] = (rel, layer, a.net or pn[term])

    # PIN/ARTWORK CO-LOCATION (the placed-stamp contract): when the placed STAMP
    # is supplied, every pin xy must sit on metal the stamp actually draws on the
    # pin's layer. The parametric cell gives the ROLE -> plate-layer mapping; its
    # tab xy however is in the parametric footprint, which a VENDOR stamp need
    # not share (the reference's 3pF cap is drawn ~29um square; the parametric
    # plate for the same value is ~54um -> the tab lands off the stamp). Reproject:
    # keep the xy when it already lies on the stamp's drawn metal on that layer,
    # else move it to a proven interior point of the largest merged conductor;
    # a terminal whose OWN
    # plate handle the stamp does not draw becomes an HONEST GAP (never a
    # floating pin, never a landing on the OTHER plate's metal -- the
    # distinct-landing obligation below). Parametric-geometry stamps are built
    # by the same builder, so their tabs lie on the drawn metal already and
    # this pass changes nothing.
    if stamp is not None and per_term:
        rings_by_layer = getattr(stamp, "polys_by_layer", {}) or {}

        # The extraction deck connects touching/overlapping polygons, not their
        # source-ring bounding boxes.  Compile the stamp artwork into those same
        # merged same-layer components so landing distinctness follows electrical
        # truth even when two raw rings touch or a third ring bridges them.
        stamp_components: Dict[_Layer, List[Any]] = {}
        for raw_layer, rings in rings_by_layer.items():
            region = kdb.Region()
            for ring in rings or ():
                points = [
                    kdb.Point(
                        int(round(float(point[0]) * _NM_PER_UM)),
                        int(round(float(point[1]) * _NM_PER_UM)),
                    )
                    for point in ring
                ]
                if len(points) >= 3:
                    region.insert(kdb.Polygon(points))
            region.merge()
            stamp_components[tuple(raw_layer)] = [
                kdb.Region(component) for component in region.each_merged()
            ]

        def _component_at(lay, point) -> Optional[int]:
            px = int(round(float(point[0]) * _NM_PER_UM))
            py = int(round(float(point[1]) * _NM_PER_UM))
            probe = kdb.Region(kdb.Box(px - 1, py - 1, px + 1, py + 1))
            for index, component in enumerate(stamp_components.get(tuple(lay), ())):
                if not (component & probe).is_empty():
                    return index
            return None

        def _component_point(lay, index: int) -> Optional[_XY]:
            """A verified on-artwork point for one merged stamp conductor.

            Prefer the bbox centre only when it is actually inside.  For a
            concave/notched polygon, use a hull point of a 1 nm inset; that point
            remains inside the original conductor instead of merely inside its
            AABB.  The final membership check is the same proof used for pins.
            """
            components = stamp_components.get(tuple(lay), ())
            if index < 0 or index >= len(components):
                return None
            component = components[index]
            inset = component.sized(-1)
            candidates = inset if not inset.is_empty() else component
            polygons = sorted(
                [kdb.Polygon(polygon) for polygon in candidates.each_merged()],
                key=lambda polygon: polygon.area(),
                reverse=True,
            )
            for polygon in polygons:
                bbox = polygon.bbox()
                points = [
                    ((bbox.left + bbox.right) // 2,
                     (bbox.bottom + bbox.top) // 2),
                    *((point.x, point.y) for point in polygon.each_point_hull()),
                ]
                for px, py in points:
                    point_um = (px / _NM_PER_UM, py / _NM_PER_UM)
                    if _component_at(lay, point_um) == index:
                        return point_um
            return None

        # A plate handle the PARAMETRIC cell exposes on a LOW metal (the external_pdk
        # top-plate METAL1 riser tip) may not exist on a VENDOR stamp, which
        # draws that plate only on its own routable HANDLE (the cover, a HIGH
        # metal). When the parametric layer is absent from the stamp, fall back
        # -- but ONLY to the terminal's OWN plate handle:
        #
        #   * a terminal whose parametric layer IS one of the spec's real plate
        #     handles (``synthetic_to_real`` values -- the exact layers the
        #     placed-stamp consistency check keys on) owns THAT handle; the
        #     stamp not drawing it is an HONEST GAP, never a hop to another
        #     metal (the other drawn handle is the OTHER plate's conductor);
        #   * a terminal whose parametric layer is an internal RISER metal (not
        #     a handle) falls back to the vendor handle of the SAME plate.  The
        #     capacitor interface defines p=bottom and n=top; those roles map to
        #     the lowest and highest plate handles in ``metal_order``.  This stays
        #     valid when BOTH plates have materialized low-metal risers.
        #
        # DISTINCT-LANDING OBLIGATION (the rings_group VOUT<->net25 regression,
        # bisected): the prior fallback walked ALL of ``metal_order`` by
        # drawn-ness alone, so a top-plate pin whose cover was absent fell
        # THROUGH the cover onto the other plate's drawn box -- two
        # different-net pins at ONE point on ONE layer, a by-construction
        # cross-net short (the router is forced through one grid column: the
        # measured n_shorts=2 at (148,81) gi2/gi3; the deck merges same-layer
        # overlap into one conductor). The module contract "the two plates are
        # emitted ONLY when they are proven distinct" binds the reprojection's
        # moved landings too: a fallback landing that shares a layer AND a
        # drawn region (or an exact point) with another terminal of this cap is
        # refused -> honest gap, never a faked co-located pin. Concentric
        # DIFFERENT-layer landings (bottom-box centre under cover-box centre --
        # the vendor MIM's plates are stacked) remain legal: each is its own
        # plate's conductor.
        _mo = [tuple(m) for m in (getattr(spec, "metal_order", ()) or ())]
        _handles = {
            tuple(v) for v in
            (getattr(spec, "synthetic_to_real", {}) or {}).values()
        }
        _ordered_handles = [m for m in _mo if m in _handles]
        _bottom = _ordered_handles[0] if _ordered_handles else None
        _cover = _ordered_handles[-1] if _ordered_handles else None
        _role_handle = {"p": _bottom, "n": _cover}
        # If two declared terminal tips fall on one placed-stamp conductor, that
        # component has no defensible single plate owner.  Refuse it for BOTH roles;
        # each may still reproject to its own signed vendor plate handle below.
        component_terms: Dict[Tuple[_Layer, int], set] = {}
        for term, (rel, layer, _tnet) in per_term.items():
            component = _component_at(layer, rel)
            if component is not None:
                component_terms.setdefault((tuple(layer), component), set()).add(term)
        ambiguous_components = {
            key for key, terms in component_terms.items() if len(terms) > 1
        }

        landed: List[Tuple[_XY, _Layer, int]] = []

        def _same_layer_conflict(pt, lay, component: Optional[int] = None) -> bool:
            if component is None:
                component = _component_at(lay, pt)
            if component is None:
                return True  # an AABB hit is not proof of on-artwork access
            if (tuple(lay), component) in ambiguous_components:
                return True
            for (cpt, clay_, ccomponent) in landed:
                if tuple(clay_) != tuple(lay):
                    continue
                if tuple(pt) == tuple(cpt):
                    return True  # co-located pins: one forced grid column
                if component == ccomponent:
                    return True  # same merged stamp conductor
            return False

        def _land_on(rel, lay, *, allow_reproject: bool):
            """A proven landing on one uncontested merged stamp conductor."""
            component = _component_at(lay, rel)
            if component is not None \
                    and not _same_layer_conflict(rel, lay, component):
                return rel, component
            if not allow_reproject:
                return None, None
            components = stamp_components.get(tuple(lay), ())
            for index in sorted(
                range(len(components)), key=lambda i: components[i].area(),
                reverse=True,
            ):
                pt = _component_point(lay, index)
                if pt is not None \
                        and not _same_layer_conflict(pt, lay, index):
                    return pt, index
            return None, None

        for term in list(per_term):
            rel, layer, tnet = per_term[term]
            layer = tuple(layer)
            pt, component = _land_on(
                rel, layer, allow_reproject=layer in _handles
            )
            own_handle = _role_handle.get(term)
            if pt is None and layer not in _handles and own_handle is not None \
                    and own_handle != layer:
                # Internal-riser terminal -> the SAME plate's vendor handle.
                cpt, ccomponent = _land_on(
                    rel, own_handle, allow_reproject=True
                )
                if cpt is not None:
                    layer, pt, component = own_handle, cpt, ccomponent
            if pt is None:
                del per_term[term]
                gaps.append((
                    name, term, pn[term],
                    f"placed stamp draws no metal on the terminal's own plate "
                    f"handle for layer {tuple(layer)} (cap terminal not wired "
                    f"-- honest gap)"
                ))
                continue
            per_term[term] = (pt, layer, tnet)
            landed.append((tuple(pt), layer, int(component)))

    if not per_term:
        return None, gaps or _all_gaps("cap has no routable plate landings")
    return per_term, gaps


@dataclass(frozen=True)
class _PassiveCapAccessSpec:
    name: str
    nets: Dict[str, str]
    value: float
    M: int
    model: str
    orient: str
    kind: str = "cap"


def compile_passive_cap_terminal_accesses(
    *,
    name: str,
    nets: Dict[str, str],
    value: float,
    multiplier: int,
    model: str,
    orient: str,
    tech: Any,
    stamp: Any = None,
):
    """Compile the one canonical MIM plate-access contract used by all stages."""

    return _passive_cap_pins(
        _PassiveCapAccessSpec(
            name=str(name),
            nets={str(role): str(net) for role, net in nets.items()},
            value=float(value),
            M=int(multiplier),
            model=str(model),
            orient=str(orient),
        ),
        tech,
        stamp=stamp,
    )


def _passive_pins_from_stamp(
    name: str,
    stamp: Any,
    pos: _XY,
    m1: _Layer,
) -> Optional[Dict[str, _XY]]:
    """Derive a 2-terminal passive's p/n M1 access points from its PLACED DEVICE STAMP.

    Used ONLY when the PDK has no parametric passive cell. ``stamp`` is the passive's
    own :class:`~clay.layout.extract.library_device.CellGeometry` device stamp
    (the captured PCell artwork -- the DEVICE CELL, not a routed wire). We read the
    stamp's OWN M1 terminal-CONTACT shapes (e.g. a poly resistor's two end contacts)
    and assign the two EXTREME M1 contacts (farthest apart) to the two terminals by
    position. This is design/device-cell geometry, NOT reference routing, so it stays
    leakage-free. Returns ``{term -> rel_xy_um}`` (centroid-relative) or ``None`` when
    the stamp has no M1 (e.g. a MIM cap whose plates are on upper metal) -- the caller
    then reports an honest gap rather than fabricating an M1 pin.

    NOTE: which extreme is 'p' vs 'n' is assigned by geometric extreme (lowest- vs
    highest- coordinate contact); both contacts ARE real on-device M1 access points
    regardless, so a swap never produces a floating/off-device pin -- only the
    role<->contact label could differ, which the router does not depend on (it joins
    all pins of a net). This labeling caveat is recorded by the caller.
    """
    if stamp is None:
        return None
    m1_rings = (getattr(stamp, "polys_by_layer", {}) or {}).get(m1)
    if not m1_rings:
        return None
    # Each M1 ring -> its centre (centroid-relative um, as the stamp is stored).
    centres: List[_XY] = []
    for ring in m1_rings:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        centres.append(((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0))
    if not centres:
        return None
    # the two contacts farthest apart are the two device terminals.
    best: Optional[Tuple[float, _XY, _XY]] = None
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            a, b = centres[i], centres[j]
            d = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
            if best is None or d > best[0]:
                best = (d, a, b)
    if best is None:
        # only one M1 contact -> both terminals share it (degenerate but on-device).
        c = centres[0]
        return {"p": c, "n": c}
    _d, a, b = best
    # assign by position: the lower (y, then x) contact is 'p', the higher is 'n'
    # (a deterministic, geometry-only rule; see the docstring caveat).
    lo, hi = sorted((a, b), key=lambda c: (c[1], c[0]))
    return {"p": lo, "n": hi}


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def extract_pins(
    intent: Any,
    placement: Dict[str, Any],
    *,
    tech: Any,
    extra_power_nets: Sequence[str] = (),
    geometry: Optional[Dict[str, Any]] = None,
    return_report: bool = False,
    include_power: bool = False,
    row_height_target_um: Optional[float] = None,
    per_device_fold: bool = False,
    fold_devices: Sequence[str] = (),
):
    """Per SIGNAL net, the device-terminal pins at absolute um on M1 (leakage-free).

    Parameters
    ----------
    intent
        The extracted intent (``clay.layout.extract.types.ExtractedIntent``): its
        ``devices`` (MOS) + ``passives`` carry the netlist topology (each device's
        ``nets`` role->net map) and the per-device parameters (kind, W, L, M / value)
        the device cell is built from.
    placement
        ``{device_name -> centroid}`` in MICRONS (as produced by
        :func:`clay.layout.extract.realize_render.realize_clean`). ``centroid`` may be
        an ``(x, y)`` tuple/list or an object with ``.x`` / ``.y``. This is the ONLY
        positional input -- leakage-free (solved from relations/order, never a stored
        reference coordinate).
    tech
        A :class:`clay.layout.tech.LayoutTech` (e.g. ``LayoutTech.resolve('open_pdk')``).
        ``tech.metal_key(1)`` gives the M1 ``(layer, datatype)`` every pin lands on.
    extra_power_nets
        Extra net names to treat as power (excluded), beyond the default vdd/vss/gnd
        vocabulary.
    geometry
        OPTIONAL ``{name -> CellGeometry}`` placed device stamps (from
        :func:`clay.layout.extract.realize_render.real_device_geometry`). Exact
        materializer-owned MOS terminal inventories are authoritative, including
        the absence of a separate source-welded/shared-well bulk access. Legacy
        stamps without an inventory use the device probe and actual stamped M1.
        Passive contacts likewise come from their own artwork, never ref routing.
    return_report
        When True, return a :class:`PinReport` (pins + power nets + honest gaps +
        synthesized-landing provenance). When False (default) return just the
        ``{net -> [Pin]}`` dict.
    row_height_target_um
        OPTIONAL serpentine-fold target (the density path). When set, the MOS
        pin-probe is built FOLD-AWARE (the same finger split + serpentine band
        count the emitted stamp uses) so each terminal pin lands on the real
        per-band M1 of the folded artwork -- the bulk 'b' tap never coincides with
        the source/drain M1. ``None`` (default) keeps the byte-identical single-row
        probe. MUST match the ``row_height_target_um`` the placement/geometry were
        folded with (the caller threads ONE fold through sizes + geometry + pins).

    Returns
    -------
    dict[str, list[Pin]]  (or :class:`PinReport` when ``return_report``)
        For each signal net, the pins joining its device terminals. Each ``Pin.layer``
        is M1 (``tech.metal_key(1)``); each ``Pin.xy_um`` is an absolute (x, y) in
        microns that lies on the placed device's own M1 artwork. Complete per net:
        one pin per device terminal the netlist binds to the net (a terminal whose
        device cell has no M1 access point is reported in the report's ``uncovered``,
        never faked).
    """
    m1: _Layer = tuple(tech.metal_key(1))  # type: ignore[assignment]
    report = PinReport()
    declared_fold_set = {str(dev_name) for dev_name in (fold_devices or ())}

    def _excl_power(net: str) -> bool:
        """Exclude this net as power? -- yes only when NOT routing the rails themselves."""
        return (not include_power) and _is_power_net(net, extra_power_nets)

    def _pos_xy(name: str) -> Optional[_XY]:
        c = placement.get(name)
        if c is None:
            return None
        if isinstance(c, (tuple, list)):
            return (float(c[0]), float(c[1]))
        x = getattr(c, "x", None)
        y = getattr(c, "y", None)
        if x is not None and y is not None:
            return (float(x), float(y))
        try:
            return (float(c[0]), float(c[1]))
        except Exception:
            return None

    # snaps under this distance (um) are sub-grid edge-clamping, not a real
    # build/vendor fold mismatch -- not recorded as a notable snap.
    _SNAP_EPS_UM = 0.02

    def _emit(
        net: str,
        device: str,
        term: str,
        entry: Tuple[_XY, float, bool],
        pos: _XY,
        layer: _Layer = m1,
        binding=None, stamp_key=None,
    ) -> None:
        (rx, ry), dist, inside = entry
        abs_nm = placed_point_nm(rx, ry, pos)
        if geometry is not None:
            stamp_name = str(device).split("@", 1)[0] if stamp_key is None else stamp_key
            stamp = geometry.get(stamp_name)
            if stamp is not None:
                abs_nm = _manufacturing_grid_access_on_stamp(
                    stamp,
                    pos,
                    tuple(layer),
                    abs_nm,
                    tech,
                )
        abs_xy = (abs_nm[0] / _NM_PER_UM, abs_nm[1] / _NM_PER_UM)
        report.pins_by_net.setdefault(net, []).append(
            Pin(
                net=net,
                device=device,
                terminal=term,
                xy_um=abs_xy,
                layer=layer,
                xy_nm=abs_nm,
                terminal_binding=binding,
            )
        )
        if dist > _SNAP_EPS_UM:
            report.snaps.append((device, term, round(dist, 4)))
            if not inside:
                report.synthesized_landings.append((device, term))

    # ---- MOS devices: build_device terminal -> M1 access, translate by placement ----
    for d in getattr(intent, "devices", []) or []:
        name = str(getattr(d, "name", ""))
        if not name:
            continue
        nets = getattr(d, "nets", {}) or {}
        pos = _pos_xy(name)
        native_terminals = getattr(d, "native_terminals", None)
        if native_terminals is not None:
            stamp = geometry.get(name) if geometry is not None else None
            accesses = getattr(stamp, "terminal_accesses_um", {}) or {}
            layers = getattr(stamp, "terminal_layers", {}) or {}
            if pos is None or set(accesses) != set(native_terminals) or set(layers) != set(native_terminals):
                raise ValueError("native device lost its complete placed terminal inventory: " + name)
            from clay.layout.native_sky130 import NativeRouteDevice
            if type(d) is NativeRouteDevice:
                # This ledger is carried only from the validated native bundle at
                # the ordinary placement bridge. Never recover identities from a
                # display name or from access coordinates.
                from clay.layout.route.terminal_identity import PhysicalTerminalPinBinding
                import json
                ledger = getattr(stamp, 'native_terminal_nodes', None)
                if type(ledger) is not dict or set(ledger) != set(native_terminals):
                    raise ValueError('native terminal source-node ledger missing: ' + name)
                authority = getattr(stamp, 'native_terminal_authority', None)
                current = tuple((role, ledger[role], tuple(accesses[role]), tuple(layers[role]),
                                 native_terminals[role]) for role in sorted(ledger))
                if type(authority) is not tuple or current != authority:
                    raise ValueError('native terminal source/access authority changed: ' + name)
                components = {}
                for term in sorted(native_terminals):
                    nodes = ledger[term]
                    if (type(nodes) is not tuple or not nodes or nodes != tuple(sorted(set(nodes))) or
                        any(type(node) is not str or not node for node in nodes) or
                        len(nodes) != len(accesses[term])):
                        raise ValueError('native terminal node/access inventory differs: ' + name + '.' + term)
                    for node, point in zip(nodes, accesses[term]):
                        net = native_terminals[term]
                        signature = (net, tuple(point), tuple(layers[term]))
                        if node in components:
                            if components[node]['signature'] != signature:
                                raise ValueError('native shared node differs across roles: ' + name)
                        else:
                            components[node] = {'signature': signature, 'roles': []}
                        components[node]['roles'].append(term)
                for node, data in sorted(components.items()):
                    net, point, layer = data['signature']
                    if not net or _excl_power(net):
                        continue
                    refs = tuple(sorted((json.dumps([name,node,role],ensure_ascii=True,separators=(',',':')),
                                         name,role) for role in data['roles']))
                    binding = PhysicalTerminalPinBinding('native:' + json.dumps(name,ensure_ascii=True),
                        'node:' + json.dumps(node,ensure_ascii=True), refs)
                    _source, owner, role = refs[0]
                    _emit(net, owner, role, (point, 0.0, True), pos, layer=layer,
                          binding=binding, stamp_key=name)
            else:
                for term, net in native_terminals.items():
                    if not net or _excl_power(net):
                        continue
                    if not accesses[term]:
                        raise ValueError("native terminal has no physical access: " + name + "." + term)
                    for index, point in enumerate(accesses[term]):
                        owner = name if index == 0 else f"{name}@{index + 1}"
                        _emit(net, owner, term, (point, 0.0, True), pos, layer=layers[term])
            continue
        # per_device_fold: build EACH MOS probe with the SAME per-device fold gate the
        # placer + route geometry bridge use (resolve_device_fold_target), so a folded
        # device's terminal pins land on the real per-band M1 of the WIDE serpentine
        # artwork the placement reserved -- not single-row coordinates that would snap
        # onto the wrong net's pad. None caller (route_cell) opts in; explicit-target
        # callers (the OTA) keep their threaded uniform fold. Default OFF = unchanged.
        _rht = row_height_target_um
        _stamp_single_row = (
            name in declared_fold_set
            or bool(
                getattr(
                    geometry.get(name) if geometry is not None else None,
                    "single_row_fold",
                    False,
                )
            )
        )
        if per_device_fold and _rht is None and not _stamp_single_row:
            try:
                from clay.layout.extract.realize_render import (
                    resolve_device_fold_target,
                )
                _rht = resolve_device_fold_target(
                    tech, str(getattr(d, "kind", "")),
                    float(getattr(d, "W", 0.0)) * 1e-6,
                    float(getattr(d, "L", 0.0)) * 1e-6,
                    int(round(float(getattr(d, "M", 1) or 1))),
                )
            except Exception:  # noqa: BLE001 - fold is an optional density refinement
                _rht = None
        geom = (
            _build_mos_cell(
                d,
                tech,
                row_height_target_um=_rht,
                substrate_rail_nets=extra_power_nets,
            )
            if pos is not None else None
        )
        if pos is None or geom is None:
            # no placement or genuine build failure -> every terminal is a gap.
            for role, term in _MOS_ROLE_TO_TERM.items():
                net = str(nets.get(role, ""))
                if net and not _excl_power(net):
                    reason = "no placement" if pos is None else "device cell build failed"
                    report.uncovered.append((name, term, net, reason))
            continue
        # Prefer snapping onto the PLACED device stamp's OWN M1 (the artwork actually
        # rendered -- vendor PCell stripes or build_device M1) so every pin lands on
        # real placed metal; fall back to the parametric cell's M1 when no stamp is
        # supplied. Both are the device cell's artwork (leakage-free), never a ref wire.
        snap_boxes = None
        if geometry is not None:
            sb = _stamp_m1_boxes_um(geometry.get(name), m1)
            if sb:
                snap_boxes = sb
        stamp = _terminal_m1_stamp(geom, m1, snap_boxes_um=snap_boxes)
        materialized_accesses = (
            getattr(geometry.get(name), "terminal_accesses_um", {}) or {}
            if geometry is not None
            else {}
        )
        exposed = set(getattr(geom, "terminals", {}) or {})
        for role, term in _MOS_ROLE_TO_TERM.items():
            net = str(nets.get(role, ""))
            if not net or _excl_power(net):
                continue
            # The device cell may not expose a separate BULK terminal: when the bulk
            # is tied through the source rail (body == source net) or a contiguous
            # SHARED WELL, ``build_device`` draws no distinct 'b' pin -- the bulk net
            # is already carried by the source pin (same net) or the well merge, so it
            # is NOT a separate routable terminal and NOT a gap. Skip a body role with
            # no exposed terminal silently. (A primary s/d/g role the cell exposes but
            # whose centre has no M1 shape IS a real gap and is reported.)
            #
            # BULK-PIN RULE (fact-keyed on the device's TAP + WELL geometry; no
            # circuit / net-name / PDK branch). A bulk terminal is a ROUTABLE pin
            # only where the placed artwork draws the body tie as the net's own
            # routable geometry:
            #
            #   (i)  LOCAL isolated WELL tap (source != bulk PMOS,
            #        ``well_tap_isolated``) CONFIRMED by the placed stamp: the tap
            #        island IS the declared body net's pin geometry -- the 'b'
            #        centre sits dist-0 inside the tap's own M1 pad, min-spaced
            #        from the source rail by construction. A well is an ISOLATED
            #        conductor: drawn metal is its ONLY tie, so this pin is a hard
            #        routing obligation. (Measured: skipping it left every s!=b
            #        tie dead artwork -- check_bulk_tie_fidelity "unattached" x4
            #        on the route_cell comparator, and the external_pdk 4-terminal LVS
            #        bulk mismatches.)
            #   (ii) source-welded tap (source == bulk): the tap M1 is welded into
            #        the source rail in-stamp; the b pin lands on that shared
            #        same-net island (long-standing behaviour, kept).
            #   (iii) SUBSTRATE-bulk device (the device cell draws NO well) with
            #        source != bulk: NOT a routable pin. The tie mechanism is the
            #        DECK-GLOBAL substrate through the device's own drawn P+ tap
            #        contact (``connect_global``) -- routed metal is not what
            #        ties it, so emitting a pin fabricates a daisy-chain routing
            #        obligation with no electrical content (the LDO SUB 16-pin
            #        band chain, where one emit-time via suppression opened the
            #        whole net). This is exactly the class
            #        check_bulk_tie_fidelity adjudicates as skipped ("deck-global
            #        tie"): routing obligations stay aligned with what the
            #        checker can hold anyone to.
            #   (iv) isolated well tap NOT confirmed by a placed stamp (a vendor
            #        PCell capture that never drew the parametric island, or no
            #        ``geometry`` supplied): NOT a pin -- a pin must land on
            #        DRAWN metal. Emitting the parametric model's island position
            #        onto artwork that never drew it hands the router a phantom
            #        target, or snaps onto arbitrary nearest M1 (the LDO cascode
            #        source->VDDA merge class).
            if term == "b":
                if getattr(geom, "nwell_bbox", None) is None and (
                    str(nets.get("body", "") or "")
                    != str(nets.get("source", "") or "")
                ):
                    continue  # (iii) substrate-bulk s!=b: deck-global tie
                if getattr(geom, "well_tap_isolated", False) and not (
                    geometry is not None
                    and getattr(geometry.get(name), "well_tap_isolated", False)
                ):
                    continue  # (iv) island not present in the placed artwork
            exact_accesses = tuple(materialized_accesses.get(term, ()) or ())
            if exact_accesses:
                for access_index, rel_xy in enumerate(exact_accesses):
                    try:
                        access_xy = (float(rel_xy[0]), float(rel_xy[1]))
                    except (IndexError, TypeError, ValueError):
                        report.uncovered.append((
                            name,
                            term,
                            net,
                            "malformed materialized terminal access",
                        ))
                        continue
                    # Topology/resource certificates require one stable
                    # provenance per exact physical access.  A multi-row stamp
                    # can draw several disconnected M1 islands for one logical
                    # MOS terminal; reusing ``pin:<device>:<terminal>`` for all
                    # of them makes the exact plan ambiguous.  The established
                    # fixed-cell BJT convention keeps the first logical name and
                    # deterministically numbers the additional physical access
                    # identities.  Net and terminal remain unchanged, so this
                    # is provenance disambiguation rather than a netlist rewrite.
                    physical_name = (
                        name
                        if access_index == 0
                        else f"{name}@{access_index + 1}"
                    )
                    _emit(
                        net,
                        physical_name,
                        term,
                        (access_xy, 0.0, True),
                        pos,
                    )
                continue
            if materialized_accesses:
                # A discrete/folded materializer owns the COMPLETE access
                # inventory. Missing 'b' means there is no separate routable
                # source-welded/shared-well tap. A standalone probe can have a
                # different tap topology; snapping its absent body terminal to
                # arbitrary nearest array M1 can land on a drain and poison the
                # real conductor's net ownership. Never invent such an access.
                if term != "b" or getattr(
                    geometry.get(name), "well_tap_isolated", False
                ):
                    report.uncovered.append((
                        name, term, net,
                        "materialized geometry exposes no such terminal access",
                    ))
                continue
            if term not in exposed:
                if term == "b":
                    continue  # bulk-via-source / shared-well: not a routable pin
                report.uncovered.append(
                    (name, term, net, "device cell exposes no such terminal")
                )
                continue
            entry = stamp.get(term)
            if entry is None:
                report.uncovered.append((name, term, net, "no M1 shape at terminal"))
                continue
            _emit(net, name, term, entry, pos)

    # ---- passives: parametric cell where the PDK has one, else placed-stamp M1 ----
    # ---- fixed-cell BJTs: exact adapter-labelled access carried on the stamp ----
    for bjt in getattr(intent, "bjts", []) or []:
        name = str(getattr(bjt, "name", ""))
        if not name:
            continue
        nets = getattr(bjt, "nets", {}) or {}
        pos = _pos_xy(name)
        stamp = geometry.get(name) if geometry is not None else None
        offsets = getattr(stamp, "terminal_offsets_um", {}) or {}
        all_offsets = getattr(stamp, "terminal_accesses_um", {}) or {}
        layers = getattr(stamp, "terminal_layers", {}) or {}
        for role, term in _BJT_ROLE_TO_TERM.items():
            net = str(nets.get(role, ""))
            if not net or _excl_power(net):
                continue
            if pos is None:
                report.uncovered.append((name, term, net, "no placement"))
                continue
            offset = offsets.get(term)
            layer = layers.get(term)
            if offset is None or layer is None:
                report.uncovered.append(
                    (name, term, net, "fixed BJT exposes no routed terminal access")
                )
                continue
            accesses = tuple(all_offsets.get(term, ()) or (offset,))
            for index, access in enumerate(accesses):
                physical_name = name if index == 0 else f"{name}@{index + 1}"
                _emit(
                    net,
                    physical_name,
                    term,
                    ((float(access[0]), float(access[1])), 0.0, True),
                    pos,
                    layer=tuple(int(v) for v in layer),
                )
        # A declared external companion tap is an additional routable terminal
        # carried by the same fixed-cell stamp.  Intrinsic-terminal adapters draw
        # no duplicate access here: their existing C/B/E terminal already owns
        # the declared net and was validated by the materializer.
        tap_net = str(getattr(bjt, "tap_net", "") or "")
        tap_offset = offsets.get("tap")
        tap_layer = layers.get("tap")
        if tap_net and not _excl_power(tap_net) \
                and tap_offset is not None and tap_layer is not None and pos is not None:
            _emit(
                tap_net,
                name,
                "tap",
                ((float(tap_offset[0]), float(tap_offset[1])), 0.0, True),
                pos,
                layer=tuple(int(v) for v in tap_layer),
            )

    # ---- passives: parametric cell where the PDK has one, else placed-stamp M1 ----
    for p in getattr(intent, "passives", []) or []:
        name = str(getattr(p, "name", ""))
        if not name:
            continue
        nets = getattr(p, "nets", {}) or {}
        pos = _pos_xy(name)
        if pos is None:
            for role, term in _PASSIVE_ROLE_TO_TERM.items():
                net = str(nets.get(role, ""))
                if net and not _excl_power(net):
                    report.uncovered.append((name, term, net, "no placement"))
            continue

        # Retained native passives never rebuild a scalar/array/bank substitute.
        # A marked but incomplete stamp cannot silently fall back to that path.
        from clay.layout.route.passive_native_pins import emit_native_passive_pins
        if emit_native_passive_pins(p, geometry=geometry, placement=placement,
                position=pos, tech=tech, emit=_emit, report=report, exclude_power=_excl_power):
            continue

        # --- MIM cap: distinct-plate terminal access (generic, DATA-driven) ----------
        # A MIM cap's two plates overlap in 2D but are TWO electrical nodes, each on its
        # OWN routing metal (the bottom plate's Metal2, the top plate's cover). The
        # generic distinct-plate extractor lands each terminal on its own plate node, so
        # the two plates are never collapsed to one layer (the reverted per-case hack
        # pinned both on Metal2 and SHORTED the cap). Handled when the passive cell has
        # no M1 access (the plates are on upper metal); a resistor (M1 contacts) still
        # takes the M1 path below.
        kind = str(getattr(p, "kind", "")).lower()
        if kind.startswith("cap") or kind == "c":
            per_term, cap_gaps = compile_passive_cap_terminal_accesses(
                name=name,
                nets={str(role): str(net) for role, net in nets.items()},
                value=float(getattr(p, "value", 0.0) or 0.0),
                multiplier=int(getattr(p, "M", 1) or 1),
                model=str(getattr(p, "model", "") or ""),
                orient=str(getattr(p, "orient", "R0") or "R0"),
                tech=tech,
                stamp=geometry.get(name) if geometry is not None else None,
            )
            if per_term is not None:
                for term, (rel_xy, rlayer, tnet) in per_term.items():
                    net = str(nets.get(term, "")) or tnet
                    if not net or _excl_power(net):
                        continue
                    _emit(net, name, term, (rel_xy, 0.0, True), pos, layer=rlayer)
                # report any honest per-terminal gaps (e.g. an unmapped plate) without
                # faking; a power-net plate excluded by policy is NOT a gap.
                for (gn, gt, gnet, greason) in cap_gaps:
                    if gnet and not _excl_power(gnet):
                        report.uncovered.append((gn, gt, gnet, greason))
                continue
            # per_term is None -> cap could not be wired distinct-plate: honest gap.
            for (gn, gt, gnet, greason) in cap_gaps:
                if gnet and not _excl_power(gnet):
                    report.uncovered.append((gn, gt, gnet, greason))
            continue

        rel_stamp: Optional[Dict[str, Tuple[_XY, float, bool]]] = None
        cell = _build_passive_cell(p, tech)
        if cell is not None:
            snap_boxes = None
            if geometry is not None:
                sb = _stamp_m1_boxes_um(geometry.get(name), m1)
                if sb:
                    snap_boxes = sb
            ts = _terminal_m1_stamp(cell, m1, snap_boxes_um=snap_boxes)
            if ts:
                rel_stamp = ts
        if rel_stamp is None and geometry is not None:
            # PDK has no parametric passive cell: use the passive's OWN placed device
            # stamp's M1 terminal contacts (device-cell artwork, not reference routing).
            # These ARE the stamp's own M1 contact centres -> exact on-device (dist 0).
            pn = _passive_pins_from_stamp(name, geometry.get(name), pos, m1)
            if pn is not None:
                rel_stamp = {t: (xy, 0.0, True) for t, xy in pn.items()}

        if rel_stamp is None:
            # no derivable M1 access (e.g. a MIM cap whose plates are on upper metal,
            # on a PDK with no parametric cap cell) -> honest gap, never a fake pin.
            for role, term in _PASSIVE_ROLE_TO_TERM.items():
                net = str(nets.get(role, ""))
                if net and not _excl_power(net):
                    report.uncovered.append(
                        (name, term, net, "no M1 access on device (no parametric cell)")
                    )
            continue

        for role, term in _PASSIVE_ROLE_TO_TERM.items():
            net = str(nets.get(role, ""))
            if not net or _excl_power(net):
                continue
            entry = rel_stamp.get(term)
            if entry is None:
                report.uncovered.append((name, term, net, "no M1 shape at terminal"))
                continue
            _emit(net, name, term, entry, pos)

    # collect the power nets we actually saw (for the report).
    seen_power = set()
    for src in (getattr(intent, "devices", []) or []):
        for v in (getattr(src, "nets", {}) or {}).values():
            if _is_power_net(str(v), extra_power_nets):
                seen_power.add(str(v))
    for src in (getattr(intent, "passives", []) or []):
        for v in (getattr(src, "nets", {}) or {}).values():
            if _is_power_net(str(v), extra_power_nets):
                seen_power.add(str(v))
    for src in (getattr(intent, "bjts", []) or []):
        for v in (getattr(src, "nets", {}) or {}).values():
            if _is_power_net(str(v), extra_power_nets):
                seen_power.add(str(v))
    report.power_nets = sorted(seen_power)

    if return_report:
        return report
    return report.pins_by_net


__all__ = ["extract_pins", "Pin", "PinReport"]
