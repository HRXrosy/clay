"""POWER DISTRIBUTION for a migrated gf180 placement -- rails + supply taps.

The problem this closes
=======================
The gf180 migration (``migrate.py``) emits ONLY per-device geometry: each device's
COMP / poly / contact / M1 / implant / nwell stamped at its migrated centroid. A real
chip layout cannot work with device geometry alone -- it needs a POWER GRID: a VDD
strap the PMOS sources tie up to, a VSS strap the NMOS sources tie down to, plus the
bulk taps (N+ in nwell to VDD, P+ in substrate to VSS) that bias the wells / latch-up
protect. This module builds exactly that geometry, generically, from:

  * the device KIND (``intent.devices[].kind`` -- 'pmos' / 'nmos');
  * the supply CONNECTIVITY (each device's ``.nets`` after the SAME sanitize the
    emitter uses -- ``vdd!`` -> ``vdd``, ``gnd!``/``0`` -> ``vss``); a device taps the
    VDD rail iff its SOURCE net sanitizes to ``vdd``, the VSS rail iff its SOURCE net
    sanitizes to ``vss``;
  * the migrated POSITIONS (``{name -> (cx, cy) um}``) + gf180 SIZES
    (``{name -> (w, h) um}``) from ``migrate.build_gf180_geometry`` +
    ``relgraph.solve_hierarchical_nonoverlap``.

WHAT IS / ISN'T CONNECTED (honest scope)
========================================
This module is POWER DISTRIBUTION only:

  * DONE: a VDD horizontal M1 strap spanning the cell width along the PMOS band's
    OUTER (top) edge; a VSS strap along the NMOS band's OUTER (bottom) edge; for each
    PMOS a metal stub + N+/nwell tap tying its SOURCE+BODY region up to VDD; for each
    NMOS a metal stub + P+ substrate tap tying SOURCE+BODY down to VSS.
  * NOT DONE (a sibling module owns this): SIGNAL routing -- gate/drain net nets,
    differential pairs, internal nodes. Nothing here routes a signal net. The taps
    connect SOURCE + BODY to the supply rails; they do NOT connect drains or gates.

gf180 layers used (gf180mcu.lyp ground truth, via ``LayoutTech`` / backend layermap)
====================================================================================
  * rails + stubs:  M1   = 34/0   (a real routing metal)
  * tap diffusion:  COMP = 22/0   (the single gf180 active layer)
  * tap contact:    CONTACT = 33/0
  * PMOS (VDD) tap: Nplus = 32/0 implant + Nwell = 21/0  (N+ in nwell = ntap)
  * NMOS (VSS) tap: Pplus = 31/0 implant (P+ in p-substrate = ptap)

All keys are resolved through ``LayoutTech.resolve('gf180mcu')`` so a different tech
(if it carries the same logical layers) flows through unchanged -- no gf180 numeric
constant is hard-coded in the geometry logic.

SCOPE / DISCIPLINE
==================
Purely compositional. Does NOT edit migrate.py / relgraph.py / fold.py /
route_migrate.py / the placers / the grammar / any design.clay. It consumes their
documented surfaces (the intent's device kind+nets, the migrated positions/sizes) and
the ``LayoutTech`` layer surface, and returns geometry the integrator splices into the
migration GDS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_NM_PER_UM = 1000.0

_XY = Tuple[float, float]  # (x, y) um
_BboxNm = Tuple[int, int, int, int]  # (x0, y0, x1, y1) nm


# --------------------------------------------------------------------------- #
# net sanitize (the SAME rule emit_clay._san applies, restricted to supplies)
# --------------------------------------------------------------------------- #
def _supply_of(raw_net: Any) -> Optional[str]:
    """Return 'vdd' / 'vss' if ``raw_net`` sanitizes to a supply, else None.

    Mirrors ``clay.layout.extract.emit_clay._san`` for the supply aliases: strip the
    Spectre global marker ``!``, then ``vdd``/``vcc`` -> 'vdd', ``gnd``/``0``/``vss``
    -> 'vss'. Any other net is a signal net (returns None) -- this module never taps a
    signal net to a rail.
    """
    s = str(raw_net).strip().rstrip("!").lower()
    if s in ("gnd", "0", "vss"):
        return "vss"
    if s in ("vdd", "vcc"):
        return "vdd"
    return None


# --------------------------------------------------------------------------- #
# data structures
# --------------------------------------------------------------------------- #
@dataclass
class _Poly:
    """One axis-aligned rectangle on a gf180 layer (nm, dbu = 1 nm).

    Mirrors the (layer, datatype, vertices) shape of ``backend.LayerPolygon`` so the
    integrator can splice these straight into the migration GDS, but defined here so
    this module never imports the heavy build path just to make a rectangle.
    """

    layer: int
    datatype: int
    bbox_nm: _BboxNm  # (x0, y0, x1, y1) nm

    def to_layer_polygon(self):
        """Convert to a ``backend.LayerPolygon`` (for GDS emission / splicing)."""
        from clay.layout.backend import LayerPolygon, Point, Polygon

        x0, y0, x1, y1 = self.bbox_nm
        return LayerPolygon(
            layer=self.layer,
            datatype=self.datatype,
            polygon=Polygon(
                vertices=[Point(x0, y0), Point(x1, y0), Point(x1, y1), Point(x0, y1)]
            ),
        )


@dataclass
class SupplyTap:
    """One device's supply tap (the geometry tying its SOURCE+BODY to a rail).

    ``device`` is the device name; ``rail`` is 'vdd' or 'vss'. ``polys`` is the tap's
    gf180 geometry (COMP + implant [+ nwell for VDD] + CONTACT + M1 stub up/down to the
    rail), all in nm. ``stub_bbox_nm`` is the M1 stub rectangle that overlaps the rail
    (the electrical join). ``source_region_nm`` is the device source-side area the tap
    sits under (so a caller / test can confirm the tap actually touches the source).
    """

    device: str
    rail: str
    polys: List[_Poly] = field(default_factory=list)
    stub_bbox_nm: _BboxNm = (0, 0, 0, 0)
    source_region_nm: _BboxNm = (0, 0, 0, 0)


@dataclass
class PowerGeom:
    """The full power-distribution geometry for a migrated placement.

    ``vdd_rail`` / ``vss_rail`` are the two supply-strap ``_Poly`` (M1) -- ``None`` if
    that band has no supply-tapping device. ``taps`` is every per-device
    :class:`SupplyTap`. ``vdd_devices`` / ``vss_devices`` list which devices tapped
    each rail. ``tech_id`` records the PDK.  ``note`` carries honesty (which devices
    had no position/size, so were skipped).
    """

    vdd_rail: Optional[_Poly] = None
    vss_rail: Optional[_Poly] = None
    taps: List[SupplyTap] = field(default_factory=list)
    vdd_devices: List[str] = field(default_factory=list)
    vss_devices: List[str] = field(default_factory=list)
    tech_id: str = "gf180mcu"
    note: str = ""

    # ---- convenience ----
    def all_polys(self) -> List[_Poly]:
        """Every power ``_Poly`` (rails + all tap geometry), rails first."""
        out: List[_Poly] = []
        if self.vdd_rail is not None:
            out.append(self.vdd_rail)
        if self.vss_rail is not None:
            out.append(self.vss_rail)
        for t in self.taps:
            out.extend(t.polys)
        return out

    def to_layer_polygons(self) -> list:
        """Every power poly as ``backend.LayerPolygon`` (for GDS splicing)."""
        return [p.to_layer_polygon() for p in self.all_polys()]

    @property
    def n_vdd_taps(self) -> int:
        return sum(1 for t in self.taps if t.rail == "vdd")

    @property
    def n_vss_taps(self) -> int:
        return sum(1 for t in self.taps if t.rail == "vss")

    def rail_bbox_nm(self, rail: str) -> Optional[_BboxNm]:
        p = self.vdd_rail if rail == "vdd" else self.vss_rail
        return None if p is None else p.bbox_nm

    def rail_bbox_um(self, rail: str) -> Optional[Tuple[float, float, float, float]]:
        bb = self.rail_bbox_nm(rail)
        if bb is None:
            return None
        return tuple(v / _NM_PER_UM for v in bb)

    def summary(self) -> str:
        def _bb(r):
            bb = self.rail_bbox_um(r)
            return "none" if bb is None else (
                f"({bb[0]:.2f},{bb[1]:.2f})..({bb[2]:.2f},{bb[3]:.2f}) um"
            )

        return (
            f"power[{self.tech_id}]: VDD rail {_bb('vdd')} + {self.n_vdd_taps} tap(s); "
            f"VSS rail {_bb('vss')} + {self.n_vss_taps} tap(s) | "
            f"PMOS->vdd {len(self.vdd_devices)}, NMOS->vss {len(self.vss_devices)}"
        )


# --------------------------------------------------------------------------- #
# gf180 tap geometry constants (sourced from the cited gf180 rules; same
# provenance the substrate_tap_tie module uses). Kept module-level + named.
# --------------------------------------------------------------------------- #
_CONTACT_SIZE_NM = 220       # CO.1 gf180 contact size
_COMP_CONTACT_ENC_NM = 60    # CO.4 COMP encloses contact
_IMPLANT_EXT_NM = 160        # Nplus/Pplus extension past COMP (NP.5a / PP.5a class)
_NWELL_ENC_NM = 160          # nwell encloses N+ comp (so ntap sits inside well)
_RAIL_WIDTH_NM = 3000        # 3 um supply strap (a real power rail)
_RAIL_MARGIN_NM = 1000       # rail overhangs the cell width by this much each side
_RAIL_GAP_NM = 1000          # gap from the band's outer device edge to the rail
_TAP_MIN_COMP_NM = 420       # DF.1a class COMP min width clamp for the tap


def _band_edges_nm(
    intent: Any,
    positions: Dict[str, _XY],
    sizes: Dict[str, _XY],
) -> Tuple[Optional[int], Optional[int], int, int]:
    """Return ``(pmos_top_nm, nmos_bot_nm, cell_x0_nm, cell_x1_nm)``.

    ``pmos_top_nm`` is the max top-edge y of any PMOS device that taps VDD (the PMOS
    band's OUTER/top edge); ``nmos_bot_nm`` is the min bottom-edge y of any NMOS that
    taps VSS (the NMOS band's OUTER/bottom edge). ``cell_x0/x1`` are the full placement
    x-extent (every placed+sized device), so the rail spans the whole cell width.
    Either band edge is ``None`` when that band has no supply-tapping device.
    """
    kind_of = {str(d.name): str(d.kind).lower() for d in (intent.devices or [])}
    supply_of: Dict[str, Optional[str]] = {}
    for d in intent.devices or []:
        supply_of[str(d.name)] = _supply_of((d.nets or {}).get("source"))

    xs: List[int] = []
    pmos_tops: List[int] = []
    nmos_bots: List[int] = []
    for name, (cx, cy) in positions.items():
        wh = sizes.get(name)
        if wh is None:
            continue
        w, h = wh
        x0 = int(round((cx - w / 2.0) * _NM_PER_UM))
        x1 = int(round((cx + w / 2.0) * _NM_PER_UM))
        y0 = int(round((cy - h / 2.0) * _NM_PER_UM))
        y1 = int(round((cy + h / 2.0) * _NM_PER_UM))
        xs.append(x0)
        xs.append(x1)
        if kind_of.get(name) == "pmos" and supply_of.get(name) == "vdd":
            pmos_tops.append(y1)
        if kind_of.get(name) == "nmos" and supply_of.get(name) == "vss":
            nmos_bots.append(y0)
    cell_x0 = min(xs) if xs else 0
    cell_x1 = max(xs) if xs else 0
    pmos_top = max(pmos_tops) if pmos_tops else None
    nmos_bot = min(nmos_bots) if nmos_bots else None
    return pmos_top, nmos_bot, cell_x0, cell_x1


def _device_source_region_nm(cx: float, cy: float, w: float, h: float, rail: str) -> _BboxNm:
    """The device's SOURCE-side region (the half of the bbox nearest its rail).

    Generic convention (no per-device branch): the source ties to the rail, so the tap
    sits on the half of the device bbox closest to the rail -- the TOP half for a VDD
    (PMOS) device, the BOTTOM half for a VSS (NMOS) device. Returned in nm.
    """
    x0 = int(round((cx - w / 2.0) * _NM_PER_UM))
    x1 = int(round((cx + w / 2.0) * _NM_PER_UM))
    y0 = int(round((cy - h / 2.0) * _NM_PER_UM))
    y1 = int(round((cy + h / 2.0) * _NM_PER_UM))
    ymid = (y0 + y1) // 2
    if rail == "vdd":
        return (x0, ymid, x1, y1)  # top half toward VDD
    return (x0, y0, x1, ymid)  # bottom half toward VSS


def _build_tap(
    tech,
    device: str,
    rail: str,
    src_region: _BboxNm,
    rail_bbox: _BboxNm,
) -> SupplyTap:
    """Build one device's supply tap: COMP + implant [+ nwell] + CONTACT + M1 stub.

    The tap sits on the device's source-side region (``src_region``), centred in x. Its
    M1 stub extends from the tap up (VDD) or down (VSS) to OVERLAP the rail. For a VDD
    (PMOS) tap the diffusion is N+ inside an nwell (an ntap biasing the well to VDD);
    for a VSS (NMOS) tap it is P+ in the p-substrate (a ptap to VSS). All keys come from
    the tech (no hard-coded gf180 number in the layer selection).
    """
    comp_key = tech.build_layer("diff")          # COMP 22/0
    contact_key = tech.contact_key()             # 33/0
    m1_key = tech.metal_key(1)                    # 34/0
    if rail == "vdd":
        implant_key = tech.build_layer("nplus")  # 32/0 -> N+ ntap
        well_key = tech.build_layer("nwell")     # 21/0
    else:
        implant_key = tech.build_layer("pplus")  # 31/0 -> P+ ptap
        well_key = None

    sx0, sy0, sx1, sy1 = src_region
    scx = (sx0 + sx1) // 2

    # COMP square clamped to a legal min width that encloses one contact.
    size = max(_TAP_MIN_COMP_NM, _CONTACT_SIZE_NM + 2 * _COMP_CONTACT_ENC_NM)
    half = size // 2
    # place the tap COMP centred on the source region (its y-centre).
    scy = (sy0 + sy1) // 2
    comp = (scx - half, scy - half, scx + half, scy + half)
    cx0, cy0, cx1, cy1 = comp

    implant = (
        cx0 - _IMPLANT_EXT_NM,
        cy0 - _IMPLANT_EXT_NM,
        cx1 + _IMPLANT_EXT_NM,
        cy1 + _IMPLANT_EXT_NM,
    )
    # one contact centred in the comp.
    cs = _CONTACT_SIZE_NM
    contact = (scx - cs // 2, scy - cs // 2, scx + cs // 2, scy + cs // 2)

    # M1 stub: a column from the tap contact reaching to the rail, overlapping it.
    rx0, ry0, rx1, ry1 = rail_bbox
    m1_half = max(cs // 2 + _COMP_CONTACT_ENC_NM, 160)
    if rail == "vdd":
        stub = (scx - m1_half, scy, scx + m1_half, ry1)
    else:
        stub = (scx - m1_half, ry0, scx + m1_half, scy)

    polys: List[_Poly] = []
    if well_key is not None:
        # nwell encloses the N+ comp so the ntap is a recognised well contact.
        well = (
            cx0 - _NWELL_ENC_NM,
            cy0 - _NWELL_ENC_NM,
            cx1 + _NWELL_ENC_NM,
            cy1 + _NWELL_ENC_NM,
        )
        polys.append(_Poly(well_key[0], well_key[1], well))
    polys.append(_Poly(comp_key[0], comp_key[1], comp))
    if implant_key is not None:
        polys.append(_Poly(implant_key[0], implant_key[1], implant))
    polys.append(_Poly(contact_key[0], contact_key[1], contact))
    polys.append(_Poly(m1_key[0], m1_key[1], stub))

    return SupplyTap(
        device=device,
        rail=rail,
        polys=polys,
        stub_bbox_nm=stub,
        source_region_nm=src_region,
    )


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def power_geometry(
    intent: Any,
    positions: Dict[str, _XY],
    sizes: Dict[str, _XY],
    *,
    tech: str = "gf180mcu",
) -> PowerGeom:
    """Build the gf180 power-distribution geometry for a migrated placement.

    Generic, driven by device KIND + SOURCE net == vdd/vss + migrated POSITION:

      * VDD rail -- a horizontal M1 strap (``_RAIL_WIDTH_NM`` tall) spanning the cell
        width (+ a margin each side) placed ``_RAIL_GAP_NM`` ABOVE the PMOS band's
        outer (top) edge.
      * VSS rail -- the mirror strap below the NMOS band's outer (bottom) edge.
      * one tap per supply-tapping device: a PMOS whose SOURCE net is ``vdd`` gets an
        N+/nwell ntap + M1 stub up to the VDD rail; an NMOS whose SOURCE is ``vss``
        gets a P+ ptap + M1 stub down to the VSS rail.

    ``positions`` is ``{name -> (cx, cy) um}``; ``sizes`` is ``{name -> (w, h) um}``
    (both from the migration). A device with no position/size is skipped + noted. No
    signal net is routed here -- this is power distribution only (see module docstring).
    """
    from clay.layout.tech import LayoutTech

    t = LayoutTech.resolve(tech)

    kind_of = {str(d.name): str(d.kind).lower() for d in (intent.devices or [])}
    src_supply: Dict[str, Optional[str]] = {
        str(d.name): _supply_of((d.nets or {}).get("source"))
        for d in (intent.devices or [])
    }

    pmos_top, nmos_bot, cx0, cx1 = _band_edges_nm(intent, positions, sizes)

    pg = PowerGeom(tech_id=tech)
    m1_key = t.metal_key(1)

    rail_x0 = cx0 - _RAIL_MARGIN_NM
    rail_x1 = cx1 + _RAIL_MARGIN_NM

    vdd_rail_bbox: Optional[_BboxNm] = None
    vss_rail_bbox: Optional[_BboxNm] = None
    if pmos_top is not None:
        ry0 = pmos_top + _RAIL_GAP_NM
        vdd_rail_bbox = (rail_x0, ry0, rail_x1, ry0 + _RAIL_WIDTH_NM)
        pg.vdd_rail = _Poly(m1_key[0], m1_key[1], vdd_rail_bbox)
    if nmos_bot is not None:
        ry1 = nmos_bot - _RAIL_GAP_NM
        vss_rail_bbox = (rail_x0, ry1 - _RAIL_WIDTH_NM, rail_x1, ry1)
        pg.vss_rail = _Poly(m1_key[0], m1_key[1], vss_rail_bbox)

    skipped: List[str] = []
    for name in sorted(positions):
        wh = sizes.get(name)
        pos = positions.get(name)
        if wh is None or pos is None:
            skipped.append(name)
            continue
        kind = kind_of.get(name)
        supply = src_supply.get(name)
        # generic gate: a PMOS sourced to vdd taps VDD; an NMOS sourced to vss taps VSS.
        if kind == "pmos" and supply == "vdd" and vdd_rail_bbox is not None:
            rail = "vdd"
            rbox = vdd_rail_bbox
        elif kind == "nmos" and supply == "vss" and vss_rail_bbox is not None:
            rail = "vss"
            rbox = vss_rail_bbox
        else:
            continue
        cx, cy = pos
        w, h = wh
        src_region = _device_source_region_nm(cx, cy, w, h, rail)
        tap = _build_tap(t, name, rail, src_region, rbox)
        pg.taps.append(tap)
        if rail == "vdd":
            pg.vdd_devices.append(name)
        else:
            pg.vss_devices.append(name)

    notes = []
    if skipped:
        notes.append(f"skipped (no position/size): {skipped}")
    if pmos_top is None:
        notes.append("no PMOS device sourced to vdd -> no VDD rail")
    if nmos_bot is None:
        notes.append("no NMOS device sourced to vss -> no VSS rail")
    notes.append(
        "scope: power rails + supply taps (source+body->rail) only; signal routing "
        "(gate/drain nets) NOT done here"
    )
    pg.note = " | ".join(notes)
    return pg


def emit_power_gds(
    power_geom: PowerGeom,
    out_gds: str,
    *,
    cell_name: str = "power",
    dbu: float = 0.001,
) -> str:
    """Write the power geometry to a flat gf180 GDS readable by klayout. Returns the path.

    Stand-alone (the rails + taps as one cell) -- the integrator more commonly splices
    ``power_geom.to_layer_polygons()`` into the migration top cell, but this lets the
    power grid be inspected / rendered on its own.
    """
    import pathlib

    import klayout.db as kdb

    ly = kdb.Layout()
    ly.dbu = dbu
    top = ly.create_cell(cell_name)
    for p in power_geom.all_polys():
        x0, y0, x1, y1 = p.bbox_nm
        li = ly.layer(p.layer, p.datatype)
        top.shapes(li).insert(
            kdb.Polygon([kdb.Point(x0, y0), kdb.Point(x1, y0), kdb.Point(x1, y1), kdb.Point(x0, y1)])
        )
    out = pathlib.Path(out_gds)
    out.parent.mkdir(parents=True, exist_ok=True)
    ly.write(str(out))
    return str(out)


# --------------------------------------------------------------------------- #
# OTA runner -- power geometry for the real OTA migration, end-to-end
# --------------------------------------------------------------------------- #
def run_ota_power(
    out_gds: str = "_review_gds/two_stage_gf180_power.gds",
) -> PowerGeom:
    """Build + print the power geometry for the real OTA gf180 migration.

    Reuses the migration's own geometry/position path: build the OTA intent, build the
    gf180 device geometry (sizes), solve hierarchical-nonoverlap positions (the clean,
    0-overlap migration), then build the power grid. Prints the rail bboxes + tap counts
    and writes the power-only GDS.
    """
    from clay.layout.extract import migrate, relgraph

    intent = migrate.build_ota_intent()
    geometry = migrate.build_gf180_geometry(intent)
    sizes = geometry.sizes_um()
    hier = relgraph.build_hierarchical_graph(intent)
    positions, _resid = relgraph.solve_hierarchical_nonoverlap(hier, sizes, (0.0, 0.0))
    positions = {n: (float(p[0]), float(p[1])) for n, p in positions.items()}

    pg = power_geometry(intent, positions, sizes, tech="gf180mcu")
    path = emit_power_gds(pg, out_gds, cell_name="OTA_power")

    print(f"[power] {pg.summary()}")
    print(f"[power] VDD devices: {pg.vdd_devices}")
    print(f"[power] VSS devices: {pg.vss_devices}")
    print(f"[power] note: {pg.note}")
    print(f"[power] GDS  : {path}")
    return pg


__all__ = [
    "PowerGeom",
    "SupplyTap",
    "power_geometry",
    "emit_power_gds",
    "run_ota_power",
]


if __name__ == "__main__":
    run_ota_power()
