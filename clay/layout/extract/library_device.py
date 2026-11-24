"""LIBRARY-BACKED DEVICE EMIT: render a placement in TRUE external_pdk geometry by
re-placing the reference's OWN resolved PCell instances.

WHY THIS EXISTS
===============
Clay's device builders are gf180-specific: it cannot synthesize external_pdk device
finger geometry from scratch. So to emit a RELAXED placement (from
``relax.place_relaxed``) in the reference's real external_pdk geometry, we do NOT rebuild
devices -- we CAPTURE the reference GDS's resolved external_pdk PCell instances (the
``pmos3v_CDNS_* / nmos3v_CDNS_* / mimcap_CDNS_* / rnhpoly_CDNS_*`` leaf cells that
carry the exact hand-drawn geometry) and re-place those captured cells at intent-
derived centroids. A relaxed placement is then nothing more than a translation of
these exact cells -- the geometry is reference-faithful by construction.

WHAT A "DEVICE" IS HERE
=======================
The netlist has 17 tracked objects (13 MOS + 2 caps + 2 res). The reference layout
FOLDS / interdigitates fingers, so one logical device is realized by one OR MORE
finger-cell INSTANCES. ``netlist_intent.correspond_to_gds`` already maps each named
device to a region (``intent.device_bboxes[name]`` in um). We collect every device-
finger instance whose centre lands inside that region as the geometry realizing the
device, store its polys per layer RELATIVE to the device centroid (a reusable blob),
and report any device whose geometry could not be cleanly captured.

ROUND-TRIP PROOF (devices only)
===============================
``emit_library_gds`` at the ANCHOR centroids (= each device's reference centroid)
must reproduce the reference DEVICE geometry exactly. ``device_layer_xor`` measures
the XOR area on the device layers (poly 13/0, OD 3/0, in-cell M1 15/0, MIM 67/5,
...) between the emitted GDS and the reference's device cells -- it is ~0 at the
anchor, which proves the captured cells are geometry-faithful. Top-level routing /
wells are OUT OF SCOPE (devices only), so the XOR is taken only on device layers and
only over the captured device-instance footprints.

This module does NOT edit relax.py / converge_egraph.py / netlist_intent.py /
geometric_intent.py / placers / grammar. It only reads the intent + reference GDS
and composes klayout.db geometry.

Public API
==========
    extract_device_geometry(intent, ref_gds) -> dict[name -> CellGeometry]
    emit_library_gds(intent, positions, ref_gds, out_gds) -> EmitReport
    device_layer_xor(gds_a, gds_b, layers=None) -> dict[(layer,dt) -> xor_area_um2]
    DEVICE_LAYERS  -- the canonical device-layer set used for XOR verification
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Constants.
# --------------------------------------------------------------------------- #

# Prefix -> logical device kind. The reference resolves Cadence external_pdk PCells to
# leaf cells named ``<kind>3v_CDNS_*`` (MOS), ``mimcap_CDNS_*``, ``rnhpoly_CDNS_*``.
_KIND_PREFIX = (
    ("pmos3v", "pmos"),
    ("nmos3v", "nmos"),
    ("mimcap", "cap"),
    ("poly_res", "res"),
)

# Canonical device layers for the round-trip XOR (GDS layer, datatype).
# These are the geometry-bearing device layers shared across the finger cells:
#   13/0 poly, 3/0 OD/diffusion, 15/0 in-cell M1, 16/0 contact, 4/0 implant select,
#   67/5 MIM cap plate, 2/0 / 7/0 pmos implant/nwell-select, 8/0 res poly select.
# Top-level routing metals (M2..M6 wrapper cells) are NOT in this set -- routing is
# out of scope; we verify DEVICE geometry only.
DEVICE_LAYERS: Tuple[Tuple[int, int], ...] = (
    (3, 0),    # OD / active
    (4, 0),    # implant select
    (13, 0),   # poly
    (15, 0),   # in-cell M1
    (16, 0),   # contact
    (67, 5),   # MIM cap top plate
)

_UM = 1.0  # device_bboxes / positions are already in um


def _kind_of(cellname: str) -> Optional[str]:
    for prefix, kind in _KIND_PREFIX:
        if cellname.startswith(prefix):
            return kind
    return None


# --------------------------------------------------------------------------- #
# Captured geometry blob.
# --------------------------------------------------------------------------- #
@dataclass
class CellGeometry:
    """The captured external_pdk geometry realizing ONE logical device.

    ``polys_by_layer`` maps ``(gds_layer, datatype) -> list[list[(x_um,y_um)]]``:
    the device's polygons RELATIVE to its captured centroid (so the blob is a
    reusable stamp -- emit translates it to any target centroid). ``ref_centroid``
    is the absolute reference centroid (um); a relaxed placement just supplies a
    NEW centroid. ``n_instances`` is how many finger-cell instances were unioned;
    ``finger_cells`` their cell names; ``clean`` is False with a ``reason`` when the
    capture is degenerate (no instance found in the device region).
    """

    name: str
    kind: str
    ref_centroid: Tuple[float, float]
    polys_by_layer: Dict[Tuple[int, int], List[List[Tuple[float, float]]]] = field(
        default_factory=dict
    )
    n_instances: int = 0
    finger_cells: List[str] = field(default_factory=list)
    clean: bool = True
    reason: str = ""
    # True when the underlying build_device DeviceGeom(s) drew a LOCAL
    # (source != bulk) well tap on the device's OWN body net rather than
    # welding the tap into the source rail (see device_cell.DeviceGeom.
    # well_tap_isolated). Additive-safe default False: every pre-existing
    # constructor call (vendor capture / no-nets synthesis) is unaffected.
    # For a multi-row fold the flag is the OR across the device's rows (the
    # rows share one tap decision).
    well_tap_isolated: bool = False

    def layers(self) -> List[Tuple[int, int]]:
        return sorted(self.polys_by_layer)

    def poly_area_um2(self, layer: Tuple[int, int]) -> float:
        """Total polygon area (um^2) captured on ``layer`` (0 if absent)."""
        total = 0.0
        for ring in self.polys_by_layer.get(layer, []):
            total += _shoelace_area(ring)
        return total


@dataclass
class EmitReport:
    """Result of ``emit_library_gds``: which devices were placed + the out path."""

    out_gds: str
    cell: str
    placed: List[str] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)  # name -> reason
    dbu: float = 0.001


# --------------------------------------------------------------------------- #
# Geometry helpers (pure; no klayout needed).
# --------------------------------------------------------------------------- #
def _shoelace_area(ring: Sequence[Tuple[float, float]]) -> float:
    n = len(ring)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


# --------------------------------------------------------------------------- #
# 1. extract_device_geometry
# --------------------------------------------------------------------------- #
def extract_device_geometry(intent, ref_gds: str) -> Dict[str, CellGeometry]:
    """Capture each device's reference finger-cell geometry as a reusable blob.

    For every name in ``intent.device_bboxes`` we collect the device-finger
    INSTANCES (pmos3v/nmos3v/mimcap/poly_res leaf cells) whose centre falls inside
    that name's region, union their per-layer polygons (transformed into top-level
    coords, then re-expressed RELATIVE to the captured centroid), and store the blob.

    Honest: a device that maps to several finger cells captures their UNION; a
    device with NO finger instance inside its region is returned with ``clean=False``
    and a ``reason`` (its bbox was a band-only inferred position with no dedicated
    bank, so it has no own geometry to capture -- e.g. an interdigitated M=1 leg).

    Returns ``{device_name -> CellGeometry}``.
    """
    import klayout.db as db

    ly = db.Layout()
    ly.read(ref_gds)
    dbu = ly.dbu
    top = ly.top_cell()

    # Enumerate every device-finger instance once, with its top-level bbox (um) and
    # the transform that places its cell into top coords.
    finger_insts: List[dict] = []
    for inst in top.each_inst():
        k = _kind_of(inst.cell.name)
        if not k:
            continue
        bb = inst.bbox()
        cx = (bb.left + bb.right) / 2.0 * dbu
        cy = (bb.bottom + bb.top) / 2.0 * dbu
        finger_insts.append(
            {
                "kind": k,
                "cell": inst.cell,
                "cellname": inst.cell.name,
                "trans": inst.trans,
                "cx": cx,
                "cy": cy,
            }
        )

    dbb = dict(getattr(intent, "device_bboxes", {}) or {})
    out: Dict[str, CellGeometry] = {}
    # Track which finger instances each device claims so a shared bank is not
    # double-counted into two devices: assign each instance to the SINGLE device
    # whose region centre it is closest to (among regions that contain it).
    # First pass: for every finger, find candidate device regions that contain it.
    claims: Dict[int, List[str]] = {}
    region_centre: Dict[str, Tuple[float, float]] = {}
    region_kind: Dict[str, Optional[str]] = {}
    for name, bb in dbb.items():
        if not bb or len(bb) < 4:
            continue
        x0, y0, x1, y1 = bb
        region_centre[name] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        # device kind for the name (MOS vs passive) to avoid cross-kind capture.
        region_kind[name] = _device_kind(intent, name)

    for fi, f in enumerate(finger_insts):
        cands: List[str] = []
        for name, bb in dbb.items():
            if not bb or len(bb) < 4:
                continue
            x0, y0, x1, y1 = bb
            if region_kind.get(name) not in (None, f["kind"]):
                continue
            if x0 <= f["cx"] <= x1 and y0 <= f["cy"] <= y1:
                cands.append(name)
        claims[fi] = cands

    # Assign each finger to its nearest claiming region centre (deterministic).
    assigned: Dict[str, List[int]] = {name: [] for name in region_centre}
    for fi, cands in claims.items():
        if not cands:
            continue
        f = finger_insts[fi]
        best = min(
            cands,
            key=lambda nm: (
                (region_centre[nm][0] - f["cx"]) ** 2
                + (region_centre[nm][1] - f["cy"]) ** 2
            ),
        )
        assigned[best].append(fi)

    # Precompute the layer-index map once.
    layer_idx: Dict[Tuple[int, int], int] = {}
    for li in ly.layer_indexes():
        info = ly.get_info(li)
        layer_idx[(info.layer, info.datatype)] = li

    for name, bb in dbb.items():
        if not bb or len(bb) < 4:
            continue
        kind = region_kind.get(name) or "?"
        fis = assigned.get(name, [])
        if not fis:
            # No dedicated finger bank in this region (band-only inferred position):
            # nothing to capture honestly.
            cx0, cy0 = region_centre[name]
            out[name] = CellGeometry(
                name=name,
                kind=kind,
                ref_centroid=(round(cx0, 4), round(cy0, 4)),
                clean=False,
                reason="no device-finger instance inside region (interdigitated / band-only bbox)",
            )
            continue

        # Capture the UNION of these instances' polygons per layer, in top coords.
        regions: Dict[Tuple[int, int], "db.Region"] = {}
        finger_cells: List[str] = []
        for fi in fis:
            f = finger_insts[fi]
            finger_cells.append(f["cellname"])
            cell = f["cell"]
            tr = f["trans"]
            for ld, li in layer_idx.items():
                shp = cell.shapes(li)
                if not shp.size():
                    continue
                reg = db.Region(shp).transformed(tr)
                if ld in regions:
                    regions[ld] += reg
                else:
                    regions[ld] = reg

        # Captured centroid = centre of the union bbox across all captured layers.
        merged = db.Region()
        for reg in regions.values():
            merged += reg
        merged.merge()
        mbb = merged.bbox()
        ccx = (mbb.left + mbb.right) / 2.0 * dbu
        ccy = (mbb.bottom + mbb.top) / 2.0 * dbu

        polys_by_layer: Dict[Tuple[int, int], List[List[Tuple[float, float]]]] = {}
        for ld, reg in regions.items():
            reg.merge()
            rings: List[List[Tuple[float, float]]] = []
            for poly in reg.each():
                ring = [
                    (round(pt.x * dbu - ccx, 4), round(pt.y * dbu - ccy, 4))
                    for pt in poly.each_point_hull()
                ]
                if len(ring) >= 3:
                    rings.append(ring)
            if rings:
                polys_by_layer[ld] = rings

        out[name] = CellGeometry(
            name=name,
            kind=kind,
            ref_centroid=(round(ccx, 4), round(ccy, 4)),
            polys_by_layer=polys_by_layer,
            n_instances=len(fis),
            finger_cells=finger_cells,
            clean=bool(polys_by_layer),
            reason="" if polys_by_layer else "captured instances carried no polygons",
        )

    return out


def _device_kind(intent, name: str) -> Optional[str]:
    """Logical kind for a tracked name (MOS kind or passive kind), else None."""
    for d in getattr(intent, "devices", []) or []:
        if getattr(d, "name", None) == name:
            return getattr(d, "kind", None)
    for p in getattr(intent, "passives", []) or []:
        if getattr(p, "name", None) == name:
            return getattr(p, "kind", None)
    return None


# --------------------------------------------------------------------------- #
# 2. emit_library_gds
# --------------------------------------------------------------------------- #
def emit_library_gds(
    intent,
    positions: Dict[str, Tuple[float, float]],
    ref_gds: str,
    out_gds: str,
    *,
    cell_name: Optional[str] = None,
    geometry: Optional[Dict[str, CellGeometry]] = None,
) -> EmitReport:
    """Place each device's captured external_pdk geometry at a target centroid into a GDS.

    ``positions`` maps ``device_name -> (cx_um, cy_um)`` (e.g. the anchor centroids
    or a ``relax.place_relaxed`` result). For each device with a CLEAN captured
    :class:`CellGeometry`, its per-layer polygons (relative to the captured centroid)
    are translated to the target centroid and written into a single flat top cell of
    ``out_gds``. Routing / wells are out of scope -- devices only.

    Returns an :class:`EmitReport` listing placed + skipped (with reasons) devices.
    """
    import klayout.db as db

    if geometry is None:
        geometry = extract_device_geometry(intent, ref_gds)

    cell = cell_name or str(getattr(intent, "cell", "LIB_DEVICES"))

    ly = db.Layout()
    ly.dbu = 0.001
    dbu = ly.dbu
    top = ly.create_cell(cell)

    layer_idx: Dict[Tuple[int, int], int] = {}

    def li_for(ld: Tuple[int, int]) -> int:
        if ld not in layer_idx:
            layer_idx[ld] = ly.layer(ld[0], ld[1])
        return layer_idx[ld]

    placed: List[str] = []
    skipped: Dict[str, str] = {}

    for name, geom in geometry.items():
        if not geom.clean or not geom.polys_by_layer:
            skipped[name] = geom.reason or "no clean geometry"
            continue
        tgt = positions.get(name)
        if tgt is None:
            skipped[name] = "no target position supplied"
            continue
        tx, ty = float(tgt[0]), float(tgt[1])
        for ld, rings in geom.polys_by_layer.items():
            li = li_for(ld)
            for ring in rings:
                pts = [
                    db.Point(round((x + tx) / dbu), round((y + ty) / dbu))
                    for (x, y) in ring
                ]
                if len(pts) >= 3:
                    top.shapes(li).insert(db.Polygon(pts))
        placed.append(name)

    ly.write(out_gds)
    return EmitReport(
        out_gds=out_gds, cell=cell, placed=sorted(placed), skipped=skipped, dbu=dbu
    )


# --------------------------------------------------------------------------- #
# 3. device_layer_xor  (the round-trip oracle)
# --------------------------------------------------------------------------- #
def device_layer_xor(
    gds_a: str,
    gds_b: str,
    layers: Optional[Iterable[Tuple[int, int]]] = None,
) -> Dict[Tuple[int, int], float]:
    """XOR area (um^2) per device layer between two GDS files' device geometry.

    Reads both layouts, and for each ``(layer, datatype)`` in ``layers`` (default
    :data:`DEVICE_LAYERS`) computes the symmetric difference (XOR) of the merged
    polygon Region from each file's TOP cell, returning its area in um^2. ~0 on
    every device layer means the two device-layer footprints coincide.

    Only DEVICE layers are compared; top-level routing layers are excluded by
    construction (devices-only scope).
    """
    import klayout.db as db

    layers = tuple(layers) if layers is not None else DEVICE_LAYERS

    def regions_of(path: str) -> Dict[Tuple[int, int], "db.Region"]:
        ly = db.Layout()
        ly.read(path)
        dbu = ly.dbu
        top = ly.top_cell()
        idx: Dict[Tuple[int, int], int] = {}
        for li in ly.layer_indexes():
            info = ly.get_info(li)
            idx[(info.layer, info.datatype)] = li
        out: Dict[Tuple[int, int], Tuple["db.Region", float]] = {}
        for ld in layers:
            li = idx.get(ld)
            if li is None:
                out[ld] = (db.Region(), dbu)
                continue
            reg = db.Region(top.begin_shapes_rec(li))
            reg.merge()
            out[ld] = (reg, dbu)
        return out

    ra = regions_of(gds_a)
    rb = regions_of(gds_b)

    result: Dict[Tuple[int, int], float] = {}
    for ld in layers:
        reg_a, dbu_a = ra[ld]
        reg_b, dbu_b = rb[ld]
        # Both layouts use the same dbu (0.001) here; areas are in dbu^2 -> um^2.
        xor = reg_a ^ reg_b
        xor.merge()
        result[ld] = xor.area() * (dbu_a * dbu_a)
    return result


# --------------------------------------------------------------------------- #
# Convenience: reference device-layer footprint extracted to its own GDS.
# --------------------------------------------------------------------------- #
def write_reference_device_gds(
    intent,
    ref_gds: str,
    out_gds: str,
    *,
    cell_name: Optional[str] = None,
    geometry: Optional[Dict[str, CellGeometry]] = None,
) -> EmitReport:
    """Emit the captured reference device geometry AT ITS OWN reference centroids.

    Equivalent to ``emit_library_gds(intent, anchor_positions, ...)`` where each
    device's target centroid IS its captured ``ref_centroid``. This is the anchor
    emit used by the round-trip proof: its device-layer XOR vs the reference's
    captured device geometry is ~0 by construction.
    """
    if geometry is None:
        geometry = extract_device_geometry(intent, ref_gds)
    anchor_positions = {
        name: geom.ref_centroid for name, geom in geometry.items() if geom.clean
    }
    return emit_library_gds(
        intent,
        anchor_positions,
        ref_gds,
        out_gds,
        cell_name=cell_name,
        geometry=geometry,
    )


def anchor_positions(geometry: Dict[str, CellGeometry]) -> Dict[str, Tuple[float, float]]:
    """The captured reference centroids = the anchor target positions for emit."""
    return {name: geom.ref_centroid for name, geom in geometry.items() if geom.clean}


__all__ = [
    "CellGeometry",
    "EmitReport",
    "DEVICE_LAYERS",
    "extract_device_geometry",
    "emit_library_gds",
    "device_layer_xor",
    "write_reference_device_gds",
    "anchor_positions",
]
