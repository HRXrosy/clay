"""Generic, leakage-free device-terminal routing-ACCESS extractor.

THE ROOT FIX for "where does a router land a wire for each device terminal?". A
device terminal can sit on ANY routing metal (not just M1), and a device (a MIM
capacitor) has SEVERAL DISTINCT electrical plates that overlap in 2D but are two
electrical nodes. This module gives, for ANY device cell (MOS / resistor / MIM
cap) on ANY PDK, a per-schematic-terminal routing access::

    role -> TerminalAccess(net, layer, xy_um, node_id)

where distinct declared terminals are GUARANTEED to land on DISTINCT electrical
nodes. For a 2-plate cap the ``p`` terminal lands on one plate's routing handle
and the ``n`` terminal on the other -- never the same node. If two declared
terminals collapse to one electrical node that is an honest ERROR reported in the
returned :class:`TerminalReport` (never silently pinned to a fabricated point).

Mechanism (root, not per-case)
==============================
The electrical nodes come from the layer-aware connectivity primitive
:mod:`clay.layout.connectivity`, fed ONLY the device cell's OWN geometry + the
PDK connect model (metals + via layer-pairs from ``tech``), plus -- for a MIM cap
-- a derived-layer recipe carried as DATA in :class:`MimConnectSpec`. The MIM
recipe (``mim_virtual`` bottom plate, ``metal2_ncap`` routable Metal2 carved out
of the plate region, ``fuse_cap`` top plate) is the reverse-engineered foundry
LVS device-recognition; it is supplied as DATA keyed by PDK, NOT as a code
branch. This is the same split (a derived "ncap" routable layer distinct from the
plate region) the gf180 / external_pdk / external_pdk decks use to keep the two cap plates on two
nodes.

Leakage discipline
==================
This reads ONLY the device-cell geometry (from the device-cell builders) and the
PDK connect model / MIM spec DATA. It never reads a reference routed layout.

No per-circuit / per-device branching
=====================================
One function for every device + PDK. The only device-specific input is the MIM
``mim_spec`` DATA (when the device is a cap), which the caller passes; the engine
itself contains no ``if kind ==`` / ``if pdk ==`` logic for terminal mapping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import klayout.db as kdb

from .connectivity import ConnectModel, DerivedLayer, extract_nodes
# open_pdk MIM-stack GDS layer-number table: the SINGLE shared source with
# clay.layout.passive_device_cell_tsmc (task B3 -- this module used to carry its
# OWN copy of this exact table because passive_device_cell_tsmc imports
# MimConnectSpec FROM this module, so importing the table back from there would
# have been circular; external_pdk is a dependency-free leaf both sides import
# instead, so the table now has exactly one declaration).

LayerKey = Tuple[int, int]


# --- output contract ---------------------------------------------------------


@dataclass(frozen=True)
class TerminalAccess:
    """The routing access of one device terminal.

    ``net``      -- the cell net the terminal binds to ('' if unbound).
    ``layer``    -- the LOWEST REAL routing-metal ``(layer, datatype)`` of this
                    terminal's electrical node a router can land a wire on.
    ``xy_um``    -- a point (x_um, y_um) ON real metal of that node on ``layer``.
    ``node_id``  -- the electrical-node index (distinct terminals -> distinct
                    node_id by the invariant; same node_id == same electrical net).
    ``role``     -- the device-local pin name ('s'/'d'/'g'/'b'/'p'/'n').
    """

    role: str
    net: str
    layer: LayerKey
    xy_um: Tuple[float, float]
    node_id: int


@dataclass(frozen=True)
class TerminalReport:
    """Diagnostics for an :func:`extract_terminal_access` call.

    ``node_count``  -- number of distinct electrical nodes in the device cell.
    ``collapsed``   -- list of ``(role_a, role_b)`` pairs of DISTINCT declared
                       terminals that resolved to the SAME electrical node -- an
                       honest correctness ERROR (e.g. a cap whose two plates
                       short). Empty when the distinct-node invariant holds.
    ``unmapped``    -- roles whose terminal center fell on NO node (no metal of
                       any node under the terminal point) -- also an honest gap.
    """

    node_count: int
    collapsed: List[Tuple[str, str]] = field(default_factory=list)
    unmapped: List[str] = field(default_factory=list)


# --- MIM connect spec (DATA, keyed by PDK) -----------------------------------


@dataclass(frozen=True)
class MimConnectSpec:
    """The PDK MIM device-recognition recipe, as DATA (no code branch).

    Carries the derived-layer recipe + connect rules that split a MIM cap's two
    plates into two electrical nodes, plus the map from each SYNTHETIC derived
    layer back to the REAL routing-metal GDS layer a router lands on (so the
    extractor can report a real routable landing, not a pseudo-layer).

    ``derived``        -- :class:`DerivedLayer` specs to materialise (mim_virtual
                          bottom plate, metal2_ncap routable Metal2, fuse_cap top).
    ``metals``         -- the node-bearing conductor layers for connectivity
                          (the derived plate/ncap/top + the real cover metal).
    ``vias``           -- via layer-pairs ``(via_key, lo, hi)``.
    ``bridges``        -- abutment ``connect(a, b)`` pairs (the gf180 missing
                          ``connect(metal2_ncap, mim_virtual)`` bridge included).
    ``synthetic_to_real`` -- map a synthetic derived ``(layer, datatype)`` to the
                          REAL routing GDS layer it physically is (mim_virtual ->
                          bottom routing metal, metal2_ncap -> bottom routing
                          metal, fuse_cap -> the FuseTop layer which is NOT a
                          routing metal so it is absent here).
    ``metal_order``    -- real routing metals low->high, so the extractor can pick
                          the LOWEST real metal of a node.
    """

    derived: Tuple[DerivedLayer, ...] = ()
    metals: Tuple[LayerKey, ...] = ()
    vias: Tuple[Tuple[LayerKey, LayerKey, LayerKey], ...] = ()
    bridges: Tuple[Tuple[LayerKey, LayerKey], ...] = ()
    synthetic_to_real: Dict[LayerKey, LayerKey] = field(default_factory=dict)
    metal_order: Tuple[LayerKey, ...] = ()
    # ``top_cover`` -- the TOP plate's cover routing metal (real GDS key). The top
    # plate's node may also touch a low-metal routable fragment that sits physically
    # OVER the bottom plate (gf180: a metal2_ncap sliver near the top via); landing
    # the top terminal there grazes the bottom plate (a real LVS short). When set,
    # the extractor lands the top-plate terminal on this cover metal (spatially above
    # the bottom plate) instead of the lowest fragment. ``None`` -> lowest-metal rule.
    top_cover: Optional[LayerKey] = None


def gf180_mim_spec(
    tech,
    *,
    mim_option: str = "A",
    metal_level: str = "5LM",
) -> MimConnectSpec:
    """Build the gf180mcu MIM :class:`MimConnectSpec` from ``tech`` (DATA, no
    reference layout). Supports Option-A (Metal2/Via2/Metal3) by default and
    Option-B (5LM Metal4/Via4/Metal5 or 4LM Metal3/Via3/Metal4) when selected.

      * ``mim_virtual`` = FuseTop.sized(1.06um) & bot_metal.interacting(that)  -- the
        MIM BOTTOM plate region on the option's bottom metal (gf180mcu.lvs).
      * ``bot_ncap`` = bot_metal - mim_virtual  -- the ROUTABLE bottom metal with the
        plate region carved out, so a router wire is a node DISTINCT from the plate.
      * ``fuse_cap``    = the FuseTop top plate.

    Connectivity: top_via rises bot_ncap<->top_cover and top_cover<->fuse_cap;
    the ``connect(bot_ncap, mim_virtual)`` bridge rejoins the routing bottom metal
    to the bottom plate so both plates remain routable on distinct electrical nodes.
    """
    from clay.pdk.gf180mcu import normalize_gf180_mim_option

    opt = normalize_gf180_mim_option(mim_option)
    if opt == "A":
        bot_idx, top_idx, via_idx = 2, 3, 2
    else:
        lvl = str(metal_level or "5LM").strip().upper()
        if lvl == "5LM":
            bot_idx, top_idx, via_idx = 4, 5, 4
        elif lvl == "4LM":
            bot_idx, top_idx, via_idx = 3, 4, 3
        else:
            raise ValueError(
                f"gf180mcu Option-B MIM supports metal_level '4LM' or '5LM', got {metal_level!r}"
            )
    bot_metal = tuple(tech.metal_key(bot_idx))
    top_metal = tuple(tech.metal_key(top_idx))
    top_via = tuple(tech.via_key(via_idx))
    fusetop = tuple(tech.marker_key("fusetop"))  # (75,0)

    # synthetic derived-layer keys (datatype-shifted so they never collide with a
    # real drawn layer): plate region / routable-metal / top-plate.
    mim_virtual = (fusetop[0], 901)
    bot_ncap = (bot_metal[0], 901)
    fuse_cap = (fusetop[0], 902)

    # 1.06um -> 1060 nm; device coords are in nm and kdb shapes carry nm directly.
    oversize_nm = 1060

    derived = (
        DerivedLayer(
            name=mim_virtual,
            ops=[
                ("sized", fusetop, oversize_nm),          # $0 = FuseTop oversized
                ("interacting", bot_metal, "$0"),         # $1 = bot_metal touching it
                ("and", "$1", "$0"),                      # $2 = that bot_metal & zone
            ],
        ),
        DerivedLayer(name=bot_ncap, ops=[("not", bot_metal, mim_virtual)]),
        # fuse_cap = the FuseTop selected by markers; the cell draws cap_mk +
        # mim_l_mk fully enclosing FuseTop, so fusetop.interacting(itself) is the
        # whole top plate (the marker selection is a no-op narrowing here).
        DerivedLayer(name=fuse_cap, ops=[("interacting", fusetop, fusetop)]),
    )
    return MimConnectSpec(
        derived=derived,
        metals=(bot_ncap, top_metal, mim_virtual, fuse_cap),
        vias=((top_via, bot_ncap, top_metal), (top_via, top_metal, fuse_cap)),
        bridges=((bot_ncap, mim_virtual),),
        synthetic_to_real={
            mim_virtual: bot_metal,    # bottom plate is physically bot_metal
            bot_ncap: bot_metal,       # routable bottom metal IS bot_metal
            # fuse_cap (FuseTop) is NOT a routing metal -> intentionally absent.
        },
        metal_order=(bot_metal, top_metal),
        # top_cover=None: send the top-plate pin to the cell's DEDICATED,
        # ISOLATED ``n`` exit TAB outside the FuseTop+1.06um plate zone.
        top_cover=None,
    )




# open_pdk MIM stack GDS numbers (DATA, cited to pdks/open_pdk/.../external_pdk.layermap
# + external_drc/external_drc.drc + external_lvs/lvs/extract.rul): CTMn = 67:n (top plate);
# METALn (bottom plate); VIA(n)(n+1) (riser); METAL(n+1) (cover). The external_pdk adapter
# exposes only the lower interconnect metals (M1..M3) via ``tech.metal_key``; the
# MIM level (default CTM4 over METAL4, the external_drc-implemented level) lives on
# layers the adapter does not publish, so the table is imported as cited DATA from
# the dependency-free ``clay.layout.external_pdk`` leaf module (module-level
# import above) -- the SAME table ``clay.layout.passive_device_cell_tsmc`` imports,
# not a second copy (task B3; the leaf module exists so neither side needs the
# circular import the historical local copy here was working around).




# --- connect-model assembly --------------------------------------------------


def _base_model(tech, *, max_metal: int = 8, max_via: int = 7) -> ConnectModel:
    """The standard routing-stack connect model from ``tech`` (MOS / resistor):
    metals 1..N + via n connecting metal n<->n+1 + the diff/poly contact bridging
    the device active/poly up to Metal1."""
    model = ConnectModel.from_tech(tech, max_metal=max_metal, max_via=max_via)
    return model


def _merge_mim(base: ConnectModel, spec: MimConnectSpec) -> ConnectModel:
    """Fold a :class:`MimConnectSpec` (DATA) into the base connect model.

    The MIM metals/vias/bridges/derived REPLACE the routing-stack ones for the
    cap (a cap cell carries no MOS contact/M1 routing); the engine then derives
    the plate/ncap/top split + connectivity entirely from the spec DATA."""
    return ConnectModel(
        metals=tuple(spec.metals),
        vias=tuple(spec.vias),
        extra_connect=base.extra_connect,
        derived=tuple(spec.derived),
        bridges=tuple(spec.bridges),
    )


# --- geometry -> kdb shapes --------------------------------------------------


def _shapes_by_layer(device_geom) -> Dict[LayerKey, List[kdb.Polygon]]:
    """Convert a :class:`~clay.layout.device_cell.DeviceGeom` to a
    ``{(layer,datatype): [kdb.Polygon]}`` map (nm coordinates carried through)."""
    out: Dict[LayerKey, List[kdb.Polygon]] = {}
    for lp in device_geom.polygons:
        pts = [kdb.Point(int(v.x), int(v.y)) for v in lp.polygon.vertices]
        out.setdefault((lp.layer, lp.datatype), []).append(kdb.Polygon(pts))
    return out


# nm database grid the CellGeometry um rings snap to (= emit_library_gds dbu).
_CELLGEOM_DBU = 0.001


def _shapes_by_layer_cellgeom(cell_geom) -> Dict[LayerKey, List[kdb.Polygon]]:
    """Convert a :class:`~clay.layout.extract.library_device.CellGeometry` STAMP
    (``polys_by_layer`` = ``{(layer,dt): [ [(x_um,y_um), ...], ... ]}``) to a
    ``{(layer,datatype): [kdb.Polygon]}`` map in nm coordinates.

    A CellGeometry carries um-ring polygons and NO terminal map (it is the geometry
    currency the passive synthesiser emits); the rings are snapped to the 1 nm grid
    so the layer-aware connectivity engine sees the same shapes the emitter stamps."""
    out: Dict[LayerKey, List[kdb.Polygon]] = {}
    for key, rings in (cell_geom.polys_by_layer or {}).items():
        for ring in rings:
            pts = [
                kdb.Point(round(x / _CELLGEOM_DBU), round(y / _CELLGEOM_DBU))
                for (x, y) in ring
            ]
            if len(pts) >= 3:
                out.setdefault(tuple(key), []).append(kdb.Polygon(pts))
    return out


def _real_layer(key: LayerKey, spec: Optional[MimConnectSpec]) -> Optional[LayerKey]:
    """Map a (possibly SYNTHETIC) node layer to its REAL routing GDS layer.

    A synthetic derived layer (mim_virtual / metal2_ncap) maps through the spec's
    ``synthetic_to_real``; a layer NOT in the map and NOT a real routing metal
    (e.g. fuse_cap/FuseTop) returns ``None`` (not routable). A plain real layer
    returns itself."""
    if spec is not None and key in spec.synthetic_to_real:
        return spec.synthetic_to_real[key]
    return key


def _metal_rank(key: LayerKey, tech, spec: Optional[MimConnectSpec]) -> Optional[int]:
    """Rank of a REAL routing metal (lower == lower metal). Uses the spec's
    ``metal_order`` when present (cap), else the tech metal stack (MOS). Returns
    ``None`` when the layer is not a known routing metal."""
    order: Sequence[LayerKey]
    if spec is not None and spec.metal_order:
        order = spec.metal_order
    else:
        order = []
        for n in range(1, 9):
            try:
                order.append(tuple(tech.metal_key(n)))
            except Exception:
                break
    for i, m in enumerate(order):
        if tuple(m) == tuple(key):
            return i
    return None


def _lowest_real_landing(
    node: Dict[LayerKey, kdb.Region],
    tech,
    spec: Optional[MimConnectSpec],
) -> Optional[Tuple[LayerKey, Tuple[int, int]]]:
    """The LOWEST real routing-metal landing of an electrical ``node``.

    For each layer the node occupies, map it to its real routing metal; among the
    real routing metals present, pick the LOWEST and return that real
    ``(layer, point_nm)`` where ``point_nm`` is the centroid of one shape of the
    node ON the layer that physically carries that real metal. Returns ``None``
    when the node has no routable real metal (an honest non-routable node)."""
    best: Optional[Tuple[int, LayerKey, Tuple[int, int]]] = None
    cover: Optional[Tuple[LayerKey, Tuple[int, int]]] = None  # a landing on the top-plate cover
    top_cover = tuple(spec.top_cover) if (spec is not None and spec.top_cover) else None
    for key, reg in node.items():
        if reg is None or reg.is_empty():
            continue
        real = _real_layer(key, spec)
        if real is None:
            continue
        rank = _metal_rank(real, tech, spec)
        if rank is None:
            continue
        # a point ON real metal of this node: centroid of the first polygon.
        first = None
        for poly in reg.each():
            first = poly
            break
        if first is None:
            continue
        bb = first.bbox()
        pt = ((bb.left + bb.right) // 2, (bb.bottom + bb.top) // 2)
        if top_cover is not None and real == top_cover:
            cover = (real, pt)
        if best is None or rank < best[0]:
            best = (rank, real, pt)
    # TOP-PLATE rule: if this node carries the cap's top-plate COVER metal, land on
    # the cover (spatially above the bottom plate) -- NOT a lower routable fragment of
    # the same node that would sit over the bottom plate and graze it (a real LVS short).
    if cover is not None:
        return cover
    if best is None:
        return None
    return (best[1], best[2])


def _node_index_for_box(
    nodes: List[Dict[LayerKey, kdb.Region]],
    box: Tuple[int, int, int, int],
) -> Optional[int]:
    """Index of the node whose ANY-layer region intersects the terminal ``box``
    (nm). Tries the bbox so a terminal pad that sits on a synthetic/real layer is
    located regardless of which derived layer carries it. Returns the node with
    the LARGEST overlap area (so a pad straddling two layers picks its own node)."""
    x0, y0, x1, y1 = (int(v) for v in box)
    tbox = kdb.Region(kdb.Box(x0, y0, x1, y1))
    best: Optional[Tuple[float, int]] = None
    for idx, node in enumerate(nodes):
        area = 0.0
        for reg in node.values():
            if reg is None or reg.is_empty():
                continue
            inter = reg & tbox
            if not inter.is_empty():
                area += inter.area()
        if area > 0 and (best is None or area > best[0]):
            best = (area, idx)
    return best[1] if best is not None else None


# --- public extractor --------------------------------------------------------

# device-local terminal roles that are SIGNAL terminals (need a routing access).
# 'b' (bulk) is included when present; a MOS bulk pin is a real tie. The engine
# never branches on the device KIND -- it just maps whatever terminals exist.
def extract_device_nodes(
    device_geom,
    *,
    tech,
    mim_spec: Optional[MimConnectSpec] = None,
) -> List[Dict[LayerKey, kdb.Region]]:
    """Electrical nodes of a terminal-bearing :class:`DeviceGeom`.

    This is the node truth consumed by :func:`extract_terminal_access`, exposed
    so routing-pin materialization can prove that a lower-layer terminal tab is
    part of the same extracted plate node before treating it as an internal riser.
    """
    if mim_spec is not None:
        model = _merge_mim(_base_model(tech), mim_spec)
    else:
        model = _base_model(tech)
    return extract_nodes(_shapes_by_layer(device_geom), model)


def extract_terminal_access(
    device_geom,
    *,
    tech,
    mim_spec: Optional[MimConnectSpec] = None,
) -> Tuple[Dict[str, TerminalAccess], TerminalReport]:
    """Map each declared terminal of ``device_geom`` to its routing access.

    Computes the device cell's electrical nodes from its OWN geometry + the PDK
    connect model (``tech``), plus -- when ``mim_spec`` is given -- the MIM
    derived-layer recipe DATA that splits a cap's two plates into two nodes. Each
    terminal's center/bbox is mapped to its node; the LOWEST real routing-metal
    landing of that node is exposed as a :class:`TerminalAccess`.

    Returns ``(access, report)``. ``access`` maps each terminal role to its
    :class:`TerminalAccess`. ``report`` is a :class:`TerminalReport`; if two
    DISTINCT declared terminals resolve to the SAME electrical node, the pair is
    listed in ``report.collapsed`` (an honest correctness error -- NOT silently
    pinned). A terminal whose center maps to no node is listed in
    ``report.unmapped``.

    Parameters
    ----------
    device_geom : DeviceGeom
        Has ``.polygons`` (LayerPolygon) and ``.terminals`` (role -> Terminal
        with ``.center`` in nm, ``.bbox``, ``.net``, ``.role``).
    tech : LayoutTech
        The PDK tech surface (connect model source).
    mim_spec : MimConnectSpec, optional
        The MIM derived-layer recipe DATA (pass for a MIM cap; see
        :func:`gf180_mim_spec`). ``None`` for a MOS / resistor.
    """
    nodes = extract_device_nodes(device_geom, tech=tech, mim_spec=mim_spec)

    access: Dict[str, TerminalAccess] = {}
    role_to_node: Dict[str, int] = {}
    unmapped: List[str] = []

    # deterministic role order for stable collapsed-pair reporting.
    terms = device_geom.terminals or {}
    for role in sorted(terms.keys()):
        term = terms[role]
        bbox = getattr(term, "bbox", None)
        if bbox is None:
            cx, cy = term.center
            bbox = (cx - 1, cy - 1, cx + 1, cy + 1)
        idx = _node_index_for_box(nodes, bbox)
        if idx is None:
            unmapped.append(role)
            continue
        role_to_node[role] = idx
        landing = _lowest_real_landing(nodes[idx], tech, mim_spec)
        if landing is None:
            # node has no routable real metal: honest non-routable terminal.
            unmapped.append(role)
            continue
        layer, pt_nm = landing
        access[role] = TerminalAccess(
            role=role,
            net=getattr(term, "net", "") or "",
            layer=tuple(layer),
            xy_um=(pt_nm[0] / 1000.0, pt_nm[1] / 1000.0),
            node_id=idx,
        )

    # distinct-node invariant: distinct declared terminals -> distinct nodes.
    collapsed: List[Tuple[str, str]] = []
    seen: Dict[int, str] = {}
    for role in sorted(role_to_node.keys()):
        nid = role_to_node[role]
        if nid in seen:
            collapsed.append((seen[nid], role))
        else:
            seen[nid] = role

    report = TerminalReport(
        node_count=len(nodes),
        collapsed=collapsed,
        unmapped=unmapped,
    )
    return access, report


def extract_cellgeom_nodes(
    cell_geom,
    *,
    tech,
    mim_spec: Optional[MimConnectSpec] = None,
) -> List[Dict[LayerKey, kdb.Region]]:
    """Electrical NODES of a :class:`CellGeometry` STAMP (no terminal map).

    The passive synthesiser (``clay.layout.passive_cell.build_mim_cap`` /
    ``build_poly_res``) emits a :class:`~clay.layout.extract.library_device.CellGeometry`
    -- centroid-relative um rings with NO ``terminals`` map -- rather than a
    :class:`~clay.layout.device_cell.DeviceGeom`. :func:`extract_terminal_access`
    needs a terminal map; for a bare stamp this returns the device's electrical
    NODES directly (each ``{layer_key: kdb.Region}``) so a caller can assert the
    distinct-plate invariant (a 2-plate MIM cap -> exactly 2 nodes) without a
    terminal map. Same root mechanism as :func:`extract_terminal_access`: the nodes
    come from the cell's OWN geometry + the PDK connect model (+ the MIM spec DATA
    when given), never a reference routed layout.

    Parameters
    ----------
    cell_geom : CellGeometry
        Has ``.polys_by_layer`` (``{(layer,dt): [um-ring, ...]}``); no terminals.
    tech : LayoutTech
        The PDK tech surface (connect-model source).
    mim_spec : MimConnectSpec, optional
        The MIM derived-layer / connect recipe DATA (e.g. :func:`external_pdk`).
    """
    if mim_spec is not None:
        model = _merge_mim(_base_model(tech), mim_spec)
    else:
        model = _base_model(tech)
    shapes = _shapes_by_layer_cellgeom(cell_geom)
    return extract_nodes(shapes, model)


__all__ = [
    "TerminalAccess",
    "TerminalReport",
    "MimConnectSpec",
    "gf180_mim_spec",
    "external_pdk",
    "external_pdk",
    "extract_device_nodes",
    "extract_terminal_access",
    "extract_cellgeom_nodes",
]
