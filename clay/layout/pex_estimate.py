"""R164 (BL-164C) -- APPROXIMATE geometric post-layout parasitic extractor (PEX).

What this is (and explicitly is NOT)
====================================
TED ships StarRC: a commercial field-solver parasitic extractor that solves the
3-D Laplace/Poisson problem on the real conductor stack to produce a calibrated
RC netlist. **Clay has no such thing.** The gf180mcu KLayout sign-off deck this
repo runs (``gf180mcu.lvs`` / ``gf180mcu.drc``) carries **no** ``extract_rc`` /
sheet-resistance / unit-capacitance section at all (grep the deck: zero ``rsh``,
``sheet`` or ``resistance`` tokens), so even the real toolchain here produces
connectivity + a device netlist but ZERO parasitics.

This module closes a *small, clearly-bounded* slice of that gap with an
**APPROXIMATE GEOMETRIC** estimator -- NOT a field solver:

  * per-net capacitance C ~= sum over metal shapes of
        area(um^2) * area_cap(aF/um^2)  +  perimeter(um) * fringe_cap(aF/um)
    using documented gf180mcu BEOL unit constants (cited below). This is a
    parallel-plate + edge-fringe approximation to substrate; it ignores
    layer-to-layer coupling, fringe shielding, conductor thickness fringe, and
    every 3-D effect a real field solver models.
  * per-net dominant wire resistance R ~= sheet_R(ohm/sq) * L_eff / W_eff, where
    L_eff is the net's metal half-perimeter wirelength (HPWL, from the Rust
    ``net-wirelength`` kernel) and W_eff is a representative wire width derived
    from the net's metal area / its bbox span. This is a single-pi-segment
    lumped estimate, NOT a distributed RC tree and NOT a multi-corner extraction.

Honesty contract (CRITICAL)
---------------------------
* Every C/R value carries ``approximation="geometric-area-fringe"`` /
  ``"sheet-R-L-over-W"`` and ``field_solver=False``. NO code path ever claims
  StarRC-equivalent accuracy or a calibrated/silicon number.
* The connectivity (which metal shapes form one net) is read from the REAL GDS
  via KLayout ``LayoutToNetlist`` -- not fabricated. When KLayout is absent the
  net partition falls back to a pure-geometry metal-touch union-find on the SAME
  emitted polygons, and the result is flagged ``connectivity_source="geom-fallback"``.
* The unit constants are documented gf180mcu / GF 0.18um Al-BEOL process-typical
  values (sourced + cited in ``GF180_METAL_PEX``), explicitly marked
  ``calibrated=False``. They are order-of-magnitude design values, not a foundry
  ITF/TLU+ extraction file.
* Pre-layout vs post-layout: "pre-layout" here is the ideal schematic view where
  every net has zero wire parasitics (R=0, C=0); "post-layout" is this geometric
  estimate. The reported delta IS the (approximate) parasitic the layout adds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .backend import LayerPolygon, NamedCell, ParameterizedLibrary

# ---------------------------------------------------------------------------
# Documented gf180mcu / GF 0.18um Al-BEOL process-typical unit constants.
#
# SOURCE / ASSUMPTION (cited, NOT a foundry extraction file):
#   gf180mcu is GlobalFoundries' open 180 nm process (the GF 0.18um node, 1P5M
#   Al metallization, ~0.53 um metal1 thickness, ~SiO2 ~k=3.9-4.2 ILD). The
#   public process-design values for an Al 0.18 um BEOL are, order of magnitude:
#     * metal1->substrate plate cap   ~ 30-40 aF/um^2   (we take 35)
#     * metal1 edge/fringe cap        ~ 30-40 aF/um     (we take 35)
#     * metal1 sheet resistance       ~ 70-95 mOhm/sq   (Al ~0.5 um) (we take 0.090 Ohm/sq)
#     * contact (CO) resistance       ~ 10-20 Ohm each  (we take 12 Ohm)
#   These match the order-of-magnitude constants the legacy ``parasitic.py``
#   already uses for metal1 area/fringe (35 / 30 aF), which were themselves
#   tagged TENTATIVE. We re-state them here with an explicit ``calibrated=False``
#   flag and a per-field provenance so no caller mistakes them for StarRC output.
#
# WHY NOT pulled from the PDK: the committed gf180mcu KLayout LVS deck has NO
# rsh/unit-cap section (verified by grep), and there is no ITF/TLU+/ICT file in
# the repo. So these constants are *documented-process-typical*, not extracted.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetalPexConstants:
    """Documented (NOT field-solver) per-metal-layer PEX unit constants.

    Attributes:
        area_cap_aF_per_um2: plate cap to substrate (aF/um^2).
        fringe_cap_aF_per_um: edge/fringe cap (aF/um).
        sheet_r_ohm_per_sq: metal sheet resistance (Ohm/square).
        contact_r_ohm: single contact/via series resistance (Ohm).
        source: one-line provenance string.
        calibrated: ALWAYS False -- these are documented design values.
    """

    area_cap_aF_per_um2: float
    fringe_cap_aF_per_um: float
    sheet_r_ohm_per_sq: float
    contact_r_ohm: float
    source: str
    calibrated: bool = False


# gf180mcu metal1 (34/0). Documented GF 0.18um Al BEOL process-typical values.
GF180_METAL_PEX: Dict[Tuple[int, int], MetalPexConstants] = {
    (34, 0): MetalPexConstants(
        area_cap_aF_per_um2=35.0,
        fringe_cap_aF_per_um=35.0,
        sheet_r_ohm_per_sq=0.090,  # ~90 mOhm/sq, Al ~0.5 um
        contact_r_ohm=12.0,
        source=(
            "gf180mcu / GF 0.18um Al 1P5M BEOL, documented process-typical "
            "(metal1 plate ~35 aF/um^2, fringe ~35 aF/um, Rsheet ~90 mOhm/sq, "
            "CO ~12 Ohm); NOT a foundry ITF/TLU+ extraction"
        ),
    ),
}

# gf180mcu GDS layer keys reused for net partition / contact counting.
GF180_METAL1_KEY: Tuple[int, int] = (34, 0)
GF180_METAL1_LABEL_KEY: Tuple[int, int] = (34, 10)
GF180_CONTACT_KEY: Tuple[int, int] = (33, 0)

DEFAULT_DBU_TO_UM = 0.001  # gf180mcu: 1 dbu = 1 nm


# ===========================================================================
# Pure geometry helpers (dbu integers in, um floats out)
# ===========================================================================


def _bbox(lp: LayerPolygon) -> Tuple[int, int, int, int]:
    xs = [v.x for v in lp.polygon.vertices]
    ys = [v.y for v in lp.polygon.vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def _rect_area_dbu2(box: Tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _rect_perim_dbu(box: Tuple[int, int, int, int]) -> int:
    return 2 * (max(0, box[2] - box[0]) + max(0, box[3] - box[1]))


def _rects_touch_or_overlap(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> bool:
    """True iff two axis-aligned rects touch OR overlap (closed intervals).

    Touching edges count as connected (a contact / metal join is a touch), which
    is the correct union-find adjacency for a metal-net partition fallback.
    """
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


# ===========================================================================
# Per-net result records
# ===========================================================================


@dataclass(frozen=True)
class NetParasitic:
    """APPROXIMATE geometric parasitic estimate for ONE net.

    Honest-labelled: ``approximation`` names the model; ``field_solver`` is
    ALWAYS False. ``cap_aF`` / ``res_ohm`` are geometric estimates, never a
    calibrated / StarRC value.
    """

    net_name: str
    metal_area_um2: float
    metal_perimeter_um: float
    wirelength_um: float
    eff_width_um: float
    contact_count: int
    area_cap_aF: float
    fringe_cap_aF: float
    cap_aF: float
    wire_res_ohm: float
    contact_res_ohm: float
    res_ohm: float
    approximation: str = "geometric-area-fringe + sheet-R-L-over-W"
    field_solver: bool = False

    @property
    def cap_fF(self) -> float:
        return self.cap_aF / 1000.0


@dataclass
class CellPexReport:
    """APPROXIMATE post-layout PEX report for one co-clean cell.

    ``connectivity_source`` is ``"klayout-l2n"`` (real GDS connectivity) or
    ``"geom-fallback"`` (metal-touch union-find on the SAME polygons when
    KLayout is unavailable). ``field_solver`` is ALWAYS False.
    """

    cell_name: str
    nets: List[NetParasitic] = field(default_factory=list)
    connectivity_source: str = "geom-fallback"
    constants_source: str = ""
    calibrated: bool = False
    field_solver: bool = False
    note: str = ""

    @property
    def total_cap_aF(self) -> float:
        return sum(n.cap_aF for n in self.nets)

    @property
    def total_cap_fF(self) -> float:
        return self.total_cap_aF / 1000.0

    @property
    def max_res_ohm(self) -> float:
        return max((n.res_ohm for n in self.nets), default=0.0)

    def net_by_name(self, name: str) -> Optional[NetParasitic]:
        for n in self.nets:
            if n.net_name == name:
                return n
        return None


@dataclass
class PreVsPostDelta:
    """Pre-layout (ideal: zero parasitics) vs post-layout (this estimate).

    ``pre`` is the schematic ideal -- every net has C=0, R=0 (no layout exists).
    ``post`` is the approximate geometric estimate. The delta IS the parasitic
    the layout adds; per the model, ``added_*`` == ``post_*`` exactly.
    """

    cell_name: str
    pre_total_cap_aF: float
    post_total_cap_aF: float
    pre_max_res_ohm: float
    post_max_res_ohm: float
    per_net_added_cap_aF: Dict[str, float] = field(default_factory=dict)
    per_net_added_res_ohm: Dict[str, float] = field(default_factory=dict)
    field_solver: bool = False
    approximation: str = "geometric (pre=ideal-zero, post=area-fringe+sheet-R)"

    @property
    def added_total_cap_aF(self) -> float:
        return self.post_total_cap_aF - self.pre_total_cap_aF

    @property
    def added_max_res_ohm(self) -> float:
        return self.post_max_res_ohm - self.pre_max_res_ohm


# ===========================================================================
# Net partition: KLayout L2N (real GDS) with pure-geometry fallback
# ===========================================================================

# A net is represented as a list of metal-shape bboxes (dbu) plus a name and a
# contact count; the same shape of data is produced by both partition paths so
# the downstream parasitic math is path-independent.
_NetShapes = Dict[str, Dict[str, object]]


def klayout_available() -> bool:
    try:
        import klayout.db as _db  # noqa: F401

        return hasattr(_db, "LayoutToNetlist")
    except Exception:  # noqa: BLE001
        return False


def partition_nets_from_gds(
    gds_path,
    *,
    topcell: Optional[str] = None,
    metal_key: Tuple[int, int] = GF180_METAL1_KEY,
    contact_key: Tuple[int, int] = GF180_CONTACT_KEY,
    label_key: Tuple[int, int] = GF180_METAL1_LABEL_KEY,
) -> Tuple[_NetShapes, str, float]:
    """Partition metal shapes into nets using REAL GDS connectivity (KLayout L2N).

    Returns ``(nets, connectivity_source, dbu_to_um)`` where ``nets`` maps a net
    name to ``{"bboxes": [..], "contacts": int}`` (bboxes in dbu). Connectivity
    is metal<->metal + metal<->contact (a contact is a metal join), which keeps
    the gate / source / drain nets DISTINCT (a contact bridges metal to ONE
    device layer, never gate-to-source). Net names are taken from metal text
    labels when present, else KLayout's anonymous ``$N``.

    This reads the SAME emitted GDS the sign-off decks run on, so the net split
    is the real layout connectivity -- not fabricated.
    """
    import klayout.db as db

    ly = db.Layout()
    ly.read(str(gds_path))
    top = ly.top_cell() if topcell is None else ly.cell(topcell)
    if top is None:
        top = ly.top_cell()
    dbu = ly.dbu

    l2n = db.LayoutToNetlist(db.RecursiveShapeIterator(ly, top, []))
    m_li = ly.layer(metal_key[0], metal_key[1])
    c_li = ly.layer(contact_key[0], contact_key[1])
    metal = l2n.make_polygon_layer(m_li, "metal")
    contact = l2n.make_polygon_layer(c_li, "contact")
    l2n.connect(metal)
    l2n.connect(contact)
    l2n.connect(metal, contact)  # contact = metal join -> nets stay distinct
    # Attach net names from metal TEXT labels (34/10) when present.
    t_li = ly.layer(label_key[0], label_key[1])
    if t_li >= 0:
        try:
            text = l2n.make_text_layer(t_li, "lbl")
            l2n.connect(metal, text)
        except Exception:  # noqa: BLE001
            pass
    l2n.extract_netlist()

    nl = l2n.netlist()
    circuits = list(nl.each_circuit())
    if not circuits:
        return {}, "klayout-l2n", dbu
    ckt = circuits[0]

    nets: _NetShapes = {}
    for n in ckt.each_net():
        name = n.name or n.expanded_name()
        reg_m = l2n.shapes_of_net(n, metal, True)
        reg_c = l2n.shapes_of_net(n, contact, True)
        bboxes: List[Tuple[int, int, int, int]] = []
        for poly in reg_m.each():
            b = poly.bbox()
            bboxes.append((b.left, b.bottom, b.right, b.top))
        contacts = reg_c.count()
        if not bboxes:
            continue
        nets[name] = {"bboxes": bboxes, "contacts": int(contacts)}
    return nets, "klayout-l2n", dbu


def partition_nets_from_polygons(
    polygons: Sequence[LayerPolygon],
    *,
    metal_key: Tuple[int, int] = GF180_METAL1_KEY,
    contact_key: Tuple[int, int] = GF180_CONTACT_KEY,
    dbu_to_um: float = DEFAULT_DBU_TO_UM,
) -> Tuple[_NetShapes, str, float]:
    """Pure-geometry net partition (KLayout-free fallback).

    Union-finds metal shapes that touch/overlap (transitive metal connectivity)
    on the SAME polygons KLayout would read. A contact is counted into whichever
    metal net its bbox touches. Net names are synthesized ``net_0, net_1, ...``
    ordered by total metal area (largest first) -- deterministic, never the real
    schematic names, so this path is honestly labelled ``geom-fallback``.
    """
    metal_boxes = [
        _bbox(lp)
        for lp in polygons
        if (lp.layer, lp.datatype) == metal_key
    ]
    contact_boxes = [
        _bbox(lp)
        for lp in polygons
        if (lp.layer, lp.datatype) == contact_key
    ]
    n = len(metal_boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if _rects_touch_or_overlap(metal_boxes[i], metal_boxes[j]):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    # Order nets by total metal area, largest first (deterministic naming).
    ordered = sorted(
        groups.values(),
        key=lambda idxs: -sum(_rect_area_dbu2(metal_boxes[i]) for i in idxs),
    )

    nets: _NetShapes = {}
    for net_idx, idxs in enumerate(ordered):
        bboxes = [metal_boxes[i] for i in idxs]
        # count contacts whose bbox touches any metal shape in this net
        contacts = 0
        for cb in contact_boxes:
            if any(_rects_touch_or_overlap(cb, metal_boxes[i]) for i in idxs):
                contacts += 1
        nets[f"net_{net_idx}"] = {"bboxes": bboxes, "contacts": contacts}
    return nets, "geom-fallback", dbu_to_um


# ===========================================================================
# Per-net parasitic math (the approximate geometric model)
# ===========================================================================


def _net_wirelength_um(
    bboxes: Sequence[Tuple[int, int, int, int]], dbu_to_um: float
) -> float:
    """Net metal HPWL (um) via the Rust ``net-wirelength`` kernel.

    Uses every metal shape's bbox-corner pins; HPWL = (span_x + span_y) of the
    net's pin bounding box. Falls back to the pure-Python mirror automatically.
    """
    from ._rs_bridge_net_wirelength import compute_net_wirelength

    pins: List[Tuple[int, int]] = []
    for b in bboxes:
        pins.append((b[0], b[1]))
        pins.append((b[2], b[3]))
    res = compute_net_wirelength([pins])
    span_dbu = res["net_total_span"][0] if res["net_total_span"] else 0
    return float(span_dbu) * dbu_to_um


def _net_parasitic(
    net_name: str,
    bboxes: Sequence[Tuple[int, int, int, int]],
    contacts: int,
    consts: MetalPexConstants,
    dbu_to_um: float,
) -> NetParasitic:
    """Compute the approximate geometric C and R for ONE net's metal shapes."""
    area_dbu2 = sum(_rect_area_dbu2(b) for b in bboxes)
    perim_dbu = sum(_rect_perim_dbu(b) for b in bboxes)
    area_um2 = float(area_dbu2) * (dbu_to_um ** 2)
    perim_um = float(perim_dbu) * dbu_to_um

    area_cap = area_um2 * consts.area_cap_aF_per_um2
    fringe_cap = perim_um * consts.fringe_cap_aF_per_um
    cap = area_cap + fringe_cap

    wl_um = _net_wirelength_um(bboxes, dbu_to_um)
    # Representative wire width: net metal area / its HPWL length (a flat-wire
    # equivalent). Guard tiny/degenerate nets with the net bbox short span.
    if wl_um > 1e-9:
        eff_w = area_um2 / wl_um
    else:
        # single-shape net: width = shorter bbox side (um)
        b = bboxes[0]
        eff_w = min(
            max(b[2] - b[0], 1) * dbu_to_um, max(b[3] - b[1], 1) * dbu_to_um
        )
    eff_w = max(eff_w, 1e-6)  # never divide by zero
    squares = wl_um / eff_w if eff_w > 0 else 0.0
    wire_r = squares * consts.sheet_r_ohm_per_sq
    # Contacts in parallel reduce series contact R (one contact = full Rc).
    contact_r = consts.contact_r_ohm / contacts if contacts > 0 else 0.0
    res = wire_r + contact_r

    return NetParasitic(
        net_name=net_name,
        metal_area_um2=area_um2,
        metal_perimeter_um=perim_um,
        wirelength_um=wl_um,
        eff_width_um=eff_w,
        contact_count=contacts,
        area_cap_aF=area_cap,
        fringe_cap_aF=fringe_cap,
        cap_aF=cap,
        wire_res_ohm=wire_r,
        contact_res_ohm=contact_r,
        res_ohm=res,
    )


def estimate_pex_from_nets(
    cell_name: str,
    nets: _NetShapes,
    connectivity_source: str,
    dbu_to_um: float,
    *,
    metal_consts: Optional[MetalPexConstants] = None,
) -> CellPexReport:
    """Build a :class:`CellPexReport` from a net partition.

    Pure: takes the partition + constants and applies the approximate model.
    """
    if metal_consts is None:
        metal_consts = GF180_METAL_PEX[GF180_METAL1_KEY]
    report = CellPexReport(
        cell_name=cell_name,
        connectivity_source=connectivity_source,
        constants_source=metal_consts.source,
        calibrated=metal_consts.calibrated,
        field_solver=False,
    )
    for name, data in nets.items():
        bboxes = data["bboxes"]  # type: ignore[assignment]
        contacts = int(data["contacts"])  # type: ignore[arg-type]
        report.nets.append(
            _net_parasitic(name, bboxes, contacts, metal_consts, dbu_to_um)
        )
    # largest-cap net first for readability
    report.nets.sort(key=lambda n: -n.cap_aF)
    report.note = (
        f"APPROXIMATE geometric PEX ({connectivity_source}); "
        f"{len(report.nets)} nets; NOT field-solver / NOT calibrated"
    )
    return report


# ===========================================================================
# Top-level: estimate PEX for a co-clean cell (real GDS or polygons)
# ===========================================================================


def estimate_pex_from_gds(
    gds_path,
    cell_name: str,
    *,
    topcell: Optional[str] = None,
    metal_consts: Optional[MetalPexConstants] = None,
) -> CellPexReport:
    """APPROXIMATE post-layout PEX for a co-clean cell from its REAL GDS.

    Reads net connectivity from the emitted GDS via KLayout L2N, then applies
    the geometric area+fringe cap + sheet-R wire model. Requires KLayout.
    """
    nets, src, dbu = partition_nets_from_gds(gds_path, topcell=topcell)
    return estimate_pex_from_nets(
        cell_name, nets, src, dbu, metal_consts=metal_consts
    )


def estimate_pex_from_polygons(
    cell_name: str,
    polygons: Sequence[LayerPolygon],
    *,
    metal_consts: Optional[MetalPexConstants] = None,
    dbu_to_um: float = DEFAULT_DBU_TO_UM,
) -> CellPexReport:
    """APPROXIMATE post-layout PEX from a polygon list (KLayout-free fallback).

    Uses the pure-geometry metal-touch net partition on the SAME polygons. The
    report is honestly flagged ``connectivity_source="geom-fallback"`` (net
    names are synthesized, not the real schematic names).
    """
    nets, src, dbu = partition_nets_from_polygons(
        polygons, dbu_to_um=dbu_to_um
    )
    return estimate_pex_from_nets(
        cell_name, nets, src, dbu, metal_consts=metal_consts
    )


def estimate_pex_for_library(
    lib: ParameterizedLibrary,
    *,
    metal_consts: Optional[MetalPexConstants] = None,
) -> CellPexReport:
    """APPROXIMATE PEX for the TOP cell of a freshly-built co-clean library.

    Convenience wrapper: no GDS round-trip needed (uses the in-memory polygons
    via the pure-geometry partition). For real-GDS connectivity use
    :func:`estimate_pex_from_gds` after emitting the GDS.
    """
    top: NamedCell = lib.cells[-1]
    dbu = lib.unit_user or DEFAULT_DBU_TO_UM
    return estimate_pex_from_polygons(
        top.name, top.polygons, metal_consts=metal_consts, dbu_to_um=dbu
    )


def compare_pre_vs_post(report: CellPexReport) -> PreVsPostDelta:
    """Pre-layout (ideal zero parasitics) vs post-layout (this estimate).

    Pre-layout is the schematic ideal: every net has C=0, R=0 (no layout). The
    delta IS the approximate parasitic the layout adds. By the model the added
    values equal the post values exactly -- this is intentional and honest, not
    a comparison against a second extraction.
    """
    delta = PreVsPostDelta(
        cell_name=report.cell_name,
        pre_total_cap_aF=0.0,
        post_total_cap_aF=report.total_cap_aF,
        pre_max_res_ohm=0.0,
        post_max_res_ohm=report.max_res_ohm,
        field_solver=False,
    )
    for n in report.nets:
        delta.per_net_added_cap_aF[n.net_name] = n.cap_aF
        delta.per_net_added_res_ohm[n.net_name] = n.res_ohm
    return delta


def honest_gap_statement() -> str:
    """One-paragraph honest statement of what this is vs TED StarRC."""
    return (
        "APPROXIMATE GEOMETRIC PEX: per-net C from area*area_cap + "
        "perimeter*fringe_cap, per-net R from sheet_R*L/W + contact R, using "
        "documented gf180mcu Al-BEOL process-typical constants (calibrated="
        "False, field_solver=False). This is an order-of-magnitude design "
        "estimate. It is NOT StarRC: no 3-D field solve, no layer-to-layer "
        "coupling, no distributed RC tree, no foundry ITF/TLU+ corner data. "
        "TED's StarRC remains the calibrated field-solver reference Clay does "
        "not yet match."
    )


__all__ = [
    "MetalPexConstants",
    "GF180_METAL_PEX",
    "GF180_METAL1_KEY",
    "GF180_METAL1_LABEL_KEY",
    "GF180_CONTACT_KEY",
    "NetParasitic",
    "CellPexReport",
    "PreVsPostDelta",
    "klayout_available",
    "partition_nets_from_gds",
    "partition_nets_from_polygons",
    "estimate_pex_from_nets",
    "estimate_pex_from_gds",
    "estimate_pex_from_polygons",
    "estimate_pex_for_library",
    "compare_pre_vs_post",
    "honest_gap_statement",
]
