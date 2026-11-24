"""GeometricPexBackend -- adapts the existing pex_estimate_v164 estimator to the
backend-neutral PexReport.

APPROXIMATE: ``calibrated=False``, ``field_solver=False``. The estimator gives a
per-net cap-to-substrate, mapped here to ``PexCap(net, ground_net, cap_aF)``. On
top of that this backend ADDS inter-net coupling caps (see
``clay.layout.pex.coupling``):

  * LATERAL (same-layer edge-to-edge): for each pair of nets whose metal shapes
    run adjacent on the same layer, a ``PexCap(net_a, net_b, cap_aF)`` from a
    real parallel-plate-side-wall + side-fringe model (run-length / spacing with
    the real per-layer thickness/spacing).
  * VERTICAL (metal-over-metal overlap): for each pair of nets whose shapes
    overlap in plan view on ADJACENT routing layers (e.g. an M1 riser under a
    foreign M2 trunk), a parallel-plate ``C = eps*area/vgap`` term. Computed only
    when the caller supplies layer-tagged shapes (the polygon path does); the
    flat-map path keeps the lateral term alone.

This is what lets a kickback / clk->regen obligation MEASURE aggressor->victim
coupling instead of reading 0.

The coupling constants are PER-PDK when a ``pdk_id`` is supplied to the
constructor: metal1 thickness / inter-metal dielectric k / min spacing /
metal1->metal2 gap are sourced from THAT PDK's own interconnect-tech file + DRC
deck (see ``clay.layout.pex.pdk_constants``), with documented node-scaled
fallbacks where the PDK is silent. With no ``pdk_id`` the gf180mcu defaults are
used (no behaviour change for existing callers). It stays a PDK-specific
geometric estimate (``calibrated=False``), NOT a field-solver / StarRC number.

Its single-pi lumped wire R is a per-net scalar (not a node-to-node element), so
it is NOT emitted into the unified report; wire R is a fast-follow handled by
Magic ``extresist``. Use ``pex_estimate_v164`` directly if the geometric lumped R
is needed.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from clay.layout import pex_estimate as geo
from clay.layout.backend import LayerPolygon
from .base import PexCap, PexReport
from .coupling import CouplingConstants, coupling_caps_from_nets


def _box(lp: LayerPolygon) -> Tuple[int, int, int, int]:
    xs = [v.x for v in lp.polygon.vertices]
    ys = [v.y for v in lp.polygon.vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def _attach_layer_bboxes(
    nets: Dict[str, Dict[str, object]],
    polygons: Sequence[LayerPolygon],
    *,
    metal_key: Tuple[int, int] = geo.GF180_METAL1_KEY,
) -> None:
    """Augment a metal1-partitioned net map IN PLACE with a ``layer_bboxes`` key.

    For each net (already partitioned on metal1 by the geometric estimator) build
    ``{(layer, dt): [Box, ...]}`` so the VERTICAL overlap term can see shapes on
    DIFFERENT layers. metal1 shapes come straight from the net's own ``bboxes``;
    a non-metal1 routing polygon is attributed to the net whose metal1 footprint
    it overlaps in plan view (a via/M2 stub physically sits over its own net's
    M1). Polygons over no metal1 net are left unattached (honest: no owner ->
    no overlap term invented).
    """
    others = [
        (lp.layer, lp.datatype, _box(lp))
        for lp in polygons
        if (lp.layer, lp.datatype) != metal_key
    ]
    # metal1 footprint per net (for owner attribution of non-metal1 shapes).
    m1_by_net = {
        name: list(data.get("bboxes", ()))  # type: ignore[arg-type]
        for name, data in nets.items()
    }
    for name, data in nets.items():
        layer_bboxes: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {
            metal_key: list(m1_by_net[name])
        }
        data["layer_bboxes"] = layer_bboxes

    def _overlap_area(a, b) -> int:
        ix = min(a[2], b[2]) - max(a[0], b[0])
        iy = min(a[3], b[3]) - max(a[1], b[1])
        return ix * iy if ix > 0 and iy > 0 else 0

    # attribute each non-metal1 routing polygon to the net whose metal1 footprint
    # it overlaps MOST (its own via-stack owner), so a foreign M2 trunk crossing a
    # net's M1 is NOT mis-assigned to that net (it overlaps its own M1 far more).
    for (l, dt, ob) in others:
        best_net, best_area = None, 0
        for name, boxes in m1_by_net.items():
            area = sum(_overlap_area(ob, mb) for mb in boxes)
            if area > best_area:
                best_net, best_area = name, area
        if best_net is not None:
            nets[best_net]["layer_bboxes"].setdefault((l, dt), []).append(ob)  # type: ignore[index]


class GeometricPexBackend:
    name = "geometric"

    def __init__(
        self,
        ground_net: str = "0",
        coupling_consts: Optional[CouplingConstants] = None,
        pdk_id: Optional[str] = None,
    ) -> None:
        """``ground_net`` + ``coupling_consts`` keep their existing meaning.

        When ``pdk_id`` is given (and no explicit ``coupling_consts``), the
        coupling constants are resolved PER-PDK from that PDK's own tech file +
        DRC deck (``CouplingConstants.from_pdk``); otherwise the gf180mcu
        defaults are used so existing callers are unchanged.
        """
        self.ground_net = ground_net
        self.pdk_id = pdk_id
        if coupling_consts is not None:
            self.coupling_consts = coupling_consts
        elif pdk_id is not None:
            self.coupling_consts = CouplingConstants.from_pdk(pdk_id)
        else:
            self.coupling_consts = CouplingConstants()

    def available(self) -> bool:
        return True

    def _estimate_substrate(self, cell_name, nets, source, dbu, *, tech=None, pdk=None):
        """Apply each actual metal's PDK stack before summing parallel net C.

        Plate C uses eps0/integral(dz/k) to the substrate. The perimeter model
        keeps an explicit unit fringe factor, C/P=eps0*k_effective. Neither
        term accounts for 3-D shielding or constitutes calibrated extraction.
        """
        from clay.pdk.pex_profile import get_profile
        from .pdk_constants import EPS0_aF_PER_UM, stack_dielectric_integral

        pdk_id = pdk or self.pdk_id
        profile = get_profile(pdk_id) if pdk_id else None
        if profile is None:
            if pdk_id not in (None, "gf180mcu"):
                raise ValueError(f"PDK {pdk_id!r} has no native substrate stack profile")
            return geo.estimate_pex_from_nets(cell_name, nets, source, dbu)
        if tech is None:
            from clay.layout.tech import LayoutTech
            tech = LayoutTech.resolve(pdk_id)
        by_layer = {}
        provenance = (f"{pdk_id} {profile.source_format}: " + "; ".join(profile.source_files)
                      + "; per-layer unshielded series-dielectric plate + unit-factor perimeter fringe; "
                      "unmapped stack gaps use nearest native dielectric; NOT calibrated / NOT field-solver")
        if profile.warnings:
            provenance += "; " + "; ".join(profile.warnings)
        for conductor in profile.conductors:
            name = conductor.name.lower()
            if name.startswith("mt") and name[2:].isdigit():
                level = int(name[2:])
            elif name.startswith("metal") and name[5:].isdigit():
                level = int(name[5:])
            else:
                continue
            integral = stack_dielectric_integral(profile, profile.substrate_height_um,
                                                  conductor.height_um)
            height = conductor.height_um - profile.substrate_height_um
            by_layer[tuple(tech.metal_key(level))] = geo.MetalPexConstants(
                area_cap_aF_per_um2=EPS0_aF_PER_UM / integral,
                fringe_cap_aF_per_um=EPS0_aF_PER_UM * height / integral,
                sheet_r_ohm_per_sq=conductor.sheet_rho_ohm_sq,
                contact_r_ohm=0., source=provenance)
        report = geo.CellPexReport(cell_name=cell_name, connectivity_source=source,
                                  constants_source=provenance, calibrated=False, field_solver=False)
        for name, data in sorted(nets.items()):
            layer_boxes = data.get("layer_bboxes")
            if not layer_boxes:
                layer_boxes = {tuple(tech.metal_key(1)): data["bboxes"]}
            for layer, boxes in sorted(layer_boxes.items()):
                if layer not in by_layer:
                    raise ValueError(f"PDK stack {pdk_id!r} has no routing layer {layer!r}")
                part = geo.estimate_pex_from_nets(
                    cell_name, {name: {"bboxes": boxes, "contacts": 0}}, source, dbu,
                    metal_consts=by_layer[layer])
                report.nets.extend(part.nets)
        return report

    def extract_from_route_snapshot(self, snapshot, *, tech=None):
        """Re-extract the same immutable route-owned geometry in a fresh process."""
        if snapshot.get("schema") != "clay.geometric-route-input/v1":
            raise ValueError("Unsupported route-owned PEX snapshot schema")
        pdk_id = str(snapshot["pdk_id"])
        if self.pdk_id is not None and pdk_id != self.pdk_id:
            raise ValueError("Route-owned PEX snapshot belongs to another PDK")
        if tech is None:
            from clay.layout.tech import LayoutTech
            tech = LayoutTech.resolve(pdk_id)
        nets = {row["net"]: {"bboxes": [tuple(box) for box in row["bboxes"]],
                             "contacts": row["contacts"],
                             "layer_bboxes": {tuple(layer["layer"]): [tuple(box) for box in layer["bboxes"]]
                                              for layer in row["layers"]}}
                for row in snapshot["nets"]}
        cell = self._estimate_substrate(snapshot["cell_name"], nets, "route-owned-full-stack",
                                         geo.DEFAULT_DBU_TO_UM, tech=tech, pdk=pdk_id)
        return self._to_report(cell, self._coupling_caps(nets, geo.DEFAULT_DBU_TO_UM))

    def _coupling_caps(
        self, nets: Dict[str, Dict[str, object]], dbu_to_um: float
    ) -> Tuple[PexCap, ...]:
        """Inter-net lateral coupling caps for an already-partitioned net map."""
        triples = coupling_caps_from_nets(nets, dbu_to_um, self.coupling_consts)
        return tuple(
            PexCap(a, b, cap_aF) for (a, b, cap_aF) in triples if cap_aF > 0.0
        )

    def _to_report(
        self, cell: geo.CellPexReport, coupling: Tuple[PexCap, ...]
    ) -> PexReport:
        substrate_caps = tuple(
            PexCap(n.net_name, self.ground_net, n.cap_aF)
            for n in cell.nets
            if n.cap_aF > 0.0
        )
        note = cell.note
        if coupling:
            pdk_tag = f"{self.pdk_id} " if self.pdk_id else "gf180 default "
            note = (
                f"{note}; + {len(coupling)} inter-net coupling caps "
                f"(geometric lateral side-wall+fringe + metal-over-metal overlap, "
                f"{pdk_tag}constants, NOT field-solver / NOT calibrated)"
            )
        return PexReport(
            cell_name=cell.cell_name,
            caps=substrate_caps + coupling,
            resistors=(),  # geometric lumped R is per-net scalar, not node-to-node
            backend=self.name,
            connectivity_source=cell.connectivity_source,
            constants_source=cell.constants_source + "; coupling: " + self.coupling_consts.source,
            calibrated=False,
            field_solver=False,
            note=note,
        )

    def extract_from_polygons(
        self, cell_name: str, polygons: Sequence[LayerPolygon]
    ) -> PexReport:
        nets, src, dbu = geo.partition_nets_from_polygons(polygons)
        cell = self._estimate_substrate(cell_name, nets, src, dbu)
        # tag each net's shapes by layer so the VERTICAL overlap term can fire
        # (the lateral term already works on the flat metal1 bboxes).
        _attach_layer_bboxes(nets, polygons)
        return self._to_report(cell, self._coupling_caps(nets, dbu))

    def extract_from_routed(self, cell_name: str, routed, tech) -> PexReport:
        """Estimate route-owned wire C while preserving explicit net ownership.

        Unlike the GDS M1-only partition, the router already records every
        emitted wire/via under its owning net across the full routing stack.
        Consuming that contract avoids anonymous ``$N`` islands and never
        guesses a schematic net from geometric proximity.
        """
        metal_keys = set()
        via_keys = set()
        for level in range(1, 10):
            try:
                metal_keys.add(tuple(tech.metal_key(level)))
            except Exception:  # noqa: BLE001 - stack ended
                pass
            try:
                via_keys.add(tuple(tech.via_key(level)))
            except Exception:  # noqa: BLE001 - stack ended
                pass
        nets: Dict[str, Dict[str, object]] = {}
        for raw_name, route in sorted(routed.nets.items()):
            bboxes: List[Tuple[int, int, int, int]] = []
            layer_bboxes: Dict[
                Tuple[int, int], List[Tuple[int, int, int, int]]
            ] = {}
            contacts = 0
            for polygon in route.wires:
                key = (int(polygon.layer), int(polygon.datatype))
                if key in metal_keys:
                    box = _box(polygon)
                    bboxes.append(box)
                    layer_bboxes.setdefault(key, []).append(box)
                elif key in via_keys:
                    contacts += 1
            if bboxes:
                nets[str(raw_name).casefold()] = {
                    "bboxes": bboxes,
                    "contacts": contacts,
                    "layer_bboxes": layer_bboxes,
                }
        if not nets:
            raise ValueError("route-owned geometric PEX found no metal")
        self.route_geometry_snapshot = {
            "schema": "clay.geometric-route-input/v1", "cell_name": cell_name,
            "ground_net": self.ground_net,
            "pdk_id": str(getattr(tech, "pdk_id", self.pdk_id or "gf180mcu")),
            "nets": [{"net": name, "bboxes": data["bboxes"], "contacts": data["contacts"],
                      "layers": [{"layer": layer, "bboxes": boxes}
                                 for layer, boxes in sorted(data["layer_bboxes"].items())]}
                     for name, data in sorted(nets.items())],
        }
        cell = self._estimate_substrate(
            cell_name, nets, "route-owned-full-stack", geo.DEFAULT_DBU_TO_UM, tech=tech
        )
        cell.note = (
            f"{cell.note}; route-owned full-stack metal only; device intrinsic "
            "parasitics remain in the PDK MOS models; NOT calibrated"
        )
        report = self._to_report(
            cell, self._coupling_caps(nets, geo.DEFAULT_DBU_TO_UM)
        )
        # ``routed.nets`` is the route connectivity authority, including nets
        # that are complete by direct terminal access and therefore need no
        # added wire polygon.  Retain those nets as explicit zero-added-wire-C
        # measurements.  They are evidence rows only: the post-PEX netlist
        # composer skips non-positive caps, so no fictitious device is injected.
        # This is materially different from inventing zero for an unknown net:
        # only nets present in the same router result are admitted here.
        present = report.nets()
        zero_caps = tuple(
            PexCap(str(raw_name).casefold(), self.ground_net, 0.0)
            for raw_name in sorted(routed.nets)
            if str(raw_name).casefold() not in present
        )
        if zero_caps:
            report = replace(
                report,
                caps=report.caps + zero_caps,
                note=(
                    f"{report.note}; {len(zero_caps)} zero-cap rows record "
                    "route-owned nets with no added wire polygon"
                ),
            )
        return report

    def extract(self, gds_path: str, cell_name: str, pdk: str = "gf180mcu") -> PexReport:
        # Resolve the metal1 / contact / metal1-label GDS keys for THIS PDK so the
        # net partition reads the right layers (external_pdk metal1 61/0, contact 50/0; the
        # gf180 defaults would read empty layers -> 0 nets -> 0 caps). Keys come from
        # the PDK's own LayoutTech; on any failure the gf180 defaults are kept so
        # existing callers are unchanged.
        metal_key = geo.GF180_METAL1_KEY
        contact_key = geo.GF180_CONTACT_KEY
        label_key = geo.GF180_METAL1_LABEL_KEY
        try:
            from clay.layout.tech import LayoutTech

            tech = LayoutTech.resolve(pdk)
            metal_key = tuple(tech.metal_key(1))
            contact_key = tuple(tech.contact_key())
            label_key = tuple(tech.label_key(1))
        except Exception:  # noqa: BLE001 - unknown PDK -> gf180 defaults (parity)
            pass
        nets, src, dbu = geo.partition_nets_from_gds(
            gds_path,
            topcell=cell_name,
            metal_key=metal_key,
            contact_key=contact_key,
            label_key=label_key,
        )
        cell = self._estimate_substrate(cell_name, nets, src, dbu, pdk=pdk)
        return self._to_report(cell, self._coupling_caps(nets, dbu))
