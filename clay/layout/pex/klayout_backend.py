"""KLayoutPexBackend — parasitic extraction using KLayout's connectivity engine.

KLayout is already used for DRC and LVS in Clay. This backend uses its
Python API to enumerate conductor geometry and estimate per-net R and C
using the PDK layer-stack profile (``PdkStackProfile``).

It serves as an independent cross-validation oracle for ``MagicPexBackend``
and ``GeometricPexBackend`` — same input GDS, same PDK constants, different
extraction engine → RC values should agree within ~50 %.

Unlike the geometric estimator, KLayout resolves the actual conductor
geometry (area, perimeter) from real GDS shapes. As with
``GeometricPexBackend``, the result is ``calibrated=False``,
``field_solver=False``; it remains a geometric estimate, but one computed
from the as-built GDS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from clay.layout.pex.base import PexCap, PexReport, PexRes


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@dataclass
class KLayoutPexBackend:
    """KLayout-based geometric parasitic extractor."""

    name: str = "klayout"
    _available: Optional[bool] = None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def available(self) -> bool:
        if self._available is None:
            try:
                import klayout.db as _db  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def supports_pdk(self, pdk: str) -> bool:
        """All PDKs with a resolvable ``PdkStackProfile`` are supported."""
        if not self.available():
            return False
        from clay.pdk.pex_profile import get_profile
        return get_profile(pdk) is not None

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        gds_path: str | Path,
        cell_name: str,
        pdk: str = "gf180mcu",
        *,
        with_resistance: bool = True,
        coupling: bool = False,
        temp_c: float = 25.0,
    ) -> PexReport:
        """Extract parasitic R and C from a GDS file.

        Returns ``PexReport`` with ``calibrated=False``.
        """
        if not self.available():
            raise RuntimeError("KLayout unavailable; install klayout Python package")

        from clay.pdk.pex_profile import get_profile

        profile = get_profile(pdk)
        if profile is None:
            raise RuntimeError(
                f"No layer-stack profile for PDK {pdk!r}"
            )

        gds_path = Path(gds_path)
        layout = self._read_layout(gds_path)
        top_cell = layout.cell(cell_name)
        if top_cell is None:
            raise RuntimeError(f"Cell {cell_name!r} not found in {gds_path}")

        layer_index = self._build_layer_index(layout, profile, pdk)
        caps, resis = self._extract_from_cell(
            top_cell, layer_index, profile, temp_c,
            with_resistance=with_resistance, coupling=coupling,
        )

        return PexReport(
            cell_name=cell_name,
            caps=tuple(caps),
            resistors=tuple(resis) if with_resistance else (),
            backend=self.name,
            connectivity_source=f"klayout_connected_components_{pdk}",
            constants_source=profile.source_format,
            calibrated=False,
            field_solver=False,
            note=(
                f"uncalibrated connected-component geometric estimate from "
                f"{profile.source_format} data; anonymous nets retain synthetic "
                "names"
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_layout(gds_path: Path):
        import klayout.db as db
        layout = db.Layout()
        layout.read(str(gds_path))
        return layout

    @staticmethod
    def _build_layer_index(layout, profile, pdk: str) -> Dict[str, int]:
        """Map conductor names → KLayout layer indices.

        Uses the PDK adapter's layermap to translate PDK layer names to
        GDS layer/datatype pairs.
        """
        index: Dict[str, int] = {}

        # Get the PDK layermap (PDK name → (gds_layer, gds_datatype))
        try:
            from clay.pdk import get_pdk
            adapter = get_pdk(pdk)
            pdk_layermap = adapter.layermap()  # {pdk_name: (gds_l, gds_dt)}
        except Exception:
            pdk_layermap = {}

        for c in profile.conductors:
            # Try PDK layermap first
            gds_key = pdk_layermap.get(c.name)
            if gds_key is None:
                canonical_name = c.name.lower()
                if canonical_name.startswith("mt") and canonical_name[2:].isdigit():
                    canonical_name = f"metal{canonical_name[2:]}"
                gds_key = pdk_layermap.get(canonical_name)
            if gds_key is None:
                # Try case-insensitive alternative names.
                for alt_name, alt_key in pdk_layermap.items():
                    if canonical_name == alt_name.lower():
                        gds_key = alt_key
                        break
            if gds_key is not None:
                gds_layer, gds_dt = gds_key
                li = layout.layer(gds_layer, gds_dt)
                if li is not None and li < layout.layers():
                    index[c.name] = li
                    continue
            # Fallback: scan by name
            for li in range(layout.layers()):
                info = layout.layer_infos()[li]
                if info is None:
                    continue
                lname = str(info.name or "").lower()
                if lname == c.name.lower():
                    index[c.name] = li
                    break
        for layer_name, gds_key in pdk_layermap.items():
            canonical = layer_name.lower()
            if not (canonical.startswith("via") or canonical == "contact"):
                continue
            li = layout.find_layer(*gds_key)
            if li is not None:
                index[canonical] = li
        return index

    @staticmethod
    def _extract_from_cell(
        top_cell, layer_index: Dict[str, int], profile,
        temp_c: float, *, with_resistance: bool, coupling: bool,
    ) -> Tuple[List[PexCap], List[PexRes]]:
        """Trace conductor connectivity and estimate RC for each physical net."""
        import klayout.db as db

        if coupling:
            raise NotImplementedError(
                "KLayout connected-component PEX does not estimate coupling"
            )

        eps0 = 8.8541878128e-3  # fF/um
        layout = top_cell.layout()
        dbu_to_um = float(layout.dbu)

        caps: List[PexCap] = []
        resis: List[PexRes] = []
        l2n = db.LayoutToNetlist(
            db.RecursiveShapeIterator(layout, top_cell, [])
        )
        conductor_handles = {}
        for conductor in profile.conductors:
            li = layer_index.get(conductor.name)
            if li is None:
                continue
            handle = l2n.make_layer(li, conductor.name)
            l2n.connect(handle)
            conductor_handles[conductor.name] = handle

        metal_handles = {}
        for conductor in profile.conductors:
            name = conductor.name.lower()
            if name.startswith("mt") and name[2:].isdigit():
                level = int(name[2:])
            elif name.startswith("metal") and name[5:].isdigit():
                level = int(name[5:])
            else:
                continue
            metal_handles[level] = conductor_handles.get(conductor.name)
        for level in sorted(metal_handles):
            lower = metal_handles.get(level)
            upper = metal_handles.get(level + 1)
            via_li = layer_index.get(f"via{level}")
            if lower is None or upper is None or via_li is None:
                continue
            via = l2n.make_layer(via_li, f"via{level}")
            l2n.connect(lower, via)
            l2n.connect(via, upper)

        l2n.extract_netlist()
        circuits = list(l2n.netlist().each_circuit())
        if not circuits:
            return caps, resis
        circuit = max(circuits, key=lambda item: sum(1 for _ in item.each_net()))
        conductors_by_name = {c.name: c for c in profile.conductors}

        for physical_net in circuit.each_net():
            expanded = str(physical_net.expanded_name() or "").strip()
            net_name = (
                expanded
                if expanded and not expanded.startswith("$")
                else f"net_{physical_net.cluster_id}"
            )
            for conductor_name, handle in conductor_handles.items():
                region = l2n.shapes_of_net(physical_net, handle, True)
                if region is None or region.is_empty():
                    continue
                c = conductors_by_name[conductor_name]

                # Find dielectric below this conductor for area capacitance
                dielectrics_below = sorted(
                    [d for d in profile.dielectrics if d.height_um <= c.height_um],
                    key=lambda d: d.height_um,
                )
                k_below = (
                    dielectrics_below[-1].dielectric_constant
                    if dielectrics_below
                    else 4.0
                )
                t_below = (
                    c.height_um - dielectrics_below[-1].height_um
                    if dielectrics_below
                    else 0.5
                )
                if t_below <= 0:
                    t_below = min(c.thickness_um * 3, 0.5)

                area_um2 = float(region.area()) * dbu_to_um**2
                perimeter_um = float(region.perimeter()) * dbu_to_um

                if c.name.lower() == "mim" and profile.mim_f_um2 is not None:
                    c_area = profile.mim_f_um2 * area_um2
                    c_fringe = 0.0
                else:
                    # Area capacitance to substrate
                    c_area = eps0 * k_below * area_um2 / t_below
                    # Fringe capacitance (simplified: ~0.05 fF per um perimeter)
                    c_fringe = eps0 * k_below * perimeter_um * 0.05
                cap_af = round((c_area + c_fringe) * 1000.0, 3)

                caps.append(PexCap(a=net_name, b="GND", cap_aF=cap_af))

                if with_resistance and area_um2 > 0 and perimeter_um > 0:
                    w_eff = 2.0 * area_um2 / perimeter_um
                    l_eff = perimeter_um / 4.0
                    seg_r = c.sheet_rho_ohm_sq * (l_eff / max(w_eff, 1e-3))
                    if c.temp_tc1 != 0:
                        seg_r *= 1.0 + c.temp_tc1 * (temp_c - 25.0)
                    resis.append(
                        PexRes(
                            a=net_name,
                            b=f"{net_name}_{c.name}_end",
                            res_ohm=round(seg_r, 4),
                        )
                    )

        # Sort for determinism
        caps.sort(key=lambda x: (x.a, x.b))
        resis.sort(key=lambda x: (x.a, x.b))
        return caps, resis


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

def get_klayout_backend() -> KLayoutPexBackend:
    """Return a singleton ``KLayoutPexBackend``."""
    return KLayoutPexBackend()
