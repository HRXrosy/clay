"""R136 (BL-136C) — formal opt-in "sane routed" lowering API.

R135 fixed the strange-looking 5T-OTA routing by composing three already-
committed pieces in a one-off demo (``tools/regen_routed_gds_v135/regen.py``):

1. connectivity-aware re-placement (``connectivity_placement.replace_placement``)
   so electrically-connected devices land adjacent instead of scattered
   corner-to-corner by the naive ``sorted(keys)`` x-spread;
2. authoritative per-device pin extraction
   (``connectivity_placement.placed_device_pins``) so net pins are read off each
   device's OWN translated geometry, never via the x-order slicing heuristic
   (which is only valid for the naive single-row layout);
3. contained Manhattan M1 net routing
   (``routing.route_m1_manhattan`` + ``route_containment.clamp_within``) so every
   routable net is wired through its real pins and clamped inside the cell bbox.

This module promotes that demo into ONE reusable, documented public function,
``route_annotated_cell_sane``. It is purely additive and OPT-IN: it imports the
committed modules read-only and modifies none of them, so the existing default
lowering (``lower_annotated_cell_to_parameterized``) and every other lowering
function behave byte-for-byte as before. Nothing here is wired into the default
path; a caller gets the sane route only by calling this function explicitly.

No coordinate is fabricated. Device positions come from
``connectivity_placement`` (derived from real device bboxes + netlist
adjacency); pin points come from authoritative placed geometry; wires come from
the real M1 router; containment only retracts vertices that already escape the
cell bbox + margin (a legal in-bbox route round-trips byte-unchanged).

Pipeline (mirrors the R135 demo, now reusable)::

    naive   = lower_annotated_cell_to_parameterized(cell, pdk, fin_pitch, sizing)
    replaced, placement = replace_placement(naive, cell, pdk, fin_pitch)
    dp      = placed_device_pins(replaced, cell, pdk, placement)   # authoritative
    netlist = build_cell_netlist(naive, cell, pdk)                 # membership only
    for net in netlist.routable_nets():
        pts   = sorted({ authoritative point of each pin })
        wires = chain route_m1_manhattan over consecutive pts
        wires = clamp_within(region, wires)                        # contain
        merge wires into the re-placed top cell

The naive x-order pin heuristic (``cell_netlist.extract_device_pins``) is used
ONLY for net *membership* (which device + which terminal belongs to a net) — the
pin *points* always come from ``placed_device_pins`` on the re-placed library, so
a reordered / stacked placement routes correctly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .annotations import AnnotatedCell
from .backend import (
    DeviceSizing,
    LayerMap,
    LayerPolygon,
    NamedCell,
    ParameterizedLibrary,
    Point,
    lower_annotated_cell_to_parameterized,
)
from .cell_netlist import DevicePins, PinKind, build_cell_netlist
from .connectivity_placement import (
    DevicePlacement,
    placed_device_pins,
    replace_placement,
)
from .route_containment import Region, clamp_within, region_from_cell, verify_within
from .routing import route_m1_manhattan


# --- result ------------------------------------------------------------------


@dataclass
class SaneRoutedResult:
    """The output of :func:`route_annotated_cell_sane`.

    Attributes
    ----------
    library:
        The connectivity-aware re-placed :class:`ParameterizedLibrary` with the
        routed M1 wires merged into its top cell. This is the routed layout the
        caller emits to GDS.
    net_wires:
        ``{net_name: [LayerPolygon, ...]}`` — the M1 wires routed for each
        routable net (already clamped into the cell region). A routable net that
        resolves to fewer than two distinct pin points is skipped (omitted).
    placement:
        The :class:`DevicePlacement` (per-device offsets + ``strategy``) that the
        re-placement produced. Lets callers/tests assert which placement rule ran
        and reconstruct authoritative pins.
    region:
        The allowed :class:`Region` (re-placed top-cell bbox + ``margin_nm``) the
        wires were clamped into; ``None`` only if the re-placed top cell is empty.
    device_pins:
        The authoritative ``{device: DevicePins}`` map (pins read off each
        device's translated geometry) used to route. Exposed so callers can
        re-measure spans without re-deriving from a reordered top cell.
    routable_net_count:
        Number of routable nets in the cell's netlist (membership-only count).
    routed_net_count:
        Number of nets actually wired (``len(net_wires)``); equals
        ``routable_net_count`` minus nets that collapsed to <2 distinct points.

    Methods give the layout-sanity facts the acceptance checks need without
    importing the metrics module: :meth:`routed_wires`, :meth:`all_within`,
    :meth:`max_net_span`.
    """

    library: ParameterizedLibrary
    net_wires: Dict[str, List[LayerPolygon]] = field(default_factory=dict)
    placement: Optional[DevicePlacement] = None
    region: Optional[Region] = None
    device_pins: Dict[str, DevicePins] = field(default_factory=dict)
    routable_net_count: int = 0
    routed_net_count: int = 0

    def routed_wires(self) -> List[LayerPolygon]:
        """Flat list of every routed wire polygon across all nets."""
        return [w for ws in self.net_wires.values() for w in ws]

    def all_within(self) -> bool:
        """True iff every routed wire vertex lies inside :attr:`region`.

        Vacuously True when there is no region (empty cell) or no wires.
        """
        if self.region is None:
            return True
        return verify_within(self.region, self.routed_wires())

    def max_net_span(self) -> int:
        """Maximum Manhattan span (bbox half-perimeter) over all routed nets.

        Each net's span is the half-perimeter of its wire-vertex bbox — the same
        wirelength proxy ``connectivity_placement`` minimises. Returns 0 when no
        net was routed. Used to compare against a naive route's worst span.
        """
        best = 0
        for ws in self.net_wires.values():
            span = _wire_span(ws)
            if span > best:
                best = span
        return best


# --- internal helpers (read-only over committed types) ----------------------


def _pin_point(
    dp: Dict[str, DevicePins], device: str, kind: PinKind
) -> Optional[Point]:
    d = dp.get(device)
    if d is None:
        return None
    if kind == PinKind.GATE:
        return d.gate
    if kind == PinKind.SOURCE:
        return d.source
    if kind == PinKind.DRAIN:
        return d.drain
    return None


def _wire_span(wires: List[LayerPolygon]) -> int:
    """Half-perimeter of the bbox enclosing one net's wire vertices (0 empty)."""
    xs = [v.x for lp in wires for v in lp.polygon.vertices]
    ys = [v.y for lp in wires for v in lp.polygon.vertices]
    if not xs:
        return 0
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


# --- public API --------------------------------------------------------------


def route_annotated_cell_sane(
    cell: AnnotatedCell,
    pdk_layers: LayerMap,
    fin_pitch_nm: float,
    sizing: DeviceSizing,
    *,
    wire_width_nm: int = 140,
    margin_nm: int = 0,
) -> SaneRoutedResult:
    """Lower ``cell`` with connectivity-aware placement + contained M1 routing.

    This is the reusable, documented form of the R135 regen demo. It does NOT
    alter any existing lowering function — it composes them read-only and
    returns a fresh :class:`SaneRoutedResult`. The default lowering remains
    byte-unchanged (this path is purely additive / opt-in).

    Parameters
    ----------
    cell:
        The annotated cell to lower (e.g. from a ``cell_templates`` factory).
    pdk_layers:
        The PDK layer map (must define POLY + diffusion + METAL1 layers).
    fin_pitch_nm:
        Fin pitch in nm (drives the inter-device gap, matching the naive path's
        ``4 * fin_pitch`` so placement is never tighter than the naive layout).
    sizing:
        Per-device ``{name: (w_um, l_um)}`` map for the naive lowering.
    wire_width_nm:
        M1 wire width in nm DBU (default 140, matching the R135 demo).
    margin_nm:
        Slack (nm) added around the re-placed cell bbox before clamping wires.
        ``0`` (default) clamps strictly to the cell footprint, so a sane route
        has an out-of-bbox fraction of exactly 0.

    Returns
    -------
    SaneRoutedResult
        The re-placed + routed library, per-net wires, placement, region, and
        authoritative device pins. See :class:`SaneRoutedResult`.

    Notes
    -----
    Net *membership* (which device terminal belongs to which net) comes from the
    topology netlist built on the NAIVE library; pin *points* come from the
    authoritative placed pins on the RE-PLACED library, so reordered / stacked
    placements route correctly. Wires are clamped into the cell region; a route
    already inside is unchanged.
    """
    # --- 1. naive lowering: supplies local-origin per-device geometry --------
    naive = lower_annotated_cell_to_parameterized(
        cell, pdk_layers, fin_pitch_nm, sizing
    )

    # --- 2. connectivity-aware re-placement (R135 A) -------------------------
    replaced, placement = replace_placement(naive, cell, pdk_layers, fin_pitch_nm)

    # --- 3. authoritative per-device pins on the re-placed cell (R135 A) -----
    dp = placed_device_pins(replaced, cell, pdk_layers, placement)

    # --- 4. net membership from the naive netlist (points are NOT used) ------
    netlist = build_cell_netlist(naive, cell, pdk_layers)

    top = replaced.cells[-1] if replaced.cells else None
    region = region_from_cell(top, margin=margin_nm) if top is not None else None

    # --- 5. route each routable net on M1, clamped into the region (R135 B) --
    net_wires: Dict[str, List[LayerPolygon]] = {}
    routable = netlist.routable_nets()
    for net in routable:
        pts: List[Tuple[int, int]] = []
        for p in net.pins:
            pt = _pin_point(dp, p.device, p.kind)
            if pt is not None:
                pts.append((pt.x, pt.y))
        pts = sorted(set(pts))
        if len(pts) < 2:
            continue
        wires: List[LayerPolygon] = []
        for i in range(len(pts) - 1):
            a = Point(*pts[i])
            b = Point(*pts[i + 1])
            wires.extend(route_m1_manhattan(a, b, wire_width_nm, pdk_layers))
        if region is not None:
            wires = clamp_within(region, wires)
        net_wires[net.name] = wires
        if top is not None:
            top.polygons.extend(wires)

    return SaneRoutedResult(
        library=replaced,
        net_wires=net_wires,
        placement=placement,
        region=region,
        device_pins=dp,
        routable_net_count=len(routable),
        routed_net_count=len(net_wires),
    )


__all__ = ["SaneRoutedResult", "route_annotated_cell_sane"]
