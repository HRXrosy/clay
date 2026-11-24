"""Unified net-tagged LayoutModel: the single physical world-view the router queries.

Dev-discipline rule 6: the router must NOT own a private physical world-view (ad-hoc
``_device_m1_obstacles`` / ``_cap_plate_obstacles``). Every device conductor and cap plate
is a net-tagged :class:`Conductor` in ONE :class:`LayoutModel`; the router's obstacle list
is a PROJECTION of that model (:meth:`LayoutModel.router_obstacles`). The legacy helpers
become thin compatibility adapters over this model (a strangler step -- the engine still
consumes obstacles for now, but they come from the single model, not a parallel build).

Leakage-free: conductors come from the DEVICE CELLS (``build_device`` / ``build_capacitor``
geometry) + the leakage-free placement, never a reference ROUTED layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from clay.layout.route.engine import Obstacle
from clay.layout.placed_geometry import placed_point_nm, placed_ring_nm
from clay.layout.tech import LayoutTech

_NM_PER_UM = 1000.0
_XY = Tuple[float, float]
_Layer = Tuple[int, int]

__all__ = ["Conductor", "LayoutModel", "ROUTING_ABSTRACTION_KINDS",
           "device_m1_conductors", "cap_plate_conductors",
           "device_upper_conductors", "device_active_projection_conductors",
           "build_layout_model"]

# Conductor KINDS that are ROUTING-MODEL ABSTRACTIONS, not literally-drawn
# artwork: the min-space-BLOATED cap-plate keep-out collars ("cap_plate", see
# ``cap_plate_conductors`` -- every emitted rect is the drawn extent grown by
# the layer min-space), the footprint-wide inter-plate via bans
# ("cap_plate_via") and the lowered ``@no_route_under`` keep-outs
# ("no_route_under"). They participate in route LEGALITY and congestion
# pricing exactly like artwork, but an engine must NEVER credit them as drawn
# metal in its emitted-geometry judgments (gap coverage / junction contact /
# same-net band detection): metal that exists only in the model cannot merge
# anything in the artifact the deck reads. The measured escape: a net25 merge
# fill rode its bloated cap-plate collar and only corner point-touched the
# TRUE drawn plate -- deck M4.W.1 x2 + M4.S.1 x2 at one kiss point, invisible
# to a coverage judgment that counted the collar as metal.
ROUTING_ABSTRACTION_KINDS = frozenset({
    "cap_plate", "cap_plate_via", "no_route_under",
})


@dataclass(frozen=True)
class Conductor:
    """One net-tagged conductor rectangle (um), on a specific layer.

    ``net`` is the owning net (a signal pin landed inside it / a cap plate's net), or
    ``None`` for an unowned conductor that is a HARD keep-out for every net (e.g. a
    body/well-tie pad). ``net`` is what lets a net route onto its OWN conductor while every
    foreign net treats it as a keep-out.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    layer: _Layer
    net: Optional[str]
    kind: str = "device"   # provenance: drawn conductor or nonblocking projection
    terminal_conductor_ref: Optional[str] = None
    # Physical producer identity, independent of electrical net ownership. Legacy
    # conductor/obstacle equality stays geometric; source-bound consumers must
    # include this field explicitly in their complete inventory comparison.
    source_stamp: Optional[str] = field(default=None, compare=False)

    def as_obstacle(self) -> Obstacle:
        return Obstacle(self.x0, self.y0, self.x1, self.y1, self.layer,
                        owner_net=self.net, kind=self.kind,
                        terminal_conductor_ref=self.terminal_conductor_ref)


def _terminal_conductor_ref(
    device: str,
    terminals: Sequence[str],
    net: str,
    component_polygon_nm: Sequence[Tuple[int, int]],
) -> str:
    """Return one deterministic compiler identity, never a geometry guess."""

    payload = {
        "device": str(device),
        "net": str(net),
        "component_polygon_nm": [list(point) for point in component_polygon_nm],
        "schema": "clay.device-terminal-conductor.v1",
        "terminals": sorted(str(terminal) for terminal in terminals),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "tc:" + hashlib.sha256(encoded).hexdigest()


def _canonical_polygon_nm(
    points: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[int, int], ...]:
    """Canonicalize one polygon ring independently of start and direction."""

    ring = tuple((int(x), int(y)) for x, y in points)
    if not ring:
        return ()
    rotations = []
    for direction in (ring, tuple(reversed(ring))):
        rotations.extend(
            direction[index:] + direction[:index]
            for index in range(len(direction))
        )
    return min(rotations)


def _upper_terminal_conductor_ref(
    device: str,
    terminals: Sequence[str],
    net: str,
    component_polygons_nm: Sequence[
        Tuple[int, Sequence[Tuple[int, int]]]
    ],
) -> str:
    """Name one deck-connected upper terminal component deterministically."""

    polygons = tuple(sorted(
        (
            int(level),
            _canonical_polygon_nm(points),
        )
        for level, points in component_polygons_nm
    ))
    payload = {
        "component_metal_polygons_nm": [
            {"level": level, "polygon": [list(point) for point in polygon]}
            for level, polygon in polygons
        ],
        "device": str(device),
        "net": str(net),
        "schema": "clay.device-upper-terminal-conductor.v1",
        "terminals": sorted(str(terminal) for terminal in terminals),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "tc:" + hashlib.sha256(encoded).hexdigest()


def _owned_merged_polygons(region: Any) -> Tuple[Any, ...]:
    """Deep-copy merged polygons before KLayout advances its borrowed iterator."""
    import klayout.db as kdb

    owned = []
    for borrowed in region.each_merged():
        owned.append(kdb.Polygon(borrowed))
    return tuple(owned)


@dataclass
class LayoutModel:
    """All net-tagged conductors of a placed layout; the router queries THIS, not ad-hoc
    obstacle builders."""

    conductors: List[Conductor] = field(default_factory=list)

    def router_obstacles(self) -> List[Obstacle]:
        """Project every conductor to a router :class:`Obstacle` (the compatibility surface
        the engine consumes today)."""
        return [c.as_obstacle() for c in self.conductors]

    def on_net(self, net: str) -> List[Conductor]:
        return [c for c in self.conductors if c.net == net]

    def on_layer(self, layer: _Layer) -> List[Conductor]:
        lk = tuple(layer)
        return [c for c in self.conductors if tuple(c.layer) == lk]


@dataclass(frozen=True)
class _DeviceArtworkProjection:
    """One actual mini-extraction shared by lower and upper projections.

    Regions are private, owned KLayout copies.  Only M1 islands connected to an
    actual graph upper metal enter ``stack_m1_components``; purely M1 terminal
    identities retain their established representation.
    """

    upper_conductors: Tuple[Conductor, ...]
    # The final flag records observed conflicting seeds. A missing local seed
    # does not veto an independently seeded global M1 electrical island.
    stack_m1_components: Tuple[Tuple[str, Any, Optional[str], Optional[str], bool], ...]


def device_m1_conductors(
    intent: Any,
    placement: Dict[str, _XY],
    geometry: Dict[str, Any],
    *,
    tech: LayoutTech,
    pins_by_net: Dict[str, List[Any]],
    strict_device_m1_provenance: bool = False,
) -> List[Conductor]:
    """Net-tagged M1 device-pad conductors. Each placed device M1 polygon becomes a
    conductor tagged with the net of the signal pin that lands inside it (or ``None`` for a
    pad covering no signal pin -> a hard keep-out).

    The historical default uses each ring's bbox and first matching pin verbatim.
    ``strict_device_m1_provenance`` is an engine capability (always true for
    Gridless, false for engines that do not consume this ownership model): it
    merges touching real M1 polygons into electrical islands, seeds them only from
    M1 pins that are inside the real polygon, and assigns an owner only when the
    island has one unique net. Actual upper-metal/via connections contribute
    the same stack ownership used by :func:`device_upper_conductors`, including
    mixed-net conflicts. The rectangular obstacle projection is decomposed
    from the real polygon, so an empty concave bbox notch never becomes owned M1.
    """
    artwork = (
        _device_artwork_projection(
            intent, placement, geometry, tech=tech, pins_by_net=pins_by_net)
        if strict_device_m1_provenance else None
    )
    return _device_m1_conductors(
        intent, placement, geometry, tech=tech, pins_by_net=pins_by_net,
        strict_device_m1_provenance=strict_device_m1_provenance,
        artwork=artwork)


def _device_m1_conductors(
    intent: Any,
    placement: Dict[str, _XY],
    geometry: Dict[str, Any],
    *,
    tech: LayoutTech,
    pins_by_net: Dict[str, List[Any]],
    strict_device_m1_provenance: bool,
    artwork: Optional[_DeviceArtworkProjection],
) -> List[Conductor]:
    m1: _Layer = tuple(tech.metal_key(1))  # type: ignore[assignment]

    if strict_device_m1_provenance:
        import klayout.db as kdb

        # KLayout uses the emitted nm integer currency.  Keep every original
        # polygon as well as the merged island graph: owners belong to islands,
        # while router obstacles remain exact rectangular pieces of the originals.
        raw: List[Tuple[str, str, Any]] = []
        all_m1 = kdb.Region()
        instance_m1: Dict[str, Any] = {}
        instance_conductor_kind: Dict[str, str] = {}
        instance_pin_device_names: Dict[str, frozenset[str]] = {}
        instance_pin_terminal_aliases: Dict[
            str,
            frozenset[Tuple[str, str]],
        ] = {}
        for name, geom in geometry.items():
            pos = placement.get(name)
            if pos is None:
                continue
            geometry_kind = str(getattr(geom, "kind", "") or "")
            geometry_source = str(getattr(geom, "source", "") or "")
            normalized_kind = geometry_kind.strip().lower().replace("_", "-")
            normalized_source = geometry_source.strip().lower().replace("_", "-")
            conductor_kind = (
                "guard-ring-m1"
                if normalized_kind == "guard-ring"
                or normalized_source == "group-guard-ring"
                else "device"
            )
            instance_conductor_kind[str(name)] = conductor_kind
            raw_composite_members = getattr(geom, "composite_members", ())
            composite_members = (
                raw_composite_members
                if type(raw_composite_members) is tuple
                and len(raw_composite_members) == len(set(raw_composite_members))
                and all(
                    type(member) is str and member
                    for member in raw_composite_members
                )
                else ()
            )
            # Shared-diffusion/common-centroid artwork is stamped once under its
            # certified group name while router pins deliberately retain their
            # original member-device identities.  The stamp's explicit
            # ``composite_members`` provenance is therefore the sole alias table
            # for terminal seams; geometry adjacency and coordinates never invent
            # a member binding.  Malformed provenance grants no aliases.
            instance_pin_device_names[str(name)] = frozenset(
                (str(name), *composite_members)
            )
            physical_terminal_aliases = set()
            raw_terminal_accesses = getattr(
                geom, "terminal_accesses_um", {}
            )
            if type(raw_terminal_accesses) is dict:
                for terminal, accesses in sorted(
                    raw_terminal_accesses.items(),
                    key=lambda item: str(item[0]),
                ):
                    if (
                        type(terminal) is not str
                        or not terminal
                        or type(accesses) is not tuple
                        or not accesses
                        or any(
                            type(access) is not tuple
                            or len(access) != 2
                            or any(
                                type(value) not in {int, float}
                                or not math.isfinite(float(value))
                                for value in access
                            )
                            for access in accesses
                        )
                    ):
                        continue
                    physical_terminal_aliases.update(
                        (
                            str(name)
                            if access_index == 0
                            else f"{name}@{access_index + 1}",
                            terminal,
                        )
                        for access_index in range(len(accesses))
                    )
            instance_pin_terminal_aliases[str(name)] = frozenset(
                physical_terminal_aliases
            )
            for ring in (getattr(geom, "polys_by_layer", {}) or {}).get(m1, []):
                if len(ring) < 3:
                    continue
                pts = [kdb.Point(x, y) for x, y in placed_ring_nm(ring, pos)]
                poly = kdb.Polygon(pts)
                if poly.area() <= 0:
                    continue
                raw.append((str(name), conductor_kind, poly))
                all_m1.insert(poly)
                instance_m1.setdefault(str(name), kdb.Region()).insert(poly)
        all_m1.merge()
        islands = _owned_merged_polygons(all_m1)

        # Pre-index M1 pins once so local-component and global-island seeding
        # avoid O(N_components * N_pins) repeated tuple/float conversions.
        m1_island_pins: List[Tuple[int, int, str]] = []
        m1_local_by_dev: Dict[str, List[Tuple[int, int, str, str]]] = {}
        m1_local_by_alias: Dict[Tuple[str, str], List[Tuple[int, int, str, str]]] = {}
        for net, pins in pins_by_net.items():
            net_str = str(net)
            for pin in pins:
                try:
                    pin_layer = tuple(pin.layer)
                except (AttributeError, TypeError):
                    continue
                if pin_layer != m1:
                    continue
                try:
                    px = round(float(pin.xy_um[0]) * _NM_PER_UM)
                    py = round(float(pin.xy_um[1]) * _NM_PER_UM)
                except (AttributeError, TypeError, ValueError):
                    continue
                m1_island_pins.append((px, py, net_str))
                try:
                    pin_device = str(pin.device)
                    pin_terminal = str(pin.terminal)
                except (AttributeError, TypeError, ValueError):
                    continue
                if pin_terminal:
                    entry = (px, py, net_str, pin_terminal)
                    m1_local_by_dev.setdefault(pin_device, []).append(entry)
                    m1_local_by_alias.setdefault((pin_device, pin_terminal), []).append(entry)

        # Compile physical seam identity inside each device instance.  Local
        # connected M1 polygons inherit an identity only from exactly one
        # declared local terminal pin inside their true polygon union.  Missing
        # or multi-terminal membership remains None (ambiguous); no bbox or
        # cross-instance adjacency is allowed to manufacture provenance.
        local_components: Dict[str, List[Tuple[Any, Any, Optional[str]]]] = {}
        for name in sorted(instance_m1):
            if instance_conductor_kind[name] not in {
                "device",
                "guard-ring-m1",
            }:
                local_components[name] = []
                continue
            cand_pins: List[Tuple[int, int, str, str]] = []
            for dev_name in instance_pin_device_names[name]:
                rows = m1_local_by_dev.get(dev_name)
                if rows:
                    cand_pins.extend(rows)
            for alias_key in instance_pin_terminal_aliases[name]:
                rows = m1_local_by_alias.get(alias_key)
                if rows:
                    cand_pins.extend(rows)
            region = instance_m1[name]
            region.merge()
            compiled: List[Tuple[Any, Any, Optional[str]]] = []
            for component in region.each_merged():
                comp_bb = component.bbox()
                cl, cb, cr, ct = comp_bb.left, comp_bb.bottom, comp_bb.right, comp_bb.top
                memberships = set()
                for px, py, net_str, pin_terminal in cand_pins:
                    if (
                        cl <= px <= cr
                        and cb <= py <= ct
                        and component.inside(kdb.Point(px, py))
                    ):
                        memberships.add((net_str, pin_terminal))
                terminal_ref = None
                membership_nets = {net for net, _terminal in memberships}
                if memberships and len(membership_nets) == 1:
                    local_net = next(iter(membership_nets))
                    local_terminals = tuple(sorted(
                        terminal for _net, terminal in memberships
                    ))
                    component_polygon_nm = tuple(
                        (int(point.x), int(point.y))
                        for point in component.each_point_hull()
                    )
                    terminal_ref = _terminal_conductor_ref(
                        name,
                        local_terminals,
                        local_net,
                        component_polygon_nm,
                    )
                # Own the component geometry independently of KLayout's borrowed
                # iterator handle so later raw-piece adjudication is stable.
                compiled.append((kdb.Region(component), comp_bb, terminal_ref))
            local_components[name] = compiled

        stack_components = artwork.stack_m1_components if artwork is not None else ()
        stack_bbs = [sr.bbox() for _n, sr, _so, _sref, _c in stack_components]
        for (name, stack_region, _stack_owner, stack_ref, _conflict), stack_bb in zip(
            stack_components, stack_bbs
        ):
            # This is the same local island from the shared actual extraction,
            # not bbox containment or an inferred cross-stamp terminal alias.
            local_components[name] = [
                (
                    region,
                    r_bb,
                    stack_ref
                    if r_bb.overlaps(stack_bb) and not (region & stack_region).is_empty()
                    else ref,
                )
                for region, r_bb, ref in local_components.get(name, ())
            ]

        island_bbs = [isl.bbox() for isl in islands]
        island_is_box = [isl.is_box() for isl in islands]
        island_regs: List[Any] = [None] * len(islands)

        owners: List[Optional[str]] = []
        for idx, island in enumerate(islands):
            ibb = island_bbs[idx]
            il, ib, ir, it = ibb.left, ibb.bottom, ibb.right, ibb.top
            nets = set()
            for px, py, net_str in m1_island_pins:
                if (
                    il <= px <= ir
                    and ib <= py <= it
                    and net_str not in nets
                    and island.inside(kdb.Point(px, py))
                ):
                    nets.add(net_str)
            conflicting_stack = False
            if stack_components:
                island_region = None
                for (_name, stack_region, stack_owner, _ref, conflict), stack_bb in zip(
                    stack_components, stack_bbs
                ):
                    if not ibb.overlaps(stack_bb):
                        continue
                    if island_region is None:
                        island_region = kdb.Region(island)
                        island_regs[idx] = island_region
                    if (island_region & stack_region).is_empty():
                        continue
                    conflicting_stack |= conflict
                    if stack_owner is not None:
                        nets.add(stack_owner)
            owners.append(
                next(iter(nets)) if len(nets) == 1 and not conflicting_stack else None)

        out: List[Conductor] = []
        for name, conductor_kind, poly in raw:
            pbb = poly.bbox()
            p_is_box = poly.is_box()
            preg = None
            owner: Optional[str] = None
            for idx, island_owner in enumerate(owners):
                if not pbb.overlaps(island_bbs[idx]):
                    continue
                if p_is_box and island_is_box[idx]:
                    owner = island_owner
                    break
                if preg is None:
                    preg = kdb.Region(poly)
                ireg = island_regs[idx]
                if ireg is None:
                    ireg = kdb.Region(islands[idx])
                    island_regs[idx] = ireg
                if not (preg & ireg).is_empty():
                    owner = island_owner
                    break
            terminal_matches = []
            for component_region, comp_bb, terminal_ref in local_components.get(name, ()):
                if not pbb.overlaps(comp_bb):
                    continue
                if preg is None:
                    preg = kdb.Region(poly)
                if not (preg & component_region).is_empty():
                    terminal_matches.append(terminal_ref)
            terminal_ref = (
                terminal_matches[0] if len(terminal_matches) == 1 else None
            )
            if p_is_box:
                out.append(Conductor(
                    pbb.left / _NM_PER_UM, pbb.bottom / _NM_PER_UM,
                    pbb.right / _NM_PER_UM, pbb.top / _NM_PER_UM,
                    m1, owner, kind=conductor_kind,
                    source_stamp=str(name),
                    terminal_conductor_ref=(
                        terminal_ref if owner is not None else None
                    )))
                continue
            if preg is None:
                preg = kdb.Region(poly)
            # Keep the KLayout iterator alive while its borrowed polygon handles
            # are inspected (materialising them into a Python list invalidates
            # those handles on some KLayout builds).
            for piece in preg.decompose_trapezoids():
                bb = piece.bbox()
                # Manhattan device M1 decomposes into rectangles.  For an
                # unexpected diagonal trapezoid, keep its bbox hard-unowned:
                # conservative safety without granting a guessed landing owner.
                piece_owner = owner if piece.area() == bb.area() else None
                out.append(Conductor(
                    bb.left / _NM_PER_UM, bb.bottom / _NM_PER_UM,
                    bb.right / _NM_PER_UM, bb.top / _NM_PER_UM,
                    m1, piece_owner, kind=conductor_kind,
                    source_stamp=str(name),
                    terminal_conductor_ref=(
                        terminal_ref if piece_owner is not None else None
                    )))
        return out

    pin_pts: List[Tuple[float, float, str]] = []
    for net, pins in pins_by_net.items():
        for p in pins:
            pin_pts.append((float(p.xy_um[0]), float(p.xy_um[1]), net))

    out: List[Conductor] = []
    for name, geom in geometry.items():
        pos = placement.get(name)
        if pos is None:
            continue
        cx, cy = float(pos[0]), float(pos[1])
        for ring in (getattr(geom, "polys_by_layer", {}) or {}).get(m1, []):
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            if not xs:
                continue
            x0 = min(xs) + cx
            y0 = min(ys) + cy
            x1 = max(xs) + cx
            y1 = max(ys) + cy
            owner: Optional[str] = None
            for (px, py, net) in pin_pts:
                if x0 - 1e-6 <= px <= x1 + 1e-6 and y0 - 1e-6 <= py <= y1 + 1e-6:
                    owner = net
                    break
            out.append(Conductor(x0, y0, x1, y1, m1, owner, kind="device",
                                 source_stamp=str(name)))
    return out


def device_active_projection_conductors(
    placement: Dict[str, _XY],
    geometry: Dict[str, Any],
    *,
    tech: LayoutTech,
) -> List[Conductor]:
    """Exact placed OD/poly footprint as nonblocking rectangle carriers.

    Layer identity comes only from :meth:`LayoutTech.build_layer`: diff/ndiff/
    pdiff aliases are deduplicated, then poly is included. Each source layer is
    unioned independently so the carrier retains its real active-layer key; the
    engine unions their intervals when scoring M2/M3. ``device_projection`` is
    metadata, never drawn conductor artwork or a hard keep-out.

    The current obstacle currency is rectangular. Generated device geometry is
    Manhattan, so KLayout's trapezoid decomposition yields exact boxes. A future
    non-Manhattan device is rejected under this opt-in path instead of silently
    receiving bbox coverage credit.
    """
    import klayout.db as kdb

    active_layers: List[_Layer] = []
    seen = set()
    for kind in ("diff", "ndiff", "pdiff", "poly"):
        try:
            key = tech.build_layer(kind)
        except KeyError:
            key = None
        if key is None:
            continue
        layer: _Layer = tuple(key)  # type: ignore[assignment]
        if layer not in seen:
            seen.add(layer)
            active_layers.append(layer)

    out: List[Conductor] = []
    for layer in active_layers:
        region = kdb.Region()
        for name, geom in geometry.items():
            pos = placement.get(name)
            if pos is None:
                continue
            cx_um = float(pos[0])
            cy_um = float(pos[1])
            rings = (getattr(geom, "polys_by_layer", {}) or {}).get(layer, ())
            for ring in rings:
                # Match the canonical full-extraction emitter: compose the
                # rigid transform before the one nm-currency quantisation.
                if len(ring) < 3:
                    continue
                pts = [kdb.Point(x, y) for x, y in placed_ring_nm(ring, (cx_um, cy_um))]
                poly = kdb.Polygon(pts)
                if poly.area() > 0:
                    region.insert(poly)
        region.merge()
        # Consume borrowed KLayout pieces inside the iterator; materialising
        # them first invalidates handles on some supported KLayout builds.
        for piece in region.decompose_trapezoids():
            bb = piece.bbox()
            if piece.area() != bb.area():
                raise ValueError(
                    "device active projection requires rectangular Manhattan "
                    f"geometry; non-rectangular piece on layer {layer}")
            out.append(Conductor(
                bb.left / _NM_PER_UM, bb.bottom / _NM_PER_UM,
                bb.right / _NM_PER_UM, bb.top / _NM_PER_UM,
                layer, None, kind="device_projection"))
    return out


def device_upper_conductors(
    intent: Any,
    placement: Dict[str, _XY],
    geometry: Dict[str, Any],
    *,
    tech: LayoutTech,
    pins_by_net: Dict[str, List[Any]],
) -> List[Conductor]:
    """Project literal above-M1 artwork using the shared actual extraction."""
    return list(_device_artwork_projection(
        intent, placement, geometry, tech=tech, pins_by_net=pins_by_net,
    ).upper_conductors)


def _device_artwork_projection(
    intent: Any,
    placement: Dict[str, _XY],
    geometry: Dict[str, Any],
    *,
    tech: LayoutTech,
    pins_by_net: Dict[str, List[Any]],
) -> _DeviceArtworkProjection:
    """Net-tagged conductors for stamp artwork ABOVE metal1 (riser pads / plate
    tabs) plus every DRAWN via cut of the stamp -- the R7-tail worldview fix.

    A passive stamp (MIM cap / resistor cell) draws a terminal RISER: pads on
    M2..M4 with via1..via3 cuts, EMITTED verbatim but historically INVISIBLE to
    the router (``build_layout_model`` projected M1 pads + plate levels only).
    The deck judges that artwork like any conductor: net-blind metal spacing
    (the measured M2.S.1/M3.S.1 same-net 232nm riser gaps) and net-blind cut
    spacing (the measured VIA2.S.1: a routed via2 170nm from a riser via2 the
    engine never saw).

    OWNERSHIP is a per-stamp MINI-EXTRACTION over the stamp's own drawn
    geometry, on REAL polygons (KLayout regions), never bboxes: same-layer
    rings merge into islands exactly as drawn (a notched bottom plate and the
    top-plate tab inside its bbox NOTCH stay two distinct islands -- a bbox
    union fused them and every component went honestly-unowned), a drawn cut
    unions the islands it overlaps under the full-extraction deck's connection
    semantics (including MIM dielectric carves), and components are seeded by
    the stamp's OWN pins (point-inside-island on the pin's layer). A
    component reachable from no pin -- or seeded by two different nets -- stays
    UNOWNED (hard keep-out for every net, never a guessed merge licence).

    EMISSION granularity: the Obstacle currency is a RECT.  Every ring uses the
    emitter's split nm rounding and exact Manhattan rectangle decomposition.  An
    unexpected non-rectangular trapezoid fails closed; its bbox is never credited
    as literal conductor artwork.

    SCOPE is the complete literal full-extraction deck declared by
    :class:`DeckConnectStack`, including conductors/cuts/markers above the fine
    routing graph.  Existing graph metal/cut ownership remains the advisory
    mini-extraction below; above-graph and marker literals remain honestly
    unowned until the deck-connected isolation view consumes them."""
    import klayout.db as kdb

    from clay.layout.route.deck_isolation import DeckConnectStack
    from clay.layout.route.engine_maze import _Tech

    _t = _Tech(tech)
    deck_stack = DeckConnectStack.from_tech(tech)
    deck_metal_keys = set(deck_stack.conductors)
    deck_cut_keys = {connection.cut for connection in deck_stack.connections}
    deck_marker_keys = {carve.top_marker for carve in deck_stack.mim_carves}
    line_end_contact_key: Optional[Tuple[int, int]] = None
    line_end_ext = tech.line_end_ext("metal1")
    if line_end_ext is not None:
        if type(line_end_ext) is not int or line_end_ext <= 0:
            raise ValueError("metal1 line-end extension must be a positive integer")
        raw_contact_key = tuple(tech.contact_key())
        if len(raw_contact_key) != 2 or any(
            type(value) is not int for value in raw_contact_key
        ):
            raise ValueError("metal1 line-end rule lacks a canonical contact layer")
        line_end_contact_key = raw_contact_key
    graph_levels = {_t._gi_to_pdk[gi] for gi in range(_t.n_layers)}
    metal_lv_by_key: Dict[Tuple[int, int], int] = {}
    for lv in sorted(graph_levels):
        try:
            metal_lv_by_key[tuple(tech.metal_key(lv))] = lv
        except Exception:
            continue
    via_lv_by_key: Dict[Tuple[int, int], int] = {}
    for lv in sorted(graph_levels):
        if (lv + 1) not in graph_levels:
            continue  # a cut to an above-graph metal: deck-owned connectivity
        try:
            vk = tech.via_key(lv)
        except Exception:
            continue
        if vk is not None:
            via_lv_by_key[tuple(vk)] = lv  # connects metal lv <-> lv+1

    pins_by_dev: Dict[
        str,
        List[Tuple[float, float, str, Optional[int], str]],
    ] = {}
    for net, pins in sorted((pins_by_net or {}).items()):
        for p in sorted(
            pins,
            key=lambda pin: (
                str(getattr(pin, "device", "")),
                str(getattr(pin, "terminal", "")),
                tuple(getattr(pin, "xy_um", ())),
            ),
        ):
            dev = str(getattr(p, "device", "") or "")
            pl = getattr(p, "layer", None)
            plv = metal_lv_by_key.get(tuple(pl)) if pl is not None else None
            terminal = str(getattr(p, "terminal", "") or "")
            pins_by_dev.setdefault(dev, []).append(
                (
                    float(p.xy_um[0]),
                    float(p.xy_um[1]),
                    str(net),
                    plv,
                    terminal,
                )
            )

    _NM = 1000.0

    def _exact_rects(poly: Any, layer: Tuple[int, int]) -> Tuple[Tuple[int, int, int, int], ...]:
        if poly.is_box():
            bbox = poly.bbox()
            return ((bbox.left, bbox.bottom, bbox.right, bbox.top),)
        rects = []
        for piece in kdb.Region(poly).decompose_trapezoids():
            bbox = piece.bbox()
            if piece.area() != bbox.area():
                raise ValueError(
                    "device upper projection requires rectangular Manhattan "
                    f"geometry; non-rectangular piece on layer {layer}"
                )
            rects.append((bbox.left, bbox.bottom, bbox.right, bbox.top))
        return tuple(sorted(rects))

    def _fast_ortho_rect_nm(
        pts: List[Tuple[int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        if len(pts) != 4:
            return None
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = pts
        if (
            (x0 == x1 and y1 == y2 and x2 == x3 and y3 == y0)
            or (y0 == y1 and x1 == x2 and y2 == y3 and x3 == x0)
        ) and x0 != x2 and y0 != y2:
            return (
                x0 if x0 < x2 else x2,
                y0 if y0 < y2 else y2,
                x2 if x0 < x2 else x0,
                y2 if y0 < y2 else y0,
            )
        return None

    _Point = kdb.Point
    _Polygon = kdb.Polygon
    _Region = kdb.Region
    _Box = kdb.Box

    out: List[Conductor] = []
    stack_m1_components: List[Tuple[str, Any, Optional[str], Optional[str], bool]] = []
    for name, geom in sorted((geometry or {}).items()):
        pos = placement.get(name)
        if pos is None:
            continue
        raw_composite_members = getattr(geom, "composite_members", ())
        composite_members = (
            raw_composite_members
            if type(raw_composite_members) is tuple
            and len(raw_composite_members) == len(set(raw_composite_members))
            and all(
                type(member) is str and member
                for member in raw_composite_members
            )
            else ()
        )
        # Shared-diffusion/common-centroid artwork is stamped under its group
        # name, while its electrical pins retain the original member-device
        # names.  Only the stamp's explicit composite-members certificate may
        # bridge that identity seam; coordinates and adjacency never invent an
        # alias.  This is the same ownership obligation used for device M1.
        stamp_pins = tuple(
            pin
            for device_name in sorted({str(name), *composite_members})
            for pin in pins_by_dev.get(device_name, ())
        )
        cx, cy = float(pos[0]), float(pos[1])
        pos_f = (cx, cy)
        # original rings (nm int polygons) per metal level + cut rects.
        ring_polys: Dict[int, List[Any]] = {}
        cuts: List[Tuple[int, Tuple[int, int], Tuple[int, int, int, int]]] = []
        literal_only: List[Tuple[Tuple[int, int], Tuple[int, int, int, int], str]] = []
        marker_polys: Dict[Tuple[int, int], List[Any]] = {}
        for key, rlist in sorted(
            (getattr(geom, "polys_by_layer", {}) or {}).items(),
            key=lambda item: (int(item[0][0]), int(item[0][1])),
        ):
            k = (int(key[0]), int(key[1]))
            mlv = metal_lv_by_key.get(k)
            vlv = via_lv_by_key.get(k)
            is_marker = k in deck_marker_keys
            if (
                k not in deck_metal_keys
                and k not in deck_cut_keys
                and not is_marker
                and k != line_end_contact_key
            ):
                continue
            if not is_marker and mlv is None:
                if vlv is not None:
                    for ring in rlist:
                        if len(ring) < 3:
                            continue
                        pts_nm = placed_ring_nm(ring, pos_f)
                        fast_rect = _fast_ortho_rect_nm(pts_nm)
                        if fast_rect is not None:
                            cuts.append((vlv, k, fast_rect))
                            continue
                        poly = _Polygon([_Point(x, y) for x, y in pts_nm])
                        if poly.area() <= 0:
                            continue
                        exact_rects = _exact_rects(poly, k)
                        cuts.extend((vlv, k, rect) for rect in exact_rects)
                else:
                    kind = (
                        "device_contact"
                        if k == line_end_contact_key
                        else "device_upper"
                        if k in deck_metal_keys
                        else "device_cut"
                        if k in deck_cut_keys
                        else "device_mim_marker"
                    )
                    for ring in rlist:
                        if len(ring) < 3:
                            continue
                        pts_nm = placed_ring_nm(ring, pos_f)
                        fast_rect = _fast_ortho_rect_nm(pts_nm)
                        if fast_rect is not None:
                            literal_only.append((k, fast_rect, kind))
                            continue
                        poly = _Polygon([_Point(x, y) for x, y in pts_nm])
                        if poly.area() <= 0:
                            continue
                        exact_rects = _exact_rects(poly, k)
                        literal_only.extend((k, rect, kind) for rect in exact_rects)
                continue
            for ring in rlist:
                if len(ring) < 3:
                    continue
                pts_nm = placed_ring_nm(ring, pos_f)
                poly = _Polygon([_Point(x, y) for x, y in pts_nm])
                if poly.area() <= 0:
                    continue
                if is_marker:
                    marker_polys.setdefault(k, []).append(poly)
                if mlv is not None:
                    ring_polys.setdefault(mlv, []).append(poly)
                elif vlv is not None:
                    exact_rects = _exact_rects(poly, k)
                    cuts.extend((vlv, k, rect) for rect in exact_rects)
                else:
                    exact_rects = _exact_rects(poly, k)
                    kind = (
                        "device_contact"
                        if k == line_end_contact_key
                        else "device_upper"
                        if k in deck_metal_keys
                        else "device_cut"
                        if k in deck_cut_keys
                        else "device_mim_marker"
                    )
                    literal_only.extend((k, rect, kind) for rect in exact_rects)
        if not cuts and not literal_only and all(lv <= 1 for lv in ring_polys):
            continue  # M1-only stamp (MOS): nothing new to project

        # ---- merged real-polygon islands per level -------------------------
        islands: List[Tuple[int, Any]] = []  # (level, kdb.Polygon)
        island_reg: List[Any] = []           # cached kdb.Region per island
        island_bbs: List[Any] = []
        island_is_box: List[bool] = []
        by_level: Dict[int, List[int]] = {}
        for lv, polys in sorted(ring_polys.items()):
            reg = _Region()
            for p in polys:
                reg.insert(p)
            reg.merge()
            for isl in _owned_merged_polygons(reg):
                idx = len(islands)
                islands.append((lv, isl))
                island_reg.append(_Region(isl))
                island_bbs.append(isl.bbox())
                island_is_box.append(isl.is_box())
                by_level.setdefault(lv, []).append(idx)

        parent = list(range(len(islands)))

        def _find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def _union(i: int, j: int) -> None:
            ri, rj = _find(i), _find(j)
            if ri != rj:
                parent[ri] = rj

        # The generic lower/cut/upper ladder is not the connectivity inside a MIM
        # dielectric.  Reuse the exact deck-declared carve already consumed by the
        # full-extraction isolation oracle: only the BOTTOM-metal complement may
        # reach the riser, while a cut in the carved plate region reaches the upper
        # cover/top marker without welding the overlapping bottom plate to it.
        #
        # Keyed entirely by deck layer facts.  A stamp without the declared marker
        # produces no carve region and retains the ordinary via behavior.
        blocked_lower_by_via: Dict[
            Tuple[int, Tuple[int, int]], Tuple[Any, Any]
        ] = {}
        for carve in deck_stack.mim_carves:
            bottom_lv = metal_lv_by_key.get(tuple(carve.bottom))
            markers = marker_polys.get(tuple(carve.top_marker), ())
            if bottom_lv is None or not markers:
                continue
            lower_region = _Region()
            for polygon in ring_polys.get(bottom_lv, ()):
                lower_region.insert(polygon)
            marker_region = _Region()
            for polygon in markers:
                marker_region.insert(polygon)
            lower_region.merge()
            marker_region.merge()
            if lower_region.is_empty() or marker_region.is_empty():
                continue
            interacting = lower_region.interacting(marker_region)
            if interacting.is_empty():
                continue
            oversize_nm = int(round(float(carve.oversize_um) * _NM))
            blocked = (
                marker_region & interacting
                if carve.marker_only else
                marker_region.sized(oversize_nm) & interacting
                if oversize_nm > 0
                else interacting
            )
            blocked.merge()
            if not blocked.is_empty():
                blocked_lower_by_via[(bottom_lv, tuple(carve.riser))] = (
                    blocked,
                    blocked.bbox(),
                )

        cut_isl: List[Optional[int]] = []
        for (vlv, cut_key, (x0, y0, x1, y1)) in cuts:
            cb = None
            first: Optional[int] = None
            for lv in (vlv, vlv + 1):
                lv_indices = by_level.get(lv, ())
                if not lv_indices:
                    continue
                blocked_entry = (
                    blocked_lower_by_via.get((vlv, cut_key))
                    if lv == vlv
                    else None
                )
                probe = None
                if blocked_entry is not None:
                    blocked_reg, bbb = blocked_entry
                    if x0 < bbb.right and bbb.left < x1 and y0 < bbb.top and bbb.bottom < y1:
                        if cb is None:
                            cb = _Region(_Box(x0, y0, x1, y1))
                        probe = cb - blocked_reg
                        if probe.is_empty():
                            continue
                for idx in lv_indices:
                    ibb = island_bbs[idx]
                    if not (x0 < ibb.right and ibb.left < x1 and y0 < ibb.top and ibb.bottom < y1):
                        continue
                    if probe is None and island_is_box[idx]:
                        hit = True
                    else:
                        if probe is None:
                            if cb is None:
                                cb = _Region(_Box(x0, y0, x1, y1))
                            probe = cb
                        hit = not (island_reg[idx] & probe).is_empty()
                    if not hit:
                        continue
                    if first is None:
                        first = idx
                    else:
                        _union(first, idx)
            cut_isl.append(first)

        comp_net: Dict[int, Optional[str]] = {}
        comp_terminals: Dict[int, set[str]] = {}
        for (
            px,
            py,
            net,
            plv,
            terminal,
        ) in stamp_pins:  # seed with certified own/member pins
            px_nm = int(round(px * _NM))
            py_nm = int(round(py * _NM))
            pt = None
            cand_indices = (
                by_level.get(plv, ()) if plv is not None else range(len(islands))
            )
            for idx in cand_indices:
                ibb = island_bbs[idx]
                if not (
                    ibb.left < px_nm + 1
                    and ibb.right > px_nm - 1
                    and ibb.bottom < py_nm + 1
                    and ibb.top > py_nm - 1
                ):
                    continue
                if not island_is_box[idx]:
                    if pt is None:
                        pt = _Region(_Box(px_nm - 1, py_nm - 1, px_nm + 1, py_nm + 1))
                    if (island_reg[idx] & pt).is_empty():
                        continue
                r = _find(idx)
                if r in comp_net and comp_net[r] not in (None, net):
                    comp_net[r] = None  # conflicting seeds: honest unowned
                    comp_terminals.pop(r, None)
                else:
                    comp_net.setdefault(r, net)
                    if comp_net[r] is not None and terminal:
                        comp_terminals.setdefault(r, set()).add(terminal)

        component_polygons: Dict[
            int,
            List[Tuple[int, Tuple[Tuple[int, int], ...]]],
        ] = {}
        for idx, (lv, island) in enumerate(islands):
            root = _find(idx)
            component_polygons.setdefault(root, []).append((
                int(lv),
                tuple(
                    (int(point.x), int(point.y))
                    for point in island.each_point_hull()
                ),
            ))
        component_terminal_refs = {
            root: _upper_terminal_conductor_ref(
                str(name),
                tuple(sorted(comp_terminals[root])),
                str(net),
                component_polygons.get(root, ()),
            )
            for root, net in comp_net.items()
            if net is not None and comp_terminals.get(root)
        }

        def _owner_of_island(idx: Optional[int]) -> Optional[str]:
            return comp_net.get(_find(idx)) if idx is not None else None

        def _terminal_ref_of_island(idx: Optional[int]) -> Optional[str]:
            return (
                component_terminal_refs.get(_find(idx))
                if idx is not None
                else None
            )

        upper_roots = {_find(idx) for idx, (lv, _island) in enumerate(islands) if lv > 1}
        for idx in by_level.get(1, ()):
            if _find(idx) in upper_roots:
                stack_m1_components.append((
                    str(name), island_reg[idx].dup(),
                    _owner_of_island(idx), _terminal_ref_of_island(idx),
                    _find(idx) in comp_net and comp_net[_find(idx)] is None))

        # emit RECTANGULAR original rings on levels >= 2, owner = island's net.
        for lv, polys in sorted(ring_polys.items()):
            if lv <= 1:
                continue  # M1 uses the shared ownership with its global conflict check.
            key = tuple(tech.metal_key(lv))
            lv_indices = by_level.get(lv, ())
            for p in polys:
                pbb = p.bbox()
                p_is_box = p.is_box()
                probe = None
                owner: Optional[str] = None
                owner_island: Optional[int] = None
                for idx in lv_indices:
                    if not pbb.overlaps(island_bbs[idx]):
                        continue
                    if p_is_box and island_is_box[idx]:
                        owner = _owner_of_island(idx)
                        owner_island = idx
                        break
                    if probe is None:
                        probe = _Region(p)
                    if not (island_reg[idx] & probe).is_empty():
                        owner = _owner_of_island(idx)
                        owner_island = idx
                        break
                tref = _terminal_ref_of_island(owner_island)
                for x0, y0, x1, y1 in _exact_rects(p, key):
                    out.append(Conductor(
                        x0 / _NM, y0 / _NM, x1 / _NM, y1 / _NM,
                        key, owner, kind="device_upper",
                        source_stamp=str(name),
                        terminal_conductor_ref=tref))  # type: ignore[arg-type]
        for (vlv, k, (x0, y0, x1, y1)), first in zip(cuts, cut_isl):
            out.append(Conductor(
                x0 / _NM, y0 / _NM, x1 / _NM, y1 / _NM,
                k, _owner_of_island(first), kind="device_cut",
                source_stamp=str(name)))  # type: ignore[arg-type]
        for k, (x0, y0, x1, y1), kind in literal_only:
            out.append(Conductor(
                x0 / _NM, y0 / _NM, x1 / _NM, y1 / _NM,
                k, None, kind=kind, source_stamp=str(name)))
    return _DeviceArtworkProjection(tuple(sorted(
        out,
        key=lambda conductor: (
            tuple(conductor.layer),
            round(conductor.x0 * _NM), round(conductor.y0 * _NM),
            round(conductor.x1 * _NM), round(conductor.y1 * _NM),
            conductor.net is not None,
            "" if conductor.net is None else conductor.net,
            conductor.kind,
        ),
    )), tuple(stack_m1_components))


def _cap_is_over_cell(intent: Any, cap_name: str, placement: Dict[str, _XY],
                      geometry: Dict[str, Any], *, tech: Any = None) -> bool:
    """True iff the SOLVED placement z-stacks ``cap_name`` OVER any device's active
    diffusion (the cap footprint overlaps a non-cap device's OD bbox). Pure geometry
    of Clay's OWN placement -- no reference coordinate, no intent-entry flag. This is
    the GENERIC condition under which a cap's plate column carries foreign
    device-region routing and therefore needs the plate-level keep-out.

    The UNIFIED all-element placement packs the cap as a no-overlap TILE, so its
    footprint never z-stacks over a device -> this returns False on the production
    path (the keep-out stays a safety net for any future over-cell arrangement,
    measured directly from geometry, with no dependency on a retired cap-mode
    intent)."""
    from clay.layout.extract.under_cap_intent import cap_footprints_um
    feet = cap_footprints_um(intent, placement, geometry or {}, tech=tech)
    cbb = feet.get(cap_name)
    if cbb is None:
        return False
    cap_names = {str(p.name) for p in (getattr(intent, "passives", []) or [])
                 if str(getattr(p, "kind", "")).lower().startswith("c")}
    diffusion_keys = (
        {tuple(key) for kind in ("ndiff", "pdiff")
         if (key := tech.build_layer(kind)) is not None}
        if tech is not None else {(3, 0)}
    )
    for name, g in (geometry or {}).items():
        if name in cap_names:
            continue
        pos = placement.get(name)
        if pos is None:
            continue
        for lay, rings in (getattr(g, "polys_by_layer", {}) or {}).items():
            if tuple(lay) not in diffusion_keys:
                continue
            for ring in rings:
                xs = [pos[0] + pt[0] for pt in ring]
                ys = [pos[1] + pt[1] for pt in ring]
                if not xs:
                    continue
                ox = min(cbb[2], max(xs)) - max(cbb[0], min(xs))
                oy = min(cbb[3], max(ys)) - max(cbb[1], min(ys))
                if ox > 0.0 and oy > 0.0:
                    return True
    return False


def _overcell_native_plate_guard(*, cell, nodes, access, mim_spec, declared_nets,
                                layer_key, owner, footprint_nm, spacing_nm,
                                terminal_layers=None):
    """Keep the column guard except at proven other-node native metal islands.

    A whole-footprint ownership label is not electrical authority for a separate
    plate's real exit on the same layer. Only the extracted node, bound back to
    the actual native terminal and declared net, may carve its own metal plus
    one native spacing halo. Actual marker/plate copper is never removed.
    """
    import klayout.db as kdb
    from clay.layout.passive_terminal_binding import _node_for_box, _probe
    from clay.layout.device_terminal import _node_index_for_box
    if type(spacing_nm) is not int or spacing_nm < 0:
        raise ValueError("native MIM spacing must be exact nonnegative nm")
    terms = cell.terminals or {}
    if set(access) != set(terms) or not access:
        raise ValueError("native MIM terminal inventory is incomplete")
    node_nets = {}
    real_nodes = []
    for node in nodes:
        real = {}
        for raw_key, region in node.items():
            key = mim_spec.synthetic_to_real.get(raw_key, raw_key)
            real.setdefault(key, kdb.Region()).insert(region)
        real_nodes.append(real)
    for role, item in access.items():
        term = terms[role]
        expected = declared_nets.get(role)
        # Retained bank terminals already carry their verified physical layer.
        # Native scalar cells retain a source pad bbox and a separate landing.
        # Never flatten the former bank's overlapping plate projections.
        source_node = (_node_for_box(real_nodes, terminal_layers[role], term.bbox)
                       if terminal_layers is not None else
                       _node_index_for_box(nodes, term.bbox))
        point = tuple(int(round(value * _NM_PER_UM)) for value in item.xy_um)
        if (not expected or item.role != role or item.net != expected or term.net != expected
                or type(item.node_id) is not int or not 0 <= item.node_id < len(nodes)
                or source_node != item.node_id
                or _node_for_box(real_nodes, item.layer, _probe(point)) != item.node_id
                or item.node_id in node_nets):
            raise ValueError("native MIM node/net does not bind its actual source terminal")
        node_nets[item.node_id] = expected
    if owner not in node_nets.values():
        raise ValueError("native MIM plate owner is not source-bound")
    actual = kdb.Region()
    marker = kdb.Region()
    routing = set(mim_spec.metal_order)
    derived_names = {row.name for row in mim_spec.derived}
    cuts = {tuple(row[0]) for row in mim_spec.vias}
    marker_keys = set(mim_spec.metals) - routing - derived_names
    for derived in mim_spec.derived:
        for operation in derived.ops:
            marker_keys.update(tuple(operand) for operand in operation[1:]
                if isinstance(operand, tuple) and len(operand) == 2
                and all(type(value) is int for value in operand))
    marker_keys -= routing | derived_names | cuts
    for polygon in cell.polygons:
        key = (int(polygon.layer), int(polygon.datatype))
        if key != layer_key and key not in marker_keys:
            continue
        shape = kdb.Polygon([kdb.Point(int(p.x), int(p.y)) for p in polygon.polygon.vertices])
        if key == layer_key:
            actual.insert(shape)
        if key in marker_keys:
            marker.insert(shape)
    actual.merge()
    marker.merge()
    own, foreign = kdb.Region(), kdb.Region()
    marker_owners = set()
    for node_id, node in enumerate(nodes):
        metal = kdb.Region()
        for raw_key, region in node.items():
            if mim_spec.synthetic_to_real.get(raw_key, raw_key) == layer_key:
                metal += region
        # Derived dielectric/marker nodes are not invented routable copper.
        metal &= actual
        net = node_nets.get(node_id)
        if not (metal & marker).is_empty():
            marker_owners.add(net)
        if net == owner:
            own += metal
        elif net is not None:
            foreign += metal
    if marker_owners and marker_owners != {owner}:
        raise ValueError("native MIM physical marker proves a different plate owner")
    guard = kdb.Region(kdb.Box(*footprint_nm)).sized(spacing_nm)
    if not foreign.is_empty():
        anchor = own + marker
        if not anchor.is_empty():
            # Do not project whole-footprint margins caused by another node's
            # exit bus into phantom guards beyond the owner's plate/marker box.
            core = kdb.Region(kdb.Box(*footprint_nm)) & kdb.Region(anchor.bbox())
            guard = core.sized(spacing_nm)
        # The router still applies the native spacing to this guard; this halo
        # permits a landing contained in the real island, not arbitrary metal
        # somewhere inside the capacitor footprint. Literal masks stay hard.
        window = foreign.sized(spacing_nm) - own - marker
        guard -= window
    return guard.merged()


def _materialized_cap_plate_conductors(passive, stamp, position, *, tech,
                                       column_guards):
    """Project the retained native bank and nodes in actual emit coordinates.

    Scalar C/M/orient cannot reconstruct a selected mixed-orientation bank.
    The source-bound passive materialization already retains every polygon,
    unit occurrence and original role. Its independent replay is mandatory.
    """
    from types import SimpleNamespace
    import klayout.db as kdb
    from clay.layout.backend import LayerPolygon, Point, Polygon
    from clay.layout.device_cell import Terminal
    from clay.layout.device_terminal import TerminalAccess
    from clay.layout.passive_terminal_binding import (
        _spec, _stamp_rows, verified_passive_route_accesses)
    facts, nodes = verified_passive_route_accesses(
        stamp, passive=passive, position=position, tech=tech)
    binding = stamp.materialized_passive_access
    spec = _spec(binding, tech)
    rows = _stamp_rows(stamp, position)
    polygons = [LayerPolygon(layer, datatype, Polygon([Point(x, y) for x,y in ring]))
                for layer, datatype, ring in rows]
    terms, access, node_nets = {}, {}, {}
    for role, net, layer, (x, y), node, _index in facts:
        if role in access:
            if access[role].node_id != node or access[role].net != net:
                raise ValueError("passive repeated native role has inconsistent nodes")
            continue
        terms[role] = Terminal(role, (x, y, x+1, y+1), net)
        access[role] = TerminalAccess(role, net, layer, (x/1000, y/1000), node)
        node_nets[node] = net
    cell = SimpleNamespace(polygons=polygons, terminals=terms)
    footprint = (min(x for _,_,ring in rows for x,y in ring),
                 min(y for _,_,ring in rows for x,y in ring),
                 max(x for _,_,ring in rows for x,y in ring),
                 max(y for _,_,ring in rows for x,y in ring))
    levels = {}
    for level in range(1, 12):
        try:
            levels[tuple(tech.metal_key(level))] = level
        except Exception:
            continue
    actual = {}
    for layer, datatype, ring in rows:
        actual.setdefault((layer, datatype), kdb.Region()).insert(
            kdb.Polygon([kdb.Point(x, y) for x,y in ring]))
    for region in actual.values():
        region.merge()
    out = []
    def emit(region, key, net, kind='cap_plate'):
        for piece in region.merged().decompose_trapezoids():
            box = piece.bbox()
            if piece.area() != box.area():
                raise ValueError("native MIM guard requires exact rectangular decomposition")
            out.append(Conductor(box.left/1000, box.bottom/1000,
                box.right/1000, box.top/1000, key, net, kind=kind,
                source_stamp=str(passive.name)))
    def metal(node, key):
        result = kdb.Region()
        for raw_key, region in node.items():
            if spec.synthetic_to_real.get(raw_key, raw_key) == key:
                result += region
        return result & actual.get(key, kdb.Region())
    # Every node's actual access-level metal, including all parallel units and
    # native buses. Unlike one-point seeding, a connected second unit is owned.
    for key in sorted({item.layer for item in access.values()}):
        spacing = int(tech.min_space(f'metal{levels[key]}') or 0)
        claimed = kdb.Region()
        for node_id, node in enumerate(nodes):
            region = metal(node, key)
            if region.is_empty():
                continue
            emit(region.sized(spacing), key, node_nets.get(node_id))
            claimed += region
        emit((actual.get(key, kdb.Region()) - claimed).sized(spacing), key, None)
    if not column_guards:
        return out
    physical_keys = tuple(spec.metal_order[-2:])
    if len(physical_keys) != 2:
        raise ValueError("native MIM recipe lacks two physical plate routing layers")
    routing, derived = set(spec.metal_order), {row.name for row in spec.derived}
    cuts = {tuple(row[0]) for row in spec.vias}
    marker_keys = set(spec.metals) - routing - derived
    for item in spec.derived:
        for operation in item.ops:
            marker_keys.update(tuple(value) for value in operation[1:]
                if isinstance(value, tuple) and len(value) == 2
                and all(type(v) is int for v in value))
    marker_keys -= routing | derived | cuts
    marker = kdb.Region()
    for key in marker_keys:
        marker += actual.get(key, kdb.Region())
    if marker.is_empty():
        raise ValueError("native MIM lacks physical plate marker")
    for key in physical_keys:
        owners = {node_nets.get(index) for index,node in enumerate(nodes)
                  if not (metal(node, key) & marker).is_empty()}
        if len(owners) != 1 or None in owners:
            raise ValueError("native MIM physical plate lacks one marker-proven owner")
        owner = next(iter(owners))
        guard = _overcell_native_plate_guard(cell=cell, nodes=nodes, access=access,
            mim_spec=spec, declared_nets=dict(passive.nets), layer_key=key,
            owner=owner, footprint_nm=footprint,
            terminal_layers={role: item.layer for role, item in access.items()},
            spacing_nm=int(tech.min_space(f'metal{levels[key]}') or 0))
        emit(guard, key, owner)
    for level in range(min(levels[key] for key in physical_keys)-1,
                       max(levels[key] for key in physical_keys)):
        try:
            key = tuple(tech.via_key(level))
        except Exception:
            continue
        emit(kdb.Region(kdb.Box(*footprint)), key, None, 'cap_plate_via')
    return out


def cap_plate_conductors(
    intent: Any,
    placement: Dict[str, _XY],
    *,
    tech: LayoutTech,
    geometry: Optional[Dict[str, Any]] = None,
    reserve_column_guards: bool = False,
) -> List[Conductor]:
    """Net-tagged cap-plate keep-out conductors (bloated by the layer min-spacing)
    so a foreign net routes AROUND a plate while the plate's own net may land on it.

    Access levels are derived GENERICALLY from the cap device cell's materialized
    terminal pins, NOT a hardcoded M2/M3.  Ownership is assigned per connected
    same-layer component, so two plate accesses may share one routing layer without
    one net overwriting the other (the old ``owner_by_layer`` defect).  This also
    closes the removed ``metal_key(2)/metal_key(3)`` gap that left an over-cell
    M5/M6 MIM plate with ZERO keep-out, so the router dropped a foreign riser into
    the plate column -> the t4 'GND,VOUT' merge).

    A retained MaterializedPassiveAccess is authoritative for selected native
    banks: its exact transformed polygons and verified role nodes are consumed
    without rebuilding C/M defaults. ``reserve_column_guards`` reserves those
    same column operands before placement, so a later over-cell overlap cannot
    add conductors absent from the source/closed-placement inventory.

    For an OVER-CELL cap (footprint z-stacked over its node devices, so foreign
    device-region routing shares the plate column), this additionally emits HARD
    keep-out conductors (``net=None`` -> off-limits to EVERY net) on the
    INTER-PLATE via levels over the footprint, so no foreign net can climb a
    via riser through one plate's column into the other plate. The plate's OWN net
    reaches its plate via the cap cell's access port, not a router riser through
    the column.

    For a top-slab parked cap (the production default, no over-cell overlap) only
    the materialized access-level keep-outs are emitted; literal upper artwork is
    projected independently by :func:`device_upper_conductors`."""
    from clay.layout.route.pins import (
        _geom_extent_center_nm, _mim_spec_for, _mim_spec_for_passive, _passive_cap_pins)
    import clay.layout.passive_device_cell as _pdc
    import klayout.db as kdb

    spec = _mim_spec_for(tech)
    if spec is None:
        return []

    def _space_um(key: _Layer) -> float:
        for lv in range(1, 12):
            try:
                if tuple(tech.metal_key(lv)) == tuple(key):
                    s = tech.min_space(f"metal{lv}")
                    return (float(s) / _NM_PER_UM) if s else 0.0
            except Exception:
                continue
        return 0.0

    out: List[Conductor] = []
    for p in getattr(intent, "passives", []) or []:
        if getattr(p, "kind", "") not in ("cap", "c"):
            continue
        name = str(getattr(p, "name", ""))
        pos = placement.get(name)
        if pos is None:
            continue
        cx, cy = float(pos[0]), float(pos[1])
        stamp = (geometry or {}).get(name)
        if getattr(stamp, 'materialized_passive_access', None) is not None:
            out.extend(_materialized_cap_plate_conductors(
                p, stamp, pos, tech=tech,
                column_guards=(reserve_column_guards or _cap_is_over_cell(
                    intent, name, placement, geometry or {}, tech=tech))))
            continue
        try:
            cell = _pdc.build_capacitor_for_tech(
                tech, float(p.value), nets=getattr(p, "nets", {}),
                m=int(getattr(p, "M", 1) or 1),
                model=str(getattr(p, "model", "") or ""),
                orient=str(getattr(p, "orient", "R0") or "R0").upper(),
            )
            per_term, pin_gaps = _passive_cap_pins(p, tech)
        except Exception:
            continue  # no cap builder for this PDK -> caps stay unrouted (honest gap)
        if per_term is None or pin_gaps:
            continue  # no proven pair of distinct physical plate accesses
        ccx_nm, ccy_nm = _geom_extent_center_nm(cell)
        ccx = ccx_nm / _NM_PER_UM
        ccy = ccy_nm / _NM_PER_UM
        # GENERIC plate levels: the cap's OWN materialized access layers, never an
        # M2/M3 constant.  Ownership is PER CONNECTED COMPONENT, not per layer:
        # two electrically distinct plate risers may both terminate on M1.  A
        # ``layer -> net`` dict would overwrite one and turn its real terminal into
        # a foreign obstacle (the external_pdk CCA.n/VOUT NO_LEGAL_ROUTE regression).
        seeds_by_layer: Dict[_Layer, List[Tuple[int, int, str]]] = {}
        for _role, (rel, layer, net) in per_term.items():
            seeds_by_layer.setdefault(tuple(layer), []).append((
                int(round(float(rel[0]) * _NM_PER_UM + ccx_nm)),
                int(round(float(rel[1]) * _NM_PER_UM + ccy_nm)),
                str(net),
            ))
        plate_keys = set(seeds_by_layer)

        components_by_layer: Dict[
            _Layer, List[Tuple[Any, Optional[str]]]
        ] = {}
        for key in sorted(plate_keys):
            region = kdb.Region()
            for lp in getattr(cell, "polygons", []) or []:
                if (int(lp.layer), int(lp.datatype)) != key:
                    continue
                region.insert(kdb.Polygon([
                    kdb.Point(int(vertex.x), int(vertex.y))
                    for vertex in lp.polygon.vertices
                ]))
            region.merge()
            compiled = []
            for component in region.each_merged():
                component_region = kdb.Region(component)
                nets = set()
                for sx, sy, net in seeds_by_layer.get(key, ()):
                    probe = kdb.Region(kdb.Box(sx - 1, sy - 1, sx + 1, sy + 1))
                    if not (component_region & probe).is_empty():
                        nets.add(net)
                compiled.append((
                    component_region,
                    next(iter(nets)) if len(nets) == 1 else None,
                ))
            components_by_layer[key] = compiled
        # per-cap footprint extent (nm->um, placement-centred) for the via keep-out
        fx0 = fy0 = 1e18
        fx1 = fy1 = -1e18
        for lp in getattr(cell, "polygons", []) or []:
            key = (int(lp.layer), int(lp.datatype))
            if key not in plate_keys:
                continue
            xs = [v.x for v in lp.polygon.vertices]
            ys = [v.y for v in lp.polygon.vertices]
            if not xs:
                continue
            b = _space_um(key)
            x0 = min(xs) / _NM_PER_UM - ccx + cx - b
            y0 = min(ys) / _NM_PER_UM - ccy + cy - b
            x1 = max(xs) / _NM_PER_UM - ccx + cx + b
            y1 = max(ys) / _NM_PER_UM - ccy + cy + b
            polygon_region = kdb.Region(kdb.Polygon([
                kdb.Point(int(vertex.x), int(vertex.y))
                for vertex in lp.polygon.vertices
            ]))
            component_owners = {
                owner
                for component_region, owner in components_by_layer.get(key, ())
                if not (polygon_region & component_region).is_empty()
            }
            owner = (
                next(iter(component_owners))
                if len(component_owners) == 1
                else None
            )
            out.append(Conductor(x0, y0, x1, y1, key,
                                 owner, kind="cap_plate", source_stamp=name))
            fx0 = min(fx0, x0); fy0 = min(fy0, y0)
            fx1 = max(fx1, x1); fy1 = max(fy1, y1)
        # OVER-CELL: a z-stacked cap's WHOLE footprint is its electrical body --
        # the bottom plate + its via-array (e.g. the 2178-cut VIA5 MIM stack)
        # extend beyond the bottom-plate POLYGON to the cap footprint edge. A
        # foreign net dropping plate-level metal or a via cut ANYWHERE in that
        # footprint climbs into the plate column (the canonical t4 breach: a VOUT
        # M5 stub at the footprint edge, OUTSIDE the M5 polygon but over the VIA5
        # array, rose into net25's plate). So over the FULL footprint emit:
        #   (i) each plate LEVEL as a keep-out OWNED BY that plate's net (the
        #       plate's own net may land; every OTHER net is kept out), and
        #   (ii) each inter-plate VIA level as a HARD keep-out (net=None) so no
        #       net punches a riser cut through the column.
        # Only when the cap is z-stacked over device cells (its column carries
        # foreign routing); a parked top-slab cap emits none (default byte-id).
        # Plate + via levels derived from the cap's access layers via the tech
        # metal/via map -- no hardcoded layer number.
        if fx1 > fx0 and (reserve_column_guards or _cap_is_over_cell(
                intent, name, placement, geometry or {}, tech=tech)):
            from clay.layout.extract.under_cap_intent import cap_footprints_um
            foot = cap_footprints_um(intent, placement, geometry or {},
                                     tech=tech).get(name)
            if foot is not None:
                gx0, gy0, gx1, gy1 = foot
                # Access tips can be internal down-risers on M1/M2. They do not
                # make the entire high-stack MIM footprint a low-metal plate.
                # The native connect recipe orders its ladder up to the actual
                # bottom plate and top cover. Select those last two real metals,
                # then prove their ownership through the real marker/node graph.
                from clay.layout.device_terminal import (
                    extract_device_nodes, extract_terminal_access)
                body_spec = _mim_spec_for_passive(tech, p)
                physical_plate_keys = tuple(body_spec.metal_order[-2:])
                if len(physical_plate_keys) != 2:
                    raise ValueError("MIM recipe lacks two physical plate routing layers")
                nodes = extract_device_nodes(cell, tech=tech, mim_spec=body_spec)
                access, report = extract_terminal_access(cell, tech=tech, mim_spec=body_spec)
                if report.collapsed or report.unmapped:
                    raise ValueError("over-cell MIM physical plate nodes are not distinct")
                node_nets = {item.node_id: item.net for item in access.values()}
                routing_keys = set(body_spec.metal_order)
                derived_keys = {item.name for item in body_spec.derived}
                cut_keys = {tuple(rule[0]) for rule in body_spec.vias}
                marker_keys = set(body_spec.metals) - routing_keys - derived_keys
                for derived in body_spec.derived:
                    for operation in derived.ops:
                        marker_keys.update(
                            tuple(operand) for operand in operation[1:]
                            if isinstance(operand, tuple) and len(operand) == 2
                            and all(type(value) is int for value in operand))
                marker_keys -= routing_keys | derived_keys | cut_keys
                marker_region = kdb.Region()
                for polygon in cell.polygons:
                    if (int(polygon.layer), int(polygon.datatype)) in marker_keys:
                        marker_region.insert(kdb.Polygon([
                            kdb.Point(int(vertex.x), int(vertex.y))
                            for vertex in polygon.polygon.vertices]))
                if marker_region.is_empty():
                    raise ValueError("over-cell MIM lacks native physical plate marker")
                physical_owners = {}
                for key in physical_plate_keys:
                    owners = set()
                    for node_id, node in enumerate(nodes):
                        metal = kdb.Region()
                        for raw_key, region in node.items():
                            if body_spec.synthetic_to_real.get(raw_key, raw_key) == key:
                                metal += region
                        if not (metal & marker_region).is_empty():
                            owners.add(node_nets.get(node_id))
                    if len(owners) != 1 or None in owners:
                        raise ValueError("MIM physical plate lacks one marker-proven owner")
                    physical_owners[key] = next(iter(owners))
                plate_levels: List[int] = []
                for key in physical_plate_keys:
                    b = _space_um(key)
                    layer_owner = physical_owners[key]
                    from clay.layout.placed_geometry import placed_coord_nm
                    ox = placed_coord_nm(-ccx, cx)
                    oy = placed_coord_nm(-ccy, cy)
                    footprint_nm = (placed_coord_nm(gx0, 0.) - ox,
                                    placed_coord_nm(gy0, 0.) - oy,
                                    placed_coord_nm(gx1, 0.) - ox,
                                    placed_coord_nm(gy1, 0.) - oy)
                    guard = _overcell_native_plate_guard(
                        cell=cell, nodes=nodes, access=access, mim_spec=body_spec,
                        declared_nets=getattr(p, "nets", {}), layer_key=key, owner=layer_owner,
                        footprint_nm=footprint_nm, spacing_nm=int(round(b * _NM_PER_UM)))
                    for piece in guard.decompose_trapezoids():
                        box = piece.bbox()
                        if piece.area() != box.area():
                            raise ValueError("native MIM guard requires exact rectangular decomposition")
                        out.append(Conductor((box.left + ox) / _NM_PER_UM,
                                             (box.bottom + oy) / _NM_PER_UM,
                                             (box.right + ox) / _NM_PER_UM,
                                             (box.top + oy) / _NM_PER_UM, key, layer_owner,
                                             kind="cap_plate", source_stamp=name))
                    for lv in range(1, 12):
                        try:
                            if tuple(tech.metal_key(lv)) == tuple(key):
                                plate_levels.append(lv)
                        except Exception:
                            break
                if plate_levels:
                    for lv in range(min(plate_levels) - 1, max(plate_levels)):
                        try:
                            vk = tuple(tech.via_key(lv))
                        except Exception:
                            continue
                        out.append(Conductor(gx0, gy0, gx1, gy1, vk, None,
                                             kind="cap_plate_via", source_stamp=name))
    # These are physical-coordinate routing operands, not unquantized display
    # floats. A half-nm centroid followed by Python's half-even round changes a
    # rigid keep-out's width after an odd-nm move. Normalize at construction,
    # using the same translation-invariant currency as actual placed stamps.
    from dataclasses import replace
    from clay.layout.placed_geometry import placed_coord_nm
    return [replace(row,
        x0=placed_coord_nm(row.x0, 0.0) / _NM_PER_UM,
        y0=placed_coord_nm(row.y0, 0.0) / _NM_PER_UM,
        x1=placed_coord_nm(row.x1, 0.0) / _NM_PER_UM,
        y1=placed_coord_nm(row.y1, 0.0) / _NM_PER_UM) for row in out]


def build_layout_model(
    intent: Any,
    placement: Dict[str, _XY],
    geometry: Dict[str, Any],
    *,
    tech: LayoutTech,
    pins_by_net: Dict[str, List[Any]],
    include_upper_artwork: bool = False,
    reserve_column_guards: Optional[bool] = None,
    strict_device_m1_provenance: bool = False,
    include_device_projection: bool = False,
) -> LayoutModel:
    """Assemble the full net-tagged LayoutModel: device M1 pads + cap plates +
    the LOWERED ``@no_route_under`` intent (hard keep-outs over cap footprints).

    ``include_upper_artwork`` (opt-in, default False -> byte-identical model)
    additionally projects every stamp's ABOVE-M1 artwork (riser pads) and DRAWN
    via cuts as net-tagged conductors (:func:`device_upper_conductors`) -- the
    R7-tail worldview fix. Keyed by the routing engine's declared capability
    (``consumes_artwork_conductors``), so the default PathFinder inputs are
    untouched. This full-artwork mode reserves native MIM column guards already
    in the source model, independently of the mutable current overlap status."""
    artwork = (
        _device_artwork_projection(
            intent, placement, geometry, tech=tech, pins_by_net=pins_by_net)
        if include_upper_artwork or strict_device_m1_provenance else None
    )
    conductors: List[Conductor] = []
    conductors.extend(_device_m1_conductors(
        intent, placement, geometry, tech=tech, pins_by_net=pins_by_net,
        strict_device_m1_provenance=strict_device_m1_provenance, artwork=artwork))
    column_guards = (
        include_upper_artwork
        if reserve_column_guards is None
        else bool(reserve_column_guards)
    )
    conductors.extend(cap_plate_conductors(intent, placement, tech=tech,
        geometry=geometry, reserve_column_guards=column_guards))
    conductors.extend(no_route_under_conductors(intent, placement, geometry,
                                                tech=tech))
    if include_upper_artwork:
        assert artwork is not None
        conductors.extend(artwork.upper_conductors)
    if include_device_projection:
        conductors.extend(device_active_projection_conductors(
            placement, geometry, tech=tech))
    return LayoutModel(conductors=conductors)


def no_route_under_conductors(
    intent: Any,
    placement: Dict[str, _XY],
    geometry: Dict[str, Any],
    *,
    tech: LayoutTech,
) -> List[Conductor]:
    """LOWER the ``@no_route_under`` intent into hard keep-out conductors.

    For every intent entry on ``intent.no_route_under`` and every matching cap,
    emit one unowned (``net=None`` -> hard keep-out for ALL nets) conductor per
    forbidden metal LEVEL covering the cap's placed footprint. The intent is
    declarable or style-extracted (``clay.layout.extract.under_cap_intent``);
    this function is pure lowering -- the model/router carries NO opinion of its
    own about routing under MIMs, and an intent-free run is byte-identical.
    Obligation (verified independently): zero foreign metal of those levels
    inside the footprints on the routed artwork."""
    entries = getattr(intent, "no_route_under", []) or []
    if not entries:
        return []
    from clay.layout.extract.under_cap_intent import (
        cap_footprints_um, matching_entries,
    )

    feet = cap_footprints_um(intent, placement, geometry, tech=tech)
    out: List[Conductor] = []
    for name, (x0, y0, x1, y1) in feet.items():
        levels: set = set()
        for e in matching_entries(intent, name):
            levels.update(int(v) for v in (e.get("forbidden_metals") or ()))
        for level in sorted(levels):
            try:
                mk: _Layer = tuple(tech.metal_key(level))  # type: ignore[assignment]
            except Exception:
                continue  # PDK has no such metal -> vacuous for this level
            out.append(Conductor(x0=x0, y0=y0, x1=x1, y1=y1, layer=mk, net=None,
                                 kind="no_route_under", source_stamp=str(name)))
    return out
