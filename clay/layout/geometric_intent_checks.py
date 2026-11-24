"""DIRECT, DETERMINISTIC geometric CHECK obligations for structural layout intent.

The placer/router is an UNTRUSTED heuristic generator: correctness comes from
mechanical CHECK obligations measured on the REAL generated layout, not inferred
from downstream DRC/LVS symptoms. Where :mod:`clay.layout.intent_obligations`
turns the NET-level parasitic budgets into obligations against a
:class:`~clay.layout.pex.base.PexReport`, and the LDE budgets into obligations
against per-device well/OD geometry, THIS module turns the STRUCTURAL placement
intents (``@interdigitated`` / common-centroid / ``@place_group``) into
obligations measured DIRECTLY on the placement / device-cluster GEOMETRY:

  * ``interdig``             -- an ``@interdigitated(A, B)`` pair is laid out as
    ONE interleaved common-centroid array whose A/B fingers actually ALTERNATE
    (not two side-by-side blocks), whose two sub-centroids COINCIDE (relative
    tol), and whose S/D columns wire to the correct schematic net per finger.
  * ``common_centroid``      -- a ``@common_centroid`` group / ``@matching`` pair
    whose member devices' centroids actually COINCIDE on the real placement
    (a DIRECT centroid measurement, NOT the MILP's y-balance proxy).
  * ``place_group_realized`` -- a ``@place_group(g, devices, row=, col=)`` whose
    devices are ACTUALLY placed in the declared row/col band on the real
    placement (their centroids fall in the right row/col region).

Every measurement is mechanical geometry math: NO LLM, NO per-circuit branching,
NO downstream-symptom proxy. Every imbalance is reduced to a RELATIVE fraction
(project-wide rule: never absolute dbu deltas), so scaling the whole layout
cannot change a verdict. A check that cannot be measured (missing cluster
provenance, an absent device, a missing placement) returns ``None`` == an
UNMEASURABLE obligation that WITHHOLDS the proof token -- never a silent pass.

The obligations are emitted as :class:`~clay.layout.intent.LayoutObligation`
(``measured_imbalance`` / ``rel_tol`` / ``missing_nets`` / ``status``) in the
SAME shape the cap / LDE obligations use, so they flow into
:func:`clay.layout.intent_obligations.check_layout_intent` -> the proof-token
gate unchanged: any FAIL / UNMEASURABLE blocks the token.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

from clay.core_contract import VerificationStatus

# Default RELATIVE tolerances for the structural geometric checks (fractions in
# (0, 1); relative-only, per the project-wide rule). A correctly built common-
# centroid cluster holds its two sub-centroids to WELL within these; a non-
# interleaved or asymmetric placement blows past them.
_DEFAULT_CENTROID_TOL = 0.05      # interdig / common-centroid sub-centroid match
_DEFAULT_PLACE_GROUP_TOL = 0.10   # row/col band realisation slack


# ─────────────────────────────────────────────────────────────────────────────
# Geometry input: a frozen, placement-derived view (mirrors LdeGeometry's shape)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlacedClusterGeom:
    """The built common-centroid cluster geometry an interdig check measures.

    Sourced from the fused :class:`~clay.layout.common_centroid_cluster.ClusterInfo`
    that ``place_*`` attaches to the placed member-A device, PLUS the per-finger
    x-positions of the realised finger poly rectangles so the centroid is measured
    on the ACTUAL geometry, not re-derived from the index.

      * ``device_a`` / ``device_b`` -- the two member labels;
      * ``finger_owner``            -- cluster-wide finger index -> owning member
        label (left-to-right), the realised interleave;
      * ``finger_x``                -- per-finger CENTRE x (same order as
        ``finger_owner``); the array's geometric finger positions;
      * ``sd_column_nets``          -- per-S/D-column net (left-to-right);
      * ``expected_drain``          -- ``{member_label: drain_net}`` from the
        schematic, against which the realised drain columns are net-checked;
      * ``shared_source_net``       -- the matched pair's shared source net.
    """

    device_a: str
    device_b: str
    finger_owner: Tuple[str, ...]
    finger_x: Tuple[float, ...]
    sd_column_nets: Tuple[str, ...]
    expected_drain: Dict[str, str]
    shared_source_net: str
    rows: int = 1


@dataclass(frozen=True)
class GeometricPlacement:
    """The placement geometry the structural geometric checks measure against.

    Mirrors :class:`clay.layout.intent_obligations.LdeGeometry`: a frozen view a
    caller builds once from a real :class:`~clay.layout.generic_placement.Placement`
    (+ the parsed cell annotations) and threads into
    :func:`clay.layout.intent_obligations.check_layout_intent`. When it is not
    supplied, every structural geometric obligation is UNMEASURABLE (token
    withheld) -- the same blocking-on-missing-substrate policy a missing net gets.

      * ``device_bboxes`` -- ``{label: (x0, y0, x1, y1)}`` placed device bboxes
        (absolute, in dbu). A fused cluster appears under its member-A label only
        (member B is absorbed), so a centroid lookup of member B falls back to the
        cluster it belongs to.
      * ``clusters``      -- ``{member_a_label: PlacedClusterGeom}`` for every
        ``@interdigitated`` pair that was fused.
      * ``cluster_member_to_a`` -- ``{member_label: member_a_label}`` so a
        centroid / place-group lookup of an absorbed member-B resolves to the
        cluster's placed bbox.
    """

    device_bboxes: Dict[str, Tuple[float, float, float, float]]
    clusters: Dict[str, PlacedClusterGeom]
    cluster_member_to_a: Dict[str, str]
    # Immutable placement-certificate authority for each declared matching pair:
    # canonical pair -> ("x"|"y", local mirror-axis coordinate in dbu).
    matching_pair_axes: Dict[Tuple[str, str], Tuple[str, float]] = None
    # Materialized orientation per member.  Missing orientation makes a direct
    # matching check unmeasurable; it is never assumed from a place-group token.
    device_orientations: Dict[str, str] = None
    # Members explicitly discarded because the materializer cannot constrain
    # their class.  This is capability absence (unsupported), not missing GDS.
    dropped_nonmos_matched: Tuple[str, ...] = ()
    # Per-pair physical PCell footprints, kept separate from the abstract
    # solver/router ``PlacedDevice`` envelope.  Only a final-master certificate
    # may populate this view.
    matching_pair_physical_bboxes: Dict[
        Tuple[str, str],
        Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]],
    ] = None
    # Exact-GDS regions named by topology groups.  They let relational intents
    # use the same endpoint vocabulary as placement without pretending a group
    # is an independently emitted device.
    named_region_bboxes: Dict[str, Tuple[float, float, float, float]] = None
    # Separately typed source/PDK evidence for channel occurrences sharing one
    # native row. It is not a collection of independent PCell footprints.
    native_row_matching_evidence: Dict[Tuple[str, str], object] = None

    def __post_init__(self) -> None:
        # Preserve old three-argument construction while exposing ordinary dicts
        # to the mechanical checkers.
        if self.matching_pair_axes is None:
            object.__setattr__(self, "matching_pair_axes", {})
        if self.matching_pair_physical_bboxes is None:
            object.__setattr__(self, "matching_pair_physical_bboxes", {})
        if self.device_orientations is None:
            object.__setattr__(self, "device_orientations", {})
        if self.named_region_bboxes is None:
            object.__setattr__(self, "named_region_bboxes", {})
        if self.native_row_matching_evidence is None:
            object.__setattr__(self, "native_row_matching_evidence", {})

    @staticmethod
    def from_placement(placement, cell_def, intent=None, *, devs=None,
                       route_geometry=None, route_placement_um=None, tech=None) -> "GeometricPlacement":
        """Build a :class:`GeometricPlacement` from a real placement + parsed cell.

        Reads the placed device bboxes off ``placement.devices`` and the fused
        common-centroid cluster provenance off each built device's
        ``__dict__['cluster']`` (the :class:`ClusterInfo` the cluster builder
        attaches). The per-finger x positions are read off the placed cluster
        device's poly rectangles so the centroid is measured on the ACTUAL placed
        geometry. ``cell_def`` supplies the schematic drain net per member (for
        the net-correctness leg of the interdig check); ``intent`` is accepted for
        symmetry with the other ``from_*`` factories and is otherwise unused here.

        ``devs`` is the built ``{label: DeviceGeom}`` the placer fused -- the
        carrier of the :class:`ClusterInfo` provenance, which the frozen
        :class:`~clay.layout.generic_placement.PlacedDevice` does not retain. A
        caller that already built it (the placer does) passes it through; when
        omitted, it is rebuilt deterministically from ``cell_def`` so the factory
        is self-sufficient from a placement alone.

        File-backed cells require their source file to remain readable: source
        declarations are inspected before native cross-row matching can be
        ruled out, even when the supplied in-memory devices are nonnative.
        Cross-row placement evidence additionally requires the actual route
        geometry, positions and current technology; without these inputs its
        matching verdict is unmeasurable.
        """
        bboxes: Dict[str, Tuple[float, float, float, float]] = {}
        for label, pd in getattr(placement, "devices", {}).items():
            bb = getattr(pd, "bbox", None)
            if bb is not None and len(bb) == 4:
                bboxes[str(label)] = (
                    float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))

        # Per-member schematic drain net + shared source net, read from the parsed
        # topology (mechanical, no inference): a MOS device exposes its terminal
        # nets via the cell's functional/topology surface.
        drain_of, source_of = _device_drain_source_nets(cell_def)

        # The ClusterInfo provenance lives on the built DeviceGeom (devs), not on
        # the placed device. Use the supplied devs, else rebuild from the cell.
        if devs is None:
            devs = _rebuild_devs(cell_def)

        clusters: Dict[str, PlacedClusterGeom] = {}
        member_to_a: Dict[str, str] = {}
        for label, pd in getattr(placement, "devices", {}).items():
            ci = _cluster_info_of(pd)
            if ci is None and devs is not None and label in devs:
                ci = _cluster_info_of(devs[label])
            if ci is None:
                continue
            da, db = str(ci.device_a), str(ci.device_b)
            # Prefer the cluster's RECORDED per-finger column x (the authoritative
            # finger positions, including a 2-row array's repeated columns, in the
            # same order as ``finger_owner``); fall back to reading the placed poly
            # rects when the provenance lacks it (legacy clusters). The recorded x is
            # in the device's own frame, but the sub-centroid metric is RELATIVE to
            # the array span, so the placement translation cancels.
            recorded_fx = tuple(
                float(x) for x in (getattr(ci, "finger_col_x", ()) or ()))
            if len(recorded_fx) == len(ci.finger_owner):
                finger_x = list(recorded_fx)
            else:
                finger_x = _finger_centre_xs(pd, len(ci.finger_owner))
            shared_src = source_of.get(da, source_of.get(db, ""))
            clusters[str(label)] = PlacedClusterGeom(
                device_a=da,
                device_b=db,
                finger_owner=tuple(str(o) for o in ci.finger_owner),
                finger_x=tuple(finger_x),
                sd_column_nets=tuple(str(n) for n in ci.sd_column_nets),
                expected_drain={
                    da: drain_of.get(da, ""),
                    db: drain_of.get(db, ""),
                },
                shared_source_net=str(shared_src),
                rows=int(getattr(ci, "rows", 1) or 1),
            )
            member_to_a[da] = str(label)
            member_to_a[db] = str(label)
        has_materialized_axis_certificate = hasattr(
            placement, "matching_pair_axes"
        )
        pair_axes: Dict[Tuple[str, str], Tuple[str, float]] = dict(
            getattr(placement, "matching_pair_axes", {}) or {}
        )
        raw_physical_pair_bboxes = getattr(
            placement, "matching_pair_physical_bboxes", {}
        )
        if type(raw_physical_pair_bboxes) is not dict:
            raise TypeError("matching physical bbox inventory must be an exact dict")
        physical_pair_bboxes = {}
        for raw_pair, raw_boxes in raw_physical_pair_bboxes.items():
            if (
                type(raw_pair) is not tuple
                or len(raw_pair) != 2
                or any(type(member) is not str or not member for member in raw_pair)
                or raw_pair[0] >= raw_pair[1]
                or type(raw_boxes) is not tuple
                or len(raw_boxes) != 2
                or any(
                    type(bbox) is not tuple
                    or len(bbox) != 4
                    or any(type(value) is not int for value in bbox)
                    for bbox in raw_boxes
                )
            ):
                raise TypeError("matching physical bbox inventory is malformed")
            if raw_pair not in pair_axes:
                raise ValueError("matching physical bbox pair lacks axis authority")
            physical_pair_bboxes[raw_pair] = tuple(
                tuple(float(value) for value in bbox) for bbox in raw_boxes
            )
        # Legacy placement objects predate per-pair certificates and exposed only
        # one solver symmetry axis.  Preserve that compatibility only when the
        # explicit certificate surface is absent.  An explicit-but-missing pair
        # means its edge was not materialized (for example an overlapping A-B/A-C
        # graph); synthesizing the global axis would turn that absence into a pass.
        constraints = getattr(placement, "constraints", None)
        if not has_materialized_axis_certificate:
            for left, right in tuple(
                getattr(constraints, "symmetry_pairs", ()) or ()
            ):
                pair = tuple(sorted((str(left), str(right))))
                pair_axes.setdefault(
                    pair,
                    ("x", float(getattr(placement, "axis_x", 0.0))),
                )

        # Orientation is evidence only when the final materialized device carries
        # it.  Source @orient (and especially an implicit R0 default) describes an
        # input request, not the placed/exact artifact, so it cannot certify this
        # obligation.
        orientations: Dict[str, str] = {
            str(label): str(orientation)
            for label, pd in getattr(placement, "devices", {}).items()
            if (orientation := getattr(pd, "orientation", ""))
        }

        from clay.layout.native_row_matching_evidence import capture_native_row_matching
        native_matching,native_inventory = capture_native_row_matching(placement, cell_def)
        from clay.layout.native_cross_row_matching_evidence import capture_native_cross_row_matching
        cross_inventory = capture_native_cross_row_matching(placement, cell_def,
            geometry=route_geometry, placement_um=route_placement_um, tech=tech,
            same_row_inventory=native_inventory)
        view_type = (_CrossBoundGeometricPlacement if cross_inventory is not None else
                     _NativeBoundGeometricPlacement if native_inventory is not None else GeometricPlacement)
        extra = ({'native_row_matching_inventory': native_inventory}
                 if native_inventory is not None or cross_inventory is not None else {})
        if cross_inventory is not None:
            extra['native_cross_row_matching_inventory'] = cross_inventory
        return view_type(
            device_bboxes=bboxes,
            clusters=clusters,
            cluster_member_to_a=member_to_a,
            matching_pair_axes=pair_axes,
            matching_pair_physical_bboxes=physical_pair_bboxes,
            native_row_matching_evidence=native_matching,
            device_orientations=orientations,
            dropped_nonmos_matched=tuple(
                getattr(placement, "dropped_nonmos_matched", ()) or ()
            ),
            **extra,
        )

    # ── centroid helpers ────────────────────────────────────────────────────
    def centroid_of(self, label: str) -> Optional[Tuple[float, float]]:
        """Placed centroid (bbox centre) of a device, resolving fused members.

        A device that was absorbed into a fused cluster (member B) has no own
        bbox; it resolves to the cluster's placed bbox (member A's key). Returns
        ``None`` when neither the device nor its cluster is placed.
        """
        bb = self.device_bboxes.get(str(label))
        if bb is None:
            a_key = self.cluster_member_to_a.get(str(label))
            if a_key is not None:
                bb = self.device_bboxes.get(a_key)
        if bb is None:
            bb = self.named_region_bboxes.get(str(label))
        if bb is None:
            return None
        x0, y0, x1, y1 = bb
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def placement_span(self) -> Optional[Tuple[float, float]]:
        """The (width, height) of the whole placement bbox, or ``None`` if empty.

        Used as the RELATIVE denominator for a centroid-offset measurement: the
        fraction of the cell's extent the two centroids differ by. Scale-invariant.
        """
        all_bboxes = tuple(self.device_bboxes.values()) + tuple(
            self.named_region_bboxes.values()
        )
        if not all_bboxes:
            return None
        x0 = min(b[0] for b in all_bboxes)
        y0 = min(b[1] for b in all_bboxes)
        x1 = max(b[2] for b in all_bboxes)
        y1 = max(b[3] for b in all_bboxes)
        return (max(x1 - x0, 0.0), max(y1 - y0, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Cluster / topology readers (deterministic, no inference)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _NativeBoundGeometricPlacement(GeometricPlacement):
    """Live native source view whose expected matching inventory cannot vanish."""
    native_row_matching_inventory: object = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if (type(self) is _NativeBoundGeometricPlacement or self.native_row_matching_inventory is not None
                or self.native_row_matching_evidence):
            from clay.layout.native_row_matching_evidence import require_native_row_matching_inventory
            require_native_row_matching_inventory(self.native_row_matching_inventory,
                                                  self.native_row_matching_evidence)


@dataclass(frozen=True)
class _CrossBoundGeometricPlacement(_NativeBoundGeometricPlacement):
    native_cross_row_matching_inventory: object = None

    def __post_init__(self) -> None:
        super().__post_init__()
        from clay.layout.native_cross_row_matching_evidence import require_native_cross_row_matching_inventory
        require_native_cross_row_matching_inventory(self.native_cross_row_matching_inventory)
        if self.native_cross_row_matching_inventory.same_row_inventory is not self.native_row_matching_inventory:
            raise ValueError('native matching inventories differ from their captured source')


def _rebuild_devs(cell_def) -> Optional[Dict[str, object]]:
    """Rebuild the fused ``{label: DeviceGeom}`` from a parsed cell, or ``None``.

    Deterministic: routes through the SAME front-end the placer uses
    (``annotated_cell_from_cell_def`` -> ``_build_devices_from_clay``) so the
    rebuilt clusters carry IDENTICAL :class:`ClusterInfo` provenance to the placed
    run. Best-effort: returns ``None`` if the placer front-end is unavailable, so
    the interdig obligations become UNMEASURABLE (blocking) rather than crash.
    """
    try:
        from .generic_placement import (
            _build_devices_from_clay,
            _resolve_tech,
            annotated_cell_from_cell_def,
        )
        annotated, _ = annotated_cell_from_cell_def(cell_def, None)
        return _build_devices_from_clay(annotated, _resolve_tech(None))
    except Exception:  # noqa: BLE001 - missing front-end -> unmeasurable, not crash
        return None


def _cluster_info_of(placed_device) -> object:
    """Return the :class:`ClusterInfo` attached to a placed device, or ``None``.

    The cluster builder attaches the provenance to the DeviceGeom via
    ``geom.__dict__['cluster']``; the placer carries it onto the placed device the
    same way when it fuses a pair. Read it off whichever object exposes it.
    """
    for obj in (placed_device, getattr(placed_device, "geom", None)):
        if obj is None:
            continue
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict) and "cluster" in d:
            return d["cluster"]
    return getattr(placed_device, "cluster", None)


def _finger_centre_xs(placed_device, n_fingers: int) -> List[float]:
    """Per-finger centre x of the placed cluster, length ``n_fingers``.

    Reads the placed poly (gate) rectangles off the device and returns the centre
    x of the ``n_fingers`` NARROWEST/tallest gate rects sorted left-to-right --
    the realised finger positions. When the polys cannot be read (no geometry),
    falls back to an empty list -> the interdig check measures centroids from the
    finger INDEX order instead (still on the realised interleave, just uniform
    pitch). Deterministic.
    """
    polys = getattr(placed_device, "polys", None)
    if not polys:
        return []
    # Collect candidate gate-finger rects: tall, narrow rectangles. We do not know
    # the PDK poly layer number here, so use a geometric signature (the finger
    # rects are the N tallest near-equal-width rectangles). To stay PDK-neutral and
    # mechanical we simply take the most common (layer, datatype) among the tall
    # narrow rects.
    rects: List[Tuple[Tuple[int, int], float, float, float]] = []
    for p in polys:
        bb = _poly_bbox(p)
        if bb is None:
            continue
        x0, y0, x1, y1 = bb
        w = x1 - x0
        h = y1 - y0
        if w <= 0 or h <= 0:
            continue
        key = (getattr(p, "layer", -1), getattr(p, "datatype", -1))
        rects.append((key, (x0 + x1) / 2.0, w, h))
    if not rects:
        return []
    # The poly fingers are the layer whose rects are tall+narrow and most numerous.
    from collections import defaultdict
    by_layer: Dict[Tuple[int, int], List[Tuple[float, float, float]]] = defaultdict(list)
    for key, cx, w, h in rects:
        by_layer[key].append((cx, w, h))
    best_layer = None
    best_score = -1.0
    for key, vals in by_layer.items():
        # finger signature: many rects with h > w (tall) of similar small width.
        tall = [v for v in vals if v[2] >= v[1]]
        if len(tall) < 2:
            continue
        score = float(len(tall))
        if score > best_score:
            best_score = score
            best_layer = key
    if best_layer is None:
        return []
    xs = sorted(cx for cx, w, h in by_layer[best_layer] if h >= w)
    # Keep the n_fingers leftmost-to-rightmost distinct finger centres. The gate
    # array has exactly n_fingers fingers; extra straps on the poly layer (gate
    # buses) are wide (w >= h) and were filtered by the h >= w test above.
    return xs[:n_fingers] if len(xs) >= n_fingers else xs


def _poly_bbox(p) -> Optional[Tuple[float, float, float, float]]:
    """Bounding box of a LayerPolygon, or ``None`` when it has no points."""
    poly = getattr(p, "polygon", None)
    pts = None
    if poly is not None:
        # LayerPolygon's Polygon exposes its corners as ``.vertices`` (Point(x,y));
        # tolerate a ``.points`` alias too.
        pts = getattr(poly, "vertices", None) or getattr(poly, "points", None)
    if not pts:
        # Some polys expose .bbox directly.
        bb = getattr(p, "bbox", None)
        if bb is not None and len(bb) == 4:
            return (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
        return None
    xs = [float(pt.x if hasattr(pt, "x") else pt[0]) for pt in pts]
    ys = [float(pt.y if hasattr(pt, "y") else pt[1]) for pt in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _device_drain_source_nets(cell_def) -> Tuple[Dict[str, str], Dict[str, str]]:
    """``({label: drain_net}, {label: source_net})`` from the parsed topology.

    Reads the per-device terminal nets straight off the parsed ``CellDef`` device
    list (mechanical -- the schematic IS the source of truth for which net a
    drain/source carries). Tolerant of the several shapes a parsed device exposes
    (``.drain``/``.source`` attrs, a ``.terminals``/``.nets`` dict, or a
    ``connections`` map). Missing -> absent key (the check then treats that net as
    unknown and the obligation is unmeasurable, never a false pass).
    """
    drain: Dict[str, str] = {}
    source: Dict[str, str] = {}
    devices = _iter_devices(cell_def)
    for dev in devices:
        label = _device_label(dev)
        if not label:
            continue
        d_net = _terminal_net(dev, ("drain", "d"))
        s_net = _terminal_net(dev, ("source", "s"))
        if d_net is not None:
            drain[label] = d_net
        if s_net is not None:
            source[label] = s_net
    return drain, source


def _iter_devices(cell_def) -> Sequence[object]:
    """Best-effort device list off a parsed CellDef (several shapes supported)."""
    for attr in ("devices", "topology_devices", "mos_devices"):
        v = getattr(cell_def, attr, None)
        if v:
            return list(v.values()) if isinstance(v, dict) else list(v)
    topo = getattr(cell_def, "topology", None)
    if topo is not None:
        for attr in ("devices", "instances"):
            v = getattr(topo, attr, None)
            if v:
                return list(v.values()) if isinstance(v, dict) else list(v)
    return []


def _device_label(dev) -> str:
    for attr in ("label", "name", "local_name", "instance_name", "id"):
        v = getattr(dev, attr, None)
        if v:
            return str(v)
    if isinstance(dev, dict):
        for k in ("label", "name", "id"):
            if dev.get(k):
                return str(dev[k])
    return ""


def _terminal_net(dev, names: Sequence[str]) -> Optional[str]:
    """The net on one of ``names`` terminals of a parsed device, or ``None``."""
    for nm in names:
        v = getattr(dev, nm, None)
        if isinstance(v, str) and v:
            return v
    for holder in ("terminals", "nets", "connections", "pins"):
        m = getattr(dev, holder, None)
        if isinstance(m, dict):
            for nm in names:
                v = m.get(nm)
                # terminal entries may be a bare net str or carry a .net
                if isinstance(v, str) and v:
                    return v
                netv = getattr(v, "net", None)
                if isinstance(netv, str) and netv:
                    return netv
    if isinstance(dev, dict):
        for nm in names:
            v = dev.get(nm)
            if isinstance(v, str) and v:
                return v
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. interdigitation check (kind 'interdig')
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InterdigVerdict:
    """The DIRECT geometric findings for one interdigitated pair (provenance).

    ``measured`` is the RELATIVE sub-centroid offset (the obligation imbalance);
    ``None`` == unmeasurable. The boolean legs explain WHY a non-zero / unmeasur-
    able verdict happened: ``interleaved`` (A/B fingers actually alternate),
    ``net_ok`` (every S/D column wires to the correct schematic net). A FAIL on
    EITHER structural leg forces ``measured`` to a sentinel over-budget value
    (1.0) so the obligation fails even if the centroid happened to coincide.
    """

    measured: Optional[float]
    interleaved: bool
    net_ok: bool
    centroid_offset_rel: Optional[float]
    reason: str = ""


def check_interdigitation(
    cluster: PlacedClusterGeom,
    *,
    rel_tol: float = _DEFAULT_CENTROID_TOL,
) -> InterdigVerdict:
    """DIRECT geometric interdigitation check on ONE built common-centroid cluster.

    Three mechanical legs, all measured on the realised cluster geometry:

      1. INTERLEAVED: the A and B fingers ALTERNATE (a common-centroid interleave),
         not two separate side-by-side blocks. Measured by counting same-owner
         adjacencies: a fully blocked ``A A A A B B B B`` has ``nf-1`` same-owner
         adjacencies per block; a proper interleave (ABAB / ABBA) has far fewer.
         The pair FAILS this leg if the two members are not interleaved at all
         (one contiguous A-run followed by one contiguous B-run).
      2. CENTROID COINCIDENCE: the A-fingers' centroid x and the B-fingers'
         centroid x COINCIDE -- measured as ``|cA - cB| / pitch`` (RELATIVE to the
         finger pitch, so the dbu unit cancels). A centro-symmetric ABAB/ABBA
         interleave drives this to ~0; an asymmetric one leaves a residual.
      3. NET CORRECTNESS: every realised S/D column wires to the right schematic
         net -- the SHARED source columns carry the shared source net, and each
         DRAIN column carries the drain net of the member owning the finger it
         drains. A mis-netted column (e.g. a drain tied to the wrong member, or a
         source column shorted to a drain net) FAILS this leg.

    Returns an :class:`InterdigVerdict`. A structural-leg failure (not interleaved,
    or mis-netted) makes the obligation FAIL regardless of centroid; a coincident,
    interleaved, correctly-netted cluster passes.
    """
    owners = cluster.finger_owner
    if len(owners) < 2:
        return InterdigVerdict(
            measured=None, interleaved=False, net_ok=False,
            centroid_offset_rel=None, reason="cluster has <2 fingers")

    a, b = cluster.device_a, cluster.device_b
    rows = max(1, int(getattr(cluster, "rows", 1) or 1))

    # ── leg 1: interleaving (alternation / 2-D common-centroid, not two blocks) ─
    # For a 1-row array, the A/B fingers must cross (not two side-by-side blocks).
    # For a MULTI-ROW array (the 2-D common-centroid), the interleave is realised
    # by the row-to-row A/B SWAP: each row carries both members and the swap places
    # the opposite device in every column, so the pair is interdigitated across rows
    # even when a single row is one device's block. We accept the array as
    # interleaved if EITHER the flat sequence crosses, OR it is a multi-row array
    # whose rows are the A/B swap (both members present, balanced) -- never two pure
    # side-by-side blocks of one device.
    if rows > 1 and (n := len(owners)) % rows == 0:
        per = n // rows
        row0 = owners[:per]
        row1 = owners[per:2 * per]
        has_a = any(o == a for o in owners)
        has_b = any(o == b for o in owners)
        # the swap interleave: row1 is row0 with A/B swapped (column-aligned).
        swapped = all(
            (row0[i] == a) == (row1[i] == b) for i in range(per)
        ) if len(row1) == per else False
        interleaved = bool(has_a and has_b and (swapped or _is_interleaved(owners, a, b)))
    else:
        interleaved = _is_interleaved(owners, a, b)

    # ── leg 2: sub-centroid coincidence (RELATIVE to finger pitch) ───────────
    centroid_rel = _subcentroid_offset_rel(cluster, a, b)

    # ── leg 3: per-column net correctness ────────────────────────────────────
    net_ok, net_reason = _columns_net_correct(cluster)

    # Reduce to a single obligation imbalance. A structural failure dominates.
    if centroid_rel is None:
        # Cannot measure the centroid (degenerate geometry) -> unmeasurable.
        return InterdigVerdict(
            measured=None, interleaved=interleaved, net_ok=net_ok,
            centroid_offset_rel=None,
            reason="centroid unmeasurable (degenerate finger geometry)")
    if not interleaved or not net_ok:
        # A blocked or mis-netted array FAILS even if the centroid coincides: the
        # obligation imbalance is forced over any tolerance (sentinel 1.0).
        reasons = []
        if not interleaved:
            reasons.append("not_interleaved")
        if not net_ok:
            reasons.append(f"net:{net_reason}")
        return InterdigVerdict(
            measured=1.0, interleaved=interleaved, net_ok=net_ok,
            centroid_offset_rel=centroid_rel, reason=";".join(reasons))
    return InterdigVerdict(
        measured=centroid_rel, interleaved=True, net_ok=True,
        centroid_offset_rel=centroid_rel, reason="ok")


def _is_interleaved(owners: Sequence[str], a: str, b: str) -> bool:
    """True iff A/B fingers ALTERNATE rather than forming two contiguous blocks.

    A genuine common-centroid interleave (ABAB / ABBA / ...) crosses between A and
    B many times along the array; two separate side-by-side devices are ONE A-run
    then ONE B-run (a single crossing). We require at least TWO owner crossings
    (so AABB -- a single A->B transition -- is rejected as blocked, while ABAB and
    ABBA pass). Deterministic, mechanical.
    """
    crossings = sum(
        1 for i in range(1, len(owners)) if owners[i] != owners[i - 1]
    )
    # A two-block layout (A...A B...B) has exactly 1 crossing. Any real interleave
    # of two members has >= 2 (it must return to A after a B somewhere, or vice
    # versa). Also require BOTH members to actually appear.
    has_a = any(o == a for o in owners)
    has_b = any(o == b for o in owners)
    return has_a and has_b and crossings >= 2


def _subcentroid_offset_rel(
    cluster: PlacedClusterGeom, a: str, b: str,
) -> Optional[float]:
    """``|centroid_x(A) - centroid_x(B)| / array_span``, or ``None``.

    Measures the two members' finger-centroid x and normalises by the ARRAY SPAN
    (the full finger extent), so the result is the fraction of the cluster width
    the two sub-centroids differ by -- RELATIVE (dbu cancels) and scale-invariant.
    A centro-symmetric ABAB/ABBA interleave drives this to ~0; an ODD-``nf`` single-
    row interleave (whose two sub-centroids cannot exactly coincide) leaves a real,
    honest residual the check reports rather than hides.

    The finger POSITIONS come from the realised per-finger x (one entry per finger,
    in the SAME order as ``finger_owner``) when the cluster carries it -- this is
    essential for a 2-ROW (odd-nf) common-centroid array, where two fingers share a
    column x in different rows, so the A/B x-centroids coincide ONLY when measured on
    the true per-finger x (collapsing to distinct columns or using the flat index
    would re-introduce the parity residual). When the per-finger x is NOT one-per-
    finger (a legacy poly-read that double-counts straps), the distinct column
    centres are used; failing that, the uniform index positions -- both give the same
    RELATIVE offset for a 1-row uniform-pitch array. ``None`` when a member has no
    fingers (degenerate).
    """
    owners = cluster.finger_owner
    n = len(owners)
    fx = list(cluster.finger_x)
    if len(fx) == n:
        # one realised x per finger (the cluster's recorded per-finger column x):
        # measure the sub-centroids on the TRUE finger positions -- correct for the
        # 2-row odd-nf array (repeated columns across rows).
        pos = [float(x) for x in fx]
    else:
        # legacy: dedup to distinct columns, else uniform index positions.
        cols = _distinct_finger_columns(cluster.finger_x)
        pos = cols if len(cols) == n else [float(i) for i in range(n)]
    a_pos = [pos[i] for i, o in enumerate(owners) if o == a]
    b_pos = [pos[i] for i, o in enumerate(owners) if o == b]
    if not a_pos or not b_pos:
        return None
    span = max(pos) - min(pos)
    if span <= 0:
        return None
    ca = sum(a_pos) / len(a_pos)
    cb = sum(b_pos) / len(b_pos)
    return abs(ca - cb) / span


def _distinct_finger_columns(finger_x: Sequence[float]) -> List[float]:
    """Collapse near-equal finger x's into distinct column centres (sorted).

    The placed gate polys carry both each finger AND its co-located gate-strap at
    the SAME x, so the raw ``finger_x`` list double-counts columns. This bins x's
    that are within a small fraction of the median gap into one column centre, so
    the result is one centre per physical finger column. Deterministic.
    """
    xs = sorted(float(x) for x in finger_x)
    if not xs:
        return []
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    pos_gaps = sorted(g for g in gaps if g > 0)
    if not pos_gaps:
        return [xs[0]]
    # bin tolerance: a fraction of the smallest positive inter-finger gap.
    tol = pos_gaps[0] * 0.25
    cols: List[float] = [xs[0]]
    bucket = [xs[0]]
    for x in xs[1:]:
        if x - bucket[-1] <= tol:
            bucket.append(x)
        else:
            cols[-1] = sum(bucket) / len(bucket)
            bucket = [x]
            cols.append(x)
    cols[-1] = sum(bucket) / len(bucket)
    return cols


def _columns_net_correct(cluster: PlacedClusterGeom) -> Tuple[bool, str]:
    """Verify every realised S/D column wires to its CORRECT schematic net.

    The interleaved array has ``2*fingers_per_row + 1`` S/D columns PER ROW (one row
    for an even-nf 1-row array, two rows for an odd-nf 2-row common-centroid). By the
    cluster builder's commoning rule, WITHIN EACH ROW: EVEN columns (0, 2, ...) are
    SHARED-SOURCE columns (must carry the shared source net); ODD columns are DRAIN
    columns owned by the member of the finger on the column's LEFT (must carry THAT
    member's drain net). ``cluster.sd_column_nets`` carries every row's columns
    concatenated (row 0 first); ``cluster.finger_owner`` carries every row's fingers
    concatenated in the same order. A column whose realised net disagrees with the
    schematic-expected net FAILS -- this is exactly where a mis-routed common source
    (tail/vdd short) or a swapped drain shows up as a DIRECT net mismatch.

    Returns ``(ok, reason)``; ``reason`` names the first offending row/column.
    """
    owners = cluster.finger_owner
    cols = cluster.sd_column_nets
    n = len(owners)
    rows = max(1, int(getattr(cluster, "rows", 1) or 1))
    if n % rows != 0:
        return False, f"finger_count {n} not divisible by rows {rows}"
    fingers_per_row = n // rows
    # each row has fingers_per_row fingers -> fingers_per_row + 1 S/D columns.
    cols_per_row = fingers_per_row + 1
    if len(cols) != rows * cols_per_row:
        return False, f"column_count {len(cols)}!={rows * cols_per_row}"
    src = cluster.shared_source_net
    if not src:
        return False, "shared_source_net_unknown"
    for ri in range(rows):
        row_owners = owners[ri * fingers_per_row:(ri + 1) * fingers_per_row]
        row_cols = cols[ri * cols_per_row:(ri + 1) * cols_per_row]
        for j, net in enumerate(row_cols):
            if j % 2 == 0:
                if net != src:
                    return False, (
                        f"row{ri}_col{j}_source_expected_{src}_got_{net or 'empty'}")
            else:
                owner = row_owners[j - 1] if (j - 1) < len(row_owners) else row_owners[-1]
                expected = cluster.expected_drain.get(owner, "")
                if not expected:
                    return False, f"row{ri}_col{j}_drain_owner_{owner}_unknown"
                if net != expected:
                    return False, (
                        f"row{ri}_col{j}_drain_owner_{owner}_expected_{expected}"
                        f"_got_{net or 'empty'}")
    return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# 2. direct matching check (kind 'matching')
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MatchingCheck:
    """Closed matching verdict plus its three mechanical sub-measurements."""

    status: str
    measured: Optional[float]
    missing: Tuple[str, ...]
    centroid_axis_residual: Optional[float]
    orientation_imbalance: Optional[float]
    footprint_imbalance: Optional[float]
    reason: str = ""


def check_matching(
    geom: GeometricPlacement,
    devices: Sequence[str],
    *,
    rel_tol: float = _DEFAULT_CENTROID_TOL,
    require_interdigitation: bool = False,
) -> MatchingCheck:
    """Measure a declared device pair on final geometry + placement authority.

    The comparator is deliberately not a place-group proxy.  It requires both
    member bboxes, the materializer-certified local mirror axis, and both member
    orientations.  The binding metric is the maximum of:

    * pair-midpoint displacement from the certified local axis and the
      perpendicular centroid skew (the complete mirror-centroid residual);
    * orientation mismatch (0 equal, 1 different);
    * relative width/height footprint spread.

    A member explicitly dropped by the materializer is ``unsupported``.  Missing
    observable geometry/axis/orientation is ``unmeasurable``.  Only a finite
    measurement is eligible for pass/fail.
    """
    if len(tuple(devices)) != 2:
        return MatchingCheck(
            VerificationStatus.UNMEASURABLE.value,
            None,
            ("matching:fewer_than_two_members",),
            None,
            None,
            None,
            "matching requires exactly two members",
        )
    pair = tuple(sorted((str(devices[0]), str(devices[1]))))
    dropped = set(str(item) for item in geom.dropped_nonmos_matched)
    unsupported = tuple(member for member in pair if member in dropped)
    if unsupported:
        return MatchingCheck(
            VerificationStatus.UNSUPPORTED.value,
            None,
            unsupported,
            None,
            None,
            None,
            "materializer dropped declared non-MOS matching member(s)",
        )

    if isinstance(geom, _NativeBoundGeometricPlacement):
        if type(geom) is _NativeBoundGeometricPlacement or geom.native_row_matching_inventory is not None:
            from clay.layout.native_row_matching_evidence import require_native_row_matching_inventory
            require_native_row_matching_inventory(geom.native_row_matching_inventory,
                                                  geom.native_row_matching_evidence)
    if isinstance(geom, _CrossBoundGeometricPlacement):
        from clay.layout.native_cross_row_matching_evidence import (
            require_native_cross_row_matching_inventory, replay_native_cross_row_matching,
        )
        inventory = require_native_cross_row_matching_inventory(geom.native_cross_row_matching_inventory)
        if pair in inventory.pairs:
            if geom.matching_pair_axes.get(pair) != inventory.placement.matching_pair_axes.get(pair):
                raise ValueError('cross-row matching geometric axis differs from live authority')
            if require_interdigitation:
                return MatchingCheck(VerificationStatus.UNMEASURABLE.value, None,
                    ('matching:interdigitation_provenance_absent',), None, None, None,
                    'cross-row proof does not establish interdigitation')
            metrics = replay_native_cross_row_matching(inventory, pair)
            if metrics is None:
                return MatchingCheck(VerificationStatus.UNMEASURABLE.value, None,
                    ('matching:cross_row_artwork_authority_absent',), None, None, None,
                    'current cross-row stamps and positions are required')
            measured = max(metrics)
            status = (VerificationStatus.PASS.value if measured <= rel_tol
                      else VerificationStatus.FAIL.value)
            return MatchingCheck(status, measured, (), *metrics,
                                 'measured on source-bound native cross-row channels')
    native_evidence = geom.native_row_matching_evidence.get(pair)
    if native_evidence is not None and not require_interdigitation:
        if not isinstance(geom, _NativeBoundGeometricPlacement):
            raise ValueError("native row matching evidence lacks its source-bound view")
        from clay.layout.native_row_matching_evidence import replay_native_row_matching
        centroid_residual, footprint, orientation = replay_native_row_matching(native_evidence, pair)
        if geom.matching_pair_axes.get(pair) != native_evidence.record[1]:
            raise ValueError("native row matching geometric axis differs from live authority")
        measured = max(centroid_residual, footprint, orientation)
        status = (VerificationStatus.PASS.value if measured <= rel_tol
                  else VerificationStatus.FAIL.value)
        return MatchingCheck(status, measured, (), centroid_residual, orientation,
                             footprint, "measured on source-bound native row channels")

    bboxes = []
    missing = []
    physical_pair = geom.matching_pair_physical_bboxes.get(pair)
    if physical_pair is not None:
        bboxes.extend(tuple(float(value) for value in bbox) for bbox in physical_pair)
    else:
        for member in pair:
            bbox = geom.device_bboxes.get(member)
            if bbox is None:
                cluster_key = geom.cluster_member_to_a.get(member)
                if cluster_key is not None:
                    bbox = geom.device_bboxes.get(cluster_key)
            if bbox is None:
                missing.append(member)
            else:
                bboxes.append(tuple(float(value) for value in bbox))
    if missing:
        return MatchingCheck(
            VerificationStatus.UNMEASURABLE.value,
            None,
            tuple(missing),
            None,
            None,
            None,
            "matching member geometry is absent",
        )

    if require_interdigitation:
        cluster = next(
            (
                item
                for item in geom.clusters.values()
                if frozenset((item.device_a, item.device_b)) == frozenset(pair)
            ),
            None,
        )
        if cluster is None:
            return MatchingCheck(
                VerificationStatus.UNMEASURABLE.value,
                None,
                ("matching:interdigitation_provenance_absent",),
                None,
                None,
                None,
                "required interdigitation cannot be reconstructed",
            )
        interdig = check_interdigitation(cluster)
        if interdig.measured is None:
            return MatchingCheck(
                VerificationStatus.UNMEASURABLE.value,
                None,
                (f"matching:interdig:{interdig.reason}",),
                None,
                None,
                None,
                interdig.reason,
            )
    else:
        interdig = None

    widths = [bbox[2] - bbox[0] for bbox in bboxes]
    heights = [bbox[3] - bbox[1] for bbox in bboxes]
    if any(value <= 0.0 for value in widths + heights):
        return MatchingCheck(
            VerificationStatus.UNMEASURABLE.value,
            None,
            ("matching:degenerate_footprint",),
            None,
            None,
            None,
            "matching member has a degenerate footprint",
        )

    axis = geom.matching_pair_axes.get(pair)
    if axis is None or axis[0] not in ("x", "y"):
        return MatchingCheck(
            VerificationStatus.UNMEASURABLE.value,
            None,
            ("matching:axis_provenance_absent",),
            None,
            None,
            None,
            "matching axis is absent from the placement certificate",
        )
    orientation_a = geom.device_orientations.get(pair[0])
    orientation_b = geom.device_orientations.get(pair[1])
    if orientation_a is None or orientation_b is None:
        missing_orientation = tuple(
            member
            for member, orientation in zip(pair, (orientation_a, orientation_b))
            if orientation is None
        )
        return MatchingCheck(
            VerificationStatus.UNMEASURABLE.value,
            None,
            tuple(f"orientation:{member}" for member in missing_orientation),
            None,
            None,
            None,
            "matching orientation provenance is absent",
        )

    centroids = [
        ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        for bbox in bboxes
    ]
    axis_index = 0 if axis[0] == "x" else 1
    perpendicular_index = 1 - axis_index
    pair_midpoint = (centroids[0][axis_index] + centroids[1][axis_index]) / 2.0
    span = max(
        max(bbox[2] for bbox in bboxes) - min(bbox[0] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes) - min(bbox[1] for bbox in bboxes),
        1.0,
    )
    centroid_residual = max(
        abs(pair_midpoint - float(axis[1])),
        abs(
            centroids[0][perpendicular_index]
            - centroids[1][perpendicular_index]
        ),
    ) / span

    def _spread(values: Sequence[float]) -> float:
        high = max(values)
        return (high - min(values)) / high if high > 0.0 else 0.0

    footprint = max(_spread(widths), _spread(heights))
    orientation = 0.0 if orientation_a == orientation_b else 1.0
    metrics = [centroid_residual, footprint, orientation]
    if interdig is not None and interdig.measured is not None:
        metrics.append(float(interdig.measured))
    measured = max(metrics)
    status = (
        VerificationStatus.PASS.value
        if measured <= rel_tol
        else VerificationStatus.FAIL.value
    )
    return MatchingCheck(
        status,
        measured,
        (),
        centroid_residual,
        orientation,
        footprint,
        "measured on final member geometry",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. common-centroid check (kind 'common_centroid')
# ─────────────────────────────────────────────────────────────────────────────


def check_common_centroid(
    geom: GeometricPlacement,
    devices: Sequence[str],
) -> Tuple[Optional[float], Tuple[str, ...]]:
    """DIRECT centroid-coincidence check for a common-centroid / matched group.

    Measures the group's devices' placed centroids and returns the RELATIVE spread
    of those centroids (the max pairwise centroid distance, normalised by the
    placement's diagonal extent so the dbu unit cancels). A genuinely common-
    centroid group has all member centroids coinciding -> ~0; a group laid out as
    separated devices has a large spread.

    This is a DIRECT measurement off the real placement, NOT the MILP's y-balance
    proxy: it reads the actual placed bbox centres. A fused common-centroid pair
    (both members in one interleaved cluster) coincides exactly (both resolve to
    the same cluster bbox), which the check correctly reports as 0.

    Returns ``(measured, missing)``; ``measured`` is ``None`` (unmeasurable) when a
    named device (or its cluster) is not placed, with the absent devices in
    ``missing``.
    """
    centroids: List[Tuple[float, float]] = []
    missing: List[str] = []
    for d in devices:
        c = geom.centroid_of(d)
        if c is None:
            missing.append(str(d))
        else:
            centroids.append(c)
    if missing or len(centroids) < 2:
        # An absent member (or a degenerate single-device group) cannot be
        # measured as a coincidence -> unmeasurable, token withheld.
        return None, tuple(missing) if missing else (("degenerate_group",))
    span = geom.placement_span()
    if span is None:
        return None, ("no_placement_extent",)
    diag = (span[0] ** 2 + span[1] ** 2) ** 0.5
    if diag <= 0:
        # All devices at one point -> trivially coincident.
        return 0.0, ()
    max_dist = 0.0
    for i in range(len(centroids)):
        for k in range(i + 1, len(centroids)):
            dx = centroids[i][0] - centroids[k][0]
            dy = centroids[i][1] - centroids[k][1]
            max_dist = max(max_dist, (dx * dx + dy * dy) ** 0.5)
    return max_dist / diag, ()


# ─────────────────────────────────────────────────────────────────────────────
# 3. place_group realisation check (kind 'place_group_realized')
# ─────────────────────────────────────────────────────────────────────────────


def check_place_group_realized(
    geom: GeometricPlacement,
    group_devices: Sequence[str],
    *,
    all_groups: Sequence[Tuple[Sequence[str], Optional[int], Optional[int]]],
    row: Optional[int],
    col: Optional[int],
    rel_tol: float = _DEFAULT_PLACE_GROUP_TOL,
) -> Tuple[Optional[float], Tuple[str, ...]]:
    """DIRECT check that a ``@place_group``'s devices realise the declared GRID ORDER.

    The declared ``row`` / ``col`` are ORDINAL band indices (row 0 below row 1; col
    0 left of col 1). Membership is DECLARED (``@place_group row= col=``), so the
    realisation we verify is that the declared GRID ORDER actually holds on the
    placement -- NOT a geometric boundary reverse-engineered from centroid medians.
    Reverse-engineering a midpoint between two columns' medians made the check
    sensitive to median SKEW: a fused/skewed column drags its median off-centre.
    The reference grammar itself is derived from anonymous REGION centroids.  A
    region may contain several functional groups, so replay first unions all
    groups in one ``(row, col)`` cell and then orders those cell centroids.

    Concretely, two mechanical group-centroid order tests, both RELATIVE (no
    absolute coordinate, no midpoint):

      * COLUMNS DON'T INTERLEAVE IN X (within a row) -- for every ordered column
        pair ``col_i < col_j`` in the SAME floorplan row, the anonymous-cell
        centroid declared in ``col_i`` is strictly left of the one in ``col_j``.
      * ROWS DON'T INTERLEAVE IN Y (within a column) -- for every ordered row pair
        ``row_i < row_j`` in the SAME column, the lower anonymous-cell centroid is
        strictly below the upper anonymous-cell centroid.

    ``measured_imbalance`` is the FRACTION of declared cell pairs involving this
    group's cell that violate the order (0.0 == perfectly ordered).  Requiring
    every subgroup or member of a wide region to precede every subgroup or member
    of the next region would invent a stronger, often impossible obligation.

    Returns ``(measured, missing)``; ``None`` (unmeasurable) when a required
    device is not placed. If neither axis has two comparable declared bands,
    there is no relative order to violate and the measurement is 0.0.
    """
    # Resolve this group's device centroids.
    centroids: List[Tuple[float, float]] = []
    missing: List[str] = []
    for d in group_devices:
        c = geom.centroid_of(d)
        if c is None:
            missing.append(str(d))
        else:
            centroids.append(c)
    if missing or not centroids:
        return None, tuple(missing) if missing else ("empty_group",)

    measured_row, missing_row = _declared_order_violation(
        geom, group_devices, all_groups, axis=1, want=row, other_axis_want=col)
    measured_col, missing_col = _declared_order_violation(
        geom, group_devices, all_groups, axis=0, want=col, other_axis_want=row)

    missing_comparison = tuple(sorted(set(missing_row) | set(missing_col)))
    if missing_comparison:
        return None, missing_comparison

    parts = [m for m in (measured_row, measured_col) if m is not None]
    if not parts:
        # Neither a row nor a col was declared (or no other band to order against)
        # -> nothing to realise (but the group's devices ARE placed): vacuously
        # satisfied (0), not unmeasurable.
        return 0.0, ()
    return max(parts), ()


def _declared_order_violation(
    geom: GeometricPlacement,
    group_devices: Sequence[str],
    all_groups: Sequence[Tuple[Sequence[str], Optional[int], Optional[int]]],
    *,
    axis: int,
    want: Optional[int],
    other_axis_want: Optional[int],
) -> Tuple[Optional[float], Tuple[str, ...]]:
    """Fraction of declared group pairs that violate this group's declared order.

    ``axis`` 1 = row (y), 0 = col (x). ``want`` is THIS group's declared band index
    along that axis. Column ordinals are local to one reference macro tier, so the
    X-order is tested WITHIN THE SAME ROW.  Symmetrically, the Y-order is tested
    WITHIN THE SAME COLUMN.  ``other_axis_want`` selects that orthogonal scope.

    The measure enumerates every ordered band pair ``(lo, hi)`` with ``lo < hi``
    and counts functional-group centroid inversions.  This exactly replays the
    anonymous region-centroid ranks carried by reference macro grammar; it does
    not invent pairwise ordering between members inside wide rigid groups.

    The measurement is ``None`` when fewer than two bands exist on this axis.
    Missing members of a comparison are returned separately so a different
    measured axis, or a vacuous axis, cannot conceal incomplete evidence.
    """
    if want is None:
        return None, ()
    # Each reference macro row owns its own compact column rank, and each column
    # owns its own compact row rank.  Never compare an ordinal with the same number
    # from another orthogonal scope.
    def _band_idx(grow: Optional[int], gcol: Optional[int]) -> Optional[int]:
        return grow if axis == 1 else gcol

    def _in_scope(grow: Optional[int], gcol: Optional[int]) -> bool:
        if other_axis_want is not None:
            return (
                gcol == other_axis_want
                if axis == 1
                else grow == other_axis_want
            )
        return True

    # The reference grammar assigns one anonymous region to each (row, column)
    # cell, while several functional groups may map to that region.  Reconstruct
    # one cell centroid from the union of its devices; comparing each subgroup
    # separately invents an order that the reference never declared.
    in_scope = [(tuple(str(d) for d in devs), int(_band_idx(grow, gcol)))
                for devs, grow, gcol in all_groups
                if _band_idx(grow, gcol) is not None and _in_scope(grow, gcol)]
    if len({index for _members, index in in_scope}) < 2:
        return None, ()
    band_members: Dict[int, Dict[str, float]] = {}
    missing_members: set[str] = set()
    for members, gidx in in_scope:
        coordinates: Dict[str, float] = {}
        for d in members:
            c = geom.centroid_of(d)
            if c is not None:
                coordinates[d] = c[axis]
        if len(coordinates) != len(set(members)) or not coordinates:
            missing_members.update(set(members) - set(coordinates))
            if not members:
                missing_members.add('empty_group')
            continue
        cell = band_members.setdefault(int(gidx), {})
        for member, coordinate in coordinates.items():
            prior = cell.setdefault(member, coordinate)
            if prior != coordinate:
                return None, (f'inconsistent_group_member:{member}',)
    if missing_members:
        return None, tuple(sorted(missing_members))
    if len(band_members) < 2 or want not in band_members:
        # Only one declared band on this axis -> no ordering to violate.
        return None, ()

    band_centroids = {
        index: math.fsum(coordinates.values()) / len(coordinates)
        for index, coordinates in band_members.items()
    }
    bands_sorted = sorted(band_centroids)
    total = 0
    bad = 0
    # Every ordered cell pair (lo < hi): the higher-ranked region centroid must
    # sit strictly beyond the lower-ranked one.  Report only pairs involving the
    # current group's cell, so each obligation retains local attribution.
    for li in range(len(bands_sorted)):
        for hj in range(li + 1, len(bands_sorted)):
            lo_idx, hi_idx = bands_sorted[li], bands_sorted[hj]
            if want not in (lo_idx, hi_idx):
                continue
            total += 1
            if not (band_centroids[lo_idx] < band_centroids[hi_idx]):
                bad += 1
    if total == 0:
        return None, ()
    return bad / total, ()


# ─────────────────────────────────────────────────────────────────────────────
# 3b. realised EQUAL-FOOTPRINT check (kind 'equal_size')
# ─────────────────────────────────────────────────────────────────────────────

# A declared @equal_size group's members must realise an IDENTICAL footprint (same
# placed width AND height) within a small RELATIVE spread. Relative-only (the dbu
# unit cancels), so scaling the whole layout cannot change the verdict.
_DEFAULT_EQUAL_SIZE_TOL = 0.02   # 2% relative footprint-spread slack (fraction in (0,1))


def check_equal_size(
    geom: GeometricPlacement,
    group_members: Sequence[str],
    *,
    rel_tol: float = _DEFAULT_EQUAL_SIZE_TOL,
) -> Tuple[Optional[float], Tuple[str, ...]]:
    """DIRECT check that an ``@equal_size`` group realises an IDENTICAL footprint.

    Reads each member's placed bbox off ``geom`` (resolving a fused-cluster member
    to its cluster bbox, exactly like ``centroid_of``) and measures the group's
    footprint SPREAD on BOTH axes:

      * ``width``  = ``x1 - x0`` of the placed bbox;
      * ``height`` = ``y1 - y0`` of the placed bbox.

    The metric is the larger of the two axes' RELATIVE spreads:

        spread_w = (max_w - min_w) / max_w
        spread_h = (max_h - min_h) / max_h
        measured = max(spread_w, spread_h)

    so a group PASSES iff EVERY member shares the same width AND the same height
    within ``rel_tol`` (``measured <= rel_tol``). Relative, so the dbu unit
    cancels and scaling the layout leaves the verdict unchanged.

    Returns ``(measured, missing)``:

      * ``measured`` is the max relative footprint spread (``0.0`` when every
        member is byte-identical; grows as footprints diverge);
      * ``missing`` names any member with NO realised bbox -- the group then
        cannot be measured honestly, so ``measured`` is ``None`` (UNMEASURABLE,
        token withheld), NEVER a false pass. A degenerate member whose bbox has a
        zero/negative width or height is also unmeasurable (no footprint to
        compare against).
    """
    widths: List[float] = []
    heights: List[float] = []
    missing: List[str] = []
    for m in group_members:
        bb = geom.device_bboxes.get(str(m))
        if bb is None:
            a_key = geom.cluster_member_to_a.get(str(m))
            if a_key is not None:
                bb = geom.device_bboxes.get(a_key)
        if bb is None:
            missing.append(str(m))
            continue
        x0, y0, x1, y1 = bb
        w = float(x1) - float(x0)
        h = float(y1) - float(y0)
        if w <= 0.0 or h <= 0.0:
            # a degenerate (zero-area) placed bbox has no footprint to compare.
            missing.append(str(m))
            continue
        widths.append(w)
        heights.append(h)

    if missing:
        return None, tuple(missing)
    if len(widths) < 2:
        # fewer than two measurable members -> nothing to compare (a single
        # device trivially equals itself). Honestly unmeasurable, not a free pass.
        return None, ("equal_size:fewer_than_two_members",)

    def _rel_spread(vals: List[float]) -> float:
        hi = max(vals)
        lo = min(vals)
        return (hi - lo) / hi if hi > 0.0 else 0.0

    measured = max(_rel_spread(widths), _rel_spread(heights))
    return measured, ()


# ─────────────────────────────────────────────────────────────────────────────
# 3c. realised RELATIVE-ORDER check (kind 'order')
# ─────────────────────────────────────────────────────────────────────────────

# A declared @order(a, b, rel) pins a's centroid left_of / above b's. The check is
# a strict inequality on the placed centroids -> the "measured" value is 0.0 when
# the order holds and the positive VIOLATION distance (relative to the cell span)
# when it is inverted, so the shared ``measured <= tol`` pass-test reads as "order
# satisfied". A tiny relative slack absorbs a one-grid tie at equal centroids.
_DEFAULT_ORDER_TOL = 0.0   # an order is satisfied iff the violation is zero


def check_order(
    geom: GeometricPlacement,
    a: str,
    b: str,
    rel: str,
    *,
    rel_tol: float = _DEFAULT_ORDER_TOL,
) -> Tuple[Optional[float], Tuple[str, ...]]:
    """DIRECT check that a declared ``@order(a, b, rel)`` holds on the placement.

    ``rel`` is ``"left_of"`` (a's centroid x is less than b's) or ``"above"`` (a's
    centroid y is greater than b's -- larger y is higher). The metric is the RELATIVE
    VIOLATION distance:

        left_of: viol = max(0, cx_a - cx_b) / span_x
        above:   viol = max(0, cy_b - cy_a) / span_y

    so ``viol == 0`` iff the order holds (a strictly on the declared side, or exactly
    equal -- a tie is not a violation) and a positive ``viol`` is the fraction of the
    cell extent by which the order is INVERTED. Relative (the dbu unit cancels via the
    cell span), so scaling the layout leaves the verdict unchanged.

    Returns ``(measured, missing)``: ``measured`` is ``None`` (UNMEASURABLE, token
    withheld -- NEVER a false pass) when either device has no realised centroid or the
    cell span on the relevant axis is degenerate (zero), with the blocking reason in
    ``missing``. ``rel`` outside the closed enum is also unmeasurable.
    """
    ca = geom.centroid_of(str(a))
    cb = geom.centroid_of(str(b))
    missing: List[str] = []
    if ca is None:
        missing.append(str(a))
    if cb is None:
        missing.append(str(b))
    if missing:
        return None, tuple(missing)
    span = geom.placement_span()
    if span is None:
        return None, (f"order:no_span:{a}:{b}",)
    span_x, span_y = span
    if rel == "left_of":
        if span_x <= 0.0:
            return None, (f"order:degenerate_span_x:{a}:{b}",)
        viol = max(0.0, ca[0] - cb[0]) / span_x
        return viol, ()
    if rel == "above":
        if span_y <= 0.0:
            return None, (f"order:degenerate_span_y:{a}:{b}",)
        # a above b => cy_a > cy_b; violation when b is higher than a.
        viol = max(0.0, cb[1] - ca[1]) / span_y
        return viol, ()
    return None, (f"order:unknown_rel:{rel}",)


# ─────────────────────────────────────────────────────────────────────────────
# 3d. realised ADJACENCY check (kind 'abut')
# ─────────────────────────────────────────────────────────────────────────────

# A declared @abut(a, b) pins a and b edge-adjacent: the gap on their FACING edge is
# at the DRC-minimum. The check measures the facing-edge gap relative to the larger
# device's facing extent and passes iff it is within a small relative slack (the two
# devices touch / nearly touch). A wider default channel gap FAILS.
_DEFAULT_ABUT_TOL = 0.15   # facing-edge gap up to 15% of the device extent counts as abutted


def check_abut(
    geom: GeometricPlacement,
    a: str,
    b: str,
    *,
    rel_tol: float = _DEFAULT_ABUT_TOL,
) -> Tuple[Optional[float], Tuple[str, ...]]:
    """DIRECT check that a declared ``@abut(a, b)`` placed the two devices adjacent.

    The two devices abut along whichever axis SEPARATES them more (the facing edge):
    if their x-extents are disjoint they face on x, else if their y-extents are
    disjoint they face on y. The facing-edge GAP is the clear distance between the
    near edges on that axis; the metric is that gap RELATIVE to the larger device's
    extent along the facing axis:

        measured = facing_gap / max(extent_a, extent_b)

    so ``measured == 0`` iff the two devices touch (zero facing gap) and grows with
    the gap; a pair packed at the DRC-minimum reads near-zero (PASS) while a pair left
    at the default routing channel reads well above the slack (FAIL). Relative (dbu
    cancels), scale-invariant.

    Returns ``(measured, missing)``: ``measured`` is ``None`` (UNMEASURABLE, blocking)
    when either device has no realised bbox, the larger facing extent is degenerate,
    or the two bboxes OVERLAP on both axes (no facing edge -- they are stacked, not
    abutted), with the reason in ``missing``. Never a silent pass.
    """
    ba = geom.device_bboxes.get(str(a))
    if ba is None:
        a_key = geom.cluster_member_to_a.get(str(a))
        if a_key is not None:
            ba = geom.device_bboxes.get(a_key)
    bb = geom.device_bboxes.get(str(b))
    if bb is None:
        b_key = geom.cluster_member_to_a.get(str(b))
        if b_key is not None:
            bb = geom.device_bboxes.get(b_key)
    missing: List[str] = []
    if ba is None:
        missing.append(str(a))
    if bb is None:
        missing.append(str(b))
    if missing:
        return None, tuple(missing)

    ax0, ay0, ax1, ay1 = ba
    bx0, by0, bx1, by1 = bb
    # disjoint on x? (one device entirely left of the other) -> facing edge is on x.
    x_disjoint = ax1 <= bx0 or bx1 <= ax0
    y_disjoint = ay1 <= by0 or by1 <= ay0
    if x_disjoint:
        gap = bx0 - ax1 if ax1 <= bx0 else ax0 - bx1
        extent = max(ax1 - ax0, bx1 - bx0)
    elif y_disjoint:
        gap = by0 - ay1 if ay1 <= by0 else ay0 - by1
        extent = max(ay1 - ay0, by1 - by0)
    else:
        # the bboxes overlap on BOTH axes -> no facing edge (stacked / overlapping).
        return None, (f"abut:overlapping_no_facing_edge:{a}:{b}",)
    if extent <= 0.0:
        return None, (f"abut:degenerate_extent:{a}:{b}",)
    measured = max(0.0, float(gap)) / float(extent)
    return measured, ()


# ─────────────────────────────────────────────────────────────────────────────
# 3e. realised ABSOLUTE-PLACEMENT check (kind 'fix')
# ─────────────────────────────────────────────────────────────────────────────

# A declared @fix(device, x, y) pins the device CENTROID to an absolute coordinate. The
# check is the ONLY geometric obligation that is genuinely ABSOLUTE (an absolute pin is a
# coordinate claim, not a relation): the realised centroid must sit within a small ABS
# tolerance of the pin. The tolerance is one manufacturing grid quantum (50 nm covers the
# gf180 / sky130 5 nm grid plus centroid half-grid rounding). The metric is the centroid
# offset divided by that tolerance so the shared ``measured <= 1.0`` pass-test reads as
# "centroid within tolerance of the pin": measured == 0 at the exact pin, 1.0 at the
# tolerance edge, > 1.0 when the device is off its pin.
_FIX_ABS_TOL_NM = 50.0   # absolute centroid-to-pin tolerance (nm); ~1 grid quantum
_DEFAULT_FIX_TOL = 1.0   # the shared rel_tol used as the pass threshold for kind 'fix'


def check_fix(
    geom: GeometricPlacement,
    device: str,
    pin_x_nm: float,
    pin_y_nm: float,
    *,
    abs_tol_nm: float = _FIX_ABS_TOL_NM,
) -> Tuple[Optional[float], Tuple[str, ...]]:
    """DIRECT check that a declared ``@fix(device, x, y)`` placed the device on its pin.

    ``pin_x_nm`` / ``pin_y_nm`` are the declared centroid in NM (the caller scales the
    microns the annotation carries). The realised centroid is read off ``geom`` (resolving
    a fused-cluster member to its cluster bbox, like ``centroid_of``). The metric is the
    Euclidean centroid-to-pin offset divided by ``abs_tol_nm`` so:

        measured = sqrt((cx - pin_x)^2 + (cy - pin_y)^2) / abs_tol_nm

    PASSES iff ``measured <= 1.0`` (the centroid is within one grid quantum of the pin).
    This is an ABSOLUTE check (an absolute pin is a coordinate claim), unlike the relative
    structural obligations -- the tolerance is a fixed grid quantum, not a fraction of the
    cell span.

    Returns ``(measured, missing)``: ``measured`` is ``None`` (UNMEASURABLE, BLOCKING --
    never a silent pass) when the device has NO realised centroid (no bbox), naming it in
    ``missing``.
    """
    c = geom.centroid_of(str(device))
    if c is None:
        return None, (str(device),)
    if abs_tol_nm <= 0.0:
        return None, (f"fix:nonpositive_tol:{device}",)
    dx = float(c[0]) - float(pin_x_nm)
    dy = float(c[1]) - float(pin_y_nm)
    offset = (dx * dx + dy * dy) ** 0.5
    return offset / float(abs_tol_nm), ()


# ─────────────────────────────────────────────────────────────────────────────
# 4. realised wire-WIDTH check (kind 'wire_width')
# ─────────────────────────────────────────────────────────────────────────────

# A declared wire width is an EXACT-OR-GREATER obligation (the router must lay the
# net AT LEAST as wide as declared), with a small RELATIVE slack so a one-grid
# rounding of the centred rectangle never trips a width that is physically met.
_DEFAULT_WIRE_WIDTH_TOL = 0.02   # 2% relative under-width slack (fraction in (0,1))
_DEFAULT_MATCH_LENGTH_TOL = 0.10  # 10% relative wire-length imbalance slack (0,1)


@dataclass(frozen=True)
class RoutedGeometry:
    """The routed-wire geometry a ``wire_width`` obligation measures against.

    A frozen view a caller builds once from a routed cell's net-tagged shapes and
    threads into :func:`clay.layout.intent_obligations.check_layout_intent`. When it
    is not supplied, every declared ``wire_width`` obligation is UNMEASURABLE (token
    withheld) -- the same blocking-on-missing-geometry policy the LDE / structural
    checks use.

      * ``net_wire_bboxes`` -- ``{net: [(x0, y0, x1, y1), ...]}`` the per-net routed
        wire rectangles (absolute, dbu). Every routed shape of a net (trunk, riser,
        rail body, pad) contributes its bbox; the check measures the net's WIDEST
        wire as the max over shapes of ``min(bbox_width, bbox_height)`` (the wire's
        cross-axis extent -- a horizontal trunk's width is its height, a vertical
        riser's width is its width), so a widened net is detected on whichever wire
        carries the declared width.
    """

    net_wire_bboxes: Dict[str, List[Tuple[float, float, float, float]]]
    # Retain the verified artifact for checks requiring layers, connectivity and
    # fixed device metal. A raw bbox view cannot authorize these checks.
    exact_artifact: object = None
    spacing_tech: object = None

    @staticmethod
    def from_exact_artifact(artifact, *, tech=None) -> "RoutedGeometry":
        """Build the routed-wire view from one exhaustive exact-GDS manifest.

        The ownership manifest is accepted only through an
        :class:`ExactEmittedArtifact`: ``validate()`` re-hashes the current GDS
        and replays the manifest/scope bindings, while ``scope.exhaustive``
        proves that the allocated, routed, emitted, and written route unions are
        identical and that no heal/fill/snap geometry was added.  Consequently
        these net-tagged rectangles are a measurement index for the exact GDS,
        not an unverified router-side estimate.
        """
        from clay.layout.route.exact_artifact_evidence import ExactEmittedArtifact

        if type(artifact) is not ExactEmittedArtifact:
            raise TypeError(
                "routed geometry requires an exact emitted artifact"
            )
        artifact.validate()
        if artifact.scope.exhaustive is not True:
            raise ValueError("exact route-emission scope is not exhaustive")
        view = RoutedGeometry.from_tagged(artifact.route_manifest.entries)
        return RoutedGeometry(view.net_wire_bboxes, exact_artifact=artifact, spacing_tech=tech)

    @staticmethod
    def from_tagged(tagged) -> "RoutedGeometry":
        """Build a :class:`RoutedGeometry` from a routed cell's ``tagged`` shapes.

        ``tagged`` is the ``[(net, (layer, datatype), bbox), ...]`` list a
        :class:`~clay.layout.generic_router.RoutedCell` carries. Mechanical: every
        tagged shape's bbox is bucketed by its net. No layer filtering -- the wire
        width of a net is the widest cross-axis extent over ALL its routed metal, so
        a declared width realised on ANY of the net's wires (trunk / riser / rail)
        satisfies the obligation.
        """
        buckets: Dict[str, List[Tuple[float, float, float, float]]] = {}
        for entry in tagged or ():
            try:
                net, _key, bbox = entry
            except (ValueError, TypeError):
                continue
            if bbox is None or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = bbox
            buckets.setdefault(str(net), []).append(
                (float(x0), float(y0), float(x1), float(y1)))
        return RoutedGeometry(net_wire_bboxes=buckets)

    def max_wire_width(self, net: str) -> Optional[float]:
        """The net's WIDEST realised wire width (dbu), or ``None`` when absent.

        For each routed rectangle of the net the wire WIDTH is its cross-axis extent
        ``min(width, height)`` (a long horizontal trunk's width is its short vertical
        extent; a tall vertical riser's width is its short horizontal extent). The
        net's realised width is the MAX of those over all its wires -- the declared
        width is met when the net has AT LEAST one wire that wide. ``None`` when the
        net has no routed geometry (the obligation is then UNMEASURABLE/blocking).
        """
        boxes = self.net_wire_bboxes.get(str(net))
        if not boxes:
            return None
        widest = None
        for (x0, y0, x1, y1) in boxes:
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            cross = min(w, h)
            if widest is None or cross > widest:
                widest = cross
        return widest

    def routed_length(self, net: str) -> Optional[float]:
        """The net's total ROUTED wire-RUN length (dbu), or ``None`` when absent.

        Convention (the wire-length analogue of :meth:`max_wire_width`): for each
        routed rectangle of the net the wire RUN length is its LONG-axis extent
        ``max(width, height)`` (a horizontal trunk's run is its width; a vertical
        riser's run is its height), and the net's routed length is the SUM of those
        runs over all its wire shapes. This is the manhattan run length of the net's
        routed metal -- the quantity whose imbalance between two matched nets makes
        their wire parasitic (R / C) differ. ``None`` when the net has no routed
        geometry (the obligation is then UNMEASURABLE / blocking).
        """
        boxes = self.net_wire_bboxes.get(str(net))
        if not boxes:
            return None
        total = 0.0
        for (x0, y0, x1, y1) in boxes:
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            total += max(w, h)
        return total


def check_match_length(
    geom: RoutedGeometry,
    net_a: str,
    net_b: str,
    *,
    rel_tol: float = _DEFAULT_MATCH_LENGTH_TOL,
) -> Tuple[Optional[float], Tuple[str, ...]]:
    """RELATIVE wire-length-balance check for one ``@match_length(a, b)`` decl.

    Measures each net's total ROUTED wire-run length off the same routed geometry
    ``check_wire_width`` uses (:meth:`RoutedGeometry.routed_length`) and reduces the
    pair to a scale-invariant relative imbalance:

        measured_imbalance = |La - Lb| / max(La, Lb)

    so ``imbalance == 0`` iff the two matched nets have identical routed length and a
    pair whose lengths diverge yields a positive imbalance proportional to the
    mismatch. This feeds the obligation's shared ``measured_imbalance <= rel_tol``
    pass test, so two matched / differential nets routed with balanced length PASS
    and a deliberately unbalanced pair FAILS, never silently passes.

    Returns ``(measured, missing)``; ``measured`` is ``None`` (UNMEASURABLE) when
    EITHER net has no routed geometry (blocking reason in ``missing``), or when both
    measured lengths are zero (degenerate -- no routed run to balance). Scale note:
    the metric is a RATIO of two dbu lengths, so the dbu unit cancels
    (scale-invariant), per the project-wide rule.
    """
    la = geom.routed_length(net_a)
    lb = geom.routed_length(net_b)
    missing: List[str] = []
    if la is None:
        missing.append(f"match_length:no_routed_geometry:{net_a}")
    if lb is None:
        missing.append(f"match_length:no_routed_geometry:{net_b}")
    if missing:
        return None, tuple(missing)
    denom = max(float(la), float(lb))
    if denom <= 0.0:
        return None, (f"match_length:zero_length:{net_a}:{net_b}",)
    imbalance = abs(float(la) - float(lb)) / denom
    return imbalance, ()


def check_wire_width(
    geom: RoutedGeometry,
    net: str,
    declared_width: float,
    *,
    rel_tol: float = _DEFAULT_WIRE_WIDTH_TOL,
) -> Tuple[Optional[float], Tuple[str, ...]]:
    """DIRECT realised-width check for one ``@wire_width(net, w)`` declaration.

    Measures the net's WIDEST realised wire width off the routed geometry and
    reduces it to a RELATIVE under-width fraction against the declared width:

        imbalance = max(0, (declared - realised) / declared)

    so ``imbalance == 0`` iff the realised width is AT-OR-ABOVE the declared width
    (exact-or-greater), and a net routed NARROWER than declared yields a positive
    imbalance proportional to the shortfall. Reducing the metric this way means the
    obligation's existing ``measured_imbalance <= rel_tol`` pass test (shared with
    every other obligation) is exactly the "realised >= declared (within slack)"
    contract -- a router that did NOT widen the net FAILS, never silently passes.

    Returns ``(measured, missing)``; ``measured`` is ``None`` (UNMEASURABLE) when
    the net has no routed geometry or the declared width is non-positive, with the
    blocking reason in ``missing``. Scale note: this metric is a RATIO of two dbu
    lengths, so the dbu unit cancels (scale-invariant), per the project-wide rule.
    """
    if declared_width <= 0:
        return None, (f"wire_width:non_positive_declared:{net}",)
    realised = geom.max_wire_width(net)
    if realised is None:
        return None, (f"wire_width:no_routed_geometry:{net}",)
    shortfall = (float(declared_width) - float(realised)) / float(declared_width)
    return max(0.0, shortfall), ()


__all__ = [
    "PlacedClusterGeom",
    "GeometricPlacement",
    "InterdigVerdict",
    "RoutedGeometry",
    "MatchingCheck",
    "check_interdigitation",
    "check_matching",
    "check_common_centroid",
    "check_place_group_realized",
    "check_equal_size",
    "check_order",
    "check_abut",
    "check_fix",
    "check_wire_width",
    "check_match_length",
]
