"""P1 -- zero-coordinate placement-intent container (QSR relation network).

Design: docs/research/placement-intent-container-design-2026-06-14.html

This module holds ONLY qualitative spatial relations between devices and
between a device and a virtual frame sentinel. It NEVER stores an absolute
coordinate or a micron-scale value. The zero-coordinate property is by
construction: there is no field that accepts, derives, or serialises a
geometric quantity. Relations are categorical symbols drawn from a fixed,
finite QSR vocabulary; constraint strength is encoded as the *cardinality*
of the allowed-relation set (single relation = anchored, full set = free).

Pure data structure. stdlib only. No pipeline coupling (P2 extraction and
P3 lowering attach later). Deterministic serialisation for golden tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# QSR relation vocabulary (the FULL set).
#
# A finite, explicit word list. Two families, both purely categorical:
#
#  * Rectangle-Algebra direction / Allen-interval relations (per-axis ordering
#    and meeting/alignment of device extents) -- the "where relative to" words.
#  * RCC-8 subset topology (containment / boundary contact / disconnection)
#    -- the "inside / touching / apart" words.
#
# "free / dont-care" == an edge whose allowed-relation set is exactly this
# full vocabulary: the relation is completely unconstrained.
# --------------------------------------------------------------------------- #

# Rectangle-Algebra / Allen direction relations.
_RA_DIRECTION = (
    "left",          # A entirely left of B (Allen before on X)
    "right",         # A entirely right of B (Allen after on X)
    "above",         # A entirely above B (Allen after on Y)
    "below",         # A entirely below B (Allen before on Y)
    "left-meets",    # A's right edge abuts B's left edge (Allen meets, X)
    "right-meets",   # A's left edge abuts B's right edge (Allen met-by, X)
    "above-meets",   # A's bottom edge abuts B's top edge (Allen meets, Y)
    "below-meets",   # A's top edge abuts B's bottom edge (Allen met-by, Y)
    "align-x",       # equal X extent (Allen equals on X: left/right edges flush)
    "align-y",       # equal Y extent (Allen equals on Y: top/bottom edges flush)
)

# RCC-8 subset topology relations.
_RCC8_SUBSET = (
    "DC",     # disconnected (no contact, apart)
    "EC",     # externally connected (abut, edges touch, interiors disjoint)
    "TPP",    # tangential proper part (inside, touching the boundary)
    "NTPP",   # non-tangential proper part (strictly interior)
)

# The complete, ordered vocabulary. Order is fixed for deterministic output.
_ALL_RELATION_SYMBOLS: Tuple[str, ...] = _RA_DIRECTION + _RCC8_SUBSET

# Frame sentinel sides.
_FRAME_SIDES: Tuple[str, ...] = ("L", "R", "T", "B")

# Strength labels (Cassowary-style constraint hierarchy). Categorical only.
_STRENGTHS = frozenset({"REQUIRED", "STRONG", "WEAK"})


def all_relation_symbols() -> FrozenSet[str]:
    """The complete finite QSR relation vocabulary.

    An edge whose allowed set equals this is `free` (dont-care)."""
    return frozenset(_ALL_RELATION_SYMBOLS)


def _classify_anchoring(symbols: FrozenSet[str]) -> str:
    """anchoring == cardinality bucket of the allowed-relation set.

    single relation        -> 'anchored'  (pinned, may not be broken for area)
    small disjunction       -> 'ordered'   (only relative order/side fixed)
    full vocabulary         -> 'free'      (dont-care filler)
    empty                   -> 'free'      (no constraint == dont-care)
    """
    full = all_relation_symbols()
    n = len(symbols)
    if n == 0:
        return "free"
    if symbols == full:
        return "free"
    if n == 1:
        return "anchored"
    # 2 <= n < |full|
    return "ordered"


# --------------------------------------------------------------------------- #
# nodes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DeviceNode:
    """A device node. All attributes are symbolic -- never a coordinate.

    `w_class` / `l_class` are symbolic size *classes* (e.g. 'wide', 'min'),
    not micron values; they default to None and are not used by P1 tests but
    keep the node surface honest about being symbol-only."""

    name: str
    kind: Optional[str] = None
    role: Optional[str] = None
    w_class: Optional[str] = None
    l_class: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "role": self.role,
            "w_class": self.w_class,
            "l_class": self.l_class,
        }


# --------------------------------------------------------------------------- #
# edges
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RelationEdge:
    """A binary QSR relation between two endpoints.

    `symbols` is the allowed-relation disjunction set (frozenset of vocabulary
    words). `strength` is a Cassowary-style categorical label. `anchoring` is
    derived from the cardinality of `symbols` -- it is NOT an independent field
    the caller can spoof; it is computed from the set."""

    a: str
    b: str
    symbols: FrozenSet[str]
    strength: str = "STRONG"

    @property
    def anchoring(self) -> str:
        return _classify_anchoring(self.symbols)

    def as_dict(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "symbols": sorted(self.symbols),
            "strength": self.strength,
            "anchoring": self.anchoring,
        }


@dataclass(frozen=True)
class Group:
    """A hyperedge over a set of devices (symmetry / common-centroid / abut
    cluster / guard-ring containment). `members` order is preserved as given,
    but serialisation sorts for determinism."""

    kind: str
    members: Tuple[str, ...]

    def as_dict(self) -> dict:
        return {"kind": self.kind, "members": list(self.members)}


# --------------------------------------------------------------------------- #
# the container
# --------------------------------------------------------------------------- #
class RelationNetwork:
    """Zero-coordinate placement-intent container.

    Holds device nodes, an optional frame sentinel (L/R/T/B), binary QSR
    relation edges (device-device and device-frame), open-side flags, and
    symmetry/common-centroid hyperedges. No coordinate ever enters."""

    def __init__(self) -> None:
        self._devices: Dict[str, DeviceNode] = {}
        self._has_frame: bool = False
        # (a, b) -> RelationEdge for device-device relations (a, b ordered key)
        self._edges: Dict[Tuple[str, str], RelationEdge] = {}
        # (dev, side) -> RelationEdge for device-frame relations
        self._frame_edges: Dict[Tuple[str, str], RelationEdge] = {}
        # set of (dev, side) flagged open (no neighbour beyond dev on that side)
        self._open_sides: set = set()
        self._groups: List[Group] = []
        # Categorical provenance tag (e.g. 'style-transfer'); never a coordinate
        # and never part of as_dict() numeric leaves. None until an extractor
        # stamps it. A hand-built P1 container leaves it None.
        self.provenance: Optional[str] = None

    # --- nodes ------------------------------------------------------------- #
    def add_device(
        self,
        name: str,
        kind: Optional[str] = None,
        role: Optional[str] = None,
        w_class: Optional[str] = None,
        l_class: Optional[str] = None,
    ) -> DeviceNode:
        node = DeviceNode(
            name=name, kind=kind, role=role, w_class=w_class, l_class=l_class
        )
        self._devices[name] = node
        return node

    def has_device(self, name: str) -> bool:
        return name in self._devices

    def device(self, name: str) -> DeviceNode:
        return self._devices[name]

    def devices(self) -> List[DeviceNode]:
        return [self._devices[n] for n in sorted(self._devices)]

    # --- frame ------------------------------------------------------------- #
    def add_frame(self) -> None:
        """Create the frame sentinel (four side sentinels L/R/T/B).

        The frame is a virtual region: it carries no coordinate, it is only
        the other endpoint of boundary/quadrant relations."""
        self._has_frame = True

    def has_frame(self) -> bool:
        return self._has_frame

    def frame_sides(self) -> FrozenSet[str]:
        if not self._has_frame:
            return frozenset()
        return frozenset(_FRAME_SIDES)

    # --- vocabulary -------------------------------------------------------- #
    @staticmethod
    def all_relation_symbols() -> FrozenSet[str]:
        """The complete finite QSR relation vocabulary (instance accessor).

        An edge whose allowed set equals this is `free` (dont-care)."""
        return all_relation_symbols()

    # --- binary device-device relations ------------------------------------ #
    @staticmethod
    def _edge_key(a: str, b: str) -> Tuple[str, str]:
        return (a, b)

    def relate(
        self,
        a: str,
        b: str,
        symbols: Iterable[str],
        strength: str = "STRONG",
    ) -> RelationEdge:
        """Add / replace the QSR relation edge between devices a and b.

        `symbols` is the allowed-relation disjunction set. Its cardinality
        determines anchoring (single = anchored, full = free)."""
        sset = frozenset(symbols)
        self._validate_symbols(sset)
        self._validate_strength(strength)
        edge = RelationEdge(a=a, b=b, symbols=sset, strength=strength)
        self._edges[self._edge_key(a, b)] = edge
        return edge

    def relation(self, a: str, b: str) -> RelationEdge:
        return self._edges[self._edge_key(a, b)]

    def has_relation(self, a: str, b: str) -> bool:
        return self._edge_key(a, b) in self._edges

    def edges(self) -> List[RelationEdge]:
        return [self._edges[k] for k in sorted(self._edges)]

    # --- device-frame relations + open sides ------------------------------- #
    def relate_frame(
        self,
        dev: str,
        side: str,
        symbols: Iterable[str],
        strength: str = "STRONG",
    ) -> RelationEdge:
        """Relate a device to a frame side (e.g. cap TPP frame@R = hug right
        boundary). Endpoint `b` is the symbolic side token, not a coordinate."""
        if side not in _FRAME_SIDES:
            raise ValueError(f"unknown frame side: {side!r}")
        sset = frozenset(symbols)
        self._validate_symbols(sset)
        self._validate_strength(strength)
        edge = RelationEdge(a=dev, b=side, symbols=sset, strength=strength)
        self._frame_edges[(dev, side)] = edge
        return edge

    def relation_frame(self, dev: str, side: str) -> RelationEdge:
        return self._frame_edges[(dev, side)]

    def has_frame_relation(self, dev: str, side: str) -> bool:
        return (dev, side) in self._frame_edges

    # P2 extractor predicate name (alias of has_frame_relation): "is there a
    # device<->frame-side relation edge for (dev, side)?". Same categorical
    # fact, the name the placement-relation tests query by.
    def has_relation_frame(self, dev: str, side: str) -> bool:
        return self.has_frame_relation(dev, side)

    def set_open_side(self, dev: str, side: str) -> None:
        """Mark a device side as open: no device sits beyond it on that side.

        Stored as a categorical (dev, side) fact -- no gap distance."""
        if side not in _FRAME_SIDES:
            raise ValueError(f"unknown frame side: {side!r}")
        self._open_sides.add((dev, side))

    def is_open_side(self, dev: str, side: str) -> bool:
        return (dev, side) in self._open_sides

    # --- hyperedges -------------------------------------------------------- #
    def add_group(self, kind: str, members: Iterable[str]) -> Group:
        g = Group(kind=kind, members=tuple(members))
        self._groups.append(g)
        return g

    def groups(self) -> List[Group]:
        return list(self._groups)

    # --- device-level anchoring -------------------------------------------- #
    def anchoring_of(self, dev: str) -> str:
        """Overall anchoring of a device = the strongest of all its edges.

        anchored > ordered > free. A device with no constraining edge (or only
        full-set / empty edges) is free -- e.g. a dummy filler. A device with
        any single-relation (REQUIRED or not) edge is anchored."""
        order = {"anchored": 2, "ordered": 1, "free": 0}
        best = "free"
        best_rank = 0
        for edge in self._iter_edges_touching(dev):
            rank = order[edge.anchoring]
            if rank > best_rank:
                best_rank = rank
                best = edge.anchoring
        return best

    def _iter_edges_touching(self, dev: str) -> Iterable[RelationEdge]:
        for edge in self._edges.values():
            if edge.a == dev or edge.b == dev:
                yield edge
        for edge in self._frame_edges.values():
            if edge.a == dev:
                yield edge

    # --- validation -------------------------------------------------------- #
    @staticmethod
    def _validate_symbols(symbols: FrozenSet[str]) -> None:
        unknown = symbols - all_relation_symbols()
        if unknown:
            raise ValueError(f"unknown QSR relation symbols: {sorted(unknown)}")

    @staticmethod
    def _validate_strength(strength: str) -> None:
        if strength not in _STRENGTHS:
            raise ValueError(
                f"unknown strength {strength!r}; expected one of {sorted(_STRENGTHS)}"
            )

    # --- serialisation ----------------------------------------------------- #
    def as_dict(self) -> dict:
        """Fully symbolic, deterministic serialisation.

        Every leaf is a string, a bool, or a list/dict of those. There is no
        numeric leaf at all -- the zero-coordinate contract holds because no
        method ever accepts or stores a coordinate, so none can be emitted."""
        return {
            "devices": [d.as_dict() for d in self.devices()],
            "frame": sorted(self.frame_sides()),
            "edges": [e.as_dict() for e in self.edges()],
            "frame_edges": [
                self._frame_edges[k].as_dict()
                for k in sorted(self._frame_edges)
            ],
            "open_sides": [
                {"device": d, "side": s} for (d, s) in sorted(self._open_sides)
            ],
            "groups": [g.as_dict() for g in self._groups],
        }


# --------------------------------------------------------------------------- #
# P2 -- E layer: extract the QSR relation network FROM a reference layout.
#
# Reads intent.device_bboxes (reference correspondence coordinates) INTERNALLY,
# computes purely RELATIVE facts (who is left/right/above/below whom, which
# device hugs the placement boundary, which boundary side is open) and emits
# ONLY relation symbols into a RelationNetwork. The coordinates are consumed and
# dropped here -- same leakage discipline as style_order._reference_rows /
# extract_io_edge_intent. provenance='style-transfer'.
#
# TRANSLATION INVARIANCE BY CONSTRUCTION: every decision below is a comparison
# of DIFFERENCES (centroid_a - centroid_b, edge - overall_extent_edge, span
# fractions). Adding the same constant offset to every bbox cancels in every
# such difference, so the emitted symbol set -- and therefore as_dict() -- is
# byte-identical under any pure translation of the reference. No absolute
# coordinate, axis value or micron span ever crosses into the container.
# --------------------------------------------------------------------------- #

# A device edge counts as HUGGING the overall placement boundary on a side when
# the edge-to-boundary margin is below this fraction of the placement extent
# span on that axis. A RELATIVE criterion (no absolute um), so it is invariant
# to translation AND scale of the reference frame.
_HUG_MARGIN_FRACTION: float = 0.02

# A device counts as LARGE-AREA passive (the cap that anchors a die edge) when
# its bbox area is at least this fraction of the largest device-area in the
# placement. Relative, translation/scale invariant.
_LARGE_AREA_FRACTION: float = 0.5


def _kind_role_of(intent) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """name -> (kind, role) from the DESIGN (netlist) only -- no geometry.

    kind: 'nmos'/'pmos' for MOS, 'cap'/'res' for passives. role: a coarse
    DESIGN-sourced tag ('dummy' when the device is in intent.free_devices,
    else None). Never derived from coordinates."""
    free = {str(n) for n in (getattr(intent, "free_devices", None) or [])}
    out: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for d in getattr(intent, "devices", []) or []:
        nm = str(d.name)
        out[nm] = (str(getattr(d, "kind", None) or "") or None,
                   "dummy" if nm in free else None)
    for p in getattr(intent, "passives", []) or []:
        nm = str(p.name)
        out[nm] = (str(getattr(p, "kind", None) or "") or None,
                   "dummy" if nm in free else None)
    return out


def _anchored_names(intent) -> set:
    """Names the DESIGN marks as anchored: any cap / large passive, and every
    member of a matched / diff / mirror pair (symmetric pairs). DESIGN-sourced
    (netlist topology), never geometry."""
    anchored: set = set()
    for p in getattr(intent, "passives", []) or []:
        anchored.add(str(p.name))  # caps + resistors are matched/anchored passives
    for pairs_attr in ("matched_pairs", "diff_pairs", "mirror_pairs"):
        for pair in getattr(intent, pairs_attr, None) or []:
            for nm in pair:
                anchored.add(str(nm))
    return anchored


def extract_placement_relations(intent, *, tech_id: str):
    """Reference layout -> P1 RelationNetwork (QSR relations only, zero coords).

    Reads ``intent.device_bboxes`` (the reference correspondence) INTERNALLY,
    derives purely relative facts and emits ONLY relation symbols. The result
    carries ``provenance='style-transfer'``; no coordinate crosses out.

    What it emits:
      * one ``add_device(kind, role)`` per corresponded device (DESIGN-sourced
        kind/role; geometry never sets these);
      * ``add_frame()`` (the virtual die boundary sentinel);
      * binary QSR relations between devices from their RELATIVE bbox position
        (strict left/right/above/below, with abut 'EC' / 'align-x' / 'align-y'
        refinements when edges meet / extents are flush) -- the symbol SET's
        cardinality encodes anchoring: a single symbol for an anchored device
        (cap / matched / symmetric pair member), a small disjunction (ordered)
        for a plain device, the FULL vocabulary (free) for a dummy / no-
        constraint device;
      * for a large-area passive (the cap) whose bbox hugs an overall-extent
        side: ``relate_frame(cap, side, {'TPP'}, 'REQUIRED')`` (boundary hug),
        and ``set_open_side(cap, side)`` when no device lies beyond the cap on
        that side (the no-neighbour fact).

    ``tech_id`` is interface surface (future PDK-specific relation rules); the
    extractor consumes no target-PDK value. Translation invariance is by
    construction (see module note) -- the leakage guard the tests assert.
    """
    net = RelationNetwork()
    net.provenance = "style-transfer"
    net.add_frame()

    kr = _kind_role_of(intent)
    bboxes = {
        str(k): tuple(float(v) for v in bb)
        for k, bb in (getattr(intent, "device_bboxes", None) or {}).items()
    }

    # nodes: DESIGN-sourced kind/role (never geometry). Deterministic order.
    for nm in sorted(set(kr) | set(bboxes)):
        kind, role = kr.get(nm, (None, None))
        net.add_device(nm, kind=kind, role=role)

    if not bboxes:
        # No reference correspondence -> nodes + frame only (honest empty
        # relation graph, never a fabricated relation).
        return net

    anchored = _anchored_names(intent)

    # overall placement extent (consumed INTERNALLY, never stored).
    ox0 = min(bb[0] for bb in bboxes.values())
    oy0 = min(bb[1] for bb in bboxes.values())
    ox1 = max(bb[2] for bb in bboxes.values())
    oy1 = max(bb[3] for bb in bboxes.values())
    span_x = ox1 - ox0
    span_y = oy1 - oy0
    # abut / align tolerance: a small RELATIVE fraction of the extent span.
    tol_x = _HUG_MARGIN_FRACTION * span_x
    tol_y = _HUG_MARGIN_FRACTION * span_y

    areas = {nm: max(0.0, (bb[2] - bb[0])) * max(0.0, (bb[3] - bb[1]))
             for nm, bb in bboxes.items()}
    max_area = max(areas.values()) if areas else 0.0

    names = sorted(bboxes)

    # --- binary device-device relations (deterministic, translation-invariant) #
    # For each unordered pair (a<b by name), emit at most one edge keyed (a,b).
    # The relation SET cardinality encodes anchoring of the edge:
    #   anchored endpoint -> single sharpest symbol;
    #   dummy endpoint    -> full vocabulary (free);
    #   plain             -> small disjunction (the side + its meets refinement).
    full = all_relation_symbols()
    for i, a in enumerate(names):
        abb = bboxes[a]
        for b in names[i + 1:]:
            bbb = bboxes[b]
            a_dummy = (kr.get(a, (None, None))[1] == "dummy")
            b_dummy = (kr.get(b, (None, None))[1] == "dummy")
            if a_dummy or b_dummy:
                # dummy / no-constraint filler: free relation (full set).
                net.relate(a, b, set(full), strength="WEAK")
                continue

            # primary axis = the one with the larger relative separation, so the
            # call is a pure comparison of differences (translation invariant).
            acx = (abb[0] + abb[2]) / 2.0
            acy = (abb[1] + abb[3]) / 2.0
            bcx = (bbb[0] + bbb[2]) / 2.0
            bcy = (bbb[1] + bbb[3]) / 2.0
            dx = acx - bcx
            dy = acy - bcy

            sym: set = set()
            if abs(dx) * max(span_y, 1e-9) >= abs(dy) * max(span_x, 1e-9):
                # x dominates: left/right of, refined by edge meeting.
                if dx <= 0.0:
                    sym.add("left")
                    if abs(abb[2] - bbb[0]) <= tol_x:
                        sym.add("left-meets")
                        sym.add("EC")
                else:
                    sym.add("right")
                    if abs(abb[0] - bbb[2]) <= tol_x:
                        sym.add("right-meets")
                        sym.add("EC")
                if abs(abb[1] - bbb[1]) <= tol_y and abs(abb[3] - bbb[3]) <= tol_y:
                    sym.add("align-y")
            else:
                # y dominates: above/below, refined by edge meeting.
                if dy >= 0.0:
                    sym.add("above")
                    if abs(abb[1] - bbb[3]) <= tol_y:
                        sym.add("above-meets")
                        sym.add("EC")
                else:
                    sym.add("below")
                    if abs(abb[3] - bbb[1]) <= tol_y:
                        sym.add("below-meets")
                        sym.add("EC")
                if abs(abb[0] - bbb[0]) <= tol_x and abs(abb[2] - bbb[2]) <= tol_x:
                    sym.add("align-x")

            # anchoring via cardinality: an anchored endpoint collapses the set
            # to its single sharpest symbol; a plain pair keeps the disjunction.
            if a in anchored or b in anchored:
                sharp = _sharpest_symbol(sym)
                net.relate(a, b, {sharp}, strength="REQUIRED")
            else:
                net.relate(a, b, sym, strength="STRONG")

    # --- large-area passive (the cap) hugging a die-boundary side -------------- #
    # A cap whose bbox edge sits within the relative hug margin of the overall
    # extent boundary on a side hugs that side (TPP frame@side, REQUIRED). The
    # side is OPEN when no OTHER device extends beyond the cap on that side.
    for nm in names:
        kind = kr.get(nm, (None, None))[0]
        if kind != "cap":
            continue
        if max_area > 0.0 and areas[nm] < _LARGE_AREA_FRACTION * max_area:
            # require LARGE-AREA passive: a small parasitic cap does not anchor.
            continue
        bb = bboxes[nm]
        # margins to each overall side (differences only -> translation invariant)
        side_margin = {
            "L": bb[0] - ox0,
            "R": ox1 - bb[2],
            "B": bb[1] - oy0,
            "T": oy1 - bb[3],
        }
        side_tol = {"L": tol_x, "R": tol_x, "B": tol_y, "T": tol_y}
        for side in ("L", "R", "T", "B"):
            if side_margin[side] <= side_tol[side]:
                net.relate_frame(nm, side, {"TPP"}, strength="REQUIRED")
                # open side: no OTHER device lies beyond the cap on that side.
                if _no_device_beyond(nm, side, bboxes):
                    net.set_open_side(nm, side)

    return net


def _sharpest_symbol(sym: set) -> str:
    """The single most specific relation symbol of a disjunction set -- used to
    collapse an anchored edge to one (anchored) relation. Preference order:
    topology contact (EC) then the meeting refinements then the bare direction;
    deterministic by the fixed vocabulary order as the final tiebreak."""
    pref = (
        "EC",
        "left-meets", "right-meets", "above-meets", "below-meets",
        "align-x", "align-y",
        "left", "right", "above", "below",
    )
    for p in pref:
        if p in sym:
            return p
    # any direction-less / empty set: a single, defensible free-direction symbol
    # from the fixed vocabulary (deterministic).
    return sorted(sym)[0] if sym else "DC"


def _no_device_beyond(dev: str, side: str, bboxes: Dict[str, tuple]) -> bool:
    """True when NO other device's bbox extends past ``dev`` on ``side``.

    A pure comparison of edge coordinates (differences) -> translation
    invariant. 'beyond on L' = some other device's left edge is left of dev's
    left edge while overlapping dev's y-extent; symmetric for R/B/T."""
    db = bboxes[dev]
    for nm, bb in bboxes.items():
        if nm == dev:
            continue
        if side == "L":
            if bb[0] < db[0] and bb[3] > db[1] and bb[1] < db[3]:
                return False
        elif side == "R":
            if bb[2] > db[2] and bb[3] > db[1] and bb[1] < db[3]:
                return False
        elif side == "B":
            if bb[1] < db[1] and bb[2] > db[0] and bb[0] < db[2]:
                return False
        elif side == "T":
            if bb[3] > db[3] and bb[2] > db[0] and bb[0] < db[2]:
                return False
    return True


__all__ = [
    "RelationNetwork",
    "DeviceNode",
    "RelationEdge",
    "Group",
    "all_relation_symbols",
    "extract_placement_relations",
]
