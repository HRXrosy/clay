"""General routed-cell data types + emit + connectivity oracle (router-agnostic).

These are the SHARED, NON-router infrastructure symbols every routed-cell
consumer needs, factored OUT of :mod:`clay.layout.generic_router` so they no
longer live behind the legacy greedy ``route_generic`` planner:

  * :class:`RoutedCell` -- the routed-result data type (``polys`` / ``tagged`` /
    ``labels`` / ``shorts`` / ``net_complete``) the production flow, the
    fault-intent diagnosis, and the DRC-violation join consume. The
    :mod:`clay.layout.route.route_cell` PathFinder result adapts INTO this shape
    (its ``to_routed_cell``), so a route_cell route flows through the SAME
    downstream consumers as a legacy route_generic result -- one canonical result
    surface, never a second divergent geometry path.
  * :func:`net_terminals_from_netlist` -- the ``net -> [(device, term), ...]``
    schematic connectivity authority (keyed on a parsed CellDef + a placement),
    used by the net-completeness gate + every router.
  * :func:`check_shorts` / :class:`ShortsCheck` / :func:`nets_joined` -- the
    INDEPENDENT union-find distinct-net (no-short) oracle over a ``tagged`` shape
    list (two shapes connect on the SAME (layer, datatype) OR through a via cut).
  * :func:`assert_net_complete` / :class:`NetCompletenessResult` /
    :class:`NetOpen` / :class:`NetCompletenessError` -- the LVS-faithfulness gate
    over the SCHEMATIC net set (every net is exactly one component carrying all
    its terminals, no short, no open).
  * :func:`emit_routed_cell` -- the KLayout GDS emit (every routed layer + net
    text labels, optionally merged with the placement's device FEOL artwork).

This module carries NO router logic (no lane/trunk planning, no PDK rule arithmetic
beyond the via-bridge layer SET) and NO per-circuit branch -- it is pure routed-cell
plumbing reusable by any router backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .backend import LayerPolygon
from .m2_crossing_route import VIA1_KEY
from .coclean_5tota import GF180_METAL1_LABEL_KEY

Bbox = Tuple[int, int, int, int]
TaggedShape = Tuple[str, Tuple[int, int], Bbox]


@dataclass
class RoutedCell:
    """Generic routed result, shape-compatible with the per-circuit generators.

    Mirrors ``comparator_router.RoutedComparator`` (``polys`` / ``tagged`` /
    ``labels``) so :func:`emit_routed_cell` reuses the existing klayout emit.

      * ``polys``  — every routed :class:`LayerPolygon` (all layers).
      * ``tagged`` — per-shape ``(net, (layer, datatype), bbox)`` for the
        union-find short check + Magic naming.
      * ``labels`` — ``net -> (x, y)`` text-label anchor points (on M1).
      * ``shorts`` — the :class:`ShortsCheck` result (populated by the router).
      * ``net_complete`` — the :class:`NetCompletenessResult` over the SCHEMATIC
        net set (populated by the router): whether EVERY schematic net is exactly
        one component carrying all its terminals (the LVS-faithfulness
        obligation), with per-net open/short diagnostics.
    """

    polys: List[LayerPolygon] = field(default_factory=list)
    tagged: List[TaggedShape] = field(default_factory=list)
    labels: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    shorts: Optional["ShortsCheck"] = None
    net_complete: Optional["NetCompletenessResult"] = None


# ── netlist -> net->terminals graph ───────────────────────────────────────────

# Map a device-IR pin key to the Placement terminal key. Body/bulk (``b``) is a
# well/substrate tie, routed to a rail rather than a signal terminal, so it is
# resolved to the matching terminal only when the placement carries it.
_PIN_TO_TERM = {"d": "d", "g": "g", "s": "s", "b": "b",
                "drain": "d", "gate": "g", "source": "s", "bulk": "b",
                # 2-terminal passive (resistor / cap) pins p/n -> the passive
                # device cell's p/n terminal map (a contact-pad M1 landing each).
                "p": "p", "n": "n", "plus": "p", "minus": "n"}


def net_terminals_from_netlist(
    netlist, placement,
) -> Dict[str, List[Tuple[str, str]]]:
    """Build ``net -> [(device_label, term_key), ...]`` from a parsed CellDef.

    Each device's ``pins`` map (``{'d':net,'g':net,'s':net,'b':net}``) names the
    net every terminal connects to. We keep only terminals the Placement actually
    exposes (``placement.devices[label].terminals`` has the key), so a body tie
    with no placed terminal is silently dropped (it is handled by the rail strap,
    not a signal riser). Determinism: nets are emitted in first-seen device order.
    """
    devices = getattr(placement, "devices", {})
    nets: Dict[str, List[Tuple[str, str]]] = {}
    for dev in getattr(netlist, "devices", []) or []:
        label = getattr(dev, "instance_name", None)
        pins = getattr(dev, "pins", None) or {}
        if label is None or label not in devices:
            continue
        placed_terms = getattr(devices[label], "terminals", {}) or {}
        for pin_key, net in pins.items():
            tk = _PIN_TO_TERM.get(str(pin_key).lower())
            if tk is None or tk not in placed_terms:
                continue
            nets.setdefault(str(net), []).append((label, tk))
    return nets


# ── union-find distinct-net (no-short) verification ───────────────────────────


@dataclass(frozen=True)
class ShortsCheck:
    """Result of the geometric distinct-net (no-short) check on ``tagged``.

      * ``distinct``      — True iff no two different nets share a component.
      * ``merged_pairs``  — sorted (net_a, net_b) pairs that wrongly merged.
      * ``net_components`` — net -> the component ids its shapes fell into.
      * ``shape_roots``   — per-shape union-find ROOT id, PARALLEL to the ``tagged``
        list the check ran on (``shape_roots[i]`` is the component id of
        ``tagged[i]``). This is what lets the net-completeness gate count PER-PIN
        coverage: a device terminal is connected only when a same-net shape OVER its
        coordinate lies in the net's connected COMPONENT, not merely anywhere in the
        net. Empty when the check was built without geometry (the hand-built unit
        cases construct ``ShortsCheck`` directly) -> the gate falls back to the
        per-net coverage it used before (no behaviour change for those callers).
    """

    distinct: bool
    merged_pairs: List[Tuple[str, str]]
    net_components: Dict[str, List[int]]
    shape_roots: Tuple[int, ...] = ()


def _touch(a: Bbox, b: Bbox) -> bool:
    """True if rectangles share any point (edge-touch counts — electrical)."""
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def check_shorts(
    tagged: Sequence[TaggedShape],
    via_keys: Optional[Sequence[Tuple[int, int]]] = None,
) -> ShortsCheck:
    """Union-find over tagged shapes: two shapes connect when they touch on the
    SAME (layer, datatype) OR one of them is on a VIA (cut) layer (the only legal
    metal<->metal bridge). A net is clean iff its component holds no OTHER net.
    Mirrors the v164 / v173 geometric short check (composed contract,
    re-implemented over the generic ``tagged`` triple list).

    ``via_keys`` is the SET of cut-layer ``(layer, datatype)`` keys that bridge two
    metals (a ``RouteTech.via_keys`` for the active PDK). It defaults to the
    gf180mcu Via1 key alone, so existing callers ``check_shorts(tagged)`` are
    unchanged; the P4 ``route_generic`` passes the tech's full via-key set so the
    rule is PDK-agnostic (``ki in via_keys``), not a hard-coded gf180 Via1."""
    via_set = frozenset(via_keys) if via_keys is not None else frozenset({VIA1_KEY})
    items = list(tagged)
    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        _ni, ki, bi = items[i]
        for j in range(i + 1, n):
            _nj, kj, bj = items[j]
            if not _touch(bi, bj):
                continue
            same_layer = ki == kj
            via_bridge = (ki in via_set) or (kj in via_set)
            if same_layer or via_bridge:
                union(i, j)

    net_components: Dict[str, set] = {}
    root_to_nets: Dict[int, set] = {}
    for idx, (name, _k, _b) in enumerate(items):
        r = find(idx)
        net_components.setdefault(name, set()).add(r)
        root_to_nets.setdefault(r, set()).add(name)

    merged: set = set()
    for _r, names in root_to_nets.items():
        if len(names) > 1:
            ordered = sorted(names)
            for a in range(len(ordered)):
                for b in range(a + 1, len(ordered)):
                    merged.add((ordered[a], ordered[b]))

    shape_roots = tuple(find(i) for i in range(n))
    return ShortsCheck(
        distinct=len(merged) == 0,
        merged_pairs=sorted(merged),
        net_components={k: sorted(v) for k, v in net_components.items()},
        shape_roots=shape_roots,
    )


def nets_joined(check: ShortsCheck, net: str) -> bool:
    """True iff every shape tagged ``net`` fell into a SINGLE component (the net's
    terminals are electrically joined). A net the router never touched is absent
    and reports False."""
    comps = check.net_components.get(net)
    return bool(comps) and len(comps) == 1


# ── the NET-COMPLETENESS gate (LVS-faithfulness obligation) ───────────────────
#
# A routed cell is LVS-faithful only if EVERY schematic net is realised as
# EXACTLY ONE electrical component carrying ALL its terminals — no inter-net SHORT
# and no per-net OPEN (a dropped or split net). ``check_shorts`` already finds
# shorts; ``nets_joined`` already finds a per-net split. The gate iterates over the
# SCHEMATIC net set (``net_terminals``) so a net the router silently DROPPED (it
# emitted no shapes for it, so it is absent from ``net_components``) is a FAILING
# obligation with a per-net diagnostic, NOT an invisible open. This is the
# difference between "no short among the shapes I happened to draw" and "every net
# the schematic demands is connected".


@dataclass(frozen=True)
class NetOpen:
    """One per-net OPEN/short diagnostic (a failing net-completeness obligation).

      * ``net``       — the schematic net name.
      * ``kind``      — ``"open"`` (the net's shapes split into >1 components, OR
        the net was never routed — 0 components), or ``"short"`` (the net's
        component also holds a DIFFERENT net).
      * ``components``— how many electrical components the net's shapes fell into
        (0 = the router emitted nothing for this net).
      * ``missing_terminals`` — the expected ``(device, term)`` pairs that are NOT
        present in the net's (largest) routed component — the terminals the router
        failed to connect (empty for a pure short).
      * ``shorted_with`` — for a ``"short"``, the OTHER net(s) sharing the
        component (the inter-net merge).
    """

    net: str
    kind: str
    components: int
    missing_terminals: Tuple[Tuple[str, str], ...] = ()
    shorted_with: Tuple[str, ...] = ()


@dataclass(frozen=True)
class NetCompletenessResult:
    """Result of :func:`assert_net_complete` over the SCHEMATIC net set.

      * ``complete``    — True iff EVERY schematic net is exactly one component,
        carries all its expected terminals, and shares it with no other net.
      * ``n_total``     — number of schematic nets checked.
      * ``n_complete``  — number that are complete.
      * ``opens``       — per-net :class:`NetOpen` diagnostics for every failing
        net (open or short), net-name sorted.
      * ``shorts``      — the underlying :class:`ShortsCheck` (the short detail).
    """

    complete: bool
    n_total: int
    n_complete: int
    opens: Tuple[NetOpen, ...]
    shorts: ShortsCheck

    @property
    def open_nets(self) -> Tuple[str, ...]:
        """Net names that are OPEN (split or never-routed), sorted."""
        return tuple(o.net for o in self.opens if o.kind == "open")

    @property
    def short_nets(self) -> Tuple[str, ...]:
        """Net names that SHORT to another net, sorted."""
        return tuple(o.net for o in self.opens if o.kind == "short")

    def summary(self) -> str:
        """A one-line honest summary: ``"X/N nets complete; open={..} short={..}"``."""
        parts = [f"{self.n_complete}/{self.n_total} nets complete"]
        if self.open_nets:
            parts.append("open=" + "{" + ", ".join(self.open_nets) + "}")
        if self.short_nets:
            parts.append("short=" + "{" + ", ".join(self.short_nets) + "}")
        return "; ".join(parts)


def _terminal_covered(net: str, xy: Tuple[int, int],
                      tagged: Sequence[TaggedShape],
                      *, member_idx: Optional[set] = None) -> bool:
    """True iff some shape tagged ``net`` that is IN the net's connected component
    contains the terminal point ``xy``.

    A terminal the router PHYSICALLY strapped has a same-net shape OVER its
    coordinate that is part of the same electrical component the net's other
    terminals join. A terminal the router DROPPED has none — OR has only a same-net
    shape that belongs to a DIFFERENT fragment (a body/well-tap pad whose via to the
    rail never landed, while the rail itself happens to span over the tap coordinate)
    -> the tap is electrically floating even though a same-net polygon covers it.

    ``member_idx`` is the set of ``tagged`` INDICES that lie in the component the
    coverage must belong to (the net's largest connected component, from
    ``ShortsCheck.shape_roots``). With it, a covering shape OUTSIDE that component is
    REJECTED — this is the per-PIN coverage rule (a net is complete only when ALL its
    device pins are connected to the SAME component, not merely overlapped somewhere
    by a same-net polygon). Without it (the hand-built unit cases that construct a
    ``ShortsCheck`` directly, with no per-shape roots) the check degrades to the
    original ANY-same-net-shape coverage, so those callers are unchanged."""
    px, py = xy
    for idx, (n, _k, (x0, y0, x1, y1)) in enumerate(tagged):
        if n != net:
            continue
        if member_idx is not None and idx not in member_idx:
            continue  # same net but a DIFFERENT component -> not a real strap
        if x0 <= px <= x1 and y0 <= py <= y1:
            return True
    return False


def assert_net_complete(
    check: ShortsCheck,
    net_terminals: Dict[str, List[Tuple[str, str]]],
    *,
    tagged: Optional[Sequence[TaggedShape]] = None,
    terminal_xy: Optional[Dict[Tuple[str, str], Tuple[int, int]]] = None,
) -> NetCompletenessResult:
    """The NET-COMPLETENESS gate: a routed cell is LVS-faithful iff EVERY schematic
    net is exactly one electrical component carrying all its terminals, with no
    inter-net short.

    ``check`` is the :class:`ShortsCheck` over the routed geometry (from
    :func:`check_shorts`); ``net_terminals`` is the SCHEMATIC net authority
    (``{net -> [(device, term), ...]}`` from ``LayoutPlan.net_terminals`` /
    :func:`net_terminals_from_netlist`). The gate iterates over ``net_terminals``
    — the SCHEMATIC set — so a net the router silently dropped (absent from
    ``check.net_components``) is a FAILING open, not an invisible gap.

    A net is COMPLETE iff ALL of:

      (a) ``check.distinct`` holds for it — its component holds NO other net
          (no SHORT); AND
      (b) ``nets_joined(check, net)`` — its shapes form EXACTLY ONE component
          (no OPEN / split); AND
      (c) every expected ``(device, term)`` in ``net_terminals`` is tagged into
          that component. With ``tagged`` + ``terminal_xy`` (a ``{(device, term)
          -> (x, y)}`` coordinate map), a terminal the router drew NO same-net
          shape over is reported as a missing terminal — a per-terminal open even
          when the net's other terminals still form one component. Without the
          coordinate map this collapses to (b)+(c)-by-component-count (a
          never-routed net has 0 components and fails as an open).

    Returns a :class:`NetCompletenessResult` with a per-net :class:`NetOpen`
    diagnostic for every failing net (which net, open vs short, how many
    components, and the expected terminals the router failed to connect).
    """
    # which OTHER net(s) each net is shorted to (from the merged pairs).
    shorted_with: Dict[str, set] = {}
    for (a, b) in check.merged_pairs:
        shorted_with.setdefault(a, set()).add(b)
        shorted_with.setdefault(b, set()).add(a)

    # PER-PIN coverage support: the per-shape union-find ROOT (parallel to ``tagged``)
    # lets clause (c) require a terminal's coverage to lie in the net's connected
    # COMPONENT, not merely anywhere in the net. Group the tagged shape INDICES by
    # (net, root) so the largest same-net component (the one the net's terminals must
    # join) is the membership set the coverage is checked against. Empty when the
    # ShortsCheck carries no per-shape roots (the hand-built unit ShortsCheck) -> the
    # gate degrades to the original ANY-same-net coverage (those callers unchanged).
    shape_roots = getattr(check, "shape_roots", ()) or ()
    net_root_members: Dict[Tuple[str, int], set] = {}
    if tagged is not None and shape_roots and len(shape_roots) == len(tagged):
        for idx, (sn, _sk, _sb) in enumerate(tagged):
            net_root_members.setdefault((sn, shape_roots[idx]), set()).add(idx)

    def _largest_component_members(net: str) -> Optional[set]:
        """Indices of the net's LARGEST connected component (the body the net's
        terminals must all reach), or ``None`` when no per-shape roots are available
        (degrade to per-net coverage)."""
        if not net_root_members:
            return None
        groups = [m for (gn, _r), m in net_root_members.items() if gn == net]
        if not groups:
            return set()
        return max(groups, key=len)

    opens: List[NetOpen] = []
    n_complete = 0
    schematic_nets = list(net_terminals.keys())
    for net in schematic_nets:
        comps = check.net_components.get(net)
        n_comps = len(comps) if comps else 0
        terms = tuple(net_terminals.get(net, ()) or ())
        is_short = net in shorted_with

        # clause (c): which expected terminals does the routed geometry NOT connect
        # to the net's COMPONENT? A terminal is covered only when a same-net shape IN
        # the net's connected component sits over its coordinate — a body/well-tap pad
        # whose strap to the rail never landed (so it is a SEPARATE fragment the rail
        # merely spans over) is therefore reported as a per-PIN open even though a
        # same-net polygon covers it. (Only checkable when both the tagged geometry
        # and the terminal coords are supplied; otherwise the 0-component branch
        # catches an unrouted net.)
        missing_terms: Tuple[Tuple[str, str], ...] = ()
        if tagged is not None and terminal_xy is not None:
            member_idx = _largest_component_members(net)
            miss = []
            for (dev, term) in terms:
                xy = terminal_xy.get((dev, term))
                if xy is None:
                    continue  # coordinate not provided -> cannot check this one
                if not _terminal_covered(net, xy, tagged, member_idx=member_idx):
                    miss.append((dev, term))
            missing_terms = tuple(miss)

        # OPEN: the net was never routed (0 comps) OR split into >1 component OR a
        # declared terminal was not drawn (a per-terminal silent open).
        if n_comps != 1 or missing_terms:
            # a never-routed net's terminals are all "missing"; a split/partial net
            # reports the per-terminal misses when known, else the full expected set.
            missing = missing_terms if missing_terms else (terms if n_comps != 1 else ())
            opens.append(
                NetOpen(
                    net=net,
                    kind="open",
                    components=n_comps,
                    missing_terminals=missing,
                    shorted_with=tuple(sorted(shorted_with.get(net, ()))),
                )
            )
            continue
        if is_short:
            opens.append(
                NetOpen(
                    net=net,
                    kind="short",
                    components=n_comps,
                    missing_terminals=(),
                    shorted_with=tuple(sorted(shorted_with[net])),
                )
            )
            continue
        n_complete += 1

    opens.sort(key=lambda o: (o.net, o.kind))
    return NetCompletenessResult(
        complete=(n_complete == len(schematic_nets)),
        n_total=len(schematic_nets),
        n_complete=n_complete,
        opens=tuple(opens),
        shorts=check,
    )


class NetCompletenessError(RuntimeError):
    """Raised by ``route_generic`` when the routed cell is NOT net-complete.

    Carries the :class:`NetCompletenessResult` so a caller can inspect the per-net
    open/short diagnostics (``.result.opens``) instead of a quietly-open cell."""

    def __init__(self, result: NetCompletenessResult, cell_name: str = ""):
        self.result = result
        self.cell_name = cell_name
        super().__init__(
            (f"cell {cell_name!r}: " if cell_name else "")
            + "routed cell is NOT net-complete: "
            + result.summary()
        )


# ── GDS emit (mirrors the per-circuit generators' klayout path) ───────────────


def emit_routed_cell(
    cell: RoutedCell, gds_path: str, *,
    cell_name: str = "routed_generic", dbu: float = 0.001,
    label_key: Tuple[int, int] = GF180_METAL1_LABEL_KEY,
    device_polys: Optional[Sequence[LayerPolygon]] = None,
    pin_nets: Optional[Sequence[str]] = None,
) -> str:
    """Emit all routed polygons (every layer) + net text labels via klayout — the
    same emit shape ``emit_routed_comparator`` uses, so a generic cell flows through
    the existing GDS path unchanged.

    ``label_key`` is the (layer, datatype) the net-name texts are written to;
    defaults to the gf180mcu M1 label key (byte-identical to the legacy emit). Pass
    ``tech.label_key(1)`` for another PDK.

    ``device_polys`` (optional) is the placement's DEVICE artwork — the laid-out
    transistor diffusion / poly / implant / well / contact / device-M1 polygons the
    placer built (``Placement.polys``). The generic place+route path routes only the
    BEOL (M1/M2/M3 + vias) over the device terminals; the device FEOL geometry lives
    on the placement, not the routed cell. Emitting both together yields a
    FULL-stack GDS that LVS can extract transistors from (without it the layout has
    routing but no devices, so extraction finds 0 MOS). Device polys are written
    FIRST (the base layers) and the routing on top; KLayout merges coincident
    same-layer same-net M1 (a device S/D pad + the router's terminal pad). Defaults
    to ``None`` so every existing caller (the per-circuit generators, whose
    ``cell.polys`` ALREADY carry their device geometry) is byte-unchanged.

    ``pin_nets`` (optional) restricts which nets receive a text label. A KLayout
    text on the top cell promotes its net to a TOP-LEVEL PIN of the extracted
    circuit (the foundry LVS deck runs with ``TOP_LVL_PINS=false``, so the pin set
    is taken from the labels alone). The schematic reference's pin set is the
    cell's ``.subckt`` port list, so labelling EVERY routed net — including
    internal nets (tail / s1out / s1dio …) — makes the layout circuit carry more
    pins than the schematic and the comparer rejects the whole match on the pin
    mismatch. Passing the cell's external port list here labels only those nets,
    so the layout pin set equals the schematic port set; internal nets stay
    unlabelled (KLayout numbers them — LVS matches by graph topology, not name).
    ``None`` keeps the legacy behaviour (label every net), so every existing caller
    is byte-unchanged."""
    import klayout.db as db

    ly = db.Layout()
    ly.dbu = dbu
    top = ly.create_cell(cell_name)
    for lp in (list(device_polys or []) + list(cell.polys)):
        li = ly.layer(lp.layer, lp.datatype)
        top.shapes(li).insert(
            db.Polygon([db.Point(int(v.x), int(v.y)) for v in lp.polygon.vertices])
        )
    lbl = ly.layer(label_key[0], label_key[1])
    allowed = None if pin_nets is None else {str(n) for n in pin_nets}
    for net, (x, y) in cell.labels.items():
        if allowed is not None and str(net) not in allowed:
            continue
        top.shapes(lbl).insert(db.Text(net, db.Trans(db.Vector(int(x), int(y)))))
    ly.write(str(gds_path))
    return str(gds_path)


__all__ = [
    "Bbox",
    "TaggedShape",
    "RoutedCell",
    "ShortsCheck",
    "NetOpen",
    "NetCompletenessResult",
    "NetCompletenessError",
    "net_terminals_from_netlist",
    "check_shorts",
    "nets_joined",
    "assert_net_complete",
    "emit_routed_cell",
]
