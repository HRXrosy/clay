"""Topology-inference of analog-layout CRAFT intent from a parsed netlist.

Stage-1 Task 2: derive the coordinate-free craft intent (``clay.layout.intent``
dataclasses) that the reference OTA's hand craft embodies, keyed ONLY on netlist
TOPOLOGY -- device kind, W / L / M, shared nets, the diode test (gate net ==
drain net), the source-rail test (source == body and that net is not a supply
rail), port role, and the functional-group device lists on the cell_def. This
module is READ-ONLY over the netlist: it never reads a coordinate, never places
or routes, and never emits geometry. The single public entry is::

    infer_craft(cell_def) -> list[<craft intent objects>]

plus :func:`group_members`, which exposes the SAME topology-derived group ->
members map so a caller (or a test) can resolve a group label back to its device
members without re-deriving the classification.

The inference rules are the per-cluster rules of
``ota_dev/analysis/reference_craft_spec.md`` (the source of truth), implemented
verbatim:

* **Diode test** -- a FET is a diode iff ``pins['g'] == pins['d']``.
* **Source-tied body** -- ``pins['s'] == pins['b']`` and that net is NOT a supply
  rail. The discriminator that both forces a private well and marks the diff pair.
* **Current mirror** -- >=2 same-polarity FETs sharing ONE gate net, source tied
  to a supply rail, where EXACTLY one member is a diode -> a shared-diffusion
  ``ArrayIntent(order='ratio')`` (``style='discrete'`` when EVERY member has
  ``M == 1``) + one rail-tied ``DummyFlankIntent`` per end.
* **Diff pair** -- two same-polarity FETs, equal W/L, equal M, sharing a source
  node that is ALSO their body (source-tied body), with distinct single-fanout
  gate nets -> ``CommonCentroidIntent(rows=2)`` + shared-diffusion
  ``ArrayIntent(order='ABBA')`` + gate-off ``DummyFlankIntent`` +
  ``WellIslandIntent(body=<the shared source/body net>)``.
* **Degenerated leg** -- a FET whose gate is a bias-mirror gate net but whose
  source ties a NON-rail node (a resistor midnode) -> its OWN shared-diffusion
  ``ArrayIntent``, no dummies.
* **Split passive** -- a resistor value realized as N equal SERIES units on an
  intermediate-node chain, or a cap value realized as N equal PARALLEL units on
  one net pair -> ``SplitIntent(n=N, match='series'|'parallel')``.
* **Banding** -- source on the top supply -> top; source on the bottom supply ->
  bottom; a pmos group with a source-tied (non-rail) body -> its own off-band
  well island (no top/bottom affinity) -> one ``BandIntent`` +
  ``RailAffinityIntent`` per banded MOS group.
* **Routing** -- a port with a power/ground role -> ``PowerPlaneIntent`` on the
  upper metals (M6 and M5); a diode-sourced gate net fanning to >=3 gates ->
  ``TrunkIntent(layer='M2', orient='H')``.
"""
from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from clay.frontend.role_model import DeviceCard, RoleDecomposition

from clay.layout.intent import (
    ArrayIntent,
    BandIntent,
    CommonCentroidIntent,
    DummyFlankIntent,
    PowerPlaneIntent,
    RailAffinityIntent,
    SplitIntent,
    TrunkIntent,
    WellIslandIntent,
)

# Port roles whose net is a supply/ground RAIL (topology-derived, not a name
# guess): a device whose source ties one of these bands to that supply.
_SUPPLY_ROLES = frozenset({"power_p", "power_n", "ground"})
# Conservative fallback rail names for a cell whose ports carry no role
# annotation. Only used to seed the rail set when ``port_roles`` is empty.
_FALLBACK_RAILS = frozenset({"vdd", "vss", "gnd", "0"})

# Upper-metal layers a supply net is distributed on as a power plane (M6 = the
# single global plane; M5 = the second plate layer -- both drawn in the
# reference). Ordered high -> low so the emitted intents read M6 then M5.
_POWER_PLANE_LAYERS: Tuple[str, ...] = ("M6", "M5")
# A diode-sourced gate net must fan out to at least this many gates to be a
# horizontal signal trunk (the reference draws npb / nbias1 as wide M2 trunks;
# the 2-gate nets nbias2 / ndio stay at the PDK minimum).
_TRUNK_FANOUT_MIN = 3


# ─────────────────────────────────────────────────────────────────────────────
# Netlist accessors (pure; no coordinate anywhere)
# ─────────────────────────────────────────────────────────────────────────────


def _polarity(device) -> Optional[str]:
    """MOS polarity ('pmos' / 'nmos') of a device, or ``None`` if not a MOS."""
    cls = str(getattr(device, "device_class", "") or "").lower()
    if "pmos" in cls or "pfet" in cls:
        return "pmos"
    if "nmos" in cls or "nfet" in cls:
        return "nmos"
    return None


def _is_mos(device) -> bool:
    return _polarity(device) is not None


def _pin(device, role: str) -> str:
    return str((getattr(device, "pins", {}) or {}).get(role, ""))


def _is_diode(device) -> bool:
    """A FET is a diode iff its gate net == its drain net."""
    return _pin(device, "g") == _pin(device, "d") != ""


def _m(device) -> int:
    try:
        return max(1, int(round(float((getattr(device, "sizing", {}) or {}).get("M", 1) or 1))))
    except (TypeError, ValueError):
        return 1


def _wl(device) -> Tuple[float, float]:
    s = getattr(device, "sizing", {}) or {}
    return float(s.get("W", 0.0) or 0.0), float(s.get("L", 0.0) or 0.0)


def _value(device) -> float:
    try:
        return float((getattr(device, "sizing", {}) or {}).get("value", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _supply_rails(cell_def) -> frozenset:
    """The set of supply/ground RAIL nets, from port roles (topology-derived)."""
    roles = getattr(cell_def, "port_roles", {}) or {}
    rails = {str(net) for net, role in roles.items() if role in _SUPPLY_ROLES}
    if not rails:
        # No role annotation -> fall back to the conventional rail names present
        # on the cell's ports so the source-rail test still has a rail set.
        ports = {str(p).lower() for p in getattr(cell_def, "ports", []) or []}
        rails = {p for p in ports if p in _FALLBACK_RAILS}
    return frozenset(rails)


def _top_bottom_supply(cell_def, rails: frozenset) -> Tuple[str, str]:
    """The (top, bottom) supply net names: power_p -> top, ground/power_n -> bottom."""
    roles = getattr(cell_def, "port_roles", {}) or {}
    top = next((str(n) for n, r in roles.items() if r == "power_p"), "")
    bottom = next((str(n) for n, r in roles.items() if r in ("ground", "power_n")), "")
    # Fallback by conventional name if a role is missing.
    if not top:
        top = next((r for r in sorted(rails) if r.lower() in ("vdd", "vcc")), "")
    if not bottom:
        bottom = next((r for r in sorted(rails) if r.lower() in ("gnd", "vss", "0")), "")
    return top, bottom


def _functional_groups(cell_def) -> Dict[str, Tuple[str, ...]]:
    """``functional_group name -> ordered device labels`` from the cell_def."""
    out: Dict[str, Tuple[str, ...]] = {}
    for fr in getattr(cell_def, "functional_roles", []) or []:
        if getattr(fr, "kind", "") != "functional_group":
            continue
        attrs = getattr(fr, "attributes", {}) or {}
        devs = tuple(str(d) for d in (attrs.get("devices") or ()))
        if devs:
            out[str(getattr(fr, "name", "") or "")] = devs
    return out


def _common_prefix(names: Sequence[str]) -> str:
    """A stable logical label for a split passive: the units' common name prefix."""
    p = os.path.commonprefix(sorted(names)).rstrip("_")
    return p.upper() if p else "_".join(sorted(names))


def _approx_equal(a: float, b: float, rel: float = 1e-6) -> bool:
    if a == b:
        return True
    m = max(abs(a), abs(b))
    return m == 0.0 or abs(a - b) / m <= rel


# ─────────────────────────────────────────────────────────────────────────────
# Topology analysis (single source of truth for infer_craft + group_members)
# ─────────────────────────────────────────────────────────────────────────────


class _Cluster:
    """One classified MOS craft cluster (topology-derived, coordinate-free)."""

    __slots__ = ("role", "label", "members", "gate", "body", "side", "discrete")

    def __init__(self, role, label, members, gate="", body="", side="", discrete=False):
        self.role = role                       # 'diff_pair'|'mirror'|'degen'|'cs_row'|'standalone'
        self.label = label                     # topology-derived group label
        self.members = tuple(members)          # device labels, declaration order
        self.gate = gate                       # shared gate net (mirror) or ''
        self.body = body                       # shared source/body net (diff pair) or ''
        self.side = side                       # 'top' | 'bottom' | 'island'
        self.discrete = discrete               # style discrete (all members M==1)


class _Split:
    __slots__ = ("label", "members", "match")

    def __init__(self, label, members, match):
        self.label = label
        self.members = tuple(members)
        self.match = match                     # 'series' | 'parallel'


class _Analysis:
    """The full topology analysis of one cell_def."""

    def __init__(self):
        self.clusters: List[_Cluster] = []
        self.splits: List[_Split] = []
        self.trunks: List[str] = []
        self.power_nets: List[str] = []
        self.top: str = ""
        self.bottom: str = ""


def _label_for(members: Sequence[str], fg_by_set: Dict[frozenset, str], fallback: str) -> str:
    """Prefer the functional-group NAME whose device set == ``members`` exactly;
    otherwise use the synthesized ``fallback`` label."""
    return fg_by_set.get(frozenset(members), fallback)


def _band_side(members, devs, rails, top, bottom) -> str:
    """Assign a banding side to a MOS group from source-to-rail connectivity.

    source on the top supply -> 'top'; source on the bottom supply -> 'bottom';
    a pmos group whose source ties its own (non-rail) body -> 'island' (its own
    off-band well); otherwise band by polarity (pmos -> top, nmos -> bottom)."""
    md = [devs[m] for m in members]
    srcs = {_pin(d, "s") for d in md}
    if srcs == {top} and top:
        return "top"
    if srcs == {bottom} and bottom:
        return "bottom"
    # non-rail source: a source-tied-body pmos group is its own island.
    if all(_polarity(d) == "pmos" for d in md) and all(
        _pin(d, "s") == _pin(d, "b") and _pin(d, "s") not in rails for d in md
    ):
        return "island"
    # otherwise the group's polarity picks its rail band.
    pol = {_polarity(d) for d in md}
    if pol == {"pmos"}:
        return "top"
    if pol == {"nmos"}:
        return "bottom"
    return ""


def _analyze(cell_def) -> _Analysis:
    """Classify every MOS device + split passive by netlist topology."""
    an = _Analysis()
    devices = list(getattr(cell_def, "devices", []) or [])
    mos = [d for d in devices if _is_mos(d)]
    by_name = {d.instance_name: d for d in devices}
    order = {d.instance_name: i for i, d in enumerate(devices)}
    rails = _supply_rails(cell_def)
    top, bottom = _top_bottom_supply(cell_def, rails)
    an.top, an.bottom = top, bottom
    fg = _functional_groups(cell_def)
    fg_by_set = {frozenset(devs): name for name, devs in fg.items()}

    # A device is an electrically-off DUMMY iff its gate ties its source (the
    # reference's rail spacers + input flank dummies): gate == source.
    dummies = {d.instance_name for d in mos if _pin(d, "g") == _pin(d, "s")}

    # The set of diode-sourced MIRROR gate nets (a diode net that ->=2 FETs share
    # as their gate): the discriminator for the degenerated-leg rule.
    gate_fanout = Counter(_pin(d, "g") for d in mos)
    diode_gate_nets = {
        _pin(d, "g")
        for d in mos
        if _is_diode(d) and _pin(d, "g") not in rails and gate_fanout[_pin(d, "g")] >= 2
    }

    claimed: set = set()

    # ── 1. Diff pairs ─────────────────────────────────────────────────────────
    # two same-polarity non-dummy FETs, equal W/L, equal M, source-tied body on a
    # NON-rail net, distinct single-fanout gate nets.
    actives = [d for d in mos if d.instance_name not in dummies]
    for i, a in enumerate(actives):
        if a.instance_name in claimed:
            continue
        for b in actives[i + 1:]:
            if b.instance_name in claimed:
                continue
            if _polarity(a) != _polarity(b):
                continue
            if _wl(a) != _wl(b) or _m(a) != _m(b):
                continue
            body_a, body_b = _pin(a, "s"), _pin(b, "s")
            if not (body_a == _pin(a, "b") and body_b == _pin(b, "b")):
                continue
            if body_a != body_b or body_a in rails or body_a == "":
                continue
            ga, gb = _pin(a, "g"), _pin(b, "g")
            if ga == gb or gate_fanout[ga] != 1 or gate_fanout[gb] != 1:
                continue
            members = sorted((a.instance_name, b.instance_name), key=lambda n: order[n])
            label = _label_for(members, fg_by_set, f"diff_pair_{body_a}")
            an.clusters.append(
                _Cluster("diff_pair", label, members, body=body_a, side="island")
            )
            claimed.update(members)
            break

    # ── 2. Current mirrors ────────────────────────────────────────────────────
    # >=2 same-polarity rail-sourced FETs sharing one gate net, exactly one diode.
    by_gate: Dict[str, List] = defaultdict(list)
    for d in mos:
        if d.instance_name in claimed or d.instance_name in dummies:
            continue
        if _pin(d, "s") in rails and _pin(d, "g") not in rails:
            by_gate[_pin(d, "g")].append(d)
    mirror_gates = sorted(by_gate, key=lambda g: min(order[x.instance_name] for x in by_gate[g]))
    for gate in mirror_gates:
        group = by_gate[gate]
        if len(group) < 2:
            continue
        if sum(1 for d in group if _is_diode(d)) != 1:
            continue
        members = sorted((d.instance_name for d in group), key=lambda n: order[n])
        discrete = all(_m(by_name[n]) == 1 for n in members)
        fallback = f"mirror_{gate}"
        label = _label_for(members, fg_by_set, fallback)
        side = _band_side(members, by_name, rails, top, bottom)
        an.clusters.append(
            _Cluster("mirror", label, members, gate=gate, side=side, discrete=discrete)
        )
        claimed.update(members)

    # ── 3. Degenerated legs ───────────────────────────────────────────────────
    # a FET on a bias-mirror gate net whose source ties a NON-rail (resistor) node.
    for d in mos:
        if d.instance_name in claimed or d.instance_name in dummies:
            continue
        if _pin(d, "g") in diode_gate_nets and _pin(d, "s") not in rails and _pin(d, "s") != "":
            label = _label_for([d.instance_name], fg_by_set, f"degen_{d.instance_name}")
            side = _band_side([d.instance_name], by_name, rails, top, bottom)
            an.clusters.append(
                _Cluster("degen", label, [d.instance_name], gate=_pin(d, "g"), side=side)
            )
            claimed.add(d.instance_name)

    # ── 4. Standalone banded MOS ──────────────────────────────────────────────
    # active MOS not in any cluster. A lone COMMON-SOURCE device -- source on a
    # rail, gate on a SIGNAL net (not a rail), M > 1 -- gets its OWN shared-OD row
    # of its M fingers + rail dummies (craft-spec cluster G, e.g. the output MNOUT).
    # Any other lone MOS (M == 1, or not rail-sourced) gets a banding affinity only.
    for d in mos:
        if d.instance_name in claimed or d.instance_name in dummies:
            continue
        label = _label_for([d.instance_name], fg_by_set, d.instance_name)
        side = _band_side([d.instance_name], by_name, rails, top, bottom)
        is_cs_row = _pin(d, "s") in rails and _pin(d, "g") not in rails and _m(d) > 1
        role = "cs_row" if is_cs_row else "standalone"
        an.clusters.append(_Cluster(role, label, [d.instance_name], side=side))
        claimed.add(d.instance_name)

    # ── 5. Split passives ─────────────────────────────────────────────────────
    an.splits = _detect_splits(devices, order)

    # ── 6. Routing: high-fanout diode-sourced gate nets -> trunks ─────────────
    trunk_seen = []
    for d in mos:
        g = _pin(d, "g")
        if _is_diode(d) and g not in rails and gate_fanout[g] >= _TRUNK_FANOUT_MIN and g not in trunk_seen:
            trunk_seen.append(g)
    an.trunks = trunk_seen

    # power/ground ports (in port declaration order) -> power planes.
    roles = getattr(cell_def, "port_roles", {}) or {}
    an.power_nets = [str(p) for p in (getattr(cell_def, "ports", []) or [])
                     if roles.get(p) in _SUPPLY_ROLES]
    return an


def _detect_splits(devices, order) -> List[_Split]:
    """Discover matched split passives: N equal PARALLEL caps on one net pair, or
    N equal SERIES resistors on an intermediate-node chain."""
    splits: List[_Split] = []
    caps = [d for d in devices if str(getattr(d, "device_class", "")) == "cap"]
    ress = [d for d in devices if str(getattr(d, "device_class", "")) == "res"]

    # Parallel caps: >=2 equal-value caps on the SAME unordered net pair.
    cap_by_pair: Dict[frozenset, List] = defaultdict(list)
    for c in caps:
        cap_by_pair[frozenset((_pin(c, "p"), _pin(c, "n")))].append(c)
    for group in cap_by_pair.values():
        if len(group) >= 2 and all(_approx_equal(_value(group[0]), _value(g)) for g in group):
            members = sorted((g.instance_name for g in group), key=lambda n: order[n])
            splits.append(_Split(_common_prefix(members), members, "parallel"))

    # Series resistors: unite equal-value resistors that share an INTERNAL chain
    # node (a net touched by exactly two resistor terminals -- a genuine midpoint,
    # never a rail or a fan-out node).
    by_name = {r.instance_name: r for r in ress}
    parent = {r.instance_name: r.instance_name for r in ress}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        parent[_find(a)] = _find(b)

    net_terms: Dict[str, List[str]] = defaultdict(list)
    for r in ress:
        for role in ("p", "n"):
            net_terms[_pin(r, role)].append(r.instance_name)
    for names in net_terms.values():
        if len(names) == 2 and names[0] != names[1]:
            a, b = names
            if _approx_equal(_value(by_name[a]), _value(by_name[b])):
                _union(a, b)

    chains: Dict[str, List[str]] = defaultdict(list)
    for r in ress:
        chains[_find(r.instance_name)].append(r.instance_name)
    for members in chains.values():
        if len(members) >= 2:
            members = sorted(members, key=lambda n: order[n])
            splits.append(_Split(_common_prefix(members), members, "series"))

    splits.sort(key=lambda s: order.get(s.members[0], 0))
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def group_members(cell_def) -> Dict[str, List[str]]:
    """The declared-or-topology-derived ``group label -> device members`` map.

    Exposes the SAME classification :func:`infer_craft` uses so a caller can
    resolve any emitted intent's ``group`` / ``target`` label back to its device
    members.  Explicit ``functional_group`` declarations are circuit DATA and
    therefore take precedence over a coincident topology-inferred label.  This
    is what lets an optimized derivative materialize a declared input pair even
    when its four-terminal body net intentionally differs from its shared source
    net.  Coordinate-free and deterministic; no device-name branch.
    """
    an = _analyze(cell_def)
    out: Dict[str, List[str]] = {}
    for c in an.clusters:
        out[c.label] = list(c.members)
    for s in an.splits:
        out[s.label] = list(s.members)
    known_devices = {
        str(getattr(device, "instance_name", ""))
        for device in getattr(cell_def, "devices", ()) or ()
    }
    for role in getattr(cell_def, "functional_roles", ()) or ():
        label = str(getattr(role, "name", "")).strip()
        attrs = getattr(role, "attributes", {}) or {}
        members = [
            str(member)
            for member in attrs.get("devices", ()) or ()
            if str(member) in known_devices
        ]
        if label and members:
            out[label] = members
    devices_by_name = {
        str(getattr(device, "instance_name", "")): device
        for device in getattr(cell_def, "devices", ()) or ()
    }
    for decl in getattr(cell_def, "layout_intents", ()) or ():
        if getattr(decl, "kind", "") != "array":
            continue
        attrs = getattr(decl, "attributes", None) or {}
        if str(attrs.get("style") or "") != "native_cmos":
            continue
        cmos_group = str(attrs.get("group") or "").strip()
        if not cmos_group or cmos_group not in out:
            continue
        def _cmos_polarity(dev) -> Optional[str]:
            token = (
                str(getattr(dev, "device_class", "") or "")
                + " "
                + str(getattr(dev, "pdk_variant", "") or "")
            ).lower()
            if "pmos" in token or "pfet" in token:
                return "pmos"
            if "nmos" in token or "nfet" in token:
                return "nmos"
            return None

        pmos_members = [
            m for m in out[cmos_group]
            if m in devices_by_name and _cmos_polarity(devices_by_name[m]) == "pmos"
        ]
        nmos_members = [
            m for m in out[cmos_group]
            if m in devices_by_name and _cmos_polarity(devices_by_name[m]) == "nmos"
        ]
        if pmos_members and nmos_members:
            out.setdefault(f"{cmos_group}__pmos", pmos_members)
            out.setdefault(f"{cmos_group}__nmos", nmos_members)
    return out


def infer_craft(cell_def) -> list:
    """Infer the Stage-1 craft intents from ``cell_def``'s netlist TOPOLOGY.

    Returns a deterministic list of ``clay.layout.intent`` dataclasses
    (``ArrayIntent`` / ``CommonCentroidIntent`` / ``DummyFlankIntent`` /
    ``WellIslandIntent`` / ``SplitIntent`` / ``BandIntent`` /
    ``RailAffinityIntent`` / ``TrunkIntent`` / ``PowerPlaneIntent``) derived
    entirely from device kind / W / L / M / shared nets / diode test /
    source-rail test / port role. No geometry, no coordinate, read-only.
    """
    an = _analyze(cell_def)
    arrays: List = []
    centroids: List = []
    dummy_flanks: List = []
    well_islands: List = []
    rail_affinities: List = []

    for c in an.clusters:
        if c.role == "diff_pair":
            arrays.append(ArrayIntent(group=c.label, style="shared_diffusion", rows=2, order="ABBA"))
            centroids.append(CommonCentroidIntent(group=c.label, rows=2))
            dummy_flanks.append(
                DummyFlankIntent(target=c.label, per_end=1, tie="gate_off", share_diffusion=True)
            )
            well_islands.append(WellIslandIntent(group=c.label, body=c.body))
            # off-band island -> no top/bottom rail affinity (WellIsland covers it).
        elif c.role == "mirror":
            style = "discrete" if c.discrete else "shared_diffusion"
            arrays.append(ArrayIntent(group=c.label, style=style, rows=1, order="ratio"))
            dummy_flanks.append(
                DummyFlankIntent(target=c.label, per_end=1, tie="rail", share_diffusion=not c.discrete)
            )
            if c.side in ("top", "bottom"):
                rail_affinities.append(RailAffinityIntent(group=c.label, side=c.side))
        elif c.role == "degen":
            arrays.append(ArrayIntent(group=c.label, style="shared_diffusion", rows=1, order=""))
            if c.side in ("top", "bottom"):
                rail_affinities.append(RailAffinityIntent(group=c.label, side=c.side))
        elif c.role == "cs_row":
            arrays.append(ArrayIntent(group=c.label, style="shared_diffusion", rows=1, order=""))
            dummy_flanks.append(
                DummyFlankIntent(target=c.label, per_end=1, tie="rail", share_diffusion=True)
            )
            if c.side in ("top", "bottom"):
                rail_affinities.append(RailAffinityIntent(group=c.label, side=c.side))
        elif c.role == "standalone":
            if c.side in ("top", "bottom"):
                rail_affinities.append(RailAffinityIntent(group=c.label, side=c.side))

    splits = [SplitIntent(passive=s.label, n=len(s.members), match=s.match) for s in an.splits]

    bands: List = []
    if an.top or an.bottom:
        bands.append(BandIntent(top=an.top, bottom=an.bottom, center="passives"))

    trunks = [TrunkIntent(net=n, layer="M2", orient="H") for n in an.trunks]

    power_planes: List = []
    for net in an.power_nets:
        for layer in _POWER_PLANE_LAYERS:
            power_planes.append(PowerPlaneIntent(net=net, layer=layer))

    # Deterministic emission order (arrays -> centroids -> dummy flanks -> well
    # islands -> splits -> band -> rail affinities -> trunks -> power planes).
    return (
        arrays
        + centroids
        + dummy_flanks
        + well_islands
        + splits
        + bands
        + rail_affinities
        + trunks
        + power_planes
    )


# ─────────────────────────────────────────────────────────────────────────────
# Topology + DC OP-aware full-flow AutoCraft synthesis
# ─────────────────────────────────────────────────────────────────────────────

_DNW_CAPABLE_PDKS = frozenset({
    "gf180mcu",
    "sky130",
    "vpdk180nm",
    "vpdk55nm",
})


@dataclass(frozen=True)
class FoldingHint:
    device: str
    w_um: float
    l_um: float
    h_max_um: float
    nf: int
    m: int = 1


@dataclass(frozen=True)
class WidthIntentSpec:
    net: str
    width_um: float
    current_a: float
    reason: str = ""


@dataclass(frozen=True)
class GuardRingSpec:
    group_name: str
    devices: Tuple[str, ...]
    well: str  # "p" | "n"
    tie: str
    role: str = "tap"


@dataclass(frozen=True)
class KelvinSpec:
    force_net: str
    sense_net: str
    target: str


@dataclass(frozen=True)
class KelvinHint:
    device: str
    force_net: str
    sense_net: str
    reason: str = ""


@dataclass(frozen=True)
class ShieldSpec:
    aggressor: str
    victim: str
    shield_net: str


@dataclass(frozen=True)
class BudgetSpec:
    kind: str
    net: str
    limit_val: float
    unit: str


@dataclass(frozen=True)
class ResistorSegmentSpec:
    device: str
    pdk: str
    model: str
    count: int
    connection: str
    width_um: float
    length_um: float
    segment_resistance_ohms: float
    value_quantum_ohms: float
    segment_resistance_quantum_ohms: float = 0.0


@dataclass(frozen=True)
class SharedOdArraySpec:
    """Grid-exact uniform-unit shared-diffusion ratio array plan."""

    group_name: str
    devices: Tuple[str, ...]
    unit_w_um: float
    nf_by_device: Tuple[Tuple[str, int], ...]
    needs_functional_group: bool = False


@dataclass(frozen=True)
class AutoCraftPlan:
    """Rich coordinate-free craft synthesis combining topology + DC OP profile."""

    cell: str
    pdk_id: str
    matched_pairs: Tuple[Tuple[str, str], ...] = ()
    ratio_groups: Tuple[Tuple[str, ...], ...] = ()
    mirror_axis: Optional[str] = None
    well_isolated: Tuple[Tuple[str, str], ...] = ()
    well_islands: Tuple[Tuple[str, Tuple[str, ...], str], ...] = ()
    guard_rings: Tuple[GuardRingSpec, ...] = ()
    folding_hints: Tuple[FoldingHint, ...] = ()
    wire_widths: Tuple[WidthIntentSpec, ...] = ()
    trunks: Tuple[TrunkIntent, ...] = ()
    kelvins: Tuple[KelvinSpec, ...] = ()
    kelvin_hints: Tuple[KelvinHint, ...] = ()
    shields: Tuple[ShieldSpec, ...] = ()
    budgets: Tuple[BudgetSpec, ...] = ()
    notes: Tuple[str, ...] = ()
    device_taps: Tuple[Tuple[str, str, str], ...] = ()
    resistor_segments: Tuple[ResistorSegmentSpec, ...] = ()
    shared_od_arrays: Tuple[SharedOdArraySpec, ...] = ()
    dense_contact_devices: Tuple[str, ...] = ()


def _pdk_grid_nm(pdk_id: str) -> int:
    try:
        from clay.layout.tech import LayoutTech
        from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

        return max(1, int(pdk_manufacturing_grid_nm(LayoutTech.resolve(pdk_id))))
    except Exception:  # noqa: BLE001
        return 5 if str(pdk_id).lower() in ("gf180mcu", "sky130") else 1


def _plan_grid_exact_fold(
    w_um: float,
    l_um: float,
    m_val: int,
    *,
    grid_nm: int,
    prefer_even: bool,
) -> Optional[Tuple[float, int]]:
    """Choose ``(h_max_um, nf)`` for ``@fold(h_max_um)`` so downstream
    :func:`folded_finger_plan` yields ``W_nm % (nf * grid_nm) == 0`` exactly.

    Eliminates LVS channel-width truncation (e.g. ``100u / 7 -> 14.285u * 7 = 99.995u``).
    """
    from clay.layout.extract.fold import folded_finger_plan

    nf_in = max(1, int(m_val))
    w_eff_um = float(w_um) * float(nf_in)
    w_eff_nm = int(round(w_eff_um * 1000.0))
    if w_eff_um <= 22.0 or w_eff_nm <= 0:
        return None

    target_h = 22.0 if w_um >= 500.0 else (16.0 if w_um >= 60.0 else 20.0)
    min_nf = max(2, nf_in, int(math.ceil(w_eff_um / 22.0)))
    max_nf = max(min_nf, int(math.floor(w_eff_um / 4.0)))
    # Cap iteration window cleanly for huge pass devices (e.g. 2500u)
    if max_nf - min_nf > 512:
        max_nf = min_nf + 512

    candidates: List[Tuple[Tuple[int, int, float, int], float, int]] = []
    for nf in range(min_nf, max_nf + 1):
        if w_eff_nm % (nf * grid_nm) != 0:
            continue
        finger_w_nm = w_eff_nm // nf
        finger_w_um = float(finger_w_nm) / 1000.0
        if finger_w_um > 22.0 or finger_w_um < 4.0:
            continue
        h_next_um = w_eff_um / float(nf - 1)
        gap = h_next_um - finger_w_um
        if gap <= 1e-6:
            continue
        # Step strictly inside (finger_w_um, h_next_um) so IEEE-754 ceil never bumps nf -> nf+1
        margin = min(0.01, 0.25 * gap)
        h_safe = round(finger_w_um + margin, 4)
        if h_safe <= finger_w_um or h_safe >= h_next_um:
            h_safe = finger_w_um + 0.5 * gap
        actual_w_um, actual_nf = folded_finger_plan(
            w_um, l_um, nf_in, h_max_um=h_safe
        )
        if actual_nf != nf or abs(actual_w_um * 1000.0 - finger_w_nm) > 1e-4:
            continue
        score = (
            1 if (prefer_even and nf % 2 != 0) else 0,
            0 if (finger_w_nm % (2 * grid_nm) == 0) else 1,
            round(abs(finger_w_um - target_h), 6),
            nf,
        )
        candidates.append((score, h_safe, nf))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    _, best_h, best_nf = candidates[0]
    return best_h, best_nf


def _sheet_res_ohm_sq(pdk_id: str, layer_num: int) -> float:
    try:
        from clay.layout.tech import LayoutTech

        tech = LayoutTech.resolve(pdk_id)
        return float(tech.calibrated_metal_sheet_resistance_ohm_sq(layer_num))
    except Exception:  # noqa: BLE001
        return 0.06 if layer_num >= 3 else 0.09


def _norm_cards_for_craft(
    decomp_or_cell,
    params: Optional[Mapping[str, str]],
):
    from clay.frontend.op_enrich import (
        _canon_kind,
        _decomp_from_cell_def,
        _is_bare_micron_mos_cir,
        _resolve_token_value,
    )

    if isinstance(decomp_or_cell, RoleDecomposition):
        decomp = decomp_or_cell
        param_map = dict(params or {})
        scale_um = _is_bare_micron_mos_cir(decomp, "", params=param_map)
        devs: List[dict] = []
        for c in decomp.devices:
            k = _canon_kind(c)
            is_mos = k in ("nmos", "pmos")
            w_m = _resolve_token_value(c.w, param_map, default_scale_um=(scale_um and is_mos))
            l_m = _resolve_token_value(c.l, param_map, default_scale_um=(scale_um and is_mos))
            m_v = _resolve_token_value(c.m, param_map, default_scale_um=False)
            devs.append({
                "name": c.name,
                "kind": k,
                "pins": dict(c.terminals),
                "w_um": (w_m * 1e6) if (k in ("nmos", "pmos") and w_m) else None,
                "l_um": (l_m * 1e6) if (k in ("nmos", "pmos") and l_m) else None,
                "val": w_m if k in ("res", "cap") else None,
                "m": max(1, int(round(m_v))) if m_v else 1,
                "model": c.source_model,
            })
        return decomp, devs

    decomp = _decomp_from_cell_def(decomp_or_cell)
    devs = []
    for raw_d in getattr(decomp_or_cell, "devices", ()) or ():
        name = str(getattr(raw_d, "instance_name", ""))
        cls = str(getattr(raw_d, "device_class", "")).lower()
        k = "pmos" if ("pmos" in cls or "pfet" in cls) else (
            "nmos" if ("nmos" in cls or "nfet" in cls) else (
                "res" if cls.startswith("res") else (
                    "cap" if cls.startswith("cap") else cls
                )
            )
        )
        pins = {str(pk): str(pv) for pk, pv in (getattr(raw_d, "pins", {}) or {}).items()}
        sz = dict(getattr(raw_d, "sizing", {}) or {})
        w_m = sz.get("W")
        l_m = sz.get("L")
        m_v = sz.get("M")
        devs.append({
            "name": name,
            "kind": k,
            "pins": pins,
            "w_um": (float(w_m) * 1e6) if (k in ("nmos", "pmos") and w_m) else None,
            "l_um": (float(l_m) * 1e6) if (k in ("nmos", "pmos") and l_m) else None,
            "val": float(sz.get("value")) if (k in ("res", "cap") and sz.get("value") is not None) else None,
            "m": max(1, int(round(float(m_v)))) if m_v else 1,
            "model": str(getattr(raw_d, "source_model", "") or ""),
        })
    return decomp, devs


def synthesize_craft(
    decomp_or_cell,
    op_profile=None,
    *,
    pdk_id: str = "gf180mcu",
    params: Optional[Mapping[str, str]] = None,
) -> AutoCraftPlan:
    """Derive topology + DC OP-aware craft synthesis without raw coordinates."""
    from clay.frontend.op_enrich import extract_op_profile
    from clay.frontend.clay_emit import infer_port_kind

    decomp, devs = _norm_cards_for_craft(decomp_or_cell, params)
    if op_profile is None:
        op_profile = extract_op_profile(
            decomp,
            pdk_id=pdk_id,
            run_ngspice=False,
            param_overrides=params,
        )

    dev_map = {d["name"]: d for d in devs}
    all_nets: set[str] = {n for d in devs for n in d["pins"].values() if n}
    role_of_dev: Dict[str, str] = {}
    for hit in decomp.roles:
        for dn in hit.devices:
            role_of_dev.setdefault(dn, hit.role)

    def _is_supply_net(net: str) -> bool:
        if not net:
            return False
        role = infer_port_kind(net)
        if role in ("power_p", "power_n", "ground"):
            return True
        low = net.lower()
        return low in {"vdd", "avdd", "dvdd", "vss", "avss", "dvss", "gnd", "agnd", "0", "vin"}

    # 1. Separate 1:1 matched pairs vs 1:K ratio mirrors/arrays
    def _aspect(d: dict) -> Optional[float]:
        w, l, m = d["w_um"], d["l_um"], d["m"]
        if w and l and l > 0:
            return (w * m) / l
        return None

    matched_pairs: List[Tuple[str, str]] = []
    ratio_groups: List[Tuple[str, ...]] = []
    ratio_seen: set[frozenset[str]] = set()

    for a, b in decomp.matched_pairs:
        da, db = dev_map.get(a), dev_map.get(b)
        if not da or not db or da["kind"] != db["kind"]:
            continue
        if da["kind"] in ("nmos", "pmos"):
            aa, ab = _aspect(da), _aspect(db)
            la, lb = da["l_um"], db["l_um"]
            if aa and ab and min(aa, ab) > 0:
                if max(aa, ab) / min(aa, ab) > 1.05:
                    grp = tuple(sorted((a, b)))
                    if frozenset(grp) not in ratio_seen:
                        ratio_seen.add(frozenset(grp))
                        ratio_groups.append(grp)
                    continue
            if la and lb and min(la, lb) > 0 and max(la, lb) / min(la, lb) > 1.05:
                grp = tuple(sorted((a, b)))
                if frozenset(grp) not in ratio_seen:
                    ratio_seen.add(frozenset(grp))
                    ratio_groups.append(grp)
                continue
        matched_pairs.append((a, b))

    # Shared-gate mirror groups (1:K ratio families) & 1:K BJT unit legs
    gate_families: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    gate_fanout: Counter[str] = Counter()
    for d in devs:
        if d["kind"] in ("nmos", "pmos"):
            g = d["pins"].get("g", "")
            if g:
                gate_fanout[g] += 1
                gate_families[(d["kind"], g)].append(d)

    for (_k, _g), group in gate_families.items():
        if len(group) < 2:
            continue
        diodes = [d for d in group if d["pins"].get("d") == d["pins"].get("g") != d["pins"].get("s")]
        if not diodes:
            continue
        aspects = [_aspect(d) for d in group]
        valid_asp = [a for a in aspects if a is not None and a > 0]
        if len(valid_asp) >= 2 and (max(valid_asp) / min(valid_asp) > 1.05 or len({d["m"] for d in group}) > 1):
            ordered = sorted(
                group,
                key=lambda x: (0 if x in diodes else 1, _aspect(x) or 0.0, x["name"]),
            )
            names = tuple(x["name"] for x in ordered)
            if frozenset(names) not in ratio_seen:
                ratio_seen.add(frozenset(names))
                ratio_groups.append(names)

    bjt_by_model: Dict[str, List[dict]] = defaultdict(list)
    for d in devs:
        if d["kind"] in ("pnp", "npn", "bjt"):
            bjt_by_model[d["model"] or d["kind"]].append(d)
    for _mod, blist in bjt_by_model.items():
        if len(blist) >= 3:
            by_branch: Dict[str, List[str]] = defaultdict(list)
            for b in blist:
                em = b["pins"].get("e", b["pins"].get("c", ""))
                by_branch[em].append(b["name"])
            if len(by_branch) >= 2:
                all_b = tuple(n for grp in by_branch.values() for n in sorted(grp))
                if frozenset(all_b) not in ratio_seen:
                    ratio_seen.add(frozenset(all_b))
                    ratio_groups.append(all_b)

    matched_partner: Dict[str, str] = {}
    for a, b in matched_pairs:
        matched_partner[a] = b
        matched_partner[b] = a

    # 2. Automatic transistor folding hints (W_um > 22.0 um), grid-exact for LVS
    grid_nm = _pdk_grid_nm(pdk_id)
    folding_hints_by_dev: Dict[str, FoldingHint] = {}
    for d in devs:
        if d["kind"] not in ("nmos", "pmos"):
            continue
        w_um, l_um = d["w_um"], d["l_um"]
        if w_um is None or w_um <= 22.0:
            continue
        # Even finger counts place both outer diffusions on the shared source/rail,
        # eliminating asymmetric single-finger M1 overhang and reducing drain area.
        planned = _plan_grid_exact_fold(
            w_um,
            l_um or 0.5,
            d["m"],
            grid_nm=grid_nm,
            prefer_even=True,
        )
        if planned is None:
            continue
        h_max, nf = planned
        folding_hints_by_dev[d["name"]] = FoldingHint(
            device=d["name"],
            w_um=round(w_um, 4),
            l_um=round(l_um or 0.5, 4),
            h_max_um=h_max,
            nf=nf,
            m=d["m"],
        )

    # 2b. Uniform-unit grid-exact shared-diffusion PCell array synthesis
    #     Groups >=2 rail-sourced same-(kind, oxide, L, gate, source, body) MOS
    #     into continuous-OD ratio strips (rows=1, even per-member nf, w_unit on
    #     2*grid_nm) while rejecting binary-weighted DAC ladders (W_max/W_min >= 8
    #     across >=3 widths), pruning extreme high-drive second-stage outliers,
    #     and preserving discrete symmetry on BJT bandgap beta-multipliers.
    from clay.frontend.clay_emit import _unique_group_names
    from clay.layout.extract.fold import folded_finger_plan
    from clay.layout.extract.netlist_intent import model_oxide

    has_bjt = any(d["kind"] in ("pnp", "npn", "bjt") for d in devs)
    roles_of_dev: Dict[str, set[str]] = defaultdict(set)
    for hit in decomp.roles:
        for dn in hit.devices:
            roles_of_dev[dn].add(hit.role)
    fg_unique_names = _unique_group_names(decomp.roles)
    fg_by_set: Dict[frozenset[str], str] = {}
    fg_role_priority = {"pass_device": 0, "current_mirror": 1, "load_mirror": 2}
    for gname, hit in sorted(
        zip(fg_unique_names, decomp.roles),
        key=lambda item: (fg_role_priority.get(item[1].role, 9), item[0]),
    ):
        key = frozenset(hit.devices)
        if key not in fg_by_set:
            fg_by_set[key] = gname

    def _safe_fold_h_for_unit(
        w_um_val: float, l_um_val: float, m_val: int, w_unit_nm: int, target_nf: int
    ) -> Optional[float]:
        nf_in = max(1, int(m_val))
        if target_nf < max(2, nf_in):
            return None
        w_eff_um = float(w_um_val) * float(nf_in)
        w_unit_um = float(w_unit_nm) / 1000.0
        h_next_um = w_eff_um / float(target_nf - 1)
        gap = h_next_um - w_unit_um
        if gap <= 1e-6:
            return None
        margin = min(0.01, 0.25 * gap)
        h_safe = round(w_unit_um + margin, 4)
        if h_safe <= w_unit_um or h_safe >= h_next_um:
            h_safe = w_unit_um + 0.5 * gap
        actual_w_um, actual_nf = folded_finger_plan(
            w_um_val, l_um_val or 0.5, nf_in, h_max_um=h_safe
        )
        if actual_nf != target_nf or abs(actual_w_um * 1000.0 - w_unit_nm) > 1e-4:
            return None
        return h_safe

    shared_od_arrays: List[SharedOdArraySpec] = []
    array_consumed: set[str] = set()
    if not has_bjt:
        sd_buckets: Dict[Tuple[str, str, int, str, str, str], List[dict]] = defaultdict(list)
        for d in devs:
            if d["kind"] not in ("nmos", "pmos"):
                continue
            w_um, l_um = d["w_um"], d["l_um"]
            if not w_um or not l_um or w_um <= 0.0 or l_um <= 0.0:
                continue
            g = d["pins"].get("g", "")
            s = d["pins"].get("s", "")
            b = d["pins"].get("b", "")
            if not g or not s or not b:
                continue
            if not _is_supply_net(s) or _is_supply_net(g) or g == s:
                continue
            if b != s and not _is_supply_net(b):
                continue
            ox = model_oxide(d["model"]) if d.get("model") else "thin"
            l_nm = int(round(l_um * 1000.0))
            sd_buckets[(d["kind"], ox, l_nm, g, s, b)].append(d)

        synth_arr_idx = 0
        for (kind_k, _ox, _lnm, _g, _s, _b), raw_grp in sorted(
            sd_buckets.items(), key=lambda kv: (kv[0][0], kv[1][0]["name"])
        ):
            if len(raw_grp) < 2:
                continue
            w_effs_all = [int(round(x["w_um"] * max(1, x["m"]) * 1000.0)) for x in raw_grp]
            w_min_all = min(w_effs_all)
            w_max_all = max(w_effs_all)
            grp = list(raw_grp)
            if w_min_all > 0 and (w_max_all / float(w_min_all)) > 4.0:
                # Prune high-drive (>=50um) second-stage output outliers first so
                # small bias mirrors stay compact and do not trip the DAC guard.
                pruned = [
                    x for x in grp
                    if not (
                        "second_stage" in roles_of_dev.get(x["name"], ())
                        and int(round(x["w_um"] * max(1, x["m"]) * 1000.0)) >= 50_000
                        and int(round(x["w_um"] * max(1, x["m"]) * 1000.0)) > 4 * w_min_all
                    )
                ]
                if len(pruned) >= 2:
                    grp = pruned
            w_effs = [int(round(x["w_um"] * max(1, x["m"]) * 1000.0)) for x in grp]
            w_min = min(w_effs)
            w_max = max(w_effs)
            # Skip multi-bit binary-weighted DAC ladders (>=3 distinct widths & >=8:1 span)
            if len(set(w_effs)) >= 3 and w_min > 0 and (w_max / float(w_min)) >= 8.0:
                continue
            if w_min <= 0 or (w_max / float(w_min)) > 4.0:
                continue

            uniform_wide_bank = (
                len(grp) >= 4
                and len(set(w_effs)) == 1
                and w_effs[0] >= 80_000
            )
            max_unit_nm = 25_000 if uniform_wide_bank else 22_000
            min_unit_nm = 1_500
            common_gcd = math.gcd(*w_effs)
            # Every member needs even nf >= 2 -> w_unit_nm must divide common_gcd // 2
            half_gcd = common_gcd // 2
            if common_gcd % 2 != 0 or half_gcd < min_unit_nm:
                continue
            # Enumerate exact divisors of half_gcd descending (prefer tallest compact fingers)
            divisors: List[int] = []
            r_lim = int(math.isqrt(half_gcd))
            for k in range(1, r_lim + 1):
                if half_gcd % k == 0:
                    divisors.append(k)
                    other = half_gcd // k
                    if other != k:
                        divisors.append(other)
            divisors.sort(reverse=True)

            chosen: Optional[Tuple[int, Dict[str, Tuple[float, int]]]] = None
            unit_grid = 2 * grid_nm
            for w_unit_nm in divisors:
                if w_unit_nm > max_unit_nm or w_unit_nm < min_unit_nm:
                    continue
                if w_unit_nm % unit_grid != 0:
                    continue
                per_dev: Dict[str, Tuple[float, int]] = {}
                total_nf = 0
                ok = True
                for x, we_nm in zip(grp, w_effs):
                    nf = we_nm // w_unit_nm
                    if nf < max(2, x["m"]) or nf % 2 != 0:
                        ok = False
                        break
                    h_safe = _safe_fold_h_for_unit(
                        x["w_um"], x["l_um"] or 0.5, x["m"], w_unit_nm, nf
                    )
                    if h_safe is None:
                        ok = False
                        break
                    per_dev[x["name"]] = (h_safe, nf)
                    total_nf += nf
                if not ok or total_nf > 32:
                    continue
                # Reject ultra-skinny horizontal noodles (e.g. 22 fingers at 2.5um height)
                if (float(total_nf) / (float(w_unit_nm) / 1000.0)) > 3.0:
                    continue
                chosen = (w_unit_nm, per_dev)
                break

            if chosen is None:
                continue
            w_unit_nm, per_dev = chosen
            names_sorted = tuple(sorted(x["name"] for x in grp))
            existing_fg = fg_by_set.get(frozenset(names_sorted))
            if existing_fg:
                gname = existing_fg
                needs_fg = False
            else:
                synth_arr_idx += 1
                gname = f"{kind_k}_mirror_array_{synth_arr_idx}"
                needs_fg = True
            shared_od_arrays.append(
                SharedOdArraySpec(
                    group_name=gname,
                    devices=names_sorted,
                    unit_w_um=round(w_unit_nm / 1000.0, 4),
                    nf_by_device=tuple((nm, per_dev[nm][1]) for nm in names_sorted),
                    needs_functional_group=needs_fg,
                )
            )
            array_consumed.update(names_sorted)
            for x in grp:
                h_safe, nf = per_dev[x["name"]]
                folding_hints_by_dev[x["name"]] = FoldingHint(
                    device=x["name"],
                    w_um=round(x["w_um"], 4),
                    l_um=round(x["l_um"] or 0.5, 4),
                    h_max_um=h_safe,
                    nf=nf,
                    m=x["m"],
                )

    # Row-height matching: when a same-(kind, oxide, L) shared-OD array sets a
    # substantial unit finger height (>= 8um), snap discrete wide (>22um)
    # non-input-pair siblings (e.g. a second-stage driver in the same polarity
    # row) to that exact unit height when evenly divisible (2 <= nf <= 16).
    arr_unit_by_dom: Dict[Tuple[str, str, int], List[int]] = defaultdict(list)
    for arr in shared_od_arrays:
        u_nm = int(round(arr.unit_w_um * 1000.0))
        if u_nm < 8_000 or not arr.devices:
            continue
        d0 = dev_map.get(arr.devices[0])
        if not d0 or not d0["l_um"]:
            continue
        ox0 = model_oxide(d0["model"]) if d0.get("model") else "thin"
        arr_unit_by_dom[(d0["kind"], ox0, int(round(d0["l_um"] * 1000.0)))].append(u_nm)
    for d in devs:
        if d["kind"] not in ("nmos", "pmos") or d["name"] in array_consumed:
            continue
        if "input_pair" in roles_of_dev.get(d["name"], ()):
            continue
        w_um, l_um = d["w_um"], d["l_um"]
        if not w_um or not l_um or w_um <= 22.0:
            continue
        ox = model_oxide(d["model"]) if d.get("model") else "thin"
        dom_units = arr_unit_by_dom.get((d["kind"], ox, int(round(l_um * 1000.0))))
        if not dom_units:
            continue
        w_eff_nm = int(round(w_um * max(1, d["m"]) * 1000.0))
        for u_nm in sorted(set(dom_units), reverse=True):
            if w_eff_nm % u_nm != 0:
                continue
            nf = w_eff_nm // u_nm
            if nf < max(2, d["m"]) or nf > 16 or nf % 2 != 0:
                continue
            h_safe = _safe_fold_h_for_unit(w_um, l_um, d["m"], u_nm, nf)
            if h_safe is not None:
                folding_hints_by_dev[d["name"]] = FoldingHint(
                    device=d["name"],
                    w_um=round(w_um, 4),
                    l_um=round(l_um, 4),
                    h_max_um=h_safe,
                    nf=nf,
                    m=d["m"],
                )
                break

    # Fold medium-width discrete matched input_pair devices (6u <= W <= 22u) to
    # clean 2-finger dense-contact strips when grid-exact on 2*grid_nm.
    for d in devs:
        if d["kind"] not in ("nmos", "pmos"):
            continue
        if d["name"] in array_consumed or d["name"] in folding_hints_by_dev:
            continue
        if "input_pair" not in roles_of_dev.get(d["name"], ()):
            continue
        w_um, l_um = d["w_um"], d["l_um"]
        if not w_um or not (6.0 <= w_um <= 22.0) or d["m"] != 1:
            continue
        w_eff_nm = int(round(w_um * 1000.0))
        if w_eff_nm % (4 * grid_nm) != 0:
            continue
        w_unit_nm = w_eff_nm // 2
        if w_unit_nm < 2_500:
            continue
        h_safe = _safe_fold_h_for_unit(w_um, l_um or 0.5, 1, w_unit_nm, 2)
        if h_safe is not None:
            folding_hints_by_dev[d["name"]] = FoldingHint(
                device=d["name"],
                w_um=round(w_um, 4),
                l_um=round(l_um or 0.5, 4),
                h_max_um=h_safe,
                nf=2,
                m=1,
            )

    # Drop redundant @matching pairs whose both devices belong to one shared-OD array
    # (prevents row matching source-axis replay mismatch on array-consumed devices).
    array_group_of: Dict[str, str] = {
        nm: spec.group_name for spec in shared_od_arrays for nm in spec.devices
    }
    matched_pairs = [
        (a, b)
        for a, b in matched_pairs
        if not (
            a in array_group_of
            and b in array_group_of
            and array_group_of[a] == array_group_of[b]
        )
    ]
    matched_partner = {}
    for a, b in matched_pairs:
        matched_partner[a] = b
        matched_partner[b] = a

    folding_hints: List[FoldingHint] = [
        folding_hints_by_dev[d["name"]]
        for d in devs
        if d["name"] in folding_hints_by_dev
    ]

    # 2c. Dense S/D contact rows on discrete (non-array) MOS devices.
    #     On BJT bandgap circuits, restrict to PMOS >= 1um so NMOS vss substrate
    #     strapping and -40C startup equilibrium stay unchanged.
    dense_contact_devices: List[str] = []
    for d in devs:
        if d["kind"] not in ("nmos", "pmos") or d["name"] in array_consumed:
            continue
        w_um = d["w_um"] or 0.0
        if has_bjt:
            if d["kind"] == "pmos" and w_um >= 1.0:
                dense_contact_devices.append(d["name"])
        else:
            if w_um >= 0.5:
                dense_contact_devices.append(d["name"])

    # 3. Well isolation (NMOS DNW only) vs PMOS/NMOS well islands vs Guard Rings
    well_isolated: List[Tuple[str, str]] = []
    well_islands: List[Tuple[str, Tuple[str, ...], str]] = []
    island_members: set[str] = set()

    dnw_ok = pdk_id.lower() in _DNW_CAPABLE_PDKS
    iso_groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for d in devs:
        if d["kind"] not in ("nmos", "pmos"):
            continue
        b = d["pins"].get("b", "")
        s = d["pins"].get("s", "")
        if b and s and b == s and not _is_supply_net(b):
            iso_groups[(d["kind"], b)].append(d["name"])

    for idx, ((pol, body_net), members) in enumerate(sorted(iso_groups.items())):
        if len(members) >= 2:
            # Match role group name if all members fall inside one functional_group
            grp_label = None
            for hit in decomp.roles:
                if set(members) <= set(hit.devices) and len(members) == len(hit.devices):
                    grp_label = hit.role
                    break
            grp_label = grp_label or f"well_iso_{pol}_{idx}"
            well_islands.append((grp_label, tuple(sorted(members)), body_net))
            island_members.update(members)
        if pol == "nmos" and dnw_ok:
            for dn in sorted(members):
                partner = matched_partner.get(dn, dn)
                well_isolated.append((dn, partner))

    # Guard rings: grouped by distinct functional cluster / shared-OD array and
    # (well, body_net) so different mirrors are never merged into one ring,
    # multi-role tail/mirror members stay with their mirror cluster, and
    # giant single-device pass FETs (>= 200um) never merge with small-signal cores.
    guard_rings: List[GuardRingSpec] = []
    sensitive_roles = {"input_pair", "cascode", "current_mirror", "load_mirror", "pass_device"}
    gr_candidates: List[Tuple[str, bool, str, str, Tuple[str, ...]]] = []
    seen_gr_sets: set[frozenset[str]] = set()

    def _add_gr_candidate(role_tag: str, is_arr: bool, member_names: Sequence[str]) -> None:
        by_wb: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        for nm in member_names:
            d = dev_map.get(nm)
            if d is None or d["kind"] not in ("nmos", "pmos") or nm in island_members:
                continue
            if (d["w_um"] or 0.0) >= 200.0:
                continue
            b = d["pins"].get("b", "")
            if not b:
                continue
            well = "n" if d["kind"] == "pmos" else "p"
            by_wb[(well, b)].append(d)
        for (well, tie), sub in sorted(by_wb.items()):
            if len(sub) < 2:
                continue
            if not is_arr and role_tag in ("current_mirror", "load_mirror"):
                w_effs_sub = [(x["w_um"] or 0.0) * max(1, x["m"]) for x in sub]
                w_min_sub = min((v for v in w_effs_sub if v > 0.0), default=0.0)
                if w_min_sub > 0.0:
                    pruned = [
                        x for x in sub
                        if not (
                            x["name"] not in array_consumed
                            and "second_stage" in roles_of_dev.get(x["name"], ())
                            and ((x["w_um"] or 0.0) * max(1, x["m"])) >= 50.0
                            and ((x["w_um"] or 0.0) * max(1, x["m"])) > 4.0 * w_min_sub
                        )
                    ]
                    if len(pruned) >= 2:
                        sub = pruned
            m_sorted = tuple(sorted(x["name"] for x in sub))
            key = frozenset(m_sorted)
            if key in seen_gr_sets:
                continue
            seen_gr_sets.add(key)
            gr_candidates.append((role_tag, is_arr, well, tie, m_sorted))

    # 1) Input pairs first so differential inputs always retain top priority.
    for hit in decomp.roles:
        if hit.role == "input_pair":
            _add_gr_candidate("input_pair", False, hit.devices)

    # 2) Synthesized shared-OD PCell arrays keep exact 1:1 guard-ring grouping.
    for arr in shared_od_arrays:
        _add_gr_candidate("array", True, arr.devices)

    # 3) Remaining sensitive functional roles (mirrors, cascodes, pass banks).
    for hit in decomp.roles:
        if hit.role in sensitive_roles and hit.role != "input_pair":
            _add_gr_candidate(hit.role, False, hit.devices)

    # 4) Fallback matched pairs not already covered by a sensitive cluster.
    covered_devs = {m for _, _, _, _, ms in gr_candidates for m in ms}
    matched_fallback: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for a, b in matched_pairs:
        if a in covered_devs or b in covered_devs:
            continue
        da, db = dev_map.get(a), dev_map.get(b)
        if not da or not db or da["kind"] != db["kind"] or da["kind"] not in ("nmos", "pmos"):
            continue
        ba, bb = da["pins"].get("b", ""), db["pins"].get("b", "")
        if ba and ba == bb:
            well = "n" if da["kind"] == "pmos" else "p"
            matched_fallback[(well, ba)].extend([a, b])
    for (well, tie), ms in sorted(matched_fallback.items()):
        _add_gr_candidate("matched_" + well, False, sorted(set(ms)))

    # Also add isolated PMOS source-tied pairs (e.g. telescopic PMOS diff pair with
    # body=ntail != vdd) as local N-well guard rings when not covered by @well_island
    for grp_label, members, body_net in well_islands:
        if len(members) >= 2 and all(dev_map[m]["kind"] == "pmos" for m in members if m in dev_map):
            guard_rings.append(
                GuardRingSpec(
                    group_name=f"gr_well_{grp_label}",
                    devices=tuple(sorted(members)),
                    well="n",
                    tie=body_net,
                    role="tap",
                )
            )

    has_clock = any(infer_port_kind(n) == "clock" for n in all_nets)
    ring_idx = 0
    for role_tag, is_arr, well, tie, members in gr_candidates:
        if len(members) < 2:
            continue
        if role_tag != "input_pair" and not is_arr and not has_clock and len(devs) < 12:
            continue
        ring_idx += 1
        guard_rings.append(
            GuardRingSpec(
                group_name=f"gr_{role_tag}_{ring_idx}",
                devices=members,
                well=well,
                tie=tie,
                role="tap",
            )
        )

    # 4. OP-aware @wire_width and @trunk sizing (symmetrize across matched pairs
    # so static single-code/single-polarity OP tests widen both differential legs).
    sym_currents: Dict[str, float] = dict(op_profile.net_currents_a)
    for a, b in matched_pairs:
        da, db = dev_map.get(a), dev_map.get(b)
        if not da or not db:
            continue
        for pin_name in ("d", "s", "p", "n", "c", "e"):
            na = da["pins"].get(pin_name, "")
            nb = db["pins"].get(pin_name, "")
            if na and nb and na != nb and infer_port_kind(na) == infer_port_kind(nb):
                ic_sym = max(sym_currents.get(na, 0.0), sym_currents.get(nb, 0.0))
                if ic_sym > 0:
                    sym_currents[na] = ic_sym
                    sym_currents[nb] = ic_sym

    r_sq_m3 = _sheet_res_ohm_sq(pdk_id, 3)
    r_sq_m2 = _sheet_res_ohm_sq(pdk_id, 2)
    v_drop_max = max(0.005, 0.005 * float(op_profile.vdd_v or 3.3))

    wire_widths: List[WidthIntentSpec] = []
    trunks: List[TrunkIntent] = []
    seen_trunk_nets: set[str] = set()
    seen_width_nets: set[str] = set()

    for net, ic in sorted(
        sym_currents.items(),
        key=lambda kv: (-kv[1], kv[0]),
    ):
        if net not in all_nets:
            continue
        port_k = infer_port_kind(net)
        is_sup = _is_supply_net(net)
        is_hi = net in op_profile.high_current_nets or ic >= max(100e-6, 0.12 * max(sym_currents.values(), default=0.0))
        if not (is_hi or (is_sup and ic >= 25e-6) or (port_k == "output" and ic >= 50e-6)):
            continue
        w_em = (ic / 0.5e-3) * 1.5
        w_ir = (ic * r_sq_m3 * 100.0) / v_drop_max
        w_req = max(0.6, w_em, w_ir)
        # Edge supply rails (vdd/vin/vss) have open boundary corridors up to 3.0um;
        # interior high-current output trunks (e.g. LDO vout) cap at 2.0um so
        # vertical M3->M4 risers clear adjacent MIM plates cleanly.
        cap_trunk_um = (
            (3.0 if is_sup else 2.0)
            if ic >= 1.5e-3
            else (2.5 if ic >= 200e-6 else 1.5)
        )
        w_trunk_um = round(min(cap_trunk_um, math.ceil(w_req * 10.0) / 10.0), 2)
        # Keep per-net pin-escape/interconnect @wire_width <= 2.0um so wide
        # power-bus drops do not pinch adjacent M1/M5 signal corridors while
        # retaining wide dedicated M3 @trunk buses.
        w_wire_um = round(min(2.0, w_trunk_um), 2)
        reason = f"OP |I|={ic*1e3:.3g}mA (EM/IR sizing)"
        if net not in seen_width_nets:
            seen_width_nets.add(net)
            wire_widths.append(WidthIntentSpec(net=net, width_um=w_wire_um, current_a=ic, reason=reason))
        if net not in seen_trunk_nets and (is_sup or (port_k == "output" and ic >= 1e-3)):
            seen_trunk_nets.add(net)
            edge = ""
            if port_k == "power_p" or net.lower() in ("vdd", "avdd", "vin"):
                edge = "top"
            elif port_k in ("power_n", "ground") or net.lower() in ("vss", "gnd", "agnd"):
                edge = "bottom"
            trunks.append(
                TrunkIntent(
                    net=net,
                    layer="M3",
                    orient="H",
                    width_um=w_trunk_um,
                    edge=edge,
                )
            )

    # Bias/mirror high-fanout horizontal M2 trunks
    for net in sorted(all_nets):
        if net in seen_trunk_nets or _is_supply_net(net):
            continue
        if gate_fanout.get(net, 0) >= 3 and (net in op_profile.bias_nets or any(
            d["pins"].get("d") == net and d["pins"].get("g") == net for d in devs if d["kind"] in ("nmos", "pmos")
        )):
            seen_trunk_nets.add(net)
            ic_bias = op_profile.net_currents_a.get(net, 10e-6)
            w_b = 0.5 if ic_bias < 100e-6 else 0.7
            trunks.append(TrunkIntent(net=net, layer="M2", orient="H", width_um=w_b, edge=""))

    # 5. Shielding: only emit @shield when aggressor and victim occupy separated
    # terminal domains (no shared device, no shared @guard_ring, no shared
    # matched pair, and no shared floorplan row in either raw or compact 2-D
    # floorplan), preventing ShieldMaterializationError.
    from clay.frontend.clay_emit import _clean_domain_place_groups
    from clay.frontend.floorplan_intent import floorplan_place_groups

    raw_pgs = floorplan_place_groups(decomp)
    clean_pgs = _clean_domain_place_groups(decomp, raw_pgs, craft_plan=True)
    dev_to_raw_row: Dict[str, int] = {}
    for pg in raw_pgs:
        for dn in pg.devices:
            dev_to_raw_row[dn] = pg.row
    dev_to_clean_row: Dict[str, int] = {}
    for pg in clean_pgs:
        for dn in pg.devices:
            dev_to_clean_row[dn] = pg.row
    net_devices: Dict[str, set[str]] = defaultdict(set)
    net_raw_rows: Dict[str, set[int]] = defaultdict(set)
    net_clean_rows: Dict[str, set[int]] = defaultdict(set)
    for d in devs:
        for net_n in d["pins"].values():
            if net_n:
                net_devices[net_n].add(d["name"])
                if d["name"] in dev_to_raw_row:
                    net_raw_rows[net_n].add(dev_to_raw_row[d["name"]])
                if d["name"] in dev_to_clean_row:
                    net_clean_rows[net_n].add(dev_to_clean_row[d["name"]])

    ring_groups: List[set[str]] = [set(gr.devices) for gr in guard_rings]
    matched_twin_nets: set[Tuple[str, str]] = set()
    for a, b in matched_pairs:
        da, db = dev_map.get(a), dev_map.get(b)
        if da and db:
            for pin_n in ("d", "g", "s", "p", "n"):
                na, nb = da["pins"].get(pin_n, ""), db["pins"].get(pin_n, "")
                if na and nb and na != nb:
                    matched_twin_nets.add((na, nb))
                    matched_twin_nets.add((nb, na))

    def _domains_separable(aggr: str, vic: str) -> bool:
        da, dv = net_devices.get(aggr, set()), net_devices.get(vic, set())
        if not da or not dv:
            return False
        if da & dv:
            # Shared device -> interdigitated/pin-adjacent, never separated
            return False
        if (aggr, vic) in matched_twin_nets:
            # Differential complementary pair -> interleaved along symmetry axis
            return False
        if net_raw_rows.get(aggr, set()) & net_raw_rows.get(vic, set()):
            # Same raw floorplan row -> horizontally interleaved pins, not separated
            return False
        if net_clean_rows.get(aggr, set()) & net_clean_rows.get(vic, set()):
            # Same compact 2-D floorplan row -> horizontally interleaved pins
            return False
        for rg in ring_groups:
            if (da & rg) and (dv & rg):
                return False
        return True

    shield_gnd = next(
        (n for n in ("vss", "gnd", "agnd", "dvss") if n in all_nets),
        next((n for n in all_nets if infer_port_kind(n) in ("power_n", "ground")), ""),
    )
    shields: List[ShieldSpec] = []
    if shield_gnd:
        aggressors = [n for n in sorted(all_nets) if infer_port_kind(n) == "clock"]
        if not aggressors:
            aggressors = [
                n for n in sorted(all_nets)
                if infer_port_kind(n) == "output" and sym_currents.get(n, 0.0) >= 1.0e-3
            ]
        # Prefer explicit input / feedback high-Z victims that are strictly domain-separable
        victims = [
            v for v in op_profile.high_z_nets
            if v in all_nets
            and not _is_supply_net(v)
            and v not in aggressors
            and v != shield_gnd
            and infer_port_kind(v) != "output"
        ]
        for aggr in aggressors[:2]:
            for vic in victims[:4]:
                if len({aggr, vic, shield_gnd}) == 3 and _domains_separable(aggr, vic):
                    shields.append(ShieldSpec(aggressor=aggr, victim=vic, shield_net=shield_gnd))
                    if len(shields) >= 4:
                        break
            if len(shields) >= 4:
                break

    # 6. Kelvin force/sense discovery (hints + strictly pin-verified specs)
    kelvin_hints: List[KelvinHint] = []
    kelvins: List[KelvinSpec] = []
    for d in devs:
        pins = set(d["pins"].values())
        outs = [n for n in pins if infer_port_kind(n) == "output"]
        if outs and d["kind"] == "res":
            out_n = outs[0]
            other = [n for n in pins if n != out_n and not _is_supply_net(n)]
            if other and (op_profile.net_currents_a.get(out_n, 0.0) >= 500e-6 or "fb" in other[0].lower()):
                kelvin_hints.append(
                    KelvinHint(
                        device=d["name"],
                        force_net=out_n,
                        sense_net=other[0],
                        reason="Output feedback resistor star/Kelvin sense candidate",
                    )
                )

    device_taps = _infer_bjt_device_taps(
        decomp,
        devs,
        pdk_id=pdk_id,
        all_nets=all_nets,
        rails=_supply_rails(decomp) if False else all_nets,
        infer_port_kind=infer_port_kind,
    )
    resistor_segments = _plan_poly_resistor_segments(
        devs,
        pdk_id=pdk_id,
    )

    notes: List[str] = [
        f"op_source={op_profile.source}",
        f"matched_pairs={len(matched_pairs)} ratio_groups={len(ratio_groups)} folds={len(folding_hints)}",
        f"shared_od_arrays={len(shared_od_arrays)} dense_contacts={len(dense_contact_devices)}",
        f"wire_widths={len(wire_widths)} trunks={len(trunks)} guard_rings={len(guard_rings)} shields={len(shields)}",
    ]

    return AutoCraftPlan(
        cell=decomp.cell,
        pdk_id=pdk_id,
        matched_pairs=tuple(matched_pairs),
        ratio_groups=tuple(ratio_groups),
        mirror_axis=decomp.mirror_axis,
        well_isolated=tuple(well_isolated),
        well_islands=tuple(well_islands),
        guard_rings=tuple(guard_rings),
        folding_hints=tuple(folding_hints),
        wire_widths=tuple(wire_widths),
        trunks=tuple(trunks),
        kelvins=tuple(kelvins),
        kelvin_hints=tuple(kelvin_hints),
        shields=tuple(shields),
        budgets=(),
        notes=tuple(notes),
        device_taps=device_taps,
        resistor_segments=resistor_segments,
        shared_od_arrays=tuple(shared_od_arrays),
        dense_contact_devices=tuple(dense_contact_devices),
    )


def _infer_bjt_device_taps(
    decomp: RoleDecomposition,
    devs: Sequence[dict],
    *,
    pdk_id: str,
    all_nets: set[str],
    rails: set[str],
    infer_port_kind,
) -> Tuple[Tuple[str, str, str], ...]:
    """Derive PDK-backed @tap(net, where=substrate) declarations for BJTs."""
    try:
        from clay.pdk import get_pdk

        pdk = get_pdk(pdk_id)
        bjt_bindings = getattr(pdk, "bjt_bindings", {}) or {}
    except Exception:  # noqa: BLE001
        return ()
    if not bjt_bindings:
        return ()

    bjt_entries: List[Tuple[str, dict]] = []
    for d in devs:
        k = d["kind"]
        if k not in ("pnp", "npn", "bjt"):
            continue
        mod = str(d.get("model") or "").lower()
        bjt_k = k if k in ("pnp", "npn") else ("pnp" if "pnp" in mod else ("npn" if "npn" in mod else ""))
        binding = bjt_bindings.get(bjt_k)
        tap_binding = getattr(binding, "tap_binding", None) if binding is not None else None
        where_opts = set(getattr(tap_binding, "where", ()) or ()) if tap_binding is not None else set()
        if "substrate" in where_opts:
            bjt_entries.append((d["name"], d))
    if not bjt_entries:
        return ()

    src_ports = list(getattr(decomp, "source_ports", ()) or ())
    sub_net = next((p for p in src_ports if p in all_nets and infer_port_kind(p) == "ground"), None)
    if not sub_net:
        nmos_bulks = [
            d["pins"].get("b", "")
            for d in devs
            if d["kind"] == "nmos" and d["pins"].get("b", "") in all_nets
        ]
        rail_bulks = [b for b in nmos_bulks if infer_port_kind(b) in ("power_n", "ground") or b.lower() in _FALLBACK_RAILS]
        if rail_bulks:
            sub_net = Counter(rail_bulks).most_common(1)[0][0]
    if not sub_net:
        sub_net = next((p for p in src_ports if p in all_nets and infer_port_kind(p) == "power_n"), None)
    if not sub_net:
        bjt_cb = [
            d["pins"].get("c") or d["pins"].get("b") or ""
            for _, d in bjt_entries
        ]
        cb_rails = [n for n in bjt_cb if n and (infer_port_kind(n) in ("power_n", "ground") or n.lower() in _FALLBACK_RAILS)]
        if cb_rails:
            sub_net = Counter(cb_rails).most_common(1)[0][0]
    if not sub_net:
        try:
            cand = pdk.substrate_net(sorted(all_nets))
            if cand:
                sub_net = next((n for n in all_nets if n.lower() == cand.lower()), None)
        except Exception:  # noqa: BLE001
            sub_net = None
    if not sub_net:
        return ()
    return tuple((name, sub_net, "substrate") for name, _ in bjt_entries)


def _plan_poly_resistor_segments(
    devs: Sequence[dict],
    *,
    pdk_id: str,
) -> Tuple[ResistorSegmentSpec, ...]:
    """Plan grid-exact segmented poly resistors from PDK sheet/rules tables."""
    pdk_key = str(pdk_id or "").strip().lower()
    try:
        from clay.layout.passive_cell import _resolve_poly_res_facts
        from clay.layout.segmented_resistor import _MAX_SEGMENTS, _MODELS_BY_PDK, _rules
        from clay.layout.tech import LayoutTech

        allowed_models = _MODELS_BY_PDK.get(pdk_key)
        if not allowed_models:
            return ()
        facts, _ = _resolve_poly_res_facts(pdk_key)
        rules = _rules(LayoutTech.resolve(pdk_key))
    except Exception:  # noqa: BLE001
        return ()

    default_model = str(facts.sheet_model or "").strip().split()[0]
    if default_model not in allowed_models:
        default_model = sorted(allowed_models)[0]
    rho = float(facts.sheet_ohm_sq)
    grid_nm = max(1, int(rules["grid_nm"]))
    min_w_nm = int(math.ceil((float(facts.body_min_w_um) * 1000.0) / grid_nm)) * grid_nm
    contact_nm = int(rules["contact_size_nm"])
    c_side = int(rules["contact_min_side_margin_nm"])
    c_space = int(rules["contact_min_space_nm"])
    if rho <= 0.0:
        return ()

    def _is_legal_width_nm(w_nm: int) -> bool:
        if w_nm < min_w_nm or w_nm % grid_nm != 0 or 2 * int(rules["m1_side_margin_nm"]) >= w_nm:
            return False
        cap = (w_nm - 2 * c_side + c_space) // (contact_nm + c_space)
        if cap < 1:
            return False
        if cap == 1:
            return ((w_nm - contact_nm) // 2) % grid_nm == 0
        residual = w_nm - 2 * c_side - cap * contact_nm
        return residual >= (cap - 1) * c_space and residual % (cap - 1) == 0

    w_base_nm = min_w_nm
    while not _is_legal_width_nm(w_base_nm):
        w_base_nm += grid_nm
    w_base_um = w_base_nm / 1000.0
    max_seg_l_um = 40.0 if w_base_um <= 0.85 else 50.0
    r_max_base = rho * max_seg_l_um / w_base_um
    r_min_base = rho * 2.0 / w_base_um

    res_devs = [
        d for d in devs
        if d["kind"] == "res"
        and d["val"] is not None
        and float(d["val"]) >= r_min_base
        and int(d["m"]) == 1
        and (not d.get("model") or d["model"] in allowed_models)
    ]
    if not res_devs:
        return ()

    # Derive a shared bank unit resistance when >= 2 resistors share an exact
    # high-resistance GCD unit (e.g., R_sum1=175k & R_sum2=1697.5k -> 17.5k).
    bank_vals = [float(d["val"]) for d in res_devs if float(d["val"]) > r_max_base]
    shared_unit_ohm = 0.0
    if len(bank_vals) >= 2:
        scaled = [int(round(v * 100.0)) for v in bank_vals]
        g_ohm = math.gcd(*scaled) / 100.0
        if g_ohm > 0:
            k_min = max(1, int(math.ceil(g_ohm / r_max_base)))
            for k in range(k_min, min(k_min + 64, 512)):
                cand = g_ohm / float(k)
                if r_max_base * 0.4 <= cand <= r_max_base:
                    # Verify segment length hits manufacturing grid cleanly
                    l_ideal_nm = (cand * w_base_um / rho) * 1000.0
                    l_grid_nm = int(round(l_ideal_nm / grid_nm)) * grid_nm
                    if abs(l_ideal_nm - l_grid_nm) <= 1e-3 and 1 <= int(round(max(bank_vals) / cand)) <= _MAX_SEGMENTS:
                        shared_unit_ohm = cand
                        break

    legal_widths_um = [
        nm / 1000.0
        for nm in range(w_base_nm, min_w_nm * 3 + grid_nm, grid_nm)
        if _is_legal_width_nm(nm)
    ]

    plans: List[ResistorSegmentSpec] = []
    for d in res_devs:
        val = float(d["val"])
        model = str(d.get("model") or "") or default_model
        if shared_unit_ohm > 0 and val >= shared_unit_ohm - 1e-6:
            n_exact = val / shared_unit_ohm
            if abs(n_exact - round(n_exact)) <= 1e-6 and 1 <= int(round(n_exact)) <= _MAX_SEGMENTS:
                cnt = int(round(n_exact))
                w_um = w_base_um
                l_um = (int(round(((shared_unit_ohm * w_um / rho) * 1000.0) / grid_nm)) * grid_nm) / 1000.0
                r_seg = round(rho * l_um / w_um, 6)
                plans.append(ResistorSegmentSpec(
                    device=d["name"], pdk=pdk_key, model=model, count=cnt,
                    connection="series", width_um=w_um, length_um=l_um,
                    segment_resistance_ohms=r_seg, value_quantum_ohms=1.0,
                ))
                continue

        cnt = max(1, int(math.ceil(val / r_max_base)))
        if cnt > _MAX_SEGMENTS:
            continue
        best: Optional[Tuple[Tuple[float, float, int], float, float, float]] = None
        w_candidates = legal_widths_um if cnt == 1 else [w_base_um]
        for w_um in w_candidates:
            r_target = val / float(cnt)
            l_ideal_nm = (r_target * w_um / rho) * 1000.0
            l_nm = max(grid_nm, int(round(l_ideal_nm / grid_nm)) * grid_nm)
            l_um = l_nm / 1000.0
            if l_um > 50.0 or l_um < 2.0:
                continue
            r_seg = round(rho * l_um / w_um, 6)
            rel_err = abs(cnt * r_seg - val) / val
            # Prefer exact/lowest error, then narrower compact width
            score = (round(rel_err, 8), w_um, l_nm)
            if best is None or score < best[0]:
                best = (score, w_um, l_um, r_seg)
        if best is None:
            w_um = w_base_um
            l_nm = max(grid_nm, int(round(((val / cnt) * w_um / rho) * 1000.0 / grid_nm)) * grid_nm)
            l_um = l_nm / 1000.0
            r_seg = round(rho * l_um / w_um, 6)
        else:
            _, w_um, l_um, r_seg = best
        err = abs(cnt * r_seg - val)
        quantum = 1.0 if err <= 0.49 else float(10 ** math.ceil(math.log10(max(err * 2.5, 1.0))))
        plans.append(ResistorSegmentSpec(
            device=d["name"], pdk=pdk_key, model=model, count=cnt,
            connection="series", width_um=w_um, length_um=l_um,
            segment_resistance_ohms=r_seg, value_quantum_ohms=quantum,
        ))
    return tuple(plans)
