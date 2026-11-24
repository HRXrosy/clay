"""Circuit-derived symmetry groups — schematic symmetry read from the topology.

The user's principle (2026-05-27): layout symmetry must be the SCHEMATIC-level
transistor symmetry (a differential pair, a cross-coupled latch, a current
mirror — devices that play mirror-image roles in the CIRCUIT), faithfully mapped
to the layout. It is NOT an arbitrary "mirror these two devices" physical hint.

So symmetry is DERIVED from the circuit structure the designer already wrote in
the ``topology`` block (device class + sizing + the nets each pin connects),
not from a separate physical ``@symmetric`` annotation. This module is that
derivation: ``derive_symmetry_groups(cell)`` reads ``cell.devices`` and returns
first-class :class:`SymmetryGroup` objects that downstream placement + routing +
parasitic-balance checks consume.

Recognised circuit-symmetric structures (all from the DECLARED topology):
  * differential pair  — two same-class, same-(W,L) MOS sharing a source net,
    with distinct gate nets and distinct drain nets. The gate nets form one
    differential-net pair (e.g. inn/inp); the drains another (outn/outp); the
    shared source is a common (on-axis) net.
  * cross-coupled pair — two same-class, same-(W,L) MOS sharing a source net,
    each one's gate tied to the other's drain (a latch). The drains form the
    differential-net pair.
Pairs that share a differential net are MERGED into one group about one axis
(a comparator's input diff pair + cross-coupled latch share outn/outp -> one
symmetric core). Unpaired MOS whose drain or source lands on a group's common
net (e.g. a tail switch) are recorded as ``self_symmetric`` (centred on axis).

``crosscheck_against_annotations`` then compares the circuit-derived groups to
any explicit ``@symmetric``/``@matching`` annotations, so an annotation that is
NOT backed by circuit structure (an arbitrary two-device pairing) is flagged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_MOS = ("nmos", "pmos")

# Supply/ground nets: a TRUE differential pair shares an INTERNAL tail node, not
# the supply rail. Every PMOS shares vdd as source and every NMOS shares vss/gnd,
# so without this exclusion the shared-source heuristic spuriously pairs any two
# same-rail devices. Cross-coupled latches are exempt (their g<->d cross-wiring is
# the signature, and double-tail latches even split the source).
_SUPPLY_NETS = frozenset({
    "vdd", "vdd!", "vcc", "vcc!", "vss", "vss!", "gnd", "gnd!", "0",
    "vsubs", "vsubs!", "vgnd", "vpwr", "vgnd!",
})


def _is_mos(dev) -> bool:
    return getattr(dev, "device_class", None) in _MOS


def _wl(dev) -> Tuple[float, float]:
    s = dev.sizing or {}
    return (round(float(s.get("W", 0.0)), 12), round(float(s.get("L", 0.0)), 12))


def _norm_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


@dataclass(frozen=True)
class SymmetryGroup:
    """A circuit-symmetric structure derived from the topology.

    ``pairs`` are matched device instance-name pairs (mirror-image circuit roles).
    ``differential_nets`` are the net pairs that mirror about the axis (e.g.
    (inn, inp), (outn, outp)). ``common_nets`` are on-axis shared nets (tail, vss).
    ``self_symmetric`` are unpaired devices centred on the axis. ``source`` records
    provenance (topology-diffpair / topology-xcouple / mirror_match / annotation),
    so a PASS reflects circuit structure, never a guessed physical pairing.
    """

    pairs: Tuple[Tuple[str, str], ...]
    axis: str = "vertical"
    differential_nets: Tuple[Tuple[str, str], ...] = ()
    common_nets: Tuple[str, ...] = ()
    self_symmetric: Tuple[str, ...] = ()
    source: str = "topology"
    kind: str = "mixed"

    @property
    def devices(self) -> Tuple[str, ...]:
        out: List[str] = []
        for a, b in self.pairs:
            out.extend((a, b))
        out.extend(self.self_symmetric)
        return tuple(out)


@dataclass(frozen=True)
class _RawPair:
    a: str
    b: str
    diff_nets: Tuple[Tuple[str, str], ...]
    common: Optional[str]
    kind: str


def _find_raw_pairs(devs: Sequence) -> List[_RawPair]:
    """Find differential and cross-coupled MOS pairs from the topology.

    Cross-coupled latches are matched by their g<->d cross-wiring (the defining,
    spurious-proof signature) WITHOUT requiring a shared source -- a double-tail
    latch even splits the source. Differential pairs require a shared NON-SUPPLY
    source (an internal tail), since every PMOS/NMOS trivially shares the supply
    rail and that must not be mistaken for a tail.
    """
    raws: List[_RawPair] = []
    used = set()
    for i, a in enumerate(devs):
        if a.instance_name in used:
            continue
        for b in devs[i + 1:]:
            if b.instance_name in used:
                continue
            if a.device_class != b.device_class or _wl(a) != _wl(b):
                continue
            sa, sb = a.pins.get("s"), b.pins.get("s")
            ga, gb = a.pins.get("g"), b.pins.get("g")
            da, db = a.pins.get("d"), b.pins.get("d")
            shared_src = sa if (sa is not None and sa == sb) else None
            # cross-coupled latch: each gate tied to the other's drain (source not
            # required to match -- the cross-wiring is the signature).
            if ga is not None and ga == db and gb == da and ga != gb:
                raws.append(_RawPair(
                    a.instance_name, b.instance_name,
                    (_norm_pair(da, db),), shared_src, "cross_coupled"))
                used.add(a.instance_name); used.add(b.instance_name)
                break
            # differential pair: shared INTERNAL (non-supply) source + distinct
            # gates AND distinct drains.
            if (shared_src is not None and shared_src.lower() not in _SUPPLY_NETS
                    and ga != gb and da != db and ga is not None and da is not None):
                raws.append(_RawPair(
                    a.instance_name, b.instance_name,
                    (_norm_pair(ga, gb), _norm_pair(da, db)), shared_src, "diff_pair"))
                used.add(a.instance_name); used.add(b.instance_name)
                break
    return raws


def derive_symmetry_groups(cell) -> List[SymmetryGroup]:
    """Derive circuit-symmetric groups from ``cell.devices`` (the topology)."""
    devs = [d for d in getattr(cell, "devices", []) if _is_mos(d)]
    raws = _find_raw_pairs(devs)
    if not raws:
        return []

    # Merge raw pairs that share a differential net into one group (one axis).
    # Union-find over raw-pair indices keyed by shared differential nets.
    parent = list(range(len(raws)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    net_to_raw: Dict[Tuple[str, str], int] = {}
    for idx, r in enumerate(raws):
        for dn in r.diff_nets:
            if dn in net_to_raw:
                union(idx, net_to_raw[dn])
            else:
                net_to_raw[dn] = idx

    clusters: Dict[int, List[int]] = {}
    for idx in range(len(raws)):
        clusters.setdefault(find(idx), []).append(idx)

    groups: List[SymmetryGroup] = []
    grouped_common = set()
    for members in clusters.values():
        pairs = tuple((raws[m].a, raws[m].b) for m in members)
        diff_nets = tuple(sorted({dn for m in members for dn in raws[m].diff_nets}))
        commons = tuple(sorted({raws[m].common for m in members if raws[m].common is not None}))
        grouped_common.update(commons)
        kinds = {raws[m].kind for m in members}
        kind = next(iter(kinds)) if len(kinds) == 1 else "mixed"
        src = "topology-diffpair" if kinds == {"diff_pair"} else (
            "topology-xcouple" if kinds == {"cross_coupled"} else "topology-mixed")
        groups.append(SymmetryGroup(
            pairs=pairs, axis="vertical", differential_nets=diff_nets,
            common_nets=commons, self_symmetric=(), source=src, kind=kind))

    # Unpaired MOS whose drain or source touches a group common net = on-axis.
    paired = {n for g in groups for n in g.devices}
    for g_idx, g in enumerate(groups):
        on_axis = []
        for d in devs:
            if d.instance_name in paired:
                continue
            if d.pins.get("d") in g.common_nets or d.pins.get("s") in g.common_nets:
                on_axis.append(d.instance_name)
        if on_axis:
            groups[g_idx] = SymmetryGroup(
                pairs=g.pairs, axis=g.axis, differential_nets=g.differential_nets,
                common_nets=g.common_nets, self_symmetric=tuple(sorted(on_axis)),
                source=g.source, kind=g.kind)
            paired.update(on_axis)
    return groups


@dataclass
class ConsistencyReport:
    """Cross-check of circuit-derived symmetry vs explicit annotations."""

    derived_pairs: Tuple[Tuple[str, str], ...]
    annotated_pairs: Tuple[Tuple[str, str], ...]
    unbacked: Tuple[Tuple[str, str], ...] = ()        # annotated but NOT circuit-symmetric
    underdeclared: Tuple[Tuple[str, str], ...] = ()    # circuit-symmetric but NOT annotated

    @property
    def consistent(self) -> bool:
        return not self.unbacked and not self.underdeclared


def crosscheck_against_annotations(
    groups: Sequence[SymmetryGroup],
    annotated_pairs: Sequence[Tuple[str, str]],
) -> ConsistencyReport:
    """Flag annotations not backed by circuit structure (``unbacked``) and
    circuit-symmetric pairs the source never annotated (``underdeclared``)."""
    derived = {_norm_pair(a, b) for g in groups for (a, b) in g.pairs}
    annot = {_norm_pair(a, b) for (a, b) in annotated_pairs}
    return ConsistencyReport(
        derived_pairs=tuple(sorted(derived)),
        annotated_pairs=tuple(sorted(annot)),
        unbacked=tuple(sorted(annot - derived)),
        underdeclared=tuple(sorted(derived - annot)),
    )
