"""Faithful circuit-schematic model for Clay cell netlists.

A :class:`SchematicModel` is a structural, netlist-faithful description of a
cell: one :class:`SchDevice` per device row (with its exact drain / gate /
source / body nets) and one :class:`SchNet` per net. It carries NO geometry --
placement (and any pretty routing) is added by :mod:`clay.schematic.placement`
and serialised by :mod:`clay.schematic.export`.

The first instance is the 17-device double-tail dynamic latch comparator whose
verbatim connectivity lives in ``clay.layout.magical_comparator`` as
``COMP_DEVICE_PLAN`` -- this module imports that plan READ-ONLY and never
hardcodes net names ("gnd"/"vdd" come from ``GND_NET`` / ``VDD_NET``). The
builder is kept generic so any ``(name, kind, drain, gate, source)`` netlist
maps to a model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple

# Net "klass" buckets for the viewer's colouring / highlight hooks.
NET_KLASSES = ("signal", "clk", "supply", "bias")

# The faithful comparator role map (from the COMP_DEVICE_PLAN comments). Each
# device name -> the schematic-level functional role it plays. Encoded once here
# so the model stays generic and any netlist can pass its own role map.
COMPARATOR_ROLES: Dict[str, str] = {
    "M7": "tail_clk_switch",
    "M5": "input_pair",
    "M6": "input_pair",
    "M18": "precharge1",
    "M2": "precharge1",
    "M4": "regen_nmos",
    "M3": "regen_nmos",
    "M14": "regen_pmos",
    "M13": "regen_pmos",
    "M1": "precharge2",
    "M12": "precharge2",
    "M16": "out_inv_m",
    "M8": "out_inv_m",
    "M17": "out_inv_p",
    "M15": "out_inv_p",
    "M0": "reset_bleed",
    "M22": "reset_bleed",
}


@dataclass(frozen=True)
class SchDevice:
    """One schematic device, faithful to a netlist row.

    ``terminals`` maps a terminal name (``drain`` / ``gate`` / ``source`` /
    ``body``) to the net it connects to, e.g.
    ``{"drain": "intern", "gate": "net0", "source": "net069", "body": "gnd"}``.

    The ``body`` (bulk) terminal stays in ``terminals`` for topological
    faithfulness, but it is NOT routed as a wire (a long bulk lead just clutters
    the schematic). Instead the viewer draws a SHORT local bulk tie on the MOS
    symbol; the net it ties to is exposed as :attr:`body_tie` and excluded from
    every net's routed terminal list (see :meth:`SchematicModel.terminals_of_net`).
    """

    id: str
    kind: str  # "nmos" | "pmos" | "cap" | "res"
    role: str
    W: float
    L: float
    nf: int
    terminals: Dict[str, str] = field(default_factory=dict)

    def net_of(self, term: str) -> str:
        return self.terminals[term]

    @property
    def body_tie(self) -> str | None:
        """The bulk net this device ties locally (a short symbol tie, NOT a
        routed wire). ``None`` for devices without a body terminal."""
        return self.terminals.get("body")


@dataclass(frozen=True)
class SchNet:
    """One net. ``klass`` in {signal, clk, supply, bias}."""

    name: str
    klass: str

    def __post_init__(self) -> None:
        if self.klass not in NET_KLASSES:
            raise ValueError(
                f"SchNet {self.name!r}: klass {self.klass!r} not in {NET_KLASSES}"
            )


@dataclass
class SchematicModel:
    """A cell's structural schematic: its devices + its nets."""

    cell: str
    devices: List[SchDevice]
    nets: List[SchNet]

    def device(self, dev_id: str) -> SchDevice:
        for d in self.devices:
            if d.id == dev_id:
                return d
        raise KeyError(dev_id)

    def net(self, name: str) -> SchNet:
        for n in self.nets:
            if n.name == name:
                return n
        raise KeyError(name)

    def net_names(self) -> List[str]:
        return [n.name for n in self.nets]

    def terminals_of_net(self, name: str) -> List[str]:
        """All ROUTED ``"<devId>.<term>"`` references that connect to ``name``,
        in device order then terminal order (deterministic).

        The ``body`` (bulk) terminal is EXCLUDED: the bulk net is tied with a
        short local symbol tie (``device.body_tie``), never a routed wire, so it
        must not appear in any net's routed terminal list. The device still keeps
        its ``body`` terminal under ``device.terminals`` for faithfulness.
        """
        refs: List[str] = []
        for d in self.devices:
            for term in sorted(d.terminals):
                if term == "body":
                    continue
                if d.terminals[term] == name:
                    refs.append(f"{d.id}.{term}")
        return refs


def classify_net(name: str, *, vdd: str, gnd: str, clk_net: str = "clk") -> str:
    """Deterministic net-class assignment.

    ``vdd`` / ``gnd`` -> supply, the clock net -> clk, everything else ->
    signal. (No ``bias`` nets in the comparator; the bucket exists in the
    contract for biased cells.)
    """
    if name in (vdd, gnd):
        return "supply"
    if name == clk_net:
        return "clk"
    return "signal"


def from_device_plan(
    plan: Sequence[Tuple],
    vdd: str,
    gnd: str,
    roles: Mapping[str, str],
    *,
    cell: str = "comparator",
    clk_net: str = "clk",
) -> SchematicModel:
    """Build a faithful :class:`SchematicModel` from a device plan.

    ``plan`` is a sequence of rows ``(name, kind, drain, gate, source, nf, W,
    L)`` (the ``COMP_DEVICE_PLAN`` shape). Every row becomes exactly one
    :class:`SchDevice` with its exact drain / gate / source nets; the body net
    is ``gnd`` for nmos and ``vdd`` for pmos (standard MOS bulk tie). Nets are
    discovered from the plan (no net invented or dropped), de-duplicated in
    first-seen order, and classified.
    """
    devices: List[SchDevice] = []
    net_order: List[str] = []
    seen: set = set()

    def _see(net: str) -> None:
        if net not in seen:
            seen.add(net)
            net_order.append(net)

    for row in plan:
        name, kind, d_net, g_net, s_net, nf, w, l = row
        body = gnd if kind == "nmos" else vdd
        terminals = {
            "drain": d_net,
            "gate": g_net,
            "source": s_net,
            "body": body,
        }
        devices.append(
            SchDevice(
                id=name,
                kind=kind,
                role=roles.get(name, "unknown"),
                W=float(w),
                L=float(l),
                nf=int(nf),
                terminals=terminals,
            )
        )
        for net in (d_net, g_net, s_net, body):
            _see(net)

    nets = [
        SchNet(name=n, klass=classify_net(n, vdd=vdd, gnd=gnd, clk_net=clk_net))
        for n in net_order
    ]
    return SchematicModel(cell=cell, devices=devices, nets=nets)


def comparator_model(cell: str = "comparator") -> SchematicModel:
    """Build the faithful 17-device comparator model from the verbatim
    ``COMP_DEVICE_PLAN`` + ``GND_NET`` / ``VDD_NET`` constants (imported,
    not hardcoded)."""
    from clay.layout.magical_comparator import (
        COMP_DEVICE_PLAN,
        GND_NET,
        VDD_NET,
    )

    return from_device_plan(
        COMP_DEVICE_PLAN,
        vdd=VDD_NET,
        gnd=GND_NET,
        roles=COMPARATOR_ROLES,
        cell=cell,
    )


__all__ = [
    "NET_KLASSES",
    "COMPARATOR_ROLES",
    "SchDevice",
    "SchNet",
    "SchematicModel",
    "classify_net",
    "from_device_plan",
    "comparator_model",
]
