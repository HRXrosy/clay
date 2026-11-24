"""R145 (BL-145B) -- schematic SPICE netlist export for gf180mcu LVS.

LVS (Layout-Versus-Schematic) compares the device list + connectivity that
KLayout *extracts from the layout* against a reference "schematic" netlist
that states what the cell is *supposed* to be. This module emits that
reference netlist for a Clay cell.

Why this is separate from the ngspice driver
=============================================
``clay/drivers/ngspice.py`` already emits MOSFET subckt-instance lines for
simulation (``x<name> d g s b <model> W=.. L=.. m=..``), and we reuse that
exact element convention here so the schematic the LVS deck sees matches
what Clay simulates. But the ngspice path builds its netlist from a
``Testbench`` (``CellSpec`` with explicit per-device pin dicts). A *layout*
cell (``AnnotatedCell`` produced by the cell-template factories) carries no
explicit net names -- it carries device layouts + spatial annotations plus
an implicit textbook topology. This module bridges that gap: it derives the
schematic connectivity from the recognised topology (the same recognition
``cell_netlist.classify_topology`` performs) and writes a flat SPICE subckt.

gf180mcu device model names
===========================
The gf180mcu KLayout LVS deck extracts MOSFETs as ``mos4`` devices named
``nmos_3p3`` / ``pmos_3p3`` (3.3 V core flavour), e.g.::

    extract_devices(mos4("nmos_3p3"), { "SD" => nsd, "G" => ngate_3p3v, ... })
    extract_devices(mos4("pmos_3p3"), { "SD" => psd, "G" => pgate_3p3v, ... })

Source: ``AnalogGym/PDK/gf180mcu_pdk/libraries/gf180mcu_fd_pr/latest/rules/
klayout/lvs/gf180mcu.lvs`` lines 1298 (pmos_3p3) and 1332 (nmos_3p3).

The matching SPICE subckts in ``sm141064.ngspice`` declare the pin order
``d g s b``::

    .subckt nmos_3p3 d g s b w=1e-5 l=2.8e-7
    .subckt pmos_3p3 d g s b w=1e-5 l=2.8e-7

So the schematic netlist instantiates each device with exactly that pin
order, and the model names are taken from the gf180mcu Clay PDK adapter's
``device_bindings`` (``nmos`` -> ``nmos_3p3``, ``pmos`` -> ``pmos_3p3``) so
no model name is hard-coded here that the PDK/deck does not recognise.

No fabricated connectivity
==========================
For a recognised topology (current mirror, 5T-OTA) the net names come from
the textbook invariants of that topology (shared gate, common source rail,
diode connection, tail node). For an unrecognised topology the export
raises rather than inventing nets it cannot justify.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .annotations import AnnotatedCell
from .cell_netlist import TopologyKind, classify_topology


# gf180mcu 3.3 V core device model names recognised by the LVS deck
# (mos4("nmos_3p3") / mos4("pmos_3p3"), gf180mcu.lvs lines 1332 / 1298) and
# declared as `.subckt <name> d g s b ...` in sm141064.ngspice.
GF180_NMOS_MODEL = "nmos_3p3"
GF180_PMOS_MODEL = "pmos_3p3"

# The subckt pin order both the model card and the Clay ngspice driver use.
MOS_PIN_ORDER: Tuple[str, str, str, str] = ("d", "g", "s", "b")


class MosType:
    NMOS = "nmos"
    PMOS = "pmos"


def _model_for(mos_type: str, bindings: Optional[Dict[str, str]]) -> str:
    """Resolve the gf180 model name for an NMOS/PMOS device.

    Prefers the PDK adapter's ``device_bindings`` (so the same code works if
    a caller swaps in the 6 V flavour) and falls back to the documented 3.3 V
    core names. Never returns a name not sourced from the deck/PDK.
    """
    if bindings:
        if mos_type == MosType.NMOS and "nmos" in bindings:
            return bindings["nmos"]
        if mos_type == MosType.PMOS and "pmos" in bindings:
            return bindings["pmos"]
    return GF180_NMOS_MODEL if mos_type == MosType.NMOS else GF180_PMOS_MODEL


@dataclass
class LvsDevice:
    """One MOSFET in the schematic netlist: its type, model, and net pins.

    ``pins`` maps the four MOS terminals (d/g/s/b) to schematic net names.
    """

    instance: str
    mos_type: str
    pins: Dict[str, str]
    w_um: float
    l_um: float
    nf: int = 1

    def model_name(self, bindings: Optional[Dict[str, str]]) -> str:
        return _model_for(self.mos_type, bindings)

    def emit(self, bindings: Optional[Dict[str, str]]) -> str:
        """Emit the SPICE subckt-instance line (matches ngspice driver)."""
        model = self.model_name(bindings)
        d, g, s, b = (self.pins[p] for p in MOS_PIN_ORDER)
        return (
            f"x{self.instance} {d} {g} {s} {b} {model} "
            f"W={self.w_um:g} L={self.l_um:g} m={int(self.nf)}"
        )


@dataclass
class LvsSchematic:
    """Device list + port list of the cell's schematic netlist."""

    cell_name: str
    ports: List[str] = field(default_factory=list)
    devices: List[LvsDevice] = field(default_factory=list)

    def nets(self) -> List[str]:
        seen: List[str] = []
        for dev in self.devices:
            for p in MOS_PIN_ORDER:
                n = dev.pins[p]
                if n not in seen:
                    seen.append(n)
        return seen

    def device_count(self) -> int:
        return len(self.devices)

    def count_by_type(self) -> Dict[str, int]:
        counts = {MosType.NMOS: 0, MosType.PMOS: 0}
        for dev in self.devices:
            counts[dev.mos_type] += 1
        return counts


def _sizing_for(
    device: str,
    sizing: Optional[Dict[str, Tuple[float, float]]],
    default_w: float,
    default_l: float,
) -> Tuple[float, float]:
    if sizing and device in sizing:
        w, l = sizing[device]
        return float(w), float(l)
    return default_w, default_l


def _nf_for(cell: AnnotatedCell, device: str) -> int:
    layout = cell.layouts.get(device)
    if layout is None:
        return 1
    return int(getattr(layout, "finger_count", 1))


def _current_mirror_schematic(
    cell: AnnotatedCell,
    sizing: Optional[Dict[str, Tuple[float, float]]],
    default_w: float,
    default_l: float,
) -> LvsSchematic:
    """Build the current-mirror schematic from its textbook topology.

    Both transistors are NMOS sharing a gate net; both sources tie to the
    common source rail (vss); the reference is diode-connected so its drain
    equals the shared gate net; the output transistor's drain is the output
    port. Bulks tie to the source rail (gf180 nmos_3p3 W terminal = sub).
    """
    nets = {
        "gate": "vbias",        # shared gate == ref drain (diode-connected)
        "vss": "vss",
        "vout": "vout",
    }
    gate = nets["gate"]
    devices: List[LvsDevice] = []
    # M_ref: diode-connected -> drain == gate net.
    w, l = _sizing_for("M_ref", sizing, default_w, default_l)
    devices.append(
        LvsDevice(
            instance="M_ref",
            mos_type=MosType.NMOS,
            pins={"d": nets["gate"], "g": nets["gate"], "s": nets["vss"], "b": nets["vss"]},
            w_um=w,
            l_um=l,
            nf=_nf_for(cell, "M_ref"),
        )
    )
    w, l = _sizing_for("M_out", sizing, default_w, default_l)
    devices.append(
        LvsDevice(
            instance="M_out",
            mos_type=MosType.NMOS,
            pins={"d": nets["vout"], "g": nets["gate"], "s": nets["vss"], "b": nets["vss"]},
            w_um=w,
            l_um=l,
            nf=_nf_for(cell, "M_out"),
        )
    )
    return LvsSchematic(
        cell_name=cell.cell_name,
        ports=[gate, nets["vout"], nets["vss"]],
        devices=devices,
    )


def _five_t_ota_schematic(
    cell: AnnotatedCell,
    sizing: Optional[Dict[str, Tuple[float, float]]],
    default_w: float,
    default_l: float,
) -> LvsSchematic:
    """Build the 5T-OTA schematic from its textbook topology.

    Devices (per five_t_ota.py): M_in_p / M_in_n NMOS input pair, M_load_p /
    M_load_n PMOS current-mirror load (diode-connected on the plus branch),
    M_tail NMOS tail source.

    Nets:
      - vdd          : PMOS load sources + PMOS bulks (nwell)
      - vss          : tail source + all NMOS bulks (sub)
      - in_p / in_n  : input gates (ports)
      - tail         : input-pair common source + tail drain
      - load_gate    : load pair shared gate == load_p drain (diode tie)
                       == left-branch drain of M_in_p
      - vout         : load_n drain == in_n drain (output port)
      - vbias        : tail gate (bias port)
    """
    nets = {
        "vdd": "vdd",
        "vss": "vss",
        "in_p": "in_p",
        "in_n": "in_n",
        "tail": "tail",
        "load_gate": "load_gate",
        "vout": "vout",
        "vbias": "vbias",
    }
    devices: List[LvsDevice] = []

    def add(instance: str, mos_type: str, d: str, g: str, s: str, b: str) -> None:
        w, l = _sizing_for(instance, sizing, default_w, default_l)
        devices.append(
            LvsDevice(
                instance=instance,
                mos_type=mos_type,
                pins={"d": d, "g": g, "s": s, "b": b},
                w_um=w,
                l_um=l,
                nf=_nf_for(cell, instance),
            )
        )

    # NMOS input pair: sources to tail node, bulks to vss.
    add("M_in_p", MosType.NMOS, d=nets["load_gate"], g=nets["in_p"], s=nets["tail"], b=nets["vss"])
    add("M_in_n", MosType.NMOS, d=nets["vout"], g=nets["in_n"], s=nets["tail"], b=nets["vss"])
    # PMOS load mirror: sources to vdd, bulks to vdd (nwell). load_p diode-tied.
    add("M_load_p", MosType.PMOS, d=nets["load_gate"], g=nets["load_gate"], s=nets["vdd"], b=nets["vdd"])
    add("M_load_n", MosType.PMOS, d=nets["vout"], g=nets["load_gate"], s=nets["vdd"], b=nets["vdd"])
    # NMOS tail: drain to tail node, gate to bias, source + bulk to vss.
    add("M_tail", MosType.NMOS, d=nets["tail"], g=nets["vbias"], s=nets["vss"], b=nets["vss"])

    return LvsSchematic(
        cell_name=cell.cell_name,
        ports=[nets["in_p"], nets["in_n"], nets["vout"], nets["vbias"], nets["vdd"], nets["vss"]],
        devices=devices,
    )


def build_lvs_schematic(
    cell: AnnotatedCell,
    sizing: Optional[Dict[str, Tuple[float, float]]] = None,
    *,
    default_w_um: float = 4.0,
    default_l_um: float = 0.5,
) -> LvsSchematic:
    """Build the schematic device/net model for a Clay layout cell.

    ``sizing`` is the ``DeviceSizing`` dict the cell-template factory returns
    (device -> (w_um, l_um)). When omitted, every device falls back to
    ``default_w_um`` / ``default_l_um``.

    Raises ``ValueError`` for a topology this exporter does not recognise --
    it never fabricates connectivity it cannot derive from a textbook
    topology.
    """
    topo = classify_topology(cell)
    if topo == TopologyKind.CURRENT_MIRROR:
        return _current_mirror_schematic(cell, sizing, default_w_um, default_l_um)
    if topo == TopologyKind.FIVE_T_OTA:
        return _five_t_ota_schematic(cell, sizing, default_w_um, default_l_um)
    raise ValueError(
        f"LVS schematic export: cell {cell.cell_name!r} has unrecognised "
        f"topology {topo.value!r}. Supported: current_mirror, five_t_ota. "
        f"Refusing to fabricate connectivity for an unknown cell."
    )


def export_lvs_netlist(
    cell: AnnotatedCell,
    sizing: Optional[Dict[str, Tuple[float, float]]] = None,
    *,
    bindings: Optional[Dict[str, str]] = None,
    default_w_um: float = 4.0,
    default_l_um: float = 0.5,
    include_header: bool = True,
) -> str:
    """Export the gf180mcu LVS *schematic* SPICE netlist for ``cell``.

    The returned string is a flat ``.subckt`` whose body is one
    ``x<name> d g s b <model> W=.. L=.. m=..`` line per MOSFET, using the
    gf180mcu model names (``nmos_3p3`` / ``pmos_3p3``) the KLayout LVS deck
    recognises. Connectivity is the textbook topology of the cell.

    Args:
        cell: the ``AnnotatedCell`` (a cell-template factory's first return).
        sizing: optional device -> (w_um, l_um) map (the factory's second
            return). Missing devices fall back to ``default_w_um/_l_um``.
        bindings: optional PDK ``device_bindings`` map (``nmos``/``pmos`` ->
            model). Defaults to the gf180mcu 3.3 V core names.
        include_header: prepend an informational comment header.

    Returns:
        The SPICE netlist as a string.
    """
    sch = build_lvs_schematic(
        cell, sizing, default_w_um=default_w_um, default_l_um=default_l_um
    )
    lines: List[str] = []
    if include_header:
        counts = sch.count_by_type()
        nmos_model = _model_for(MosType.NMOS, bindings)
        pmos_model = _model_for(MosType.PMOS, bindings)
        lines.append(f"* gf180mcu LVS schematic netlist for cell {sch.cell_name!r}")
        lines.append(
            f"* devices: {sch.device_count()} "
            f"(nmos={counts[MosType.NMOS]}, pmos={counts[MosType.PMOS]}); "
            f"models: {nmos_model} / {pmos_model}; pin order d g s b"
        )
    lines.append(f".subckt {sch.cell_name} {' '.join(sch.ports)}")
    for dev in sch.devices:
        lines.append("  " + dev.emit(bindings))
    lines.append(".ends")
    return "\n".join(lines) + "\n"


__all__ = [
    "GF180_NMOS_MODEL",
    "GF180_PMOS_MODEL",
    "MOS_PIN_ORDER",
    "MosType",
    "LvsDevice",
    "LvsSchematic",
    "build_lvs_schematic",
    "export_lvs_netlist",
]
