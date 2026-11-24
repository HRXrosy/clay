"""Serialise a placed :class:`SchematicModel` to the ``schematic.json`` contract.

``to_schematic_dict(model, placement)`` emits EXACTLY the agreed schema (the
contract shared with the web viewer); ``write_schematic_json`` writes it with a
stable key order + trailing newline so re-export is byte-idempotent. Device ids
and net names are the verbatim netlist ids (``M5`` ..., ``intern`` ...) -- the
cross-highlight hooks for the viewer and any future layout / 3D layers.

CLI::

    python -m clay.schematic.export \\
        --cell comparator --out out/schematic/comparator.schematic.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .model import SchematicModel, comparator_model
from .placement import Placement, place

# The signal-path spine + differential-pair groupings of the comparator (part
# of the contract; faithful to the topology comments in COMP_DEVICE_PLAN).
COMPARATOR_SIGNAL_PATH = [
    "net0", "net1", "intern", "interp", "crossn", "crossp", "outm", "outp",
]
COMPARATOR_CLK_NET = "clk"
COMPARATOR_DIFF_PAIRS = [
    ["intern", "interp"],
    ["crossn", "crossp"],
    ["outm", "outp"],
    ["net0", "net1"],
]


def _terminal_block(model: SchematicModel, pl: Placement) -> Dict[str, Dict[str, Any]]:
    """Per-device terminals: {term: {net, x, y}} in a stable term order."""
    blocks: Dict[str, Dict[str, Any]] = {}
    term_order = ("drain", "gate", "source", "body")
    for d in model.devices:
        dp = pl.devices[d.id]
        ordered = [t for t in term_order if t in d.terminals]
        ordered += [t for t in sorted(d.terminals) if t not in term_order]
        blocks[d.id] = {
            term: {
                "net": d.terminals[term],
                "x": dp.pins[term][0],
                "y": dp.pins[term][1],
            }
            for term in ordered
        }
    return blocks


def to_schematic_dict(
    model: SchematicModel,
    placement: Placement,
    *,
    signal_path: List[str] | None = None,
    clk_net: str = COMPARATOR_CLK_NET,
    diff_pairs: List[List[str]] | None = None,
) -> Dict[str, Any]:
    """Emit the schematic dict EXACTLY in the contract shape."""
    term_blocks = _terminal_block(model, placement)

    devices: List[Dict[str, Any]] = []
    for d in model.devices:
        dp = placement.devices[d.id]
        devices.append(
            {
                "id": d.id,
                "kind": d.kind,
                "role": d.role,
                "W": d.W,
                "L": d.L,
                "nf": d.nf,
                "x": dp.x,
                "y": dp.y,
                "flip": bool(dp.flip),
                # Orientation hint for the viewer: "MX" = mirrored about X
                # (vertical flip, source-up), "R0" = unflipped. Derived from flip
                # so it can never disagree with the pin geometry.
                "orient": "MX" if dp.flip else "R0",
                "box": [dp.box[0], dp.box[1]],
                # The bulk net is tied with a SHORT local symbol tie, not a
                # routed wire (see SchDevice.body_tie / terminals_of_net).
                "body_tie": d.body_tie,
                "terminals": term_blocks[d.id],
            }
        )

    nets: List[Dict[str, Any]] = []
    for n in model.nets:
        lx, ly = placement.net_labels[n.name]
        wires = [[list(pt) for pt in poly] for poly in placement.net_wires.get(n.name, [])]
        nets.append(
            {
                "name": n.name,
                "klass": n.klass,
                "terminals": model.terminals_of_net(n.name),
                "wires": wires,
                "label": {"x": lx, "y": ly},
            }
        )

    rails = [
        {"net": r.net, "x0": r.x0, "x1": r.x1, "y": r.y} for r in placement.rails
    ]

    ports = [
        {"net": p.net, "kind": p.kind, "x": p.x, "y": p.y, "dir": p.dir}
        for p in placement.ports
    ]

    junctions = [
        {"net": j.net, "x": j.x, "y": j.y} for j in placement.junctions
    ]

    vb = placement.viewbox
    return {
        "cell": model.cell,
        "viewbox": [vb[0], vb[1], vb[2], vb[3]],
        "rails": rails,
        "ports": ports,
        "junctions": junctions,
        "devices": devices,
        "nets": nets,
        "signal_path": list(signal_path) if signal_path is not None else list(COMPARATOR_SIGNAL_PATH),
        "clk_net": clk_net,
        "diff_pairs": [list(p) for p in (diff_pairs if diff_pairs is not None else COMPARATOR_DIFF_PAIRS)],
    }


def schematic_json_text(data: Dict[str, Any]) -> str:
    """Canonical JSON text: 2-space indent, keys NOT re-sorted (the dict order
    is already deterministic), trailing newline -> byte-idempotent re-export."""
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def write_schematic_json(path: str | Path, data: Dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(schematic_json_text(data), encoding="utf-8")
    return out


def build_comparator_schematic_dict(cell: str = "comparator") -> Dict[str, Any]:
    """Convenience: faithful comparator model -> placement -> schematic dict."""
    from clay.layout.magical_comparator import GND_NET, VDD_NET

    model = comparator_model(cell=cell)
    pl = place(model, vdd=VDD_NET, gnd=GND_NET)
    return to_schematic_dict(model, pl)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clay.schematic.export",
        description="Export a faithful Clay cell schematic to JSON.",
    )
    parser.add_argument("--cell", default="comparator", help="cell name (default: comparator)")
    parser.add_argument(
        "--out",
        default="out/schematic/comparator.schematic.json",
        help="output JSON path",
    )
    args = parser.parse_args(argv)

    if args.cell != "comparator":
        parser.error(f"unknown cell {args.cell!r}; only 'comparator' is wired so far")

    data = build_comparator_schematic_dict(cell=args.cell)
    out = write_schematic_json(args.out, data)
    print(
        f"wrote {out} ({len(data['devices'])} devices, {len(data['nets'])} nets)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPARATOR_SIGNAL_PATH",
    "COMPARATOR_CLK_NET",
    "COMPARATOR_DIFF_PAIRS",
    "to_schematic_dict",
    "schematic_json_text",
    "write_schematic_json",
    "build_comparator_schematic_dict",
    "main",
]
