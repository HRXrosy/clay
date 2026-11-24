"""Clay faithful circuit-schematic exporter.

Turns a Clay cell netlist into a deterministic ``schematic.json`` that a web
viewer renders. The model is netlist-faithful (every device row -> one device,
every terminal -> its exact net); placement and JSON export are pure functions
so re-export is byte-idempotent.

Public API:
    from_device_plan(plan, vdd, gnd, roles) -> SchematicModel
    comparator_model() -> SchematicModel   (the first faithful instance)
    place(model) -> Placement
    to_schematic_dict(model, placement) -> dict
    write_schematic_json(path, data)
    CLI: python -m clay.schematic.export --cell comparator --out <path>
"""
from __future__ import annotations

from .model import (
    COMPARATOR_ROLES,
    NET_KLASSES,
    SchDevice,
    SchNet,
    SchematicModel,
    classify_net,
    comparator_model,
    from_device_plan,
)
from .symbols import (
    GRID,
    SYMBOLS,
    pin_offsets,
    snap,
    symbol_box,
)
from .placement import (
    DevicePlacement,
    Junction,
    Placement,
    Port,
    Rail,
    place,
)
from .export import (
    COMPARATOR_CLK_NET,
    COMPARATOR_DIFF_PAIRS,
    COMPARATOR_SIGNAL_PATH,
    build_comparator_schematic_dict,
    schematic_json_text,
    to_schematic_dict,
    write_schematic_json,
)

__all__ = [
    # symbols
    "GRID",
    "SYMBOLS",
    "pin_offsets",
    "snap",
    "symbol_box",
    # model
    "NET_KLASSES",
    "COMPARATOR_ROLES",
    "SchDevice",
    "SchNet",
    "SchematicModel",
    "classify_net",
    "from_device_plan",
    "comparator_model",
    # placement
    "DevicePlacement",
    "Rail",
    "Port",
    "Junction",
    "Placement",
    "place",
    # export
    "COMPARATOR_SIGNAL_PATH",
    "COMPARATOR_CLK_NET",
    "COMPARATOR_DIFF_PAIRS",
    "to_schematic_dict",
    "schematic_json_text",
    "write_schematic_json",
    "build_comparator_schematic_dict",
]
