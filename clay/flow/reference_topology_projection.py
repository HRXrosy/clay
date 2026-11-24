"""Compare explicit source connectivity with a bound canonical topology.

Names and ports come from the two inputs, never from a circuit registry. This
contract checks the named graph only, not sizing, geometry, or netlist LVS.
"""
from __future__ import annotations

import hashlib
import json

from clay.flow.circuit_definition import DesignInputError
from clay.flow.reference_evidence import CanonicalTopology

_PINS = {"pmos": ("d", "g", "s", "b"), "nmos": ("d", "g", "s", "b"),
         "pnp": ("c", "b", "e"), "resistor": ("p", "n"), "capacitor": ("p", "n")}


def primitive_kind(raw: str) -> str:
    kind = raw.casefold()
    if kind.startswith("pmos"):
        return "pmos"
    if kind.startswith("nmos"):
        return "nmos"
    return {"res": "resistor", "cap": "capacitor"}.get(kind, kind)


def verify_primitive_reference_topology(cell, canonical: CanonicalTopology) -> str:
    if cell.instances or cell.topology or cell.combinator_block is not None or cell.bias_tree:
        raise DesignInputError("TOPOLOGY_PROJECTION_UNSUPPORTED", "reference graph projection requires explicit primitive topology")
    ports = tuple(str(p).casefold() for p in cell.ports)
    if ports != tuple(p.casefold() for p in canonical.ports):
        raise DesignInputError("REFERENCE_TOPOLOGY_MISMATCH", "ordered source ports differ from reference; an explicit port mapping is required")
    devices = []
    names = set()
    for device in cell.devices:
        name = device.instance_name.casefold()
        if name in names:
            raise DesignInputError("REFERENCE_TOPOLOGY_MISMATCH", f"duplicate source instance {name}")
        names.add(name)
        kind = primitive_kind(device.device_class)
        if kind not in _PINS:
            raise DesignInputError("TOPOLOGY_PROJECTION_UNSUPPORTED", f"unsupported primitive family {device.device_class}")
        pins = {k.casefold(): str(v).casefold() for k, v in device.pins.items()}
        if set(pins) != set(_PINS[kind]):
            raise DesignInputError("REFERENCE_TOPOLOGY_MISMATCH", f"incomplete terminals for {name}")
        devices.append({"instance": name, "kind": kind, "nodes": [pins[k] for k in _PINS[kind]]})
    projection = {"top_cell": canonical.top_cell.casefold(), "ports": list(ports),
                  "devices": sorted(devices, key=lambda d: d["instance"])}
    digest = hashlib.sha256(json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if digest != canonical.normalized_connectivity_sha256:
        raise DesignInputError("REFERENCE_TOPOLOGY_MISMATCH", "named primitive connectivity differs from reference")
    return digest
