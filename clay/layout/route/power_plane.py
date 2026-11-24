"""Coordinate-free materialization of declared upper-metal power planes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from clay.layout.intent import replay_power_plane_inventory


@dataclass(frozen=True)
class MaterializedPowerPlaneFact:
    """Canonical write fact replayed later by the exact-GDS checker."""

    declaration_sha256: str
    content_id: str
    net: str
    layer_key: Tuple[int, int]
    bbox_nm: Tuple[int, int, int, int]
    access_xy_nm: Tuple[int, int]


def compile_power_plane_facts(
    declarations: object,
    envelope_nm: object,
    tech: Any,
) -> Tuple[MaterializedPowerPlaneFact, ...]:
    """Compile declarations against one final, coordinate-free cell envelope."""

    planes = replay_power_plane_inventory(declarations)
    if not planes:
        return ()
    if (
        type(envelope_nm) is not tuple
        or len(envelope_nm) != 4
        or any(type(value) is not int for value in envelope_nm)
    ):
        raise TypeError("power-plane envelope must be an exact four-int tuple")
    x0, y0, x1, y1 = envelope_nm
    if x1 <= x0 or y1 <= y0:
        raise ValueError("power-plane envelope must have positive area")

    owners_by_layer: Dict[Tuple[int, int], str] = {}
    facts: List[MaterializedPowerPlaneFact] = []
    for plane in planes:
        raw_key = tech.metal_key(plane.layer_index)
        if (
            type(raw_key) is not tuple
            or len(raw_key) != 2
            or any(type(value) is not int for value in raw_key)
        ):
            raise TypeError("installed power-plane layer key is not canonical")
        layer_key = raw_key
        prior_owner = owners_by_layer.setdefault(layer_key, plane.net)
        if prior_owner != plane.net:
            raise ValueError("power planes place different nets on one physical layer")
        identity = json.dumps(
            {
                "bbox_nm": envelope_nm,
                "declaration_sha256": plane.sha256,
                "layer_key": layer_key,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        facts.append(
            MaterializedPowerPlaneFact(
                declaration_sha256=plane.sha256,
                content_id=(
                    "PPLANE_"
                    + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
                ),
                net=plane.net,
                layer_key=layer_key,
                bbox_nm=envelope_nm,
                access_xy_nm=((x0 + x1) // 2, (y0 + y1) // 2),
            )
        )
    return tuple(facts)


def materialize_power_planes(
    declarations: object,
    envelope_nm: object,
    tech: Any,
    *,
    geometry: Dict[str, Any],
    placement_um: Dict[str, Tuple[float, float]],
    pins_by_net: Dict[str, List[Any]],
) -> Tuple[MaterializedPowerPlaneFact, ...]:
    """Atomically write full-envelope plane artwork and one owned access pin."""

    from clay.layout.extract.library_device import CellGeometry
    from clay.layout.route.engine import Pin

    facts = compile_power_plane_facts(declarations, envelope_nm, tech)
    for fact in facts:
        if fact.content_id in geometry or fact.content_id in placement_um:
            raise ValueError(f"power plane {fact.content_id!r} was already emitted")
        if any(
            str(getattr(pin, "device", "")) == fact.content_id
            for pins in pins_by_net.values()
            for pin in pins
        ):
            raise ValueError(f"power plane access {fact.content_id!r} already exists")

    prepared = []
    for fact in facts:
        x0, y0, x1, y1 = fact.bbox_nm
        ring = [[
            (x0 / 1000.0, y0 / 1000.0),
            (x1 / 1000.0, y0 / 1000.0),
            (x1 / 1000.0, y1 / 1000.0),
            (x0 / 1000.0, y1 / 1000.0),
        ]]
        stamp = CellGeometry(
            name=fact.content_id,
            kind="power_plane",
            ref_centroid=(0.0, 0.0),
            polys_by_layer={fact.layer_key: ring},
        )
        stamp.source = "power_plane"
        stamp.power_plane_declaration_sha256 = fact.declaration_sha256
        pin = Pin(
            net=fact.net,
            device=fact.content_id,
            terminal="plane",
            xy_um=(
                fact.access_xy_nm[0] / 1000.0,
                fact.access_xy_nm[1] / 1000.0,
            ),
            layer=fact.layer_key,
            xy_nm=fact.access_xy_nm,
        )
        prepared.append((fact, stamp, pin))

    for fact, stamp, pin in prepared:
        geometry[fact.content_id] = stamp
        placement_um[fact.content_id] = (0.0, 0.0)
        pins_by_net.setdefault(fact.net, []).append(pin)
    return facts


__all__ = [
    "MaterializedPowerPlaneFact",
    "compile_power_plane_facts",
    "materialize_power_planes",
]
