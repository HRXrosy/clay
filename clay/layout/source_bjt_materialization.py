"""Bind BJT construction to the same effective source intent at every build."""
from __future__ import annotations

from typing import Dict


class BjtMaterializationUnavailable(RuntimeError):
    """A declared BJT reached the canonical placer without a proven C/B/E cell."""


def _bjt_array_declarations(cell_def, intent):
    from clay.layout.intent import LayoutIntent, layout_intent_from_cell_def, replay_array_intent

    intent = layout_intent_from_cell_def(cell_def) if intent is None else intent
    if type(intent) is not LayoutIntent:
        raise TypeError("BJT array requires exact LayoutIntent")
    bjts = {str(device.instance_name) for device in cell_def.devices
            if str(device.device_class).lower() in ("pnp", "npn")}
    result = {}
    for value in intent.arrays:
        declared = replay_array_intent(value)
        if declared.group not in bjts:
            if declared.unit_pattern is not None:
                raise BjtMaterializationUnavailable("unit pattern array target must be an actual BJT")
            continue
        if declared.group in result:
            raise BjtMaterializationUnavailable("BJT has duplicate array authority")
        if declared.unit_pattern is None:
            raise BjtMaterializationUnavailable("BJT array requires an explicit occupied unit pattern")
        if declared.align or declared.pin_facing or declared.bank_interconnect is not None:
            raise BjtMaterializationUnavailable("BJT array has unsupported alignment or interconnect authority")
        result[declared.group] = declared
    return result

def _collect_bjts(cell_def, tech, intent=None) -> Dict[str, Dict[str, object]]:
    """Collect declared PNP/NPN cells after proving their adapter geometry.

    ``area`` and ``M`` are electrical parallel-unit multipliers. Their product
    must be a positive integer and becomes a compact bank of exact fixed cells;
    every unit C/B/E access is separately routed.
    """
    from clay.layout.bjt_device_cell import (
        BjtCellMaterializationError,
        build_bjt_cell,
    )
    from clay.pdk import get_pdk, load_all

    arrays = _bjt_array_declarations(cell_def, intent)
    load_all()
    adapter = get_pdk(str(getattr(tech, "pdk_id", "")))
    from clay.layout.annotations import Tap

    annotations = getattr(cell_def, "__dict__", {}).get("_layout_annotations", {}) or {}
    out: Dict[str, Dict[str, object]] = {}
    for dev in getattr(cell_def, "devices", []) or []:
        kind = str(getattr(dev, "device_class", "")).lower()
        if kind not in ("pnp", "npn"):
            continue
        sizing = getattr(dev, "sizing", None) or {}
        area = float(sizing.get("area", 1.0) or 1.0)
        mult = float(sizing.get("M", 1.0) or 1.0)
        units = area * mult
        unit_count = int(round(units))
        if units <= 0.0 or abs(units - unit_count) > 1e-9:
            raise BjtMaterializationUnavailable(
                f"declared BJT {dev.instance_name!r} requests non-integral "
                f"fixed-cell units area*M={units:g}; fail-closed"
            )
        declarations = [
            item
            for item in annotations.get(str(dev.instance_name), ())
            if isinstance(item, Tap)
        ]
        if len(declarations) > 1:
            raise BjtMaterializationUnavailable(
                f"declared BJT {dev.instance_name!r} has multiple @tap obligations"
            )
        tap_declaration = declarations[0] if declarations else None
        binding = (getattr(adapter, "bjt_bindings", {}) or {}).get(kind)
        tap_binding = getattr(binding, "tap_binding", None) if binding is not None else None
        pins = getattr(dev, "pins", None) or {}
        if tap_declaration is not None and tap_binding is not None \
                and tap_binding.mode == "intrinsic_terminal":
            intrinsic_net = str(pins.get(tap_binding.terminal_role, ""))
            if intrinsic_net != str(tap_declaration.net):
                raise BjtMaterializationUnavailable(
                    f"declared BJT {dev.instance_name!r} @tap net {tap_declaration.net!r} "
                    f"does not match intrinsic {tap_binding.terminal_role} net "
                    f"{intrinsic_net!r}"
                )
        try:
            # Build now so an OA-only/missing/digest-mismatched binding stops before
            # the MOS solver or any plausible partial GDS can run.
            build_bjt_cell(
                adapter,
                kind,
                units=unit_count,
                tap_declaration=tap_declaration,
                unit_pattern=arrays[str(dev.instance_name)].unit_pattern if str(dev.instance_name) in arrays else None,
            )
        except BjtCellMaterializationError as exc:
            raise BjtMaterializationUnavailable(
                f"declared BJT {dev.instance_name!r} has no proven C/B/E physical "
                f"materializer for PDK {adapter.pdk_id!r}: {exc}; fail-closed"
            ) from exc
        if set(pins) != {"c", "b", "e"}:
            raise BjtMaterializationUnavailable(
                f"declared BJT {dev.instance_name!r} lacks exact C/B/E binding"
            )
        out[str(dev.instance_name)] = {
            "kind": kind,
            "pins": {role: str(pins[role]) for role in ("c", "b", "e")},
            "area": area,
            "M": mult,
            "source_model": str(getattr(dev, "source_model", "") or ""),
            "units": unit_count,
            "tap_declaration": tap_declaration,
        }
        if str(dev.instance_name) in arrays:
            declared = arrays[str(dev.instance_name)]
            out[str(dev.instance_name)].update(unit_pattern=declared.unit_pattern, array_intent_sha256=declared.sha256)
    return out


