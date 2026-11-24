"""Shared lowering helpers for declaration-driven private MOS wells."""

from __future__ import annotations

from typing import Any


def declared_well_isolated_devices(cell_def: Any) -> tuple[str, ...]:
    """Return device targets carrying the existing ``@well_isolated`` intent."""

    if cell_def is None:
        return ()
    from clay.layout.annotations import WellIsolated

    annotations = getattr(cell_def, "__dict__", {}).get(
        "_layout_annotations", {}
    ) or {}
    names = {
        str(target)
        for target, items in annotations.items()
        if str(target) != "__cell__"
        and any(isinstance(item, WellIsolated) for item in (items or ()))
    }
    return tuple(sorted(names))


def deep_well_lvs_model_overrides(cell_def: Any, pdk: str) -> dict[str, str]:
    """Resolve isolated instances to the PDK's extracted deep-well models.

    Unsupported device classes and unsupported PDKs are contract failures; an
    ordinary instance receives no override and keeps the baseline LVS model.
    """

    isolated = set(declared_well_isolated_devices(cell_def))
    if not isolated:
        return {}
    from clay.pdk import get_pdk

    rules = get_pdk(pdk).deep_nwell()
    if rules is None:
        raise ValueError(
            f"{pdk}: @well_isolated requires a proven deep-N-well recipe"
        )
    devices = {
        str(getattr(device, "instance_name", "")): device
        for device in (getattr(cell_def, "devices", ()) or ())
    }
    missing = sorted(isolated - set(devices))
    if missing:
        raise ValueError(f"@well_isolated targets absent devices: {missing}")
    out: dict[str, str] = {}
    for name in sorted(isolated):
        device_class = str(getattr(devices[name], "device_class", "")).casefold()
        model = rules.lvs_models.get(device_class)
        if not model:
            raise ValueError(
                f"{pdk}: no deep-N-well LVS model for {device_class!r} ({name})"
            )
        out[name] = str(model)
    return out


__all__ = [
    "declared_well_isolated_devices",
    "deep_well_lvs_model_overrides",
]
