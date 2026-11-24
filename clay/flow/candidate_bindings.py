"""One source-declared parameter binding for candidate IR and native SPICE.

Optimization values are SI. Native tokens use the PDK model's declared scales;
model names, terminal maps, unbound tokens and expression dependencies survive.
This module grants no geometry or signoff capability.
"""
from __future__ import annotations

from copy import deepcopy
import math

from clay.frontend.native_spice_import import validate_native_device


def native_semantic_envelope(value):
    """Normalize all parsed native intent, including parser-added declarations.

    Source locations move when literals change. The cell's _clay_file is only
    a backlink to the containing file, whose complete contents are traversed
    by callers; neither field contributes an independent source declaration.
    """
    from dataclasses import fields, is_dataclass
    from enum import Enum

    if isinstance(value, Enum):
        return (type(value).__qualname__, value.value)
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, dict):
        return {key: native_semantic_envelope(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(native_semantic_envelope(item) for item in value))
    if hasattr(value, "__dict__") or is_dataclass(value):
        members = vars(value) if hasattr(value, "__dict__") else {f.name: getattr(value, f.name) for f in fields(value)}
        return (type(value).__module__, type(value).__qualname__, {
            key: native_semantic_envelope(item) for key, item in members.items()
            if key not in {"source_span", "_clay_file"}
        })
    raise TypeError(f"unsupported native semantic value {type(value).__name__}")


def native_parameter_token(device, field: str, value: float, *, pdk_id: str):
    """Return the existing native key and its PDK-scaled numeric token."""
    if not pdk_id:
        raise ValueError("native sizing requires an explicit PDK identity")
    info = validate_native_device(device, pdk_id=pdk_id)
    fields = {key.casefold(): key for key, _ in device.native_parameters}
    low = field.casefold()
    if low not in fields:
        raise ValueError(f"native sizing names missing parameter {field!r}")
    if low not in dict(info.numeric_parameters):
        raise ValueError(f"native sizing cannot overwrite expression parameter {field!r}")
    scales = dict(info.contract.length_scales)
    if low not in scales and low not in info.contract.integer_parameters:
        raise ValueError(f"native sizing parameter {field!r} has no declared optimization unit contract")
    numeric = float(value) / scales.get(low, 1.0)
    if not math.isfinite(numeric):
        raise ValueError(f"native sizing parameter {field!r} is nonfinite")
    if low in info.contract.integer_parameters and not numeric.is_integer():
        raise ValueError(f"native sizing parameter {field!r} must be an integer")
    token = format(numeric, ".12g")
    # The shared validator also enforces signs, all required fields and scope.
    probe = deepcopy(device)
    probe.native_parameters = tuple(
        (key, token if key.casefold() == low else old)
        for key, old in device.native_parameters
    )
    validate_native_device(probe, pdk_id=pdk_id)
    return fields[low], token


def value_for_binding(variable, binding: str, value: float) -> float:
    """Apply the source-declared dimensionless ratio to a binding exactly once."""
    scales = dict(getattr(variable, "binding_scales", ()))
    if set(scales) - set(variable.bindings):
        raise ValueError(f"{variable.name}: binding scale has no matching binding")
    scale = float(scales.get(binding, 1.0))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"{variable.name}: invalid positive dimensionless binding scale")
    return float(format(float(value) * scale, ".12g"))


def apply_candidate_device_bindings(cell_def, optimization, values, *, pdk_id=None):
    """Apply the complete binding table atomically, preserving native semantics."""
    if optimization is None:
        raise ValueError("candidate sizing supplied without optimization intent")
    supplied = {str(key): float(value) for key, value in dict(values).items()}
    known = {variable.name for variable in optimization.variables}
    unknown = sorted(set(supplied) - known)
    if unknown:
        raise ValueError(f"candidate sizing has unknown variables: {unknown}")
    devices = {device.instance_name: device for device in cell_def.devices}
    staged = {name: deepcopy(device) for name, device in devices.items()}
    owners = {}
    for variable in optimization.variables:
        value = supplied.get(variable.name, variable.default)
        if math.isfinite(value):
            if value < variable.low and math.isclose(value, variable.low, rel_tol=1e-12, abs_tol=1e-24):
                value = variable.low
            elif value > variable.high and math.isclose(value, variable.high, rel_tol=1e-12, abs_tol=1e-24):
                value = variable.high
        if not math.isfinite(value) or not variable.low <= value <= variable.high:
            raise ValueError(
                f"candidate {variable.name}={value:g} outside "
                f"[{variable.low:g}, {variable.high:g}]"
            )
        step = getattr(variable, "step", None)
        if step is not None and not math.isclose(value / step, round(value / step), rel_tol=0, abs_tol=1e-7):
            raise ValueError(f"candidate {variable.name} is not on its declared step")
        for binding in variable.bindings:
            parts = binding.split(".")
            if not parts or parts[0] != "device":
                continue
            if len(parts) != 3:
                raise ValueError(f"invalid device sizing binding {binding!r}")
            _, instance_name, field = parts
            device = staged.get(instance_name)
            if device is None:
                raise ValueError(f"sizing binding {binding!r} names missing device {instance_name!r}")
            is_native = device.device_class == "native"
            target = (instance_name, field.casefold() if is_native else field)
            if target in owners:
                raise ValueError(f"sizing target {binding!r} has multiple bindings")
            owners[target] = variable.name
            bound_value = value_for_binding(variable, binding, value)
            if is_native:
                key, token = native_parameter_token(device, field, bound_value, pdk_id=pdk_id)
                device.native_parameters = tuple(
                    (old_key, token if old_key == key else old_value)
                    for old_key, old_value in device.native_parameters
                )
            else:
                if field not in device.sizing:
                    raise ValueError(f"sizing binding {binding!r} names missing field {field!r}")
                existing = device.sizing[field]
                if not (
                    isinstance(existing, (int, float))
                    and math.isclose(float(existing), float(bound_value), rel_tol=1e-12, abs_tol=1e-24)
                ):
                    device.sizing[field] = bound_value
    for name, staged_device in staged.items():
        devices[name].sizing = staged_device.sizing
        devices[name].native_parameters = staged_device.native_parameters


def native_bound_variables(cell_def, optimization):
    """Identify native bindings without interpreting variable-name prefixes."""
    native = {device.instance_name for device in cell_def.devices
              if device.device_class == "native"}
    return {
        variable.name for variable in optimization.variables
        if any(len(parts := binding.split(".")) == 3
               and parts[0] == "device" and parts[1] in native
               for binding in variable.bindings)
    }
