"""Shared source checks for installed physical providers; no physical execution.

Portable device construction and historical layout providers retain their own
existing gates. Native support is resolved from the same PDK binding that the
canonical layout step executes, independently of circuit names.
"""
from dataclasses import dataclass
from collections.abc import Mapping

from clay.flow.readiness_result import ReadinessCheck
from clay.pdk.geometry import GeometryInputError, GeometryUnsupportedError, NativeGeometryProvider


def _location(device, artifact):
    location = dict(getattr(device, "source_span", None) or {})
    if artifact is not None:
        location["artifact"] = str(artifact)
    return (location,) if location else ()


@dataclass(frozen=True)
class PhysicalSupport:
    checks: tuple[ReadinessCheck, ...]
    native_provider: NativeGeometryProvider | None = None

    @property
    def ready(self):
        return all(check.status == "pass" for check in self.checks)


def check_device_support(cell, pdk_id, *, physical=False, source_artifact=None):
    """Validate source devices, optionally checking native geometry support.

This does not inspect a runtime manifest or claim that a backend is runnable.
The returned provider is an installed callable, not an admission certificate.
"""
    from clay.pdk import get_pdk
    try:
        from clay.pdk.native_spice import NativeSpiceModelUnavailable
        from clay.frontend.native_spice_import import NativeSpiceImportError, validate_native_device
    except Exception as exc:
        return PhysicalSupport((ReadinessCheck(
            "source.devices", "error", "SOURCE_DEVICE_CHECK_ERROR", str(exc),
            detail={"exception_type": type(exc).__name__}),))

    try:
        adapter = get_pdk(pdk_id)
    except KeyError as exc:
        return PhysicalSupport((ReadinessCheck(
            "source.pdk", "unsupported", "PDK_ADAPTER_UNAVAILABLE", str(exc)),))
    except Exception as exc:
        return PhysicalSupport((ReadinessCheck(
            "source.pdk", "error", "PDK_ADAPTER_ERROR", str(exc),
            detail={"exception_type": type(exc).__name__}),))

    devices = tuple(getattr(cell, "devices", ()) or ())
    native = tuple(device for device in devices if device.device_class == "native")
    checks = []
    annotations_by_device = getattr(cell, '__dict__', {}).get('_layout_annotations', {}) or {}
    targets = []
    if physical and annotations_by_device:
        from clay.layout.annotations import NativeBodyTap
        by_name = {device.instance_name: device for device in devices}
        targets = [target for target, annotations in
                   annotations_by_device.items()
                   if any(isinstance(annotation, NativeBodyTap) for annotation in annotations)]
        for target in targets:
            device = by_name.get(target)
            if device is None or device.device_class != 'native' or pdk_id != 'sky130':
                checks.append(ReadinessCheck(
                    'placement.devices', 'unsupported', 'NATIVE_BODY_TAP_UNSUPPORTED',
                    f'@native_body_tap({target}) requires a SKY130 native device',
                    _location(device, source_artifact) if device is not None else ()))
    valid_native = []
    for device in devices:
        locations = _location(device, source_artifact)
        portable_compile_errors = ()
        try:
            if device.device_class == "native":
                validate_native_device(device, pdk_id=pdk_id)
                valid_native.append(device)
            else:
                from clay.compiler import CellCompileError, _validate_primitive_device_class

                portable_compile_errors = (CellCompileError,)
                _validate_primitive_device_class(device, pdk_id=pdk_id, cell_name=cell.local_name)
        except NativeSpiceModelUnavailable as exc:
            checks.append(ReadinessCheck("source.devices", "unsupported", "NATIVE_MODEL_UNAVAILABLE", str(exc), locations))
        except NativeSpiceImportError as exc:
            checks.append(ReadinessCheck("source.devices", "fail", "SOURCE_DEVICE_INVALID", str(exc), locations))
        except Exception as exc:
            # The portable compiler is loaded only for portable devices.
            if isinstance(exc, portable_compile_errors):
                checks.append(ReadinessCheck("source.devices", "fail", "SOURCE_DEVICE_INVALID", str(exc), locations))
            else:
                checks.append(ReadinessCheck("source.devices", "error", "SOURCE_DEVICE_CHECK_ERROR", str(exc), locations,
                                             {"exception_type": type(exc).__name__}))
    if not checks:
        checks.append(ReadinessCheck("source.devices", "pass", "SOURCE_DEVICES_VERIFIED",
                                     "device declarations satisfy the selected PDK source contract"))
    if not physical or not native:
        return PhysicalSupport(tuple(checks))
    if len(native) != len(devices):
        checks.append(ReadinessCheck("placement.devices", "unsupported", "NATIVE_MIXED_TOPOLOGY_UNSUPPORTED",
                                     "native geometry requires a complete native cell; mixed materialization is unavailable"))
    provider = getattr(adapter, "native_geometry_provider", None)
    if provider is None:
        checks.append(ReadinessCheck("placement.devices", "unsupported", "NATIVE_GEOMETRY_PROVIDER_UNAVAILABLE",
                                     f"PDK {pdk_id!r} has no installed native geometry provider"))
        return PhysicalSupport(tuple(checks))
    try:
        if not isinstance(provider, NativeGeometryProvider):
            raise TypeError("PDK native geometry binding must be a NativeGeometryProvider")
        provider.__post_init__()
    except Exception as exc:
        checks.append(ReadinessCheck("placement.devices", "error", "NATIVE_GEOMETRY_PROVIDER_ERROR", str(exc),
                                     detail={"exception_type": type(exc).__name__}))
        return PhysicalSupport(tuple(checks))
    cell_specs = None
    if targets and pdk_id == 'sky130' and len(valid_native) == len(devices):
        try:
            from clay.pdk.sky130_native_geometry import native_cell_geometry_specs
            cell_specs = native_cell_geometry_specs(cell, validate_device=provider.validate_device)
        except GeometryUnsupportedError as exc:
            checks.append(ReadinessCheck('placement.devices', 'unsupported',
                                         'NATIVE_BODY_TAP_UNSUPPORTED', str(exc)))
        except GeometryInputError as exc:
            checks.append(ReadinessCheck('placement.devices', 'fail',
                                         'NATIVE_BODY_TAP_INVALID', str(exc)))
        except Exception as exc:
            checks.append(ReadinessCheck('placement.devices', 'error',
                                         'NATIVE_GEOMETRY_PROVIDER_ERROR', str(exc),
                                         detail={'exception_type': type(exc).__name__}))
    for device in valid_native:
        locations = _location(device, source_artifact)
        try:
            spec = cell_specs[device.instance_name] if cell_specs is not None else provider.validate_device(device)
            if not isinstance(spec, Mapping) or not spec:
                raise TypeError("native geometry validator returned no source specification")
        except GeometryUnsupportedError as exc:
            checks.append(ReadinessCheck("placement.devices", "unsupported", "NATIVE_GEOMETRY_UNSUPPORTED", str(exc), locations))
        except (GeometryInputError, NativeSpiceImportError) as exc:
            checks.append(ReadinessCheck("placement.devices", "fail", "NATIVE_GEOMETRY_INPUT_INVALID", str(exc), locations))
        except Exception as exc:
            checks.append(ReadinessCheck("placement.devices", "error", "NATIVE_GEOMETRY_PROVIDER_ERROR", str(exc), locations,
                                         {"exception_type": type(exc).__name__}))
    if all(check.status == "pass" for check in checks):
        checks.append(ReadinessCheck("placement.devices", "pass", "NATIVE_GEOMETRY_SUPPORTED",
                                     "native source parameters are supported by the installed geometry provider",
                                     detail={"provider": provider.name, "scope": "source_support_only",
                                             "physical_admission": False, "device_count": len(native)}))
    return PhysicalSupport(tuple(checks), provider)
