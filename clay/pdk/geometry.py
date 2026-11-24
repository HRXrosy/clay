"""Installed native geometry providers and their source-only error contract.

The validator must not generate geometry, inspect runtime artifacts or start
tools. A successful validation is support evidence only, never signoff.
"""
from dataclasses import dataclass
from typing import Any, Callable, Mapping


class GeometryInputError(ValueError):
    """The declared dimensions cannot describe a valid provider input."""


class GeometryUnsupportedError(ValueError):
    """Valid source semantics need a geometry capability not implemented yet."""


@dataclass(frozen=True)
class NativePrimitiveObservationCapability:
    version: str
    materialize: Callable[..., Any]

    def __post_init__(self):
        if self.version != 'clay.native-primitive-observation/v1' or not callable(self.materialize):
            raise TypeError('invalid native primitive observation capability')


@dataclass(frozen=True)
class NativeGeometryProvider:
    name: str
    validate_device: Callable[[Any], Mapping[str, Any]]
    materialize: Callable[..., Any]
    primitive_observation: NativePrimitiveObservationCapability | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise TypeError("native geometry provider requires an explicit name")
        if not callable(self.validate_device) or not callable(self.materialize):
            raise TypeError("native geometry provider requires validation and materialization callables")
        if self.primitive_observation is not None:
            if type(self.primitive_observation) is not NativePrimitiveObservationCapability:
                raise TypeError("invalid primitive observation capability")
            self.primitive_observation.__post_init__()
