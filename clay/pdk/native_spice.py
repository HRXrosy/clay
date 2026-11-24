"""PDK-owned contracts for lossless native subcircuit declarations.

These contracts describe simulation terminals and parameter units. They do not
grant a device geometry builder or a physical signoff capability.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class NativeSpiceFamily(str, Enum):
    MOS = "mos"
    VARACTOR = "varactor"
    MIM = "mim"


class NativeSpiceModelUnavailable(ValueError):
    """No installed model contract exists for the requested native model."""


@dataclass(frozen=True, slots=True)
class NativeSpiceModelContract:
    model: str
    family: NativeSpiceFamily
    # Ordered pairs: actual model terminal, explicit Clay pin role.
    terminals: tuple[tuple[str, str], ...]
    parameters: tuple[str, ...]
    required_parameters: tuple[str, ...]
    positive_parameters: tuple[str, ...]
    integer_parameters: tuple[str, ...]
    expression_parameters: tuple[str, ...]
    # Native numeric parameter -> SI length; no source-dependent unit guesses.
    length_scales: tuple[tuple[str, float], ...]
    provenance: str
    # Width is a total width split across nf for the SKY130 MOS subcircuits.
    total_width_parameter: str | None = None
    finger_parameter: str | None = None

    def __post_init__(self):
        if type(self.model) is not str or not self.model or type(self.family) is not NativeSpiceFamily:
            raise ValueError("native model requires an explicit model and typed family")
        if type(self.terminals) is not tuple or not self.terminals:
            raise ValueError("native model requires ordered terminal roles")
        for index in (0, 1):
            names = [pair[index] for pair in self.terminals]
            if len(names) != len(set(names)) or any(type(name) is not str or not name for name in names):
                raise ValueError("native model terminal roles must be unique")
        for values in (self.parameters, self.required_parameters, self.positive_parameters,
                       self.integer_parameters, self.expression_parameters):
            if (type(values) is not tuple or len(values) != len(set(values))
                    or any(type(name) is not str or name != name.casefold() for name in values)):
                raise ValueError("native model parameter names must be unique lowercase tuples")
        declared = set(self.parameters)
        if any(not set(values) <= declared for values in
               (self.required_parameters, self.positive_parameters,
                self.integer_parameters, self.expression_parameters)):
            raise ValueError("native model parameter contract references unknown parameters")
        if any(key not in declared or not math.isfinite(scale) or scale <= 0
               for key, scale in self.length_scales):
            raise ValueError("native model length scales must be explicit positive SI factors")
        if not self.provenance:
            raise ValueError("native model contract requires PDK provenance")
        if self.total_width_parameter is not None and self.total_width_parameter not in declared:
            raise ValueError("native model total-width parameter is undeclared")
        if self.finger_parameter is not None and self.finger_parameter not in declared:
            raise ValueError("native model finger parameter is undeclared")

    def canonical_data(self):
        return {"model": self.model, "family": self.family.value,
                "terminals": [list(pair) for pair in self.terminals],
                "parameters": list(self.parameters), "required_parameters": list(self.required_parameters),
                "positive_parameters": list(self.positive_parameters),
                "integer_parameters": list(self.integer_parameters),
                "expression_parameters": list(self.expression_parameters),
                "length_scales": [list(pair) for pair in self.length_scales],
                "total_width_parameter": self.total_width_parameter,
                "finger_parameter": self.finger_parameter, "provenance": self.provenance}


def native_spice_model_contract(pdk_id: str, model: str) -> NativeSpiceModelContract:
    from clay.pdk import get_pdk

    adapter = get_pdk(pdk_id)
    matches = [contract for name, contract in adapter.native_spice_models.items()
               if name.casefold() == str(model).casefold()]
    if not matches:
        raise NativeSpiceModelUnavailable(
            f"PDK {pdk_id!r} has no unique native SPICE model contract for {model!r}"
        )
    if len(matches) != 1 or type(matches[0]) is not NativeSpiceModelContract:
        raise ValueError(f"PDK {pdk_id!r} has no unique native SPICE model contract for {model!r}")
    result = matches[0]
    result.__post_init__()
    if result.model.casefold() != model.casefold():
        raise ValueError("PDK native model registry key differs from its contract")
    return result
