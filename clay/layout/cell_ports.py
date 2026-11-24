"""Declarative cell port definitions.

Each predefined ``CellPortSet`` describes the named external terminals of a
layout cell template (diff pair, current mirror, 5T OTA, folded cascode,
telescopic cascode, two-stage Miller, cascoded current mirror, bandgap
reference). Ports carry metadata about direction (input/output/power/ground)
and the GDS layer/datatype on which the pin label lives (sky130 M1 = 67/20).

The module intentionally stays minimal and pure-Python: it only declares
data and provides two lookup helpers (``get_ports_for`` and ``list_ports``).
Layout backends, DRC, and netlist exporters consume these definitions as
the single source of truth for cell terminals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# sky130 M1 pin layer/datatype defaults.
_M1_LAYER = 67
_M1_DATATYPE = 20

# Allowed direction values. Keep frozen-set so callers can validate quickly.
VALID_DIRECTIONS: frozenset = frozenset({"input", "output", "power", "ground"})


@dataclass(frozen=True)
class CellPort:
    """A single named external terminal on a cell template."""

    name: str
    direction: str
    layer: int
    datatype: int

    def __post_init__(self) -> None:  # pragma: no cover - trivial guard
        if self.direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"CellPort.direction must be one of {sorted(VALID_DIRECTIONS)}, "
                f"got {self.direction!r}"
            )
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("CellPort.name must be a non-empty string")
        if not isinstance(self.layer, int) or self.layer < 0:
            raise ValueError("CellPort.layer must be a non-negative int")
        if not isinstance(self.datatype, int) or self.datatype < 0:
            raise ValueError("CellPort.datatype must be a non-negative int")


@dataclass(frozen=True)
class CellPortSet:
    """The full set of ports for one cell template."""

    template_name: str
    ports: Tuple[CellPort, ...] = field(default_factory=tuple)

    def __init__(self, template_name: str, ports):
        # Coerce list input into an immutable tuple while preserving the
        # public API documented as ``List[CellPort]``.
        object.__setattr__(self, "template_name", template_name)
        object.__setattr__(self, "ports", tuple(ports))

    def names(self) -> List[str]:
        return [p.name for p in self.ports]


# ---------------------------------------------------------------------------
# Predefined port sets
# ---------------------------------------------------------------------------

DIFF_PAIR_PORTS = CellPortSet(
    "diff_pair",
    [
        CellPort("VINP", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("VINN", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("VOUTP", "output", _M1_LAYER, _M1_DATATYPE),
        CellPort("VOUTN", "output", _M1_LAYER, _M1_DATATYPE),
        CellPort("VTAIL", "input", _M1_LAYER, _M1_DATATYPE),
    ],
)

CURRENT_MIRROR_PORTS = CellPortSet(
    "current_mirror",
    [
        CellPort("IREF", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("IOUT", "output", _M1_LAYER, _M1_DATATYPE),
        CellPort("VSS", "ground", _M1_LAYER, _M1_DATATYPE),
    ],
)

_5T_OTA_PORTS = CellPortSet(
    "5t_ota",
    [
        CellPort("VINP", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("VINN", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("VOUT", "output", _M1_LAYER, _M1_DATATYPE),
        CellPort("VDD", "power", _M1_LAYER, _M1_DATATYPE),
        CellPort("VSS", "ground", _M1_LAYER, _M1_DATATYPE),
    ],
)

FOLDED_CASCODE_PORTS = CellPortSet(
    "folded_cascode",
    [
        CellPort("VINP", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("VINN", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("VOUT", "output", _M1_LAYER, _M1_DATATYPE),
        CellPort("VBP", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("VBN", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("VDD", "power", _M1_LAYER, _M1_DATATYPE),
        CellPort("VSS", "ground", _M1_LAYER, _M1_DATATYPE),
    ],
)

TELESCOPIC_CASCODE_PORTS = CellPortSet(
    "telescopic_cascode",
    [
        CellPort("VINP", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("VINN", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("VOUTP", "output", _M1_LAYER, _M1_DATATYPE),
        CellPort("VOUTN", "output", _M1_LAYER, _M1_DATATYPE),
        CellPort("VBP", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("VBN", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("VTAIL", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("VDD", "power", _M1_LAYER, _M1_DATATYPE),
        CellPort("VSS", "ground", _M1_LAYER, _M1_DATATYPE),
    ],
)

TWO_STAGE_MILLER_PORTS = CellPortSet(
    "two_stage_miller",
    [
        CellPort("VINP", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("VINN", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("VOUT", "output", _M1_LAYER, _M1_DATATYPE),
        CellPort("VBIAS", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("VDD", "power", _M1_LAYER, _M1_DATATYPE),
        CellPort("VSS", "ground", _M1_LAYER, _M1_DATATYPE),
    ],
)

CASCODED_CURRENT_MIRROR_PORTS = CellPortSet(
    "cascoded_current_mirror",
    [
        CellPort("IREF", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("IOUT", "output", _M1_LAYER, _M1_DATATYPE),
        CellPort("VCASC", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("VSS", "ground", _M1_LAYER, _M1_DATATYPE),
    ],
)

BANDGAP_REFERENCE_PORTS = CellPortSet(
    "bandgap_reference",
    [
        CellPort("VREF", "output", _M1_LAYER, _M1_DATATYPE),
        CellPort("VSTART", "input", _M1_LAYER, _M1_DATATYPE),
        CellPort("VDD", "power", _M1_LAYER, _M1_DATATYPE),
        CellPort("VSS", "ground", _M1_LAYER, _M1_DATATYPE),
    ],
)


_REGISTRY: Dict[str, CellPortSet] = {
    s.template_name: s
    for s in (
        DIFF_PAIR_PORTS,
        CURRENT_MIRROR_PORTS,
        _5T_OTA_PORTS,
        FOLDED_CASCODE_PORTS,
        TELESCOPIC_CASCODE_PORTS,
        TWO_STAGE_MILLER_PORTS,
        CASCODED_CURRENT_MIRROR_PORTS,
        BANDGAP_REFERENCE_PORTS,
    )
}


def get_ports_for(template_name: str) -> CellPortSet:
    """Return the ``CellPortSet`` for ``template_name``.

    Raises ``KeyError`` if the template is not registered.
    """
    try:
        return _REGISTRY[template_name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(
            f"Unknown cell template {template_name!r}; known templates: {known}"
        ) from exc


def list_ports(
    template_name: str,
    direction: Optional[str] = None,
) -> List[CellPort]:
    """Return the ports of ``template_name``, optionally filtered by direction."""
    port_set = get_ports_for(template_name)
    if direction is None:
        return list(port_set.ports)
    if direction not in VALID_DIRECTIONS:
        raise ValueError(
            f"direction filter must be one of {sorted(VALID_DIRECTIONS)} or None, "
            f"got {direction!r}"
        )
    return [p for p in port_set.ports if p.direction == direction]


def list_templates() -> List[str]:
    """Return the sorted list of registered template names."""
    return sorted(_REGISTRY)


__all__ = [
    "CellPort",
    "CellPortSet",
    "VALID_DIRECTIONS",
    "DIFF_PAIR_PORTS",
    "CURRENT_MIRROR_PORTS",
    "_5T_OTA_PORTS",
    "FOLDED_CASCODE_PORTS",
    "TELESCOPIC_CASCODE_PORTS",
    "TWO_STAGE_MILLER_PORTS",
    "CASCODED_CURRENT_MIRROR_PORTS",
    "BANDGAP_REFERENCE_PORTS",
    "get_ports_for",
    "list_ports",
    "list_templates",
]
