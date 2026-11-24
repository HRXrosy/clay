"""Stable physical terminal identity and its separate exact access facts.

Bindings are immutable data, not source or execution authority. Their complete
source ledger must be checked by the retained row/artwork consumer. Geometry,
PDK extraction and placement certificates do not belong in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math


_MAX_SOURCE_REFS = 262_144
_MAX_TEXT = 4_096
_MAX_NM = 1_000_000_000


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("physical terminal Pin " + message)


def _text(value: object, field: str, *, canonical: bool = True) -> None:
    _require(type(value) is str and 0 < len(value) <= _MAX_TEXT,
             field + " must be a bounded nonempty string")
    if canonical:
        _require(value == value.strip(), field + " must be canonical")


@dataclass(frozen=True, slots=True)
class PhysicalTerminalPinBinding:
    """A stamp-qualified component and every original terminal it represents."""

    stamp_id: str
    component_id: str
    source_refs: tuple[tuple[str, str, str], ...]

    def __post_init__(self) -> None:
        _text(self.stamp_id, "binding stamp id")
        _text(self.component_id, "binding component id")
        _require(type(self.source_refs) is tuple and 0 < len(self.source_refs) <= _MAX_SOURCE_REFS,
                 "binding source refs must be a nonempty bounded exact tuple")
        for source in self.source_refs:
            _require(type(source) is tuple and len(source) == 3,
                     "binding source ref must contain source id, member and role")
            for value, field in zip(source, ("source id", "source member", "source role")):
                _text(value, "binding " + field)
        _require(self.source_refs == tuple(sorted(self.source_refs)),
                 "binding source refs must be canonical")
        _require(len({source[0] for source in self.source_refs}) == len(self.source_refs),
                 "binding source ids must be unique")

    def canonical_tuple(self) -> tuple:
        self.__post_init__()
        return self.stamp_id, self.component_id, self.source_refs

    def payload(self) -> dict:
        self.__post_init__()
        return {"schema": "clay.physical-terminal-pin-binding/v1",
                "stamp_id": self.stamp_id, "component_id": self.component_id,
                "source_refs": self.source_refs}


def _binding(pin: object) -> PhysicalTerminalPinBinding | None:
    value = getattr(pin, "terminal_binding", None)
    if value is None:
        return None
    _require(type(value) is PhysicalTerminalPinBinding,
             "binding must be an exact typed PhysicalTerminalPinBinding")
    value.__post_init__()
    alias = (getattr(pin, "device", None), getattr(pin, "terminal", None))
    _require(all(type(item) is str for item in alias),
             "binding display alias must contain exact strings")
    _require(alias == value.source_refs[0][1:],
             "display alias must equal the canonical binding source alias")
    return value


def _qualified_provenance(binding: PhysicalTerminalPinBinding) -> str:
    # Explicit JSON tuple framing avoids collisions between unrestricted source
    # names. Coordinates and source aliases are deliberately not endpoint IDs.
    encoded = json.dumps((binding.stamp_id, binding.component_id),
                         ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return "pin-component:" + hashlib.sha256(encoded).hexdigest()


def _legacy_alias(pin: object) -> tuple[str, str]:
    return (str(getattr(pin, "device", "") or "<external>"),
            str(getattr(pin, "terminal", "") or "terminal"))


def pin_provenance(pin: object) -> str:
    """Name one physical endpoint, preserving legacy terminal strings exactly."""
    binding = _binding(pin)
    if binding is not None:
        return _qualified_provenance(binding)
    device, terminal = _legacy_alias(pin)
    return f"pin:{device}:{terminal}"


def pin_source_aliases(pin: object) -> tuple[tuple[str, str], ...]:
    binding = _binding(pin)
    if binding is None:
        return (_legacy_alias(pin),)
    return tuple(sorted({(member, role) for _source, member, role in binding.source_refs}))


def pin_matches_source(pin: object, member: str, role: str | None = None) -> bool:
    """Select by logical source membership without duplicating a shared endpoint."""
    return any(alias_member == member and (role is None or alias_role == role)
               for alias_member, alias_role in pin_source_aliases(pin))


def pin_binding_payload(pin: object) -> dict | None:
    binding = _binding(pin)
    return None if binding is None else binding.payload()


def _layer(value: object) -> tuple[int, int]:
    _require(type(value) is tuple and len(value) == 2
             and all(type(item) is int and 0 <= item <= 65_535 for item in value),
             "fact layer must be an exact resolved integer pair")
    return value


def _point(pin: object) -> tuple[int, int]:
    raw_um = getattr(pin, "xy_um", None)
    _require(type(raw_um) in (tuple, list) and len(raw_um) == 2
             and all(type(value) in (int, float) and math.isfinite(value) for value in raw_um),
             "fact coordinates require a finite micron pair")
    rounded = tuple(int(round(value * 1000.0)) for value in raw_um)
    point = getattr(pin, "xy_nm", None)
    if point is None:
        point = rounded
    _require(type(point) is tuple and len(point) == 2
             and all(type(value) is int and abs(value) <= _MAX_NM for value in point),
             "fact coordinates must be exact bounded integer nanometres")
    _require(point == rounded, "fact nanometre/micron coordinates differ")
    return point


def _base_fact(value: object, *, legacy_identity: bool = False) -> tuple:
    _require(type(value) in (tuple, list) and len(value) == 7,
             "fact requires seven canonical access fields")
    base = tuple(value)
    for item, name in zip(base[:3], ("net", "device", "terminal")):
        if legacy_identity:
            # Old world facts may carry a str subclass for an unresolved
            # identity. Preserve it for the downstream UNKNOWN result, without
            # executing its conversion, formatting or length overrides.
            _require(isinstance(item, str) and 0 < str.__len__(item) <= _MAX_TEXT,
                     "fact " + name + " must be a bounded nonempty string")
        else:
            _text(item, "fact " + name, canonical=False)
    _require(all(type(item) is int and abs(item) <= _MAX_NM for item in base[3:5]),
             "fact coordinate must be an exact bounded integer")
    _layer(base[5:7])
    return base


def pin_fact(pin: object, *, layer_key: tuple[int, int] | None = None) -> tuple:
    """Return the old seven fields, with a complete eighth binding when typed.

    Legacy named layers must be resolved by the caller using its actual PDK.
    Canonical duck inputs remain supported; source authority is checked upstream.
    """
    binding = _binding(pin)
    layer = _layer(getattr(pin, "layer", None) if layer_key is None else layer_key)
    original_layer = getattr(pin, "layer", None)
    if layer_key is not None and type(original_layer) is tuple:
        _require(_layer(original_layer) == layer, "fact resolved layer differs from the Pin")
    base = _base_fact((getattr(pin, "net", None), getattr(pin, "device", None),
                       getattr(pin, "terminal", None), *_point(pin), *layer))
    return base if binding is None else (*base, binding.canonical_tuple())


def unpack_pin_fact(fact: object) -> tuple[tuple, PhysicalTerminalPinBinding | None]:
    """Validate a canonical access fact without reconstructing a source Pin."""
    _require(type(fact) in (tuple, list) and len(fact) in (7, 8),
             "fact must contain seven legacy or eight typed fields")
    base = _base_fact(fact[:7], legacy_identity=len(fact) == 7)
    if len(fact) == 7:
        return base, None
    raw = fact[7]
    _require(type(raw) is tuple and len(raw) == 3,
             "fact binding must contain its complete canonical tuple")
    binding = PhysicalTerminalPinBinding(*raw)
    _require(base[1:3] == binding.source_refs[0][1:],
             "fact display alias differs from the canonical binding source alias")
    return base, binding


def pin_fact_provenance(fact: object) -> str:
    base, binding = unpack_pin_fact(fact)
    _require(all(type(value) is str for value in base[1:3]),
             "fact provenance requires exact string identity")
    return f"pin:{base[1]}:{base[2]}" if binding is None else _qualified_provenance(binding)


__all__ = ["PhysicalTerminalPinBinding", "pin_provenance", "pin_source_aliases",
           "pin_matches_source", "pin_binding_payload", "pin_fact", "unpack_pin_fact",
           "pin_fact_provenance"]
