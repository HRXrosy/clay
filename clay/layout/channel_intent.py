"""Source-bound, ordered horizontal channels between two declared device rows.

This is a hard construction declaration, separate from the existing soft
``@trunk`` preference and the placement-only ``@channel`` corridor. Dimensions
on this craft surface are micrometres; the canonical representation is exact
integer nanometres. No circuit identity, inferred net role, or reference
coordinate enters this type.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re


def _names(value, field, *, minimum=0):
    if type(value) is not tuple or len(value) < minimum:
        raise ValueError(f"channel {field} must be a tuple of at least {minimum} names")
    if any(type(n) is not str or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n) is None
           for n in value):
        raise ValueError(f"channel {field} contains a noncanonical name")
    if len(set(value)) != len(value):
        raise ValueError(f"channel {field} contains duplicate names")
    return value


def _layer(value):
    if type(value) is not str or re.fullmatch(r"M[1-9][0-9]*", value) is None:
        raise ValueError("channel layer must be a canonical M<number> name")
    return value


def _nm(value, field):
    if type(value) not in (int, float, str, Decimal):
        raise ValueError(f"channel {field} must be a positive finite micrometre value")
    try:
        number = Decimal(str(value)) * 1000
    except InvalidOperation as exc:
        raise ValueError(f"invalid channel {field}") from exc
    if not number.is_finite() or number <= 0 or number != number.to_integral_value():
        raise ValueError(f"channel {field} must be positive and exactly representable in nm")
    return int(number)


@dataclass(frozen=True)
class ChannelRouteIntent:
    nets: tuple[str, ...]
    between: tuple[str, str]
    trunk_layer: str = "M2"
    branch_layer: str = "M3"
    width_um: float = 0.4
    pitch_um: float = 1.3
    centered_nets: tuple[str, ...] = ()
    branch_layers: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        _names(self.nets, "nets", minimum=1)
        _names(self.between, "between", minimum=2)
        if len(self.between) != 2:
            raise ValueError("channel between requires exactly two distinct placement groups")
        _layer(self.trunk_layer)
        _layer(self.branch_layer)
        if self.trunk_layer == self.branch_layer:
            raise ValueError("channel trunk and branch roles require distinct layers")
        width = _nm(self.width_um, "width")
        pitch = _nm(self.pitch_um, "pitch")
        if pitch <= width:
            raise ValueError("channel pitch must exceed wire width")
        object.__setattr__(self, "width_um", width / 1000)
        object.__setattr__(self, "pitch_um", pitch / 1000)
        _names(self.centered_nets, "centered_nets")
        if not set(self.centered_nets).issubset(self.nets):
            raise ValueError("centered channel nets must be members of that channel")
        if type(self.branch_layers) is not tuple:
            raise ValueError("channel branch overrides must be a tuple")
        overrides = []
        seen = set()
        for pair in self.branch_layers:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError("channel branch override must contain a net and layer")
            net, layer = pair
            if net not in self.nets or net in seen:
                raise ValueError("channel branch override is absent from channel or duplicated")
            seen.add(net)
            _layer(layer)
            if layer == self.trunk_layer:
                raise ValueError("channel branch override cannot equal trunk layer")
            overrides.append(pair)
        object.__setattr__(self, "branch_layers", tuple(sorted(overrides)))

    @property
    def width_nm(self):
        return _nm(self.width_um, "width")

    @property
    def pitch_nm(self):
        return _nm(self.pitch_um, "pitch")

    def canonical_data(self):
        return {"nets": list(self.nets), "between": list(self.between),
                "trunk_layer": self.trunk_layer, "branch_layer": self.branch_layer,
                "width_nm": self.width_nm, "pitch_nm": self.pitch_nm,
                "centered_nets": sorted(self.centered_nets),
                "branch_layers": [list(pair) for pair in self.branch_layers]}

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.canonical_data(), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()

    def assert_authoritative(self):
        if replace(self) != self:
            raise ValueError("noncanonical channel declaration")


def canonical_channel_routes(value):
    if type(value) is not tuple:
        raise ValueError("channel route inventory must be a tuple")
    nets = set()
    corridors = set()
    for channel in value:
        if type(channel) is not ChannelRouteIntent:
            raise ValueError("channel route inventory contains an untyped declaration")
        channel.assert_authoritative()
        if nets.intersection(channel.nets):
            raise ValueError("a net cannot be assigned to multiple ordered channels")
        if channel.between in corridors:
            raise ValueError("one corridor requires one complete ordered-channel declaration")
        nets.update(channel.nets)
        corridors.add(channel.between)
    return value


def channel_routes_from_declarations(declarations, place_groups):
    channels = []
    branches = {}
    allowed = {"nets", "between", "trunk_layer", "branch_layer", "width", "pitch", "centered_nets"}
    groups = {group.group: group for group in place_groups}
    for decl in declarations:
        attrs = decl.attributes or {}
        if decl.kind == "channel_route":
            if set(attrs) - allowed or not {"nets", "between", "width", "pitch"}.issubset(attrs):
                raise ValueError("channel_route requires nets, between, width, pitch and known role fields")
            channel = ChannelRouteIntent(
                nets=attrs["nets"], between=attrs["between"],
                trunk_layer=attrs.get("trunk_layer", "M2"),
                branch_layer=attrs.get("branch_layer", "M3"),
                width_um=attrs["width"], pitch_um=attrs["pitch"],
                centered_nets=attrs.get("centered_nets", ()),
            )
            if any(name not in groups for name in channel.between):
                raise ValueError("channel between references an undeclared placement group")
            lower, upper = (groups[name] for name in channel.between)
            if (lower.row_min is None or upper.row_min is None
                    or lower.row_min != lower.row_max or upper.row_min != upper.row_max
                    or lower.row_min >= upper.row_min):
                raise ValueError("channel between must bind distinct exact rows in lower-to-upper order")
            if set(lower.devices).intersection(upper.devices):
                raise ValueError("channel boundary rows cannot share a device")
            channels.append(channel)
        elif decl.kind == "channel_branch":
            if len(decl.nets) != 1 or set(attrs) != {"layer"}:
                raise ValueError("channel_branch requires one net and one layer")
            net, = decl.nets
            if net in branches:
                raise ValueError("duplicate channel_branch declaration")
            branches[net] = _layer(attrs["layer"])
    channel_nets = {net for channel in channels for net in channel.nets}
    if set(branches) - channel_nets:
        raise ValueError("channel_branch names a net without a channel_route")
    return canonical_channel_routes(tuple(
        replace(channel, branch_layers=tuple((net, branches[net])
                for net in channel.nets if net in branches))
        for channel in channels))
