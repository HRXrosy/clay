"""Routing-layer pitch ruleset for sky130 and gf180mcu PDKs.

Per-layer min_width / min_spacing / pitch constants with lookup helpers.
Pitch is defined as min_width + min_spacing (canonical track pitch).

Values are taken from the public skywater-pdk and gf180mcu PDK design rule
manuals. Units are nanometres (integers) to avoid floating point drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class LayerPitchRule:
    """Routing-layer pitch rule for a single layer.

    Attributes:
        layer_name: Logical layer name (e.g. "M1", "POLY", "DIFF", "NWELL").
        min_width_nm: Minimum drawn width of a shape on this layer (nm).
        min_spacing_nm: Minimum edge-to-edge spacing between shapes (nm).
        pitch_nm: Canonical routing track pitch = min_width + min_spacing (nm).
    """

    layer_name: str
    min_width_nm: int
    min_spacing_nm: int
    pitch_nm: int


def _rule(name: str, w: int, s: int) -> LayerPitchRule:
    return LayerPitchRule(
        layer_name=name,
        min_width_nm=w,
        min_spacing_nm=s,
        pitch_nm=w + s,
    )


# Sky130 rule set (canonical values from skywater PDK docs).
SKY130_LAYER_PITCH: Dict[str, LayerPitchRule] = {
    "POLY": _rule("POLY", 150, 210),
    "DIFF": _rule("DIFF", 290, 270),
    "NWELL": _rule("NWELL", 840, 1270),
    "M1": _rule("M1", 140, 140),
    "M2": _rule("M2", 140, 140),
    "M3": _rule("M3", 300, 300),
}


# Gf180mcu rule set (coarser 180 nm process).
GF180MCU_LAYER_PITCH: Dict[str, LayerPitchRule] = {
    "POLY": _rule("POLY", 180, 240),
    "DIFF": _rule("DIFF", 300, 280),
    "NWELL": _rule("NWELL", 860, 1400),
    "M1": _rule("M1", 230, 230),
    "M2": _rule("M2", 280, 280),
    "M3": _rule("M3", 280, 280),
}


_PDK_REGISTRY: Dict[str, Dict[str, LayerPitchRule]] = {
    "sky130": SKY130_LAYER_PITCH,
    "gf180mcu": GF180MCU_LAYER_PITCH,
}


def _resolve_pdk(pdk_name: str) -> Dict[str, LayerPitchRule]:
    key = pdk_name.lower()
    if key not in _PDK_REGISTRY:
        raise KeyError(
            f"Unknown PDK {pdk_name!r}; known PDKs: {sorted(_PDK_REGISTRY)}"
        )
    return _PDK_REGISTRY[key]


def get_layer_pitch_rule(pdk_name: str, layer_name: str) -> LayerPitchRule:
    """Return the LayerPitchRule for (pdk_name, layer_name).

    Args:
        pdk_name: PDK identifier; one of {"sky130", "gf180mcu"} (case-insensitive).
        layer_name: Logical layer name (case-insensitive, e.g. "m1" -> "M1").

    Raises:
        KeyError: If the PDK or the layer is unknown for that PDK.
    """
    rules = _resolve_pdk(pdk_name)
    key = layer_name.upper()
    if key not in rules:
        raise KeyError(
            f"Unknown layer {layer_name!r} for PDK {pdk_name!r}; "
            f"known layers: {sorted(rules)}"
        )
    return rules[key]


def list_layers_for_pdk(pdk_name: str) -> List[str]:
    """Return sorted layer names available for the given PDK."""
    rules = _resolve_pdk(pdk_name)
    return sorted(rules.keys())


def compute_track_count(
    pdk_name: str, layer_name: str, channel_width_um: float,
) -> int:
    """Return how many routing tracks of the given layer fit in a channel.

    The track count is floor(channel_width_nm / pitch_nm), where
    channel_width_nm = channel_width_um * 1000.

    Args:
        pdk_name: PDK identifier (case-insensitive).
        layer_name: Layer name (case-insensitive).
        channel_width_um: Channel width in micrometres. Negative values yield 0.

    Returns:
        Non-negative integer count of routing tracks.
    """
    rule = get_layer_pitch_rule(pdk_name, layer_name)
    if channel_width_um <= 0:
        return 0
    channel_width_nm = channel_width_um * 1000.0
    return int(channel_width_nm // rule.pitch_nm)
