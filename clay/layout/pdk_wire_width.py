"""R140 (BL-140A) — PDK-aware routing wire width.

R139's real gf180mcu DRC signoff found 30 ``M1.1`` violations: the routing
flows (:func:`clay.layout.sane_routed_lowering.route_annotated_cell_sane` and
:func:`clay.layout.reroute_contain.route_contained`) default
``wire_width_nm=140``, but the gf180mcu foundry deck requires a minimum M1
width of 230 nm (rule ``M1.1``). A 140 nm M1 wire is therefore narrower than
the minimum and every routed segment fired ``M1.1``.

This module supplies the missing piece: a single helper that returns the
*correct* routing wire width per PDK, read from the real per-PDK rule tables —
never a fabricated number:

* gf180mcu M1 minimum width = 230 nm. Sourced from
  :data:`clay.layout.gf180_rules` rule ``M1.1`` (deck line 2394, 0.23 µm),
  via ``gf180_rules.min_width_nm("M1")``.
* sky130 M1 minimum width = 140 nm. Sourced from
  :data:`clay.layout.layer_pitch.SKY130_LAYER_PITCH` ``M1`` ``min_width_nm``.

The committed routing flows are NOT modified (they keep their additive,
byte-stable default). Instead callers pass the PDK-correct width explicitly::

    from clay.layout.pdk_wire_width import wire_width_nm_for_pdk
    from clay.layout.sane_routed_lowering import route_annotated_cell_sane

    width = wire_width_nm_for_pdk(pdk_layers)          # 230 for gf180, 140 sky130
    result = route_annotated_cell_sane(
        cell, pdk_layers, fin_pitch_nm, sizing, wire_width_nm=width,
    )

PDK detection mirrors the established pattern in
:func:`clay.layout.backend.min_poly_space_nm_for_pdk_layers` and
:func:`clay.layout.layout_drc._pdk_name` (recognise the committed
``Sky130LayerMap`` / ``Gf180mcuLayerMap`` by class name). Every committed module
imported here is read-only.
"""
from __future__ import annotations

from .backend import LayerMap
from . import gf180_rules
from .layer_pitch import SKY130_LAYER_PITCH


# Logical routing layer this helper sizes. Routing is M1-only in the current
# flows (see clay/layout/routing.py), so the wire width is the M1 minimum width.
_ROUTING_LAYER = "M1"


def _pdk_name(pdk_layers: LayerMap) -> str:
    """Classify a PDK LayerMap as 'sky130' / 'gf180mcu' by its class name.

    Mirrors ``backend.min_poly_space_nm_for_pdk_layers`` and
    ``layout_drc._pdk_name`` so the three places that need PDK detection agree.

    Raises:
        ValueError: if the LayerMap type is neither sky130 nor gf180mcu.
    """
    name = type(pdk_layers).__name__.lower()
    if "sky130" in name:
        return "sky130"
    if "gf180" in name:
        return "gf180mcu"
    raise ValueError(
        f"unknown PDK LayerMap type {type(pdk_layers).__name__!r}; "
        f"pdk_wire_width supports sky130 / gf180mcu — pass wire_width_nm "
        f"explicitly to the routing flow for other PDKs"
    )


def gf180_m1_min_width_nm() -> int:
    """gf180mcu M1 minimum width in nm, read from the real DRC rule table.

    Returns ``gf180_rules.min_width_nm("M1")`` — rule ``M1.1`` (0.23 µm,
    deck line 2394) == 230 nm. No value is hard-coded here; it comes straight
    from the committed extracted-rules module.
    """
    return gf180_rules.min_width_nm(_ROUTING_LAYER)


def sky130_m1_min_width_nm() -> int:
    """sky130 M1 minimum width in nm, read from the real layer-pitch table.

    Returns ``SKY130_LAYER_PITCH["M1"].min_width_nm`` == 140 nm. No value is
    hard-coded here; it comes from the committed layer-pitch module.
    """
    return SKY130_LAYER_PITCH[_ROUTING_LAYER].min_width_nm


def wire_width_nm_for_pdk(pdk_layers: LayerMap) -> int:
    """Return the PDK-correct M1 routing wire width in nm.

    The returned width is the PDK's real M1 minimum width, so a wire routed at
    this width passes the M1 min-width rule (``M1.1`` for gf180mcu). Pass it as
    the ``wire_width_nm`` argument of the committed routing flows
    (``route_annotated_cell_sane`` / ``route_contained``) which otherwise
    default to 140 nm — correct for sky130, but a violation for gf180mcu.

    Args:
        pdk_layers: a PDK LayerMap (``Sky130LayerMap`` / ``Gf180mcuLayerMap``).

    Returns:
        230 for gf180mcu (from gf180_rules ``M1.1``), 140 for sky130 (from
        ``SKY130_LAYER_PITCH``).

    Raises:
        ValueError: if the PDK LayerMap type is unrecognised; the caller should
            then pass ``wire_width_nm`` to the routing flow explicitly.
    """
    pdk = _pdk_name(pdk_layers)
    if pdk == "gf180mcu":
        return gf180_m1_min_width_nm()
    # pdk == "sky130" (the only other value _pdk_name can return)
    return sky130_m1_min_width_nm()


__all__ = [
    "wire_width_nm_for_pdk",
    "gf180_m1_min_width_nm",
    "sky130_m1_min_width_nm",
]
