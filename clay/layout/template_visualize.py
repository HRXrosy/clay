"""Phase 23 cell template ASCII visualization helpers.

One-call convenience layer combining three established APIs:

* Phase 6 cell template factories (``build_diff_pair``, ``build_5t_ota`` ...)
* Phase 5.3 DRC-complete lowering (``lower_annotated_cell_drc_complete``)
* Phase 15 ASCII renderer (``render_ascii``)

Use case: an LLM (or human) wants to "see" what a named cell looks like
without manually wiring up the build/lower/render pipeline. Pass a
``template_name`` plus a PDK layer map and get a multi-line ASCII string back.

This module does NOT modify any underlying API. It only registers
factories with reasonable default minimum sizing args (so e.g. the
``diff_pair`` registry entry knows it needs ``w_um=4.0``, ``l_um=0.5``)
and threads ``build_kwargs`` through to the underlying factory so the
caller can override anything.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from .ascii_render import render_ascii
from .annotations import AnnotatedCell
from .backend import DeviceSizing, lower_annotated_cell_drc_complete
from .cell_templates import (
    build_5t_ota,
    build_bandgap_reference,
    build_cascoded_current_mirror,
    build_current_mirror,
    build_diff_pair,
    build_folded_cascode,
    build_telescopic_cascode,
    build_two_stage_miller,
)


# Registry of (factory, default minimum sizing kwargs). Defaults supply
# anything the factory requires positionally; callers can override via
# ``build_kwargs``. Keep this in sync with cell_templates/__init__.py.
_TEMPLATE_REGISTRY: Dict[
    str,
    Tuple[Callable[..., Tuple[AnnotatedCell, DeviceSizing]], Dict[str, Any]],
] = {
    "diff_pair": (
        build_diff_pair,
        {"w_um": 4.0, "l_um": 0.5, "finger_count": 4},
    ),
    "current_mirror": (
        build_current_mirror,
        {
            "w_um": 4.0,
            "l_um": 0.5,
            "finger_count_ref": 4,
            "finger_count_out": 4,
        },
    ),
    "cascoded_mirror": (
        build_cascoded_current_mirror,
        {
            "w_um_low": 4.0,
            "l_um_low": 0.5,
            "w_um_cas": 4.0,
            "l_um_cas": 0.5,
            "finger_count_low": 4,
            "finger_count_cas": 4,
        },
    ),
    "5t_ota": (
        build_5t_ota,
        {"input_w_um": 4.0, "input_l_um": 0.5, "input_finger_count": 4},
    ),
    "folded_cascode": (
        build_folded_cascode,
        {"input_w_um": 4.0, "input_l_um": 0.5, "input_finger_count": 4},
    ),
    "telescopic_cascode": (
        build_telescopic_cascode,
        {"input_w_um": 4.0, "input_l_um": 0.5, "input_finger_count": 4},
    ),
    "two_stage_miller": (
        build_two_stage_miller,
        {"input_w_um": 4.0, "input_l_um": 0.5, "input_finger_count": 4},
    ),
    "bandgap_reference": (
        build_bandgap_reference,
        {},  # All defaults; bandgap factory has no required positional sizing.
    ),
}


def _registry_keys() -> Tuple[str, ...]:
    """Return the registered template names (stable iteration order)."""
    return tuple(_TEMPLATE_REGISTRY.keys())


def visualize_template(
    template_name: str,
    pdk_layers: Any,
    fin_pitch_nm: float,
    width_chars: int = 80,
    **build_kwargs: Any,
) -> str:
    """One-call cell template visualization.

    Builds the named cell via its Phase 6 factory, lowers it through
    Phase 5.3 DRC-complete (so well enclosures, DRC poly spacing, etc.
    are baked into the geometry), and renders the result via Phase 15's
    ``render_ascii``.

    Args:
        template_name: one of the eight known template names; see
            ``_TEMPLATE_REGISTRY`` keys. Raises ``ValueError`` on unknown
            names.
        pdk_layers: PDK layer map (e.g. ``Sky130LayerMap()``).
        fin_pitch_nm: fin pitch in nanometers (e.g. ``SKY130_FIN_PITCH_NM``).
        width_chars: ASCII grid column count. Default 80.
        **build_kwargs: keyword arguments forwarded to the underlying
            factory. Overrides any default sizing supplied by the
            registry. Must NOT include ``name`` (the cell name is fixed
            to ``template_name`` so it matches what the renderer looks
            up in the lowered library).

    Returns:
        Multi-line ASCII string (same shape ``render_ascii`` returns).

    Raises:
        ValueError: unknown ``template_name`` or caller passed ``name``
            in ``build_kwargs``.
    """
    if template_name not in _TEMPLATE_REGISTRY:
        available = ", ".join(_registry_keys())
        raise ValueError(
            f"unknown template_name {template_name!r}; "
            f"available: {available}"
        )
    if "name" in build_kwargs:
        raise ValueError(
            "build_kwargs must not include 'name'; it is fixed to "
            "template_name so the renderer can find the cell in the "
            "lowered library."
        )

    build_fn, default_kwargs = _TEMPLATE_REGISTRY[template_name]
    merged_kwargs: Dict[str, Any] = dict(default_kwargs)
    merged_kwargs.update(build_kwargs)

    cell, sizing = build_fn(name=template_name, **merged_kwargs)
    lib = lower_annotated_cell_drc_complete(
        cell, pdk_layers, fin_pitch_nm, sizing
    )
    return render_ascii(lib, template_name, width_chars=width_chars)


def visualize_all_templates(
    pdk_layers: Any,
    fin_pitch_nm: float,
    width_chars: int = 40,
) -> Dict[str, str]:
    """Render every registered cell template at default sizing.

    Args:
        pdk_layers: PDK layer map (e.g. ``Sky130LayerMap()``).
        fin_pitch_nm: fin pitch in nanometers.
        width_chars: ASCII grid width. Default 40 (more compact than
            the per-template default so a wall of 8 renderings fits on
            one screen).

    Returns:
        Dict from template name to its ASCII rendering. Iteration order
        matches ``_TEMPLATE_REGISTRY`` insertion order: diff_pair,
        current_mirror, cascoded_mirror, 5t_ota, folded_cascode,
        telescopic_cascode, two_stage_miller, bandgap_reference.
    """
    out: Dict[str, str] = {}
    for name in _registry_keys():
        out[name] = visualize_template(
            template_name=name,
            pdk_layers=pdk_layers,
            fin_pitch_nm=fin_pitch_nm,
            width_chars=width_chars,
        )
    return out
