"""Quick cell-area estimator (no Phase 5 lowering).

Given a SymbolicLayout-bearing ``AnnotatedCell`` plus a ``DeviceSizing`` map,
compute a coarse per-device area and a cell-total area in um^2 without
running the DRC-aware lowering pipeline. Useful for quick template
comparison, sweeping w/l/finger_count, and feeding sizing hooks that want
an area proxy before paying for full layout generation.

Model (sky130-ish, intentionally simple):

  device_area_um2 = w_um * l_um * finger_count + 4 * fin_pitch_um

where ``fin_pitch_um = 0.048`` (sky130 fin pitch = 48 nm). The constant
``4 * fin_pitch_um`` is a small fixed overhead per device that captures
diffusion-edge / end-of-active overhead which does not scale with w*l*nf.

  total_area_um2 = sum(device_area_um2) * 1.20

The 20 percent multiplier covers spacing, routing channels, and well
boundary overhead. This is a back-of-envelope estimator, not a substitute
for the full lowering area.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

# sky130 fin pitch in micrometres (48 nm).
_FIN_PITCH_UM: float = 0.048

# Per-device edge / end-of-active overhead.
_PER_DEVICE_OVERHEAD_UM2: float = 4.0 * _FIN_PITCH_UM

# Cell-level routing / spacing / well overhead.
_CELL_OVERHEAD_MULTIPLIER: float = 1.20


@dataclass(frozen=True)
class CellAreaEstimate:
    """Result of :func:`estimate_cell_area`.

    Attributes
    ----------
    total_area_um2:
        Cell-level total area including the 20 percent overhead.
    device_areas_um2:
        Per-device area (w * l * nf + edge overhead), keyed by device id.
    n_devices:
        Number of devices contributing to the estimate.
    """

    total_area_um2: float
    device_areas_um2: Dict[str, float] = field(default_factory=dict)
    n_devices: int = 0


def _device_area_um2(w_um: float, l_um: float, finger_count: int) -> float:
    return w_um * l_um * finger_count + _PER_DEVICE_OVERHEAD_UM2


def estimate_cell_area(cell, sizing) -> CellAreaEstimate:
    """Estimate footprint area of ``cell`` under ``sizing``.

    Parameters
    ----------
    cell:
        An :class:`~clay.layout.annotations.AnnotatedCell` (or any object
        exposing a ``layouts`` mapping from device id to a SymbolicLayout
        with a ``finger_count`` attribute).
    sizing:
        :class:`~clay.layout.backend.DeviceSizing` — mapping from device
        id to ``(w_um, l_um)`` tuple.

    Returns
    -------
    CellAreaEstimate

    Notes
    -----
    A device is counted only if it appears in *both* ``cell.layouts`` and
    ``sizing``. Devices present in just one side are silently skipped;
    this matches how lowering tolerates partial sizing during sweeps.
    """
    layouts = getattr(cell, "layouts", {}) or {}
    device_areas: Dict[str, float] = {}

    for device_id, layout in layouts.items():
        if device_id not in sizing:
            continue
        w_um, l_um = sizing[device_id]
        finger_count = int(getattr(layout, "finger_count", 1))
        if w_um <= 0 or l_um <= 0:
            raise ValueError(
                f"device {device_id!r} has non-positive sizing "
                f"(w={w_um}, l={l_um})"
            )
        device_areas[device_id] = _device_area_um2(w_um, l_um, finger_count)

    sum_devices = sum(device_areas.values())
    total = sum_devices * _CELL_OVERHEAD_MULTIPLIER

    return CellAreaEstimate(
        total_area_um2=total,
        device_areas_um2=device_areas,
        n_devices=len(device_areas),
    )


def _template_registry():
    """Lazy import so the area_estimator module stays cheap to import."""
    from .cell_templates import (
        build_diff_pair,
        build_current_mirror,
        build_cascoded_current_mirror,
        build_5t_ota,
        build_folded_cascode,
        build_telescopic_cascode,
        build_two_stage_miller,
        build_bandgap_reference,
    )

    return {
        "build_diff_pair": build_diff_pair,
        "build_current_mirror": build_current_mirror,
        "build_cascoded_current_mirror": build_cascoded_current_mirror,
        "build_5t_ota": build_5t_ota,
        "build_folded_cascode": build_folded_cascode,
        "build_telescopic_cascode": build_telescopic_cascode,
        "build_two_stage_miller": build_two_stage_miller,
        "build_bandgap_reference": build_bandgap_reference,
    }


def _default_build_kwargs(template_name: str) -> Dict[str, object]:
    """Minimal kwargs to drive each template factory if caller omitted them."""
    name = "estim_" + template_name
    if template_name == "build_diff_pair":
        return {"name": name, "w_um": 1.0, "l_um": 0.5, "finger_count": 4}
    if template_name == "build_current_mirror":
        return {
            "name": name,
            "w_um": 1.0,
            "l_um": 0.5,
            "finger_count_ref": 4,
            "finger_count_out": 4,
        }
    if template_name == "build_cascoded_current_mirror":
        return {
            "name": name,
            "w_um_low": 1.0,
            "l_um_low": 0.5,
            "w_um_cas": 1.0,
            "l_um_cas": 0.5,
        }
    if template_name in (
        "build_5t_ota",
        "build_folded_cascode",
        "build_telescopic_cascode",
        "build_two_stage_miller",
    ):
        return {"name": name, "input_w_um": 1.0, "input_l_um": 0.5}
    if template_name == "build_bandgap_reference":
        return {"name": name}
    return {"name": name}


def estimate_template_area(template_name: str, **build_kwargs) -> CellAreaEstimate:
    """Build a cell via its factory then run :func:`estimate_cell_area`.

    Parameters
    ----------
    template_name:
        One of the registered ``build_*`` factory names (see
        :func:`_template_registry`).
    **build_kwargs:
        Forwarded to the factory. Missing required kwargs are filled in
        from :func:`_default_build_kwargs`.

    Raises
    ------
    ValueError:
        If ``template_name`` is not a known factory.
    """
    registry = _template_registry()
    if template_name not in registry:
        known = ", ".join(sorted(registry))
        raise ValueError(
            f"unknown template {template_name!r}; known templates: {known}"
        )

    defaults = _default_build_kwargs(template_name)
    merged = {**defaults, **build_kwargs}

    factory = registry[template_name]
    cell, sizing = factory(**merged)
    return estimate_cell_area(cell, sizing)


def compare_template_areas(
    template_a: str, template_b: str, **build_kwargs
) -> Dict[str, float]:
    """Estimate two templates with the same build_kwargs and report the ratio.

    Returns a dict with keys ``a_area``, ``b_area``, ``ratio`` (=a/b).
    The same ``build_kwargs`` are passed to both factories, so callers
    should keep the kwargs limited to ones both accept (``name``, ``w_um``,
    ``l_um``, ...).
    """
    a = estimate_template_area(template_a, **build_kwargs)
    b = estimate_template_area(template_b, **build_kwargs)
    if b.total_area_um2 == 0:
        raise ZeroDivisionError(
            f"template {template_b!r} estimated zero area; cannot ratio"
        )
    return {
        "a_area": a.total_area_um2,
        "b_area": b.total_area_um2,
        "ratio": a.total_area_um2 / b.total_area_um2,
    }
