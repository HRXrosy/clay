"""Closed-loop sizing optimizer for Clay layout cell templates.

Given a target Vth mismatch sigma and an area budget, search a discrete
(W, L, finger_count) grid for the minimum-area point that satisfies both
constraints. Wraps the existing ``joint_sweep`` Pareto surface and adds
constraint filtering plus a deterministic tie-break (min area, then min
sigma, then lexicographic on W/L/finger_count).

This module is purely numerical and side-effect free; it does not mutate
its PDK argument or perform any I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .cosizing import joint_sweep
from .pdk_dims import DeviceType, PdkDims


@dataclass(frozen=True)
class SizingConstraints:
    """Target envelope for the sizing optimizer.

    Both ``target_sigma_mv`` and ``target_max_area_um2`` are hard upper
    bounds. Any candidate exceeding either is rejected. The three range
    lists span the (W, L, finger) search grid; the optimizer enumerates
    every combination via ``joint_sweep`` and filters in Python.
    """

    target_sigma_mv: float
    target_max_area_um2: float
    w_range: List[float] = field(default_factory=lambda: [2.0, 4.0, 8.0])
    l_range: List[float] = field(default_factory=lambda: [0.5, 1.0])
    finger_range: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])


@dataclass
class SizingOptimumResult:
    """Outcome of a single ``optimize_sizing`` call.

    On success (``found=True``) the geometric fields carry the chosen
    point. On failure (``found=False``) the geometric fields stay at
    their zero defaults and ``rejected_reason`` is a short diagnostic.
    """

    found: bool
    w_um: float = 0.0
    l_um: float = 0.0
    finger_count: int = 0
    sigma_mv: float = 0.0
    area_um2: float = 0.0
    rejected_reason: Optional[str] = None


def _validate_constraints(c: SizingConstraints) -> Optional[str]:
    if c.target_sigma_mv <= 0:
        return f"target_sigma_mv must be > 0, got {c.target_sigma_mv}"
    if c.target_max_area_um2 <= 0:
        return f"target_max_area_um2 must be > 0, got {c.target_max_area_um2}"
    if not c.w_range:
        return "w_range is empty"
    if not c.l_range:
        return "l_range is empty"
    if not c.finger_range:
        return "finger_range is empty"
    return None


def optimize_sizing(
    constraints: SizingConstraints,
    pdk: PdkDims,
    pdk_layers: Any,
    device_type: DeviceType,
    fin_pitch_nm: float,
) -> SizingOptimumResult:
    """Find the minimum-area (W, L, finger) point satisfying constraints.

    Sweeps the discrete grid encoded in ``constraints`` via
    ``joint_sweep``, filters to points where both sigma and area meet
    their targets, and returns the survivor with the smallest area. Ties
    on area are broken by smaller sigma, then by lexicographic
    (W, L, finger_count). If no point survives, returns a result with
    ``found=False`` and an explanatory ``rejected_reason``.

    The ``pdk_layers`` and ``fin_pitch_nm`` parameters are part of the
    forward-compatible signature; they are accepted so callers can pass
    a full layout context, but the Pelgrom-based cost in this Phase 1
    optimizer does not yet consume them.
    """
    # pdk_layers and fin_pitch_nm are reserved for richer cost models
    # (parasitic-aware, fin-quantized area). Phase 1 grid sweep does not
    # need them; we still validate the argument shape minimally.
    del pdk_layers
    if fin_pitch_nm <= 0:
        return SizingOptimumResult(
            found=False,
            rejected_reason=f"fin_pitch_nm must be > 0, got {fin_pitch_nm}",
        )

    err = _validate_constraints(constraints)
    if err is not None:
        return SizingOptimumResult(found=False, rejected_reason=err)

    points = joint_sweep(
        list(constraints.w_range),
        list(constraints.l_range),
        list(constraints.finger_range),
        pdk,
        device_type,
    )

    sigma_ok = [p for p in points if p.sigma_mv <= constraints.target_sigma_mv]
    area_ok = [p for p in points if p.area_um2 <= constraints.target_max_area_um2]
    satisfying = [p for p in sigma_ok if p.area_um2 <= constraints.target_max_area_um2]

    if not satisfying:
        if not sigma_ok and not area_ok:
            reason = (
                f"no point meets sigma<={constraints.target_sigma_mv} mV "
                f"or area<={constraints.target_max_area_um2} um^2"
            )
        elif not sigma_ok:
            reason = (
                f"no point meets sigma<={constraints.target_sigma_mv} mV "
                f"(min achievable sigma="
                f"{min(p.sigma_mv for p in points):.6g} mV)"
            )
        elif not area_ok:
            reason = (
                f"no point meets area<={constraints.target_max_area_um2} um^2 "
                f"(min achievable area="
                f"{min(p.area_um2 for p in points):.6g} um^2)"
            )
        else:
            reason = (
                f"no point jointly satisfies sigma<={constraints.target_sigma_mv} mV "
                f"and area<={constraints.target_max_area_um2} um^2"
            )
        return SizingOptimumResult(found=False, rejected_reason=reason)

    # Sort by (area, sigma, W, L, finger_count) so ties break
    # deterministically toward smaller geometry.
    satisfying.sort(
        key=lambda p: (p.area_um2, p.sigma_mv, p.w_um, p.l_um, p.finger_count)
    )
    best = satisfying[0]
    return SizingOptimumResult(
        found=True,
        w_um=best.w_um,
        l_um=best.l_um,
        finger_count=best.finger_count,
        sigma_mv=best.sigma_mv,
        area_um2=best.area_um2,
        rejected_reason=None,
    )
