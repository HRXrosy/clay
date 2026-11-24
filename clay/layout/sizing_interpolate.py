"""Linear interpolation between two sizing endpoints.

Produces ``n_steps`` points along a straight line in (W, L) space, with
``finger_count`` rounded to the nearest integer (>= 1). Also supplies a
helper that interpolates two :class:`SymbolicLayout` endpoints by
``finger_count`` / ``pitch_multiplier`` / ``stripe_count``.

The interpolator is intentionally small and dependency-free: callers that
want richer Pareto-front behaviour can build it on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List

from .symbolic_layout import SymbolicLayout


@dataclass(frozen=True)
class InterpolatedPoint:
    """One point on the interpolated sizing trajectory.

    Attributes
    ----------
    t:
        Parameter in ``[0.0, 1.0]`` measuring progress from start to end.
    w_um:
        Linearly interpolated transistor width in micrometres.
    l_um:
        Linearly interpolated transistor length in micrometres.
    finger_count:
        Finger count rounded to nearest integer, clamped to ``>= 1``.
    """

    t: float
    w_um: float
    l_um: float
    finger_count: int


def _round_finger(value: float) -> int:
    """Round ``value`` to nearest integer and clamp to ``>= 1``."""
    rounded = int(round(value))
    return rounded if rounded >= 1 else 1


def interpolate_sizing(
    w_start_um: float,
    l_start_um: float,
    finger_start: int,
    w_end_um: float,
    l_end_um: float,
    finger_end: int,
    n_steps: int = 5,
) -> List[InterpolatedPoint]:
    """Produce ``n_steps`` interpolated sizing points from start to end.

    ``W`` and ``L`` are linearly interpolated. ``finger_count`` is also
    linearly interpolated then rounded to the nearest integer (with a
    minimum of one). The first point has ``t == 0.0`` and matches the
    start endpoint; the last point has ``t == 1.0`` and matches the end
    endpoint.

    Adjacent points that collapse onto the same ``finger_count`` after
    rounding are kept; callers are expected to dedupe if needed.
    """
    if n_steps < 2:
        raise ValueError(
            f"n_steps must be >= 2 to span both endpoints, got {n_steps}"
        )
    if w_start_um <= 0.0 or w_end_um <= 0.0:
        raise ValueError(
            "W must be > 0 at both endpoints, "
            f"got w_start={w_start_um} w_end={w_end_um}"
        )
    if l_start_um <= 0.0 or l_end_um <= 0.0:
        raise ValueError(
            "L must be > 0 at both endpoints, "
            f"got l_start={l_start_um} l_end={l_end_um}"
        )
    if finger_start < 1 or finger_end < 1:
        raise ValueError(
            "finger_count must be >= 1 at both endpoints, "
            f"got finger_start={finger_start} finger_end={finger_end}"
        )

    points: List[InterpolatedPoint] = []
    denom = float(n_steps - 1)
    for i in range(n_steps):
        t = i / denom
        w = w_start_um + (w_end_um - w_start_um) * t
        l_val = l_start_um + (l_end_um - l_start_um) * t
        finger = _round_finger(finger_start + (finger_end - finger_start) * t)
        points.append(
            InterpolatedPoint(t=t, w_um=w, l_um=l_val, finger_count=finger)
        )
    # Pin endpoints exactly to avoid float drift on t=1.0.
    points[0] = InterpolatedPoint(
        t=0.0,
        w_um=w_start_um,
        l_um=l_start_um,
        finger_count=_round_finger(finger_start),
    )
    points[-1] = InterpolatedPoint(
        t=1.0,
        w_um=w_end_um,
        l_um=l_end_um,
        finger_count=_round_finger(finger_end),
    )
    return points


def interpolate_symbolic_layout(
    start: SymbolicLayout,
    end: SymbolicLayout,
    n_steps: int = 5,
) -> List[SymbolicLayout]:
    """Interpolate two :class:`SymbolicLayout` endpoints.

    ``finger_count`` and ``stripe_count`` are linearly interpolated and
    rounded to the nearest integer (clamped to ``>= 1``).
    ``pitch_multiplier`` is interpolated as a float then snapped back to
    a :class:`fractions.Fraction` with a denominator bound so trivial
    cases such as ``1/1 -> 1/1`` round-trip exactly.
    """
    if n_steps < 2:
        raise ValueError(
            f"n_steps must be >= 2 to span both endpoints, got {n_steps}"
        )

    pm_start = float(start.pitch_multiplier)
    pm_end = float(end.pitch_multiplier)

    layouts: List[SymbolicLayout] = []
    denom = float(n_steps - 1)
    for i in range(n_steps):
        t = i / denom
        finger = _round_finger(
            start.finger_count + (end.finger_count - start.finger_count) * t
        )
        stripe = _round_finger(
            start.stripe_count + (end.stripe_count - start.stripe_count) * t
        )
        pm_value = pm_start + (pm_end - pm_start) * t
        pm = Fraction(pm_value).limit_denominator(1000)
        layouts.append(
            SymbolicLayout(
                finger_count=finger,
                pitch_multiplier=pm,
                stripe_count=stripe,
            )
        )
    # Pin endpoints exactly so equality with caller inputs holds.
    layouts[0] = SymbolicLayout(
        finger_count=start.finger_count,
        pitch_multiplier=start.pitch_multiplier,
        stripe_count=start.stripe_count,
    )
    layouts[-1] = SymbolicLayout(
        finger_count=end.finger_count,
        pitch_multiplier=end.pitch_multiplier,
        stripe_count=end.stripe_count,
    )
    return layouts
