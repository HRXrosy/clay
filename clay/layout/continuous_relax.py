"""Phase 4.x - Continuous relaxation of finger_count for gradient sizing.

Phase 4 cosizing uses integer finger_count enumeration over a 3D grid.
For gradient-based sizing (e.g. Adam optimizer), finger_count needs to
be differentiable. Phase 4.x extends the Pelgrom mismatch formula to
accept real-valued n_finger_real in [1, inf), provides analytic gradients,
and offers a vanilla gradient descent to minimize sigma subject to range
constraints. At execution time, n_finger_real rounds to nearest int.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

from .pdk_dims import PdkDims, DeviceType


def sigma_delta_vth_continuous(
    w_um: float, l_um: float, n_finger_real: float, avt_mv_um: float,
) -> float:
    """Continuous version of Pelgrom sigma = A_VT / sqrt(W * L * N).

    n_finger_real may be a non-integer real >= 1.0.

    Raises ValueError on non-positive inputs.
    """
    if w_um <= 0 or l_um <= 0:
        raise ValueError("w_um and l_um must be > 0")
    if n_finger_real < 1.0:
        raise ValueError(f"n_finger_real must be >= 1.0, got {n_finger_real}")
    if avt_mv_um <= 0:
        raise ValueError("avt_mv_um must be > 0")
    return avt_mv_um / math.sqrt(w_um * l_um * n_finger_real)


def grad_sigma_wrt_n_finger(
    w_um: float, l_um: float, n_finger_real: float, avt_mv_um: float,
) -> float:
    """Analytic d_sigma/d_N = -A_VT * WL / (2 * (WLN)^(3/2)).

    Derivation: d/dN (A * (WLN)^(-1/2)) = -A * 0.5 * (WLN)^(-3/2) * WL
                                        = -A * WL / (2 * (WLN)^(3/2))
    """
    wln = w_um * l_um * n_finger_real
    return -avt_mv_um * w_um * l_um / (2.0 * wln ** 1.5)


def grad_sigma_wrt_w(
    w_um: float, l_um: float, n_finger_real: float, avt_mv_um: float,
) -> float:
    """d_sigma/d_W = -A_VT * L * N / (2 * (WLN)^(3/2))."""
    wln = w_um * l_um * n_finger_real
    return -avt_mv_um * l_um * n_finger_real / (2.0 * wln ** 1.5)


def grad_sigma_wrt_l(
    w_um: float, l_um: float, n_finger_real: float, avt_mv_um: float,
) -> float:
    """d_sigma/d_L = -A_VT * W * N / (2 * (WLN)^(3/2))."""
    wln = w_um * l_um * n_finger_real
    return -avt_mv_um * w_um * n_finger_real / (2.0 * wln ** 1.5)


@dataclass(frozen=True, eq=True)
class ContinuousSizingPoint:
    w_um: float
    l_um: float
    n_finger_real: float
    sigma_mv: float


def gradient_descent_minimize_sigma(
    w_init: float,
    l_init: float,
    n_init: float,
    target_sigma_mv: float,
    pdk: PdkDims,
    device_type: DeviceType,
    learning_rate: float = 0.01,
    max_iters: int = 1000,
    w_range: tuple = (1.0, 20.0),
    l_range: tuple = (0.1, 5.0),
    n_range: tuple = (1.0, 100.0),
    convergence_tol: float = 1e-6,
) -> ContinuousSizingPoint:
    """Minimize sigma subject to range constraints via plain gradient descent.

    Stops early if |sigma - target_sigma_mv| < convergence_tol or max_iters reached.

    All three dimensions updated each step; clamped to ranges. learning_rate
    is uniform across (W, L, N) for simplicity; in practice scale by parameter
    range. Convergence not guaranteed for poor initializations; Phase 4 grid
    sweep + Pareto select remains preferred for production.
    """
    avt = pdk.avt_mv_um(device_type)
    w, l, n = w_init, l_init, n_init

    for _ in range(max_iters):
        sigma = sigma_delta_vth_continuous(w, l, n, avt)
        if abs(sigma - target_sigma_mv) < convergence_tol:
            break

        gw = grad_sigma_wrt_w(w, l, n, avt)
        gl = grad_sigma_wrt_l(w, l, n, avt)
        gn = grad_sigma_wrt_n_finger(w, l, n, avt)

        # If sigma > target, we want to decrease sigma -> step in negative gradient direction
        # (gradient of sigma w.r.t. each is negative; step *= -1 effectively positive)
        sign = 1.0 if sigma > target_sigma_mv else -1.0

        w_new = w - sign * learning_rate * gw
        l_new = l - sign * learning_rate * gl
        n_new = n - sign * learning_rate * gn

        # Clamp to ranges
        w = max(w_range[0], min(w_range[1], w_new))
        l = max(l_range[0], min(l_range[1], l_new))
        n = max(n_range[0], min(n_range[1], n_new))

    final_sigma = sigma_delta_vth_continuous(w, l, n, avt)
    return ContinuousSizingPoint(
        w_um=w, l_um=l, n_finger_real=n, sigma_mv=final_sigma,
    )


def round_to_integer_finger(point: ContinuousSizingPoint) -> int:
    """Round n_finger_real to nearest integer >= 1."""
    rounded = int(round(point.n_finger_real))
    return max(1, rounded)
