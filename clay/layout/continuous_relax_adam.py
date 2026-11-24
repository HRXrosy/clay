"""Phase 4.y - Adam optimizer for continuous relaxation sizing.

Phase 4.x (clay.layout.continuous_relax) provides vanilla gradient descent
over the continuous-finger Pelgrom surface. Phase 4.y upgrades the search
to an Adam-style optimizer with per-parameter first and second moment
estimates. Adam typically converges in fewer iterations than vanilla GD
across the (W, L, N) sizing surface because the second moment normalizes
the step size per parameter without manual learning-rate scaling.

The signed step convention from Phase 4.x is preserved: when the current
sigma exceeds target, we descend (increase W/L/N drives sigma down because
each partial is negative); when sigma is already below target, we ascend
to bring sigma back up. This keeps adam_minimize_sigma a target-tracking
solver rather than a pure minimizer.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

from .continuous_relax import (
    sigma_delta_vth_continuous,
    grad_sigma_wrt_n_finger,
    grad_sigma_wrt_w,
    grad_sigma_wrt_l,
)
from .pdk_dims import PdkDims, DeviceType


@dataclass(frozen=True, eq=True)
class AdamSizingResult:
    """Outcome of an Adam-driven sizing search.

    iters_used: number of optimizer steps actually taken (<= max_iters)
    converged: True iff |sigma - target| < convergence_tol at termination
    """
    w_um: float
    l_um: float
    n_finger_real: float
    sigma_mv: float
    iters_used: int
    converged: bool


def adam_minimize_sigma(
    w_init: float,
    l_init: float,
    n_init: float,
    target_sigma_mv: float,
    pdk: PdkDims,
    device_type: DeviceType,
    learning_rate: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    max_iters: int = 1000,
    convergence_tol: float = 1e-6,
    w_range: tuple = (1.0, 20.0),
    l_range: tuple = (0.1, 5.0),
    n_range: tuple = (1.0, 100.0),
) -> AdamSizingResult:
    """Adam optimizer over (W, L, N) tracking |sigma - target| -> 0.

    Tracks first moment ``m`` and second moment ``v`` per parameter with
    bias-correction (Kingma & Ba 2014). Step direction is signed by
    ``sign = +1 if sigma > target else -1`` so that sigma is pushed
    toward target rather than away.

    All three dimensions are clamped to their ranges after every update.
    Convergence is declared when |sigma - target_sigma_mv| < convergence_tol.

    Raises ValueError on invalid hyperparameters (negative learning_rate,
    out-of-range beta1/beta2, non-positive eps, max_iters <= 0).
    """
    # Hyperparameter validation
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
    if not (0.0 <= beta1 < 1.0):
        raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
    if not (0.0 <= beta2 < 1.0):
        raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if max_iters <= 0:
        raise ValueError(f"max_iters must be > 0, got {max_iters}")
    if convergence_tol <= 0:
        raise ValueError(f"convergence_tol must be > 0, got {convergence_tol}")

    avt = pdk.avt_mv_um(device_type)
    w, l, n = w_init, l_init, n_init

    # First and second moment estimates per parameter
    m_w = m_l = m_n = 0.0
    v_w = v_l = v_n = 0.0

    iters_used = 0
    converged = False

    for step in range(1, max_iters + 1):
        sigma = sigma_delta_vth_continuous(w, l, n, avt)
        if abs(sigma - target_sigma_mv) < convergence_tol:
            converged = True
            break

        gw = grad_sigma_wrt_w(w, l, n, avt)
        gl = grad_sigma_wrt_l(w, l, n, avt)
        gn = grad_sigma_wrt_n_finger(w, l, n, avt)

        sign = 1.0 if sigma > target_sigma_mv else -1.0
        # Effective objective gradient (we want to decrease |sigma - target|;
        # multiplying raw sigma-gradient by sign yields a descent direction).
        gw_eff = sign * gw
        gl_eff = sign * gl
        gn_eff = sign * gn

        # Update biased first moment
        m_w = beta1 * m_w + (1.0 - beta1) * gw_eff
        m_l = beta1 * m_l + (1.0 - beta1) * gl_eff
        m_n = beta1 * m_n + (1.0 - beta1) * gn_eff

        # Update biased second raw moment
        v_w = beta2 * v_w + (1.0 - beta2) * gw_eff * gw_eff
        v_l = beta2 * v_l + (1.0 - beta2) * gl_eff * gl_eff
        v_n = beta2 * v_n + (1.0 - beta2) * gn_eff * gn_eff

        # Bias correction
        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step

        m_w_hat = m_w / bc1
        m_l_hat = m_l / bc1
        m_n_hat = m_n / bc1
        v_w_hat = v_w / bc2
        v_l_hat = v_l / bc2
        v_n_hat = v_n / bc2

        # Adam step (descend along gw_eff): subtract lr * m_hat / (sqrt(v_hat) + eps)
        w_new = w - learning_rate * m_w_hat / (math.sqrt(v_w_hat) + eps)
        l_new = l - learning_rate * m_l_hat / (math.sqrt(v_l_hat) + eps)
        n_new = n - learning_rate * m_n_hat / (math.sqrt(v_n_hat) + eps)

        # Clamp to ranges
        w = max(w_range[0], min(w_range[1], w_new))
        l = max(l_range[0], min(l_range[1], l_new))
        n = max(n_range[0], min(n_range[1], n_new))

        iters_used = step

    # Final sigma at terminating (w, l, n)
    final_sigma = sigma_delta_vth_continuous(w, l, n, avt)
    if not converged and abs(final_sigma - target_sigma_mv) < convergence_tol:
        converged = True

    return AdamSizingResult(
        w_um=w,
        l_um=l,
        n_finger_real=n,
        sigma_mv=final_sigma,
        iters_used=iters_used,
        converged=converged,
    )
