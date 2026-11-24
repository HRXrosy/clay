"""Monte Carlo mismatch sampler for Clay Phase 22 distribution-aware sizing.

This module draws random ΔV_TH samples from the Gaussian distribution
implied by Phase 1's Pelgrom sigma (``A_VT / sqrt(W * L * N)``) and
reports empirical mean / std / quantiles plus a 3-sigma yield estimator.

The sampler is built on top of the existing
``clay.layout.mismatch.sigma_delta_vth_mv`` helper; it does NOT modify
that API. It uses only the Python standard library (``random`` +
``statistics``) so it adds no numpy dependency.
"""

from dataclasses import dataclass, field
from random import Random
from statistics import mean as _mean, pstdev as _pstdev
from typing import Dict, List, Optional

from .mismatch import sigma_delta_vth_mv
from .pdk_dims import DeviceType, PdkDims
from .symbolic_layout import SymbolicLayout

_DEFAULT_QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)


@dataclass
class MonteCarloMismatchResult:
    """Result bundle from a Monte Carlo mismatch sampling run.

    Attributes
    ----------
    samples:
        Raw ΔV_TH samples in mV (length == ``n_samples``).
    mean:
        Empirical sample mean in mV.
    std:
        Empirical sample standard deviation (population) in mV.
    quantiles:
        Mapping ``{q: value}`` for q in {0.01, 0.05, 0.5, 0.95, 0.99}
        giving the corresponding percentile of ``samples``.
    """

    samples: List[float]
    mean: float
    std: float
    quantiles: Dict[float, float] = field(default_factory=dict)


def _empirical_quantile(sorted_samples: List[float], q: float) -> float:
    """Return the q-th quantile of ``sorted_samples`` via linear interp.

    ``sorted_samples`` must be sorted ascending and non-empty. ``q`` is
    in [0, 1]. Uses the common "linear" rule (numpy default): index
    ``q * (n - 1)`` interpolated between adjacent samples.
    """
    n = len(sorted_samples)
    if n == 1:
        return sorted_samples[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_samples[lo] * (1.0 - frac) + sorted_samples[hi] * frac


def sample_delta_vth_mv(
    w_um: float,
    l_um: float,
    n_finger: int,
    avt_mv_um: float,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> MonteCarloMismatchResult:
    """Draw ``n_samples`` random ΔV_TH values ~ ``Normal(0, sigma)``.

    ``sigma`` is computed from the Pelgrom law via
    ``sigma_delta_vth_mv(w_um, l_um, n_finger, avt_mv_um)``. Validation
    of the geometric arguments is delegated to that helper.

    The returned ``MonteCarloMismatchResult`` carries the raw samples
    plus empirical mean, population std, and the 1 / 5 / 50 / 95 / 99
    quantiles.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    sigma = sigma_delta_vth_mv(w_um, l_um, n_finger, avt_mv_um)
    rng = Random(seed)
    samples = [rng.gauss(0.0, sigma) for _ in range(n_samples)]

    sample_mean = _mean(samples)
    sample_std = _pstdev(samples) if len(samples) > 1 else 0.0

    sorted_samples = sorted(samples)
    quantiles: Dict[float, float] = {
        q: _empirical_quantile(sorted_samples, q) for q in _DEFAULT_QUANTILES
    }
    return MonteCarloMismatchResult(
        samples=samples,
        mean=sample_mean,
        std=sample_std,
        quantiles=quantiles,
    )


def sample_diff_pair_offset(
    layout: SymbolicLayout,
    w_um: float,
    l_um: float,
    pdk: PdkDims,
    device: DeviceType,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> MonteCarloMismatchResult:
    """Monte Carlo ΔV_TH offset samples for a diff-pair leg.

    Convenience wrapper that pulls ``n_finger`` from ``layout`` and
    ``A_VT`` from ``pdk.avt_mv_um(device)``, then defers to
    ``sample_delta_vth_mv``.
    """
    return sample_delta_vth_mv(
        w_um=w_um,
        l_um=l_um,
        n_finger=layout.finger_count,
        avt_mv_um=pdk.avt_mv_um(device),
        n_samples=n_samples,
        seed=seed,
    )


def estimate_3sigma_yield(
    samples: List[float],
    spec_max_mv: float,
) -> float:
    """Empirical yield: fraction of samples with ``|ΔV_TH| <= spec_max_mv``.

    Parameters
    ----------
    samples:
        Non-empty list of ΔV_TH samples in mV.
    spec_max_mv:
        Absolute mismatch budget in mV. Must be >= 0.

    Returns
    -------
    float
        Yield in ``[0.0, 1.0]``.
    """
    if not samples:
        raise ValueError("samples must be non-empty to estimate yield")
    if spec_max_mv < 0.0:
        raise ValueError(f"spec_max_mv must be >= 0, got {spec_max_mv}")
    n_pass = sum(1 for v in samples if abs(v) <= spec_max_mv)
    return n_pass / len(samples)
