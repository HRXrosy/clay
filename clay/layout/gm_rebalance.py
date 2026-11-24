"""Phase 7.4 — g_m_eff rebalance helpers.

When changing finger_count M, scale W to maintain approximate gm
invariance. Two modes:

- 'strict': W' = W * M / M'  (gm_eff exactly invariant; assumes I_D
  also tracks M, which is true for current-mirror legs)
- 'sqrt': W' = W * sqrt(M / M')  (keeps gm * sqrt(area) constant;
  matching-friendly sizing)

Per main prompt section 13.6.3 — R94 B `rebalance="g_m_eff"` request.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class GmRebalanceMode(Enum):
    """gm rebalance mode when finger_count M changes."""
    STRICT = "strict"       # W' = W * M / M' (exact gm_eff invariance)
    SQRT = "sqrt"           # W' = W * sqrt(M / M') (matching-friendly)


@dataclass(frozen=True, eq=True)
class GmRebalanceResult:
    """Result of a gm rebalance call."""
    w_um_new: float
    l_um_new: float
    finger_count_new: int
    mode: GmRebalanceMode
    gm_ratio_estimate: float  # gm_new / gm_orig estimate (1.0 = invariant)


def rebalance_for_finger_change(
    w_um_orig: float,
    l_um_orig: float,
    finger_count_orig: int,
    finger_count_new: int,
    mode: GmRebalanceMode = GmRebalanceMode.SQRT,
) -> GmRebalanceResult:
    """Compute new (W, L, M) preserving gm characteristic.

    L is held constant (sizing change is purely in W and M). Only W and
    M change.

    Args:
        w_um_orig, l_um_orig: original device W and L in micrometers
        finger_count_orig: original M
        finger_count_new: new M (target)
        mode: which rebalance flavor

    Returns:
        GmRebalanceResult with new W (L unchanged) and gm_ratio_estimate

    Raises:
        ValueError on non-positive inputs
    """
    if w_um_orig <= 0 or l_um_orig <= 0:
        raise ValueError("w_um and l_um must be > 0")
    if finger_count_orig < 1 or finger_count_new < 1:
        raise ValueError("finger_count must be >= 1")

    if mode == GmRebalanceMode.STRICT:
        w_new = w_um_orig * finger_count_orig / finger_count_new
        # gm_strict: gm_eff_new = gm_eff_orig exactly (by construction)
        gm_ratio = 1.0
    elif mode == GmRebalanceMode.SQRT:
        import math
        w_new = w_um_orig * math.sqrt(finger_count_orig / finger_count_new)
        # gm_eff_new / gm_eff_orig = sqrt(M_new * W_new / (M_orig * W_orig))
        # = sqrt(M_new * W_orig * sqrt(M_orig/M_new) / (M_orig * W_orig))
        # = sqrt(M_new/M_orig * sqrt(M_orig/M_new))
        # = (M_new/M_orig)^(1/2) * (M_orig/M_new)^(1/4)
        # = (M_new/M_orig)^(1/4)
        gm_ratio = (finger_count_new / finger_count_orig) ** 0.25
    else:
        raise ValueError(f"unknown rebalance mode: {mode}")

    return GmRebalanceResult(
        w_um_new=w_new,
        l_um_new=l_um_orig,
        finger_count_new=finger_count_new,
        mode=mode,
        gm_ratio_estimate=gm_ratio,
    )


def rebalance_strict(
    w_um_orig: float,
    l_um_orig: float,
    finger_count_orig: int,
    finger_count_new: int,
) -> Tuple[float, float, int]:
    """Convenience wrapper returning just (W_new, L_new, M_new) tuple — strict mode."""
    r = rebalance_for_finger_change(
        w_um_orig, l_um_orig, finger_count_orig, finger_count_new,
        mode=GmRebalanceMode.STRICT,
    )
    return (r.w_um_new, r.l_um_new, r.finger_count_new)


def rebalance_sqrt(
    w_um_orig: float,
    l_um_orig: float,
    finger_count_orig: int,
    finger_count_new: int,
) -> Tuple[float, float, int]:
    """Convenience wrapper returning just (W_new, L_new, M_new) tuple — sqrt mode."""
    r = rebalance_for_finger_change(
        w_um_orig, l_um_orig, finger_count_orig, finger_count_new,
        mode=GmRebalanceMode.SQRT,
    )
    return (r.w_um_new, r.l_um_new, r.finger_count_new)
