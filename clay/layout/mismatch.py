"""Pelgrom-style mismatch helpers for Clay Phase 1 layout-as-co-variable.

These pure functions translate ``SymbolicLayout`` + ``PdkDims`` into a
Vth mismatch sigma in millivolts. They are deterministic, side-effect
free, and intended to be used both inside the differentiable cost graph
and in plain numerical analysis.
"""

import math

from .pdk_dims import DeviceType, PdkDims
from .symbolic_layout import SymbolicLayout


def sigma_delta_vth_mv(
    w_um: float,
    l_um: float,
    n_finger: int,
    avt_mv_um: float,
) -> float:
    """Pelgrom Vth mismatch sigma in mV.

    Computes ``A_VT / sqrt(W * L * N_finger)``. All inputs must be
    strictly positive (n_finger must be >= 1); otherwise ``ValueError``
    is raised.
    """
    if w_um <= 0:
        raise ValueError(f"w_um must be > 0, got {w_um}")
    if l_um <= 0:
        raise ValueError(f"l_um must be > 0, got {l_um}")
    if n_finger < 1:
        raise ValueError(f"n_finger must be >= 1, got {n_finger}")
    if avt_mv_um <= 0:
        raise ValueError(f"avt_mv_um must be > 0, got {avt_mv_um}")
    return avt_mv_um / math.sqrt(w_um * l_um * n_finger)


def diff_pair_mismatch_mv(
    layout: SymbolicLayout,
    w_um: float,
    l_um: float,
    pdk: PdkDims,
    device: DeviceType,
) -> float:
    """Vth mismatch sigma for a differential pair leg, in mV."""
    return sigma_delta_vth_mv(
        w_um,
        l_um,
        layout.finger_count,
        pdk.avt_mv_um(device),
    )
