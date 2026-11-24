"""Deterministic coordinate currency for rigid placed geometry.

Device stamps use centroid-relative microns while placement uses absolute
microns.  A physical vertex is quantized only after composing those two values.
The composed float can nevertheless carry binary representation dust around an
exact half-nm tie (for example ``128156.50000000003``).  Stabilize only that
non-physical dust before the canonical translation-invariant integer-nm
quantization.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Tuple

from clay._native import load_native


_NM_PER_UM = 1000.0
_BINARY_NOISE_DECIMALS_NM = 9
_floor = math.floor


def _placed_coord_nm_py(local_um: float, placement_um: float) -> int:
    composed_nm = (float(local_um) + float(placement_um)) * _NM_PER_UM
    if -1e6 < composed_nm < 1e6:
        m = _floor(composed_nm)
        eps = composed_nm - (m + 0.5)
        if eps > -3e-10:
            return m + 1
        if eps < -7e-10:
            return m
    elif -1e8 < composed_nm < 1e8:
        m = _floor(composed_nm)
        eps = composed_nm - (m + 0.5)
        if eps >= 0.0:
            return m + 1
        if eps < -1e-7:
            return m
    stable_nm = round(composed_nm, _BINARY_NOISE_DECIMALS_NM)
    return _floor(stable_nm + 0.5)


def placed_coord_nm(local_um: float, placement_um: float) -> int:
    """Return one rigidly placed coordinate in the canonical integer-nm currency.

    Nine decimal places in nm is one attometre: far below any layout coordinate
    while comfortably above IEEE-754 addition dust at the design's dimensions.
    Exact half-nanometre ties round to the upper integer.  Unlike half-even,
    that rule commutes with every integer-nanometre translation, so moving a
    rigid stamp cannot grow or shrink its quantized bbox by one nanometre.
    """
    return _placed_coord_nm_py(local_um, placement_um)


def placed_point_nm(
    x_um: float,
    y_um: float,
    placement_um: Tuple[float, float],
) -> Tuple[int, int]:
    """Return one composed placed point in canonical integer-nm currency."""
    cx_nm = (float(x_um) + float(placement_um[0])) * _NM_PER_UM
    if -1e6 < cx_nm < 1e6:
        mx = _floor(cx_nm)
        ex = cx_nm - (mx + 0.5)
        if ex > -3e-10:
            ix = mx + 1
        elif ex < -7e-10:
            ix = mx
        else:
            ix = _floor(round(cx_nm, _BINARY_NOISE_DECIMALS_NM) + 0.5)
    elif -1e8 < cx_nm < 1e8:
        mx = _floor(cx_nm)
        ex = cx_nm - (mx + 0.5)
        if ex >= 0.0:
            ix = mx + 1
        elif ex < -1e-7:
            ix = mx
        else:
            ix = _floor(round(cx_nm, _BINARY_NOISE_DECIMALS_NM) + 0.5)
    else:
        ix = _floor(round(cx_nm, _BINARY_NOISE_DECIMALS_NM) + 0.5)

    cy_nm = (float(y_um) + float(placement_um[1])) * _NM_PER_UM
    if -1e6 < cy_nm < 1e6:
        my = _floor(cy_nm)
        ey = cy_nm - (my + 0.5)
        if ey > -3e-10:
            iy = my + 1
        elif ey < -7e-10:
            iy = my
        else:
            iy = _floor(round(cy_nm, _BINARY_NOISE_DECIMALS_NM) + 0.5)
    elif -1e8 < cy_nm < 1e8:
        my = _floor(cy_nm)
        ey = cy_nm - (my + 0.5)
        if ey >= 0.0:
            iy = my + 1
        elif ey < -1e-7:
            iy = my
        else:
            iy = _floor(round(cy_nm, _BINARY_NOISE_DECIMALS_NM) + 0.5)
    else:
        iy = _floor(round(cy_nm, _BINARY_NOISE_DECIMALS_NM) + 0.5)

    return (ix, iy)


_PLACED_RING_NM_CPP = None
_PLACED_RING_INIT = False


def placed_ring_nm(
    ring: Iterable[Tuple[float, float]],
    placement_um: Tuple[float, float],
) -> List[Tuple[int, int]]:
    """Batch-quantize one centroid-relative um ring into integer-nm pairs."""
    global _PLACED_RING_NM_CPP, _PLACED_RING_INIT
    if not _PLACED_RING_INIT:
        native = load_native("_place_cpp")
        _PLACED_RING_NM_CPP = getattr(native, "placed_ring_nm_cpp", None) if native is not None else None
        _PLACED_RING_INIT = True
    if _PLACED_RING_NM_CPP is not None:
        return _PLACED_RING_NM_CPP(ring, placement_um)
    cx = float(placement_um[0])
    cy = float(placement_um[1])
    out: List[Tuple[int, int]] = []
    for pt in ring:
        out.append((_placed_coord_nm_py(pt[0], cx), _placed_coord_nm_py(pt[1], cy)))
    return out
