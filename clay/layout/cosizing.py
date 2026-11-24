"""Phase 4 - co-variable closure demo (pure Python mirror of Rust cosizing).

Joint (W, L, finger_count) Pareto sweep using Phase 1 sigma + area as cost surface.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional

from .mismatch import sigma_delta_vth_mv
from .pdk_dims import PdkDims, DeviceType


@dataclass(frozen=True, eq=True)
class CosizingPoint:
    w_um: float
    l_um: float
    finger_count: int
    sigma_mv: float
    area_um2: float


def joint_sweep(
    w_range: List[float],
    l_range: List[float],
    finger_count_range: List[int],
    pdk: PdkDims,
    device_type: DeviceType,
) -> List[CosizingPoint]:
    avt = pdk.avt_mv_um(device_type)
    out: List[CosizingPoint] = []
    for w in w_range:
        for l in l_range:
            for n in finger_count_range:
                sigma = sigma_delta_vth_mv(w, l, n, avt)
                area = w * l * n
                out.append(CosizingPoint(w_um=w, l_um=l, finger_count=n, sigma_mv=sigma, area_um2=area))
    return out


def _dominates(d: CosizingPoint, v: CosizingPoint) -> bool:
    sigma_le = d.sigma_mv <= v.sigma_mv
    area_le = d.area_um2 <= v.area_um2
    sigma_lt = d.sigma_mv < v.sigma_mv
    area_lt = d.area_um2 < v.area_um2
    return sigma_le and area_le and (sigma_lt or area_lt)


def pareto_optimal(points: List[CosizingPoint]) -> List[CosizingPoint]:
    out: List[CosizingPoint] = []
    for p in points:
        dominated = any(_dominates(q, p) for q in points)
        if not dominated:
            out.append(p)
    out.sort(key=lambda p: p.sigma_mv)
    # Dedup exact-equal points
    deduped: List[CosizingPoint] = []
    for p in out:
        if deduped and abs(deduped[-1].sigma_mv - p.sigma_mv) < 1e-12 and abs(deduped[-1].area_um2 - p.area_um2) < 1e-12:
            continue
        deduped.append(p)
    return deduped


def select_optimal(
    pareto: List[CosizingPoint],
    cost_fn: Callable[[CosizingPoint], float],
) -> Optional[CosizingPoint]:
    if not pareto:
        return None
    best = pareto[0]
    best_c = cost_fn(best)
    for p in pareto[1:]:
        c = cost_fn(p)
        if c < best_c:
            best = p
            best_c = c
    return best
