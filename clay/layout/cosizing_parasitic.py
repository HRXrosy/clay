"""Phase 7.2 - parasitic-aware Pareto sweep.

Extends Phase 4 (joint_sweep + pareto_optimal on sigma, area) by adding
parasitic capacitance as a third objective. Each grid point evaluates:

1. sigma(Delta V_TH) via Phase 1 mismatch model (Pelgrom A_VT / sqrt(W*L*N))
2. Layout area via W * L * N_finger
3. Parasitic capacitance via Phase 7 extract_parasitic on a synthetic
   diff pair built from the template at (W, L, N) and lowered through
   Phase 5 DRC-complete pipeline.

The 3-objective Pareto front is non-dominated points in (sigma, area, C_parasitic)
space - all three minimized.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional

from .cosizing import joint_sweep, CosizingPoint
from .symbolic_layout import SymbolicLayout
from .pdk_dims import PdkDims, DeviceType
from .backend import LayerMap, lower_annotated_cell_drc_complete
from .annotations import AnnotatedCell, Matching, MatchingKind
from .cell_templates.diff_pair import build_diff_pair
from .parasitic import extract_parasitic, parasitic_params_for_pdk_layers


@dataclass(frozen=True, eq=True)
class CosizingPointWithParasitic:
    """Phase 7.2 extension of CosizingPoint with parasitic capacitance."""
    w_um: float
    l_um: float
    finger_count: int
    sigma_mv: float
    area_um2: float
    parasitic_fF: float


def _build_synthetic_diff_pair_lib(
    w_um: float, l_um: float, finger_count: int,
    pdk_layers: LayerMap,
    fin_pitch_nm: float,
):
    """Construct a minimal diff pair AnnotatedCell + lower it via Phase 5."""
    cell, sizing = build_diff_pair(
        f"_pareto_synth_w{w_um}_l{l_um}_n{finger_count}",
        w_um=w_um, l_um=l_um, finger_count=finger_count,
    )
    lib = lower_annotated_cell_drc_complete(
        cell, pdk_layers, fin_pitch_nm, sizing,
    )
    return lib


def joint_sweep_with_parasitic(
    w_range: List[float],
    l_range: List[float],
    finger_count_range: List[int],
    pdk: PdkDims,
    pdk_layers: LayerMap,
    device_type: DeviceType,
    fin_pitch_nm: float,
) -> List[CosizingPointWithParasitic]:
    """Joint sweep adding parasitic estimate at each grid point.

    For each (W, L, N) combination:
    1. Compute sigma + area via Phase 4 joint_sweep semantics.
    2. Build a synthetic diff pair via build_diff_pair + lower via Phase 5.
    3. Extract per-library parasitic total via Phase 7 extract_parasitic.
    4. Use the top cell's total parasitic (not double-counted device cells).

    Args:
        w_range, l_range, finger_count_range: grid axes
        pdk: PdkDims (Sky130() / Gf180mcu())
        pdk_layers: LayerMap (Sky130LayerMap / Gf180mcuLayerMap)
        device_type: DeviceType.NMOS / PMOS
        fin_pitch_nm: PDK fin pitch in nm (48 sky130 / 280 gf180)

    Returns:
        List of CosizingPointWithParasitic, one per grid point.
    """
    base_points = joint_sweep(w_range, l_range, finger_count_range, pdk, device_type)
    out: List[CosizingPointWithParasitic] = []
    for p in base_points:
        try:
            lib = _build_synthetic_diff_pair_lib(
                p.w_um, p.l_um, p.finger_count, pdk_layers, fin_pitch_nm,
            )
            report = extract_parasitic(lib, pdk_layers)
            # Use only the top cell parasitic to avoid double-counting per-device cells
            top_cell = next(
                (c for c in report.cells if c.cell_name.startswith("_pareto_synth")),
                None,
            )
            parasitic_aF = top_cell.total_cap_aF if top_cell else 0.0
            parasitic_fF = parasitic_aF / 1000.0
        except Exception:
            parasitic_fF = float("inf")  # mark as infeasible
        out.append(CosizingPointWithParasitic(
            w_um=p.w_um,
            l_um=p.l_um,
            finger_count=p.finger_count,
            sigma_mv=p.sigma_mv,
            area_um2=p.area_um2,
            parasitic_fF=parasitic_fF,
        ))
    return out


def _dominates_3obj(d: CosizingPointWithParasitic, v: CosizingPointWithParasitic) -> bool:
    """3-objective Pareto dominance: d dominates v if d <= v on all three
    (sigma, area, parasitic) and < on at least one."""
    le_sigma = d.sigma_mv <= v.sigma_mv
    le_area = d.area_um2 <= v.area_um2
    le_para = d.parasitic_fF <= v.parasitic_fF
    lt_sigma = d.sigma_mv < v.sigma_mv
    lt_area = d.area_um2 < v.area_um2
    lt_para = d.parasitic_fF < v.parasitic_fF
    return le_sigma and le_area and le_para and (lt_sigma or lt_area or lt_para)


def pareto_optimal_3obj(
    points: List[CosizingPointWithParasitic],
) -> List[CosizingPointWithParasitic]:
    """Extract 3-objective Pareto front (minimize sigma, area, parasitic).

    O(n^2) brute force; sufficient for typical 3-100 point grids.
    Returns points sorted by sigma ascending.
    """
    out: List[CosizingPointWithParasitic] = []
    for p in points:
        dominated = any(_dominates_3obj(q, p) for q in points)
        if not dominated:
            out.append(p)
    out.sort(key=lambda p: (p.sigma_mv, p.area_um2, p.parasitic_fF))
    return out


def select_optimal_3obj(
    pareto: List[CosizingPointWithParasitic],
    cost_fn: Callable[[CosizingPointWithParasitic], float],
) -> Optional[CosizingPointWithParasitic]:
    """Pick the cost-minimizing point on the 3-objective Pareto front.

    cost_fn is user-supplied; typical form:
        lambda p: p.sigma_mv + lam_a * p.area_um2 + lam_c * p.parasitic_fF
    """
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
