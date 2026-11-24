"""Phase 7.3 — parasitic-aware sizing cost hook.

Extends Phase 1's DefaultMismatchCostHook by adding a parasitic
capacitance penalty term to the cost function. The combined cost is:

    cost = max(0, target_cmrr_db - mismatch_cmrr_degradation_db)
         + lambda_parasitic_fF * parasitic_fF

where lambda_parasitic_fF is a user-supplied weight (units: dB / fF).

The parasitic estimate at a given (W, L, finger_count) point is
extracted by building a synthetic diff pair via build_diff_pair +
lowering via Phase 5 drc_complete + running Phase 7 extract_parasitic.
Per-call cost is dominated by the lowering + extraction pipeline
(~0.1-0.2 ms per call on typical sky130 inputs per Phase 7.2 benchmarks).
"""
from __future__ import annotations
from typing import Optional, Dict

from .symbolic_layout import SymbolicLayout
from .pdk_dims import PdkDims, DeviceType
from .sizing_hook import DefaultMismatchCostHook
from .backend import LayerMap, lower_annotated_cell_drc_complete
from .cell_templates.diff_pair import build_diff_pair
from .parasitic import extract_parasitic, LayerParasiticParams, LogicalLayer


class ParasiticAwareMismatchCostHook:
    """Sizing cost hook combining Phase 1 mismatch + Phase 7 parasitic.

    Args:
        lambda_parasitic_fF: weight on parasitic_fF term (dB per fF).
            Setting to 0.0 reduces to DefaultMismatchCostHook behavior.
        pdk_layers: required for parasitic extraction
        fin_pitch_nm: PDK fin pitch (e.g. 48 for sky130)
        parasitic_params_override: optional override of per-layer parasitic
            constants; None auto-looks up
    """

    def __init__(
        self,
        lambda_parasitic_fF: float,
        pdk_layers: LayerMap,
        fin_pitch_nm: float,
        parasitic_params_override: Optional[Dict[LogicalLayer, LayerParasiticParams]] = None,
    ):
        if lambda_parasitic_fF < 0.0:
            raise ValueError(
                f"lambda_parasitic_fF must be non-negative, got {lambda_parasitic_fF}"
            )
        self.lambda_parasitic_fF = lambda_parasitic_fF
        self.pdk_layers = pdk_layers
        self.fin_pitch_nm = fin_pitch_nm
        self.parasitic_params_override = parasitic_params_override
        self._mismatch_hook = DefaultMismatchCostHook()

    def mismatch_cost(
        self,
        layout: SymbolicLayout,
        w_um: float,
        l_um: float,
        pdk: PdkDims,
        device: DeviceType,
        target_cmrr_db: float,
    ) -> float:
        """Combined mismatch + parasitic cost.

        Reuses DefaultMismatchCostHook for the σ → CMRR degradation portion
        + adds lambda_parasitic_fF * parasitic_fF.
        """
        base = self._mismatch_hook.mismatch_cost(
            layout, w_um, l_um, pdk, device, target_cmrr_db
        )
        if self.lambda_parasitic_fF == 0.0:
            return base

        parasitic_fF = self._estimate_parasitic_fF(
            layout.finger_count, w_um, l_um,
        )
        return base + self.lambda_parasitic_fF * parasitic_fF

    def _estimate_parasitic_fF(
        self,
        finger_count: int,
        w_um: float,
        l_um: float,
    ) -> float:
        """Extract parasitic for a synthetic diff pair at (W, L, finger_count)."""
        cell, sizing = build_diff_pair(
            f"_parasitic_hook_w{w_um}_l{l_um}_n{finger_count}",
            w_um=w_um, l_um=l_um, finger_count=finger_count,
        )
        try:
            lib = lower_annotated_cell_drc_complete(
                cell, self.pdk_layers, self.fin_pitch_nm, sizing,
            )
        except Exception:
            return 0.0  # graceful fallback if lowering fails
        report = extract_parasitic(
            lib, self.pdk_layers,
            parasitic_params=self.parasitic_params_override,
        )
        # Use only the top cell parasitic (avoid double-counting per-device cells)
        top_cell = next(
            (c for c in report.cells if c.cell_name.startswith("_parasitic_hook")),
            None,
        )
        if top_cell is None:
            return 0.0
        return top_cell.total_cap_aF / 1000.0  # aF → fF
