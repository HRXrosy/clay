"""Sizing-cost hook protocol for Clay Phase 1 layout-as-co-variable.

The hook lets the sizer ask "given this symbolic layout, this PDK, this
device flavor and a target CMRR, what's the achievable CMRR after
factoring in Vth mismatch?". Concrete hooks can plug in measured silicon
data or richer device models without changing the sizer contract.
"""

from typing import Protocol, runtime_checkable

from .mismatch import diff_pair_mismatch_mv
from .pdk_dims import DeviceType, PdkDims
from .symbolic_layout import SymbolicLayout


@runtime_checkable
class LayoutCostHook(Protocol):
    """Structural protocol for layout-aware sizing cost contributions."""

    def mismatch_cost(
        self,
        layout: SymbolicLayout,
        w_um: float,
        l_um: float,
        pdk: PdkDims,
        device: DeviceType,
        target_cmrr_db: float,
    ) -> float:
        ...


class DefaultMismatchCostHook:
    """Reference Phase 1 hook: linear CMRR derate from Pelgrom sigma.

    Uses a typical 100 mV input swing scale and converts the mV-domain
    sigma into a dB degradation, then subtracts from the target CMRR.
    The returned value is clamped at zero so the cost stays well-defined
    even for absurdly tiny devices.
    """

    def mismatch_cost(
        self,
        layout: SymbolicLayout,
        w_um: float,
        l_um: float,
        pdk: PdkDims,
        device: DeviceType,
        target_cmrr_db: float,
    ) -> float:
        sigma_mv = diff_pair_mismatch_mv(layout, w_um, l_um, pdk, device)
        v_swing_typ_mv = 100.0
        cmrr_degradation_db = 20.0 * (sigma_mv / v_swing_typ_mv)
        return max(0.0, target_cmrr_db - cmrr_degradation_db)
