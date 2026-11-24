"""Phase 11 - Layout-aware netlist back-annotation.

Threads Phase 4 (cosizing) / Phase 7.2 (parasitic Pareto) selected
optima back into SPICE-style device parameter strings. Used when
downstream consumers (sizing algorithms, schematic exporters) want
to apply layout-aware (W, L, finger_count) decisions to simulation
netlists.

Supports ngspice / hspice / spectre flavors of W/L/M syntax.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .cosizing import CosizingPoint, select_optimal


@dataclass(frozen=True, eq=True)
class NetlistDeviceParams:
    """SPICE-flavored device parameters."""
    w_um: float
    l_um: float
    multiplier: int
    name: str


def cosizing_point_to_netlist_params(
    cp: CosizingPoint, name: str,
) -> NetlistDeviceParams:
    """Convert CosizingPoint (W, L, finger_count) to NetlistDeviceParams.

    finger_count maps to SPICE M multiplier directly.
    """
    return NetlistDeviceParams(
        w_um=cp.w_um, l_um=cp.l_um,
        multiplier=cp.finger_count, name=name,
    )


def format_spice_param_string(
    params: NetlistDeviceParams,
    *,
    style: str = "ngspice",
) -> str:
    """Format as SPICE parameter string (without leading device name)."""
    style_lower = style.lower()
    if style_lower == "ngspice":
        # ngspice: W=4u L=0.5u M=4
        return f"W={params.w_um}u L={params.l_um}u M={params.multiplier}"
    if style_lower == "hspice":
        # hspice: W=4e-6 L=5e-7 M=4 (SI units)
        w_m = params.w_um * 1e-6
        l_m = params.l_um * 1e-6
        return f"W={w_m:.3e} L={l_m:.3e} M={params.multiplier}"
    if style_lower == "spectre":
        # spectre: w=4u l=0.5u m=4 (lowercase, similar to ngspice)
        return f"w={params.w_um}u l={params.l_um}u m={params.multiplier}"
    raise ValueError(f"unknown SPICE style: {style!r} (expected ngspice/hspice/spectre)")


def back_annotate_pareto_choice(
    pareto: List[CosizingPoint],
    cost_fn: Callable[[CosizingPoint], float],
    device_name: str,
) -> Optional[NetlistDeviceParams]:
    """Pick the cost-optimal Pareto point + convert to NetlistDeviceParams."""
    best = select_optimal(pareto, cost_fn)
    if best is None:
        return None
    return cosizing_point_to_netlist_params(best, device_name)


def batch_back_annotate(
    selections: Dict[str, CosizingPoint],
) -> List[NetlistDeviceParams]:
    """Convert {device_name: CosizingPoint} to a list of NetlistDeviceParams.

    Order preserved by sorted device_name (deterministic).
    """
    return [
        cosizing_point_to_netlist_params(cp, name)
        for name, cp in sorted(selections.items())
    ]


def format_spice_device_lines(
    params_list: List[NetlistDeviceParams],
    *,
    style: str = "ngspice",
) -> List[str]:
    """Render SPICE device declaration lines (sizing only, no connectivity).

    Caller appends node connectivity (drain/gate/source/body) + device
    model name to construct full netlist line.

    Example output: ['M1 W=4u L=0.5u M=4', 'M_in_p W=2u L=0.5u M=8']
    """
    return [
        f"{p.name} {format_spice_param_string(p, style=style)}"
        for p in params_list
    ]
