"""
Per-cell supply-voltage resolution helper.

Testbench builders should not hardcode 1.8 V — the cell's `pdk_id`
determines what nominal supply ngspice should be driven at. sky130 is
1.8 V, gf180mcu is 3.3 V; future PDK adapters (open_pdk, etc.) can have
their own values.

v0.0.40: testbench `supply` parameter defaults to `None`. When the
builder body sees `supply is None`, it calls `resolve_supply(cell)` to
fetch the PDK adapter's `vdd_nom`. Callers can still pass an explicit
`supply` to override (useful for sweeping headroom or stress-testing
the cell at a non-nominal voltage).

This keeps every existing test that doesn't pass `supply=` working
unchanged on sky130 (auto-resolves to 1.8 V), and unlocks gf180mcu
e2e by giving every testbench the right rail without per-template
edits at call sites.
"""

from __future__ import annotations

from typing import Optional

from ..ir import Cell


# Default if the cell's pdk_id is missing or the PDK adapter cannot
# be loaded. Picks 1.8 V to match the existing sky130-only behavior.
_FALLBACK_VDD = 1.8


def resolve_supply(cell: Cell, default: Optional[float] = None) -> float:
    """Return the nominal supply voltage for `cell` based on its PDK.

    Looks up `cell.pdk_id` in the PDK registry. If the registered
    adapter exposes a `vdd_nom` attribute, returns that. Otherwise
    falls back to `default` (which itself defaults to 1.8 V).

    Never raises — testbench construction must be best-effort. If
    every layer of resolution fails, returns 1.8 V silently so the
    sky130 baseline keeps working in the absence of a PDK adapter
    (e.g., parser-only unit tests that build a Cell without a real
    PDK behind it).
    """
    if default is None:
        default = _FALLBACK_VDD
    pdk_id = getattr(cell, "pdk_id", None)
    if not pdk_id:
        return default
    try:
        from ..pdk import get_pdk
        adapter = get_pdk(pdk_id)
    except Exception:
        return default
    vdd = getattr(adapter, "vdd_nom", None)
    if vdd is None:
        return default
    return float(vdd)
