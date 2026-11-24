"""The canonical AsBuiltBundle: the ONE layout artifact every checker consumes.

Production routing (:func:`clay.layout.route.anchor_ota.route_amp_leakfree`) returns this
exact object so the LVS gate, the electrical gate, and the P6 reconcile all verify the
SAME geometry the route built -- no re-derived, path-drifted copy. It holds the LIVE route
objects (not a serialised snapshot): the routed result, the device geometry, the
placement, the per-net pins, the routing obstacles, the resolved tech, and the intent.

Naming it (codex landing step 2) turns the transitional ``_bundle`` dict into a
first-class typed artifact in the Intent->Obligation->Evidence->Verdict chain: checkers
reference ``bundle.routed`` / ``bundle.pins_by_net`` rather than ad-hoc dict keys.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AsBuiltBundle:
    """The live as-built layout surface shared by all checkers."""

    routed: Any                       # the router's routed result (nets -> wires)
    geometry: Dict[str, Any]          # device-cell geometry by device name
    placement: Dict[str, Any]         # device placement (um) by device name
    pins_by_net: Dict[str, List[Any]] # per-net routing pins (access points)
    obstacles: Any                    # routing obstacles consumed by the engine
    tech: Any                         # resolved LayoutTech
    intent: Any                       # the design intent the layout realises
    layout_model: Optional[Any] = None  # the unified net-tagged LayoutModel (rule 6 source)
