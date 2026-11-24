"""Inject a PexReport's parasitics into a testbench as IR-native devices.

Mirrors clay.selfboot.pex_aware_flywheel._inject_output_parasitic: uses the
same ``Device(device_class="cap"/"res", pins={"p","n"}, sizing={"value": ...})``
shape the SPICE emitter already handles. Every PexCap becomes a two-terminal cap
(aF -> Farads); every PexRes becomes a two-terminal resistor (ohm). The flat
node-to-node model means no substrate/coupling special-casing -- a cap to a
supply net is simply a cap to that node. Never mutates the input testbench.
"""
from __future__ import annotations

import copy
from typing import Iterable, Optional

from .base import PexReport


def inject_pex_into_testbench(
    tb, report: PexReport, skip_nets: Optional[Iterable[str]] = None
):
    """Return a DEEP COPY of ``tb`` with the report's caps + resistors added.

    ``skip_nets``: any parasitic element touching one of these nets is skipped
    (e.g. a supply already modelled ideally in the deck). aF values are converted
    to Farads (1 aF = 1e-18 F).
    """
    from clay.ir import Device

    skip = set(skip_nets or ())
    tb2 = copy.deepcopy(tb)

    for i, c in enumerate(report.caps):
        if c.cap_aF <= 0.0 or c.a in skip or c.b in skip:
            continue
        tb2.extra_devices.append(Device(
            instance_name=f"Cpar_{i}_{c.a}_{c.b}",
            device_class="cap",
            pdk_variant=None,
            pins={"p": c.a, "n": c.b},
            sizing={"value": c.cap_aF * 1e-18},
        ))

    for i, r in enumerate(report.resistors):
        if r.res_ohm <= 0.0 or r.a in skip or r.b in skip:
            continue
        tb2.extra_devices.append(Device(
            instance_name=f"Rpar_{i}_{r.a}_{r.b}",
            device_class="res",
            pdk_variant=None,
            pins={"p": r.a, "n": r.b},
            sizing={"value": r.res_ohm},
        ))
    return tb2
