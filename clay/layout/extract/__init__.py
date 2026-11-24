"""Consistent intent extractor: the dual of the placer.

Mechanically derives declared layout intent FROM a reference design so the
round-trip ``reference -> extract intent -> generic place -> fidelity`` can be
measured with no human eyeballing.

This package exposes the shared contract (``types``) plus the netlist half
(``netlist_intent``). The geometric half and emit/roundtrip live in sibling
modules.
"""

from __future__ import annotations

from . import types  # noqa: F401
from .types import ExtractedDevice, ExtractedIntent, ExtractedPassive  # noqa: F401

__all__ = [
    "types",
    "ExtractedDevice",
    "ExtractedPassive",
    "ExtractedIntent",
]

# netlist_intent imports klayout.db lazily inside correspond_to_gds, so importing
# the module here is cheap and lets callers do ``from clay.layout.extract import
# netlist_intent``.
try:  # pragma: no cover - defensive
    from . import netlist_intent  # noqa: F401

    __all__.append("netlist_intent")
except Exception:  # pragma: no cover
    pass
