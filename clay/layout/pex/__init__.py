"""Clay PEX backends: a replaceable parasitic-extraction abstraction.

``get_backend("auto")`` prefers the calibrated Magic backend when available and
falls back to the always-available geometric estimator otherwise.
"""
from __future__ import annotations

from .base import PexBackend, PexCap, PexRes, PexReport
from .geometric_backend import GeometricPexBackend
from .klayout_backend import KLayoutPexBackend
from .magic_backend import MagicPexBackend

__all__ = [
    "PexCap", "PexRes", "PexReport", "PexBackend",
    "GeometricPexBackend", "KLayoutPexBackend", "MagicPexBackend", "get_backend",
]


def get_backend(
    kind: str = "auto", *, pdk_id: str | None = None, ground_net: str = "0"
) -> PexBackend:
    if kind == "geometric":
        return GeometricPexBackend(ground_net=ground_net, pdk_id=pdk_id)
    if kind == "klayout":
        return KLayoutPexBackend()
    if kind == "magic":
        return MagicPexBackend()
    if kind == "auto":
        magic = MagicPexBackend()
        supported = pdk_id is None or magic.supports_pdk(pdk_id)
        return (
            magic
            if magic.available() and supported
            else GeometricPexBackend(ground_net=ground_net, pdk_id=pdk_id)
        )
    raise ValueError(f"unknown PEX backend kind: {kind!r}")
