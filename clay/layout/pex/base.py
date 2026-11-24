"""Backend-neutral PEX data model and the PexBackend protocol.

The report is a faithful flat list of two-terminal parasitic elements
(``PexCap`` / ``PexRes``), exactly the shape Magic ``ext2spice`` emits: caps and
resistors between named nodes, with NO canonical ground/substrate node (the
substrate cap simply lands on whatever net the bulk is tied to). Callers that
want the coupling subset pass their known ground/supply nets to
:meth:`PexReport.coupling_caps`.

``field_solver`` is False for every current backend (Clay has no 3-D field
solver). ``calibrated`` distinguishes the geometric estimate (False) from a
PDK-tech-file-calibrated Magic extraction (True). Magic is calibrated 2.5D
analytic extraction, NOT StarRC.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Optional, Protocol, Tuple, runtime_checkable


@dataclass(frozen=True)
class PexCap:
    """One two-terminal parasitic capacitor between nodes ``a`` and ``b`` (aF)."""

    a: str
    b: str
    cap_aF: float


@dataclass(frozen=True)
class PexRes:
    """One two-terminal parasitic resistor between nodes ``a`` and ``b`` (ohm)."""

    a: str
    b: str
    res_ohm: float


@dataclass(frozen=True)
class PexReport:
    """A cell's parasitics from one backend, as a flat element list."""

    cell_name: str
    caps: Tuple[PexCap, ...]
    resistors: Tuple[PexRes, ...]
    backend: str
    connectivity_source: str
    constants_source: str
    calibrated: bool
    field_solver: bool = False
    note: str = ""

    @property
    def total_cap_aF(self) -> float:
        return sum(c.cap_aF for c in self.caps)

    @property
    def total_cap_fF(self) -> float:
        return self.total_cap_aF / 1000.0

    def nets(self) -> FrozenSet[str]:
        out = set()
        for c in self.caps:
            out.add(c.a)
            out.add(c.b)
        for r in self.resistors:
            out.add(r.a)
            out.add(r.b)
        return frozenset(out)

    def net_total_cap_aF(self, net: str) -> float:
        return sum(c.cap_aF for c in self.caps if net in (c.a, c.b))

    def coupling_caps(self, ground_nets: Iterable[str]) -> Tuple[PexCap, ...]:
        """Caps where NEITHER terminal is a ground/supply net (signal coupling)."""
        g = set(ground_nets)
        return tuple(c for c in self.caps if c.a not in g and c.b not in g)


@runtime_checkable
class PexBackend(Protocol):
    """A replaceable parasitic-extraction backend."""

    name: str

    def available(self) -> bool:
        ...

    def extract(self, gds_path: str, cell_name: str, pdk: str) -> PexReport:
        ...
