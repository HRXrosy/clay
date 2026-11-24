"""Parser: Magic ``ext2spice`` SPICE output -> backend-neutral PexReport.

Pure function (no I/O), fully testable against the committed golden fixture
``tests/fixtures/pex/sample_nmos_3p3.pex.spice``. It records every two-terminal
``C`` line as a faithful node-to-node :class:`PexCap` and every ``R`` line as a
:class:`PexRes`. Active devices (subckt instances ``X``, and ``M/Q/D/J``),
comments (``*``) and directives (``.``) are ignored. There is NO substrate-node
guessing -- see ``tests/fixtures/pex/README.md``: Magic emits caps to the net the
bulk is tied to, not to a canonical ground name.
"""
from __future__ import annotations

import re

from .base import PexCap, PexRes, PexReport

# SPICE engineering suffixes -> multiplier (longest match for 'meg').
_SUFFIX = {
    "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
    "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12, "a": 1e-18,
}

_NUM = re.compile(r"^([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s*([a-zA-Z]*)$")


def _parse_spice_value(token: str) -> float:
    """Parse a SPICE numeric value with optional engineering suffix -> float."""
    token = token.strip()
    m = _NUM.match(token)
    if not m:
        raise ValueError(f"unparseable SPICE value: {token!r}")
    mant, suf = m.group(1), m.group(2).lower()
    val = float(mant)
    if not suf:
        return val
    if suf.startswith("meg"):
        return val * _SUFFIX["meg"]
    return val * _SUFFIX.get(suf[0], 1.0)


def parse_ext2spice(
    text: str,
    cell_name: str,
    pdk: str,
    *,
    calibrated: bool | None = None,
) -> PexReport:
    """Parse Magic ext2spice text into a PexReport (caps + resistors)."""
    if calibrated is None:
        calibrated = pdk in {"gf180mcu", "sky130"}
    caps = []
    resistors = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in "*.":
            continue
        head = line[0].upper()
        parts = line.split()
        if head == "C" and len(parts) >= 4:
            caps.append(PexCap(parts[1], parts[2], _parse_spice_value(parts[3]) * 1e18))
        elif head == "R" and len(parts) >= 4:
            resistors.append(PexRes(parts[1], parts[2], _parse_spice_value(parts[3])))
        # else: active device / directive -> ignore

    return PexReport(
        cell_name=cell_name,
        caps=tuple(caps),
        resistors=tuple(resistors),
        backend="magic",
        connectivity_source="magic-ext2spice",
        constants_source=f"{pdk} Magic tech-file",
        calibrated=calibrated,
        field_solver=False,
        note=(
            f"{'calibrated' if calibrated else 'uncalibrated'} 2.5D Magic "
            f"ext2spice ({len(caps)} caps, {len(resistors)} resistors); "
            "NOT field-solver / NOT StarRC"
        ),
    )
