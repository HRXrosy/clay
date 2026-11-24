"""Per-kind schematic symbol geometry: bounding box + pin offsets.

Single source of truth for how big each device symbol is and where each of its
terminals sits relative to the symbol CENTER. The viewer draws symbol leads from
these same offsets, so emitting absolute pin coords as ``center + offset`` makes
the symbol leads meet the routed wires exactly.

Everything is in integer schematic units on a shared grid (:data:`GRID`):

* ``box`` = ``(w, h)`` of the symbol's bounding box, centred on the device.
* ``pins`` = ``{terminal_name: (dx, dy)}`` offset of that terminal from the
  CENTER. ``+y`` is DOWN (SVG convention), so ``-dy`` is toward the top.

MOSFET convention (``nmos`` / ``pmos``): the BASE table draws drain on TOP,
source on BOTTOM, gate on the LEFT, body on the RIGHT. A device's ``flip`` is a
REAL vertical mirror -- :func:`pin_offsets` negates the pin ``dy`` when
``flip=True``, swapping which of drain / source sits at the top vs bottom pin.
The supply-side convention (see :func:`default_flip`) flips PMOS so its source
faces the vdd rail (source up) and keeps NMOS source-down toward gnd.
"""
from __future__ import annotations

from typing import Dict, Tuple

# Shared grid. Every device center and every pin coordinate is a multiple of it.
GRID = 20

SymbolSpec = Dict[str, object]

# kind -> {"box": (w, h), "pins": {term: (dx, dy)}}
SYMBOLS: Dict[str, SymbolSpec] = {
    "nmos": {
        "box": (64, 88),
        "pins": {
            "drain": (20, -40),   # TOP
            "source": (20, 40),   # BOTTOM
            "gate": (-40, 0),     # LEFT
            "body": (40, 0),      # RIGHT
        },
    },
    "pmos": {
        "box": (64, 88),
        "pins": {
            "drain": (20, -40),   # TOP
            "source": (20, 40),   # BOTTOM
            "gate": (-40, 0),     # LEFT
            "body": (40, 0),      # RIGHT
        },
    },
    "cap": {
        "box": (48, 56),
        "pins": {
            "p": (0, -40),        # TOP
            "n": (0, 40),         # BOTTOM
        },
    },
    "res": {
        "box": (40, 64),
        "pins": {
            "a": (0, -40),        # TOP
            "p": (0, -40),        # TOP (alias)
            "b": (0, 40),         # BOTTOM
            "n": (0, 40),         # BOTTOM (alias)
        },
    },
}


def snap(v: int | float) -> int:
    """Snap a coordinate to the nearest :data:`GRID` multiple (integer)."""
    return int(round(v / GRID)) * GRID


def symbol_box(kind: str) -> Tuple[int, int]:
    """``(w, h)`` bounding box for ``kind`` (falls back to the nmos box)."""
    spec = SYMBOLS.get(kind, SYMBOLS["nmos"])
    w, h = spec["box"]  # type: ignore[misc]
    return int(w), int(h)


def pin_offsets(kind: str, flip: bool = False) -> Dict[str, Tuple[int, int]]:
    """``{term: (dx, dy)}`` offsets for ``kind`` (falls back to the nmos pins).

    ``flip`` is a VERTICAL mirror of the symbol: it negates the ``dy`` of every
    pin, which SWAPS which of drain / source sits at the TOP vs BOTTOM (the
    horizontal pins gate / body keep their dy=0, so they are unaffected). This is
    the real orientation degree-of-freedom -- the per-device ``flip`` choice
    drives the actual pin Y, not just a render hint.

    Convention (default / anchor): a MOS's SOURCE should sit toward its OWN
    supply rail. The base table draws drain on TOP, source on BOTTOM, so:

    * PMOS (upper band, vdd rail above) -> ``flip=True`` puts SOURCE UP toward
      vdd and DRAIN DOWN into the signal channel; its body/bulk tie to vdd is
      then a short source-side lead.
    * NMOS (lower band, gnd rail below) -> ``flip=False`` keeps SOURCE DOWN
      toward gnd and DRAIN UP into the channel.
    """
    spec = SYMBOLS.get(kind, SYMBOLS["nmos"])
    pins = dict(spec["pins"])  # type: ignore[arg-type]
    if not flip:
        return pins
    return {term: (dx, -dy) for term, (dx, dy) in pins.items()}


def default_flip(kind: str) -> bool:
    """The supply-side-convention default orientation for a kind: PMOS flips
    (source up toward vdd), NMOS does not (source down toward gnd)."""
    return kind == "pmos"


__all__ = ["GRID", "SYMBOLS", "snap", "symbol_box", "pin_offsets", "default_flip"]
