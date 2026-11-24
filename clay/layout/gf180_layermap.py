"""Real gf180mcu layer map (R139, BL-139C).

Unlike the APPROXIMATE ``Gf180mcuLayerMap`` in ``clay/layout/backend.py``,
this module derives every (layer, datatype) pair from the **actual** gf180mcu
PDK KLayout layer-properties file:

    AnalogGym/PDK/gf180mcu_pdk/libraries/gf180mcu_fd_pr/latest/tech/klayout/gf180mcu.lyp

In that XML each physical layer is a ``<properties>`` element whose
``<source>`` text has the form ``[NAME ]LAYER/DATATYPE[@COMPONENT]`` (the
``<name/>`` element itself is empty in this PDK, so the human-readable layer
name lives in the ``<source>`` prefix). For example::

    <source>COMP 22/0@1</source>     -> name=COMP,  layer=22, datatype=0
    <source>Nwell 21/0@1</source>    -> name=Nwell, layer=21, datatype=0
    <source>Poly2 30/0@1</source>    -> name=Poly2, layer=30, datatype=0
    <source>Contact 33/0@1</source>  -> name=Contact, layer=33, datatype=0
    <source>Metal1 34/0@1</source>   -> name=Metal1, layer=34, datatype=0
    <source>Via1 35/0@1</source>     -> name=Via1, layer=35, datatype=0
    <source>Metal2 36/0@1</source>   -> name=Metal2, layer=36, datatype=0
    <source>Via2 38/0@1</source>     -> name=Via2, layer=38, datatype=0
    <source>Metal3 42/0@1</source>   -> name=Metal3, layer=42, datatype=0

``Gf180RealLayerMap`` parses the .lyp at construction time (preferred, so the
mapping stays traceable to the PDK) and exposes the same accessor as
``Sky130LayerMap`` / ``Gf180mcuLayerMap``: ``gds_layer(LogicalLayer) ->
Optional[GdsLayer]``. It is therefore drop-in compatible with the backend
``LayerMap`` Protocol and can be passed to
``lower_annotated_cell_to_parameterized`` and friends.

The verified (layer, datatype) values (gf180mcu.lyp line numbers cited) are
also recorded in ``GF180_VERIFIED_SOURCES`` so tests can assert against the
PDK source even on systems where the .lyp file is not present.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Tuple
from xml.etree import ElementTree as ET

from .backend import GdsLayer, LogicalLayer


# Default location of the real gf180mcu KLayout layer-properties file,
# resolved relative to the repository root (this file lives at
# clay/layout/gf180_layermap.py, so the repo root is three levels up).
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
DEFAULT_LYP_PATH = os.path.join(
    _REPO_ROOT,
    "AnalogGym",
    "PDK",
    "gf180mcu_pdk",
    "libraries",
    "gf180mcu_fd_pr",
    "latest",
    "tech",
    "klayout",
    "gf180mcu.lyp",
)


# Verified (layer, datatype) values extracted from gf180mcu.lyp, with the
# source line number in that file. These are the GROUND TRUTH — tests compare
# the parsed map against these, and they let the module fall back to a
# documented hardcoded table if the .lyp is unavailable. Each entry:
#   gf180 layer name -> (layer, datatype, lyp_line_number, source_text)
# Source: gf180mcu.lyp (AnalogGym/PDK/.../tech/klayout/gf180mcu.lyp).
GF180_VERIFIED_SOURCES: Dict[str, Tuple[int, int, int, str]] = {
    "COMP": (22, 0, 205, "COMP 22/0@1"),  # diffusion / OD (active area)
    "DNWELL": (12, 0, 222, "DNWELL 12/0@1"),  # deep nwell
    "Nwell": (21, 0, 239, "Nwell 21/0@1"),
    "LVPWELL": (204, 0, 256, "LVPWELL 204/0@1"),  # low-voltage p-well
    "Poly2": (30, 0, 290, "Poly2 30/0@1"),  # gate poly
    "Nplus": (32, 0, 307, "Nplus 32/0@1"),  # n+ implant
    "Pplus": (31, 0, 324, "Pplus 31/0@1"),  # p+ implant
    "Contact": (33, 0, 375, "Contact 33/0@1"),  # diff/poly -> metal1 contact
    "Metal1": (34, 0, 392, "Metal1 34/0@1"),
    "Via1": (35, 0, 409, "Via1 35/0@1"),
    "Metal2": (36, 0, 426, "Metal2 36/0@1"),
    "Via2": (38, 0, 443, "Via2 38/0@1"),
    "Metal3": (42, 0, 460, "Metal3 42/0@1"),
    "Via3": (40, 0, 477, "Via3 40/0@1"),
    "Metal4": (46, 0, 494, "Metal4 46/0@1"),
    "Via4": (41, 0, 511, "Via4 41/0@1"),
    "Metal5": (81, 0, 528, "Metal5 81/0@1"),
}


# How each Clay LogicalLayer maps onto a real gf180mcu .lyp layer NAME.
#
# In gf180mcu there is ONE diffusion/active layer, COMP (22/0); n-channel and
# p-channel devices are distinguished by the Nplus (32/0) / Pplus (31/0)
# implant layers drawn over COMP, not by separate diffusion GDS layers. The
# Clay logical model has separate NDIFF/PDIFF, so both map to COMP — matching
# how the approximate Gf180mcuLayerMap also maps both to the diffusion layer.
# (The implant distinction lives in dedicated logical-layer-free geometry and
# is not part of the LogicalLayer enum.)
#
# LogicalLayer.PWELL_MARKER -> None: gf180mcu's substrate is the default
# p-type bulk, so an explicit drawn p-well marker is not required for the
# baseline (non-deep-nwell) flow. LVPWELL (204/0) exists but is only used
# inside a DNWELL isolated region, which is not what Clay's PWELL_MARKER
# guard-ring annotation targets. Documented as None, same as the approximate
# map and the sky130 map.
_LOGICAL_TO_GF180_NAME: Dict[LogicalLayer, Optional[str]] = {
    LogicalLayer.NDIFF: "COMP",
    LogicalLayer.PDIFF: "COMP",
    LogicalLayer.POLY: "Poly2",
    LogicalLayer.CONTACT: "Contact",
    LogicalLayer.METAL1: "Metal1",
    LogicalLayer.NWELL: "Nwell",
    LogicalLayer.PWELL_MARKER: None,
    # P0 implant + well-isolation members, traced to the .lyp NAMES (the verified
    # GF180_VERIFIED_SOURCES table): Nplus 32/0, Pplus 31/0, LVPWELL 204/0,
    # DNWELL 12/0. The drawn p-well (PWELL) is gf180's LVPWELL, used only inside a
    # deep-nwell region; it resolves here so an implant-aware builder can request
    # it explicitly.
    LogicalLayer.NPLUS: "Nplus",
    LogicalLayer.PPLUS: "Pplus",
    LogicalLayer.PWELL: "LVPWELL",
    LogicalLayer.DNWELL: "DNWELL",
}


# Matches a KLayout <source> body: optional "name " prefix, then layer/datatype,
# then optional "@component". Examples: "COMP 22/0@1", "5/222@1", "Nwell 21/0".
_SOURCE_RE = re.compile(
    r"^\s*(?:(?P<name>\S+)\s+)?(?P<layer>\d+)/(?P<datatype>\d+)(?:@\d+)?\s*$"
)


def parse_lyp_sources(lyp_path: str) -> Dict[str, Tuple[int, int]]:
    """Parse a KLayout .lyp file into ``{layer_name: (layer, datatype)}``.

    Only ``<properties>`` whose ``<source>`` carries a non-empty name prefix
    are recorded (anonymous decorative entries like ``5/222@1`` are skipped
    because they have no usable logical identity). If a name appears more than
    once, the FIRST occurrence wins (the canonical drawing layer in this PDK
    precedes any later marker/derived reuse).

    Raises ``FileNotFoundError`` if the file is missing and
    ``ValueError`` on malformed XML.
    """
    try:
        tree = ET.parse(lyp_path)
    except ET.ParseError as exc:  # pragma: no cover - defensive
        raise ValueError(f"malformed .lyp XML: {exc}") from exc
    root = tree.getroot()

    out: Dict[str, Tuple[int, int]] = {}
    for props in root.iter("properties"):
        src_el = props.find("source")
        if src_el is None or src_el.text is None:
            continue
        m = _SOURCE_RE.match(src_el.text)
        if m is None:
            continue
        name = m.group("name")
        if not name:
            continue
        layer = int(m.group("layer"))
        datatype = int(m.group("datatype"))
        out.setdefault(name, (layer, datatype))
    return out


class Gf180RealLayerMap:
    """Real gf180mcu LayerMap, derived from the actual PDK gf180mcu.lyp.

    Interface-compatible with ``backend.LayerMap`` (and with
    ``Sky130LayerMap`` / ``Gf180mcuLayerMap``): exposes
    ``gds_layer(LogicalLayer) -> Optional[GdsLayer]``.

    By default it parses the PDK .lyp at construction so the mapping is
    traceable to source. If the .lyp cannot be read (file missing on a given
    system) it falls back to the verified hardcoded ``GF180_VERIFIED_SOURCES``
    table, which itself cites the .lyp line numbers.

    Pass ``lyp_path=None`` to force the hardcoded table without touching disk.
    """

    def __init__(self, lyp_path: Optional[str] = DEFAULT_LYP_PATH) -> None:
        self.lyp_path = lyp_path
        self.source: str
        sources: Dict[str, Tuple[int, int]]
        if lyp_path is not None and os.path.isfile(lyp_path):
            sources = parse_lyp_sources(lyp_path)
            self.source = f"parsed:{lyp_path}"
        else:
            sources = {
                name: (layer, dt)
                for name, (layer, dt, _line, _txt) in GF180_VERIFIED_SOURCES.items()
            }
            self.source = "hardcoded:GF180_VERIFIED_SOURCES (gf180mcu.lyp)"

        self._lyp_sources = sources

        table: Dict[LogicalLayer, Optional[GdsLayer]] = {}
        for logical, gf_name in _LOGICAL_TO_GF180_NAME.items():
            if gf_name is None:
                table[logical] = None
                continue
            if gf_name not in sources:
                raise ValueError(
                    f"gf180mcu layer {gf_name!r} (for {logical}) not found in "
                    f"{self.source}"
                )
            layer, datatype = sources[gf_name]
            table[logical] = GdsLayer(layer, datatype)
        self._table = table

    def gds_layer(self, logical: LogicalLayer) -> Optional[GdsLayer]:
        return self._table.get(logical)

    def gf180_layer(self, gf_name: str) -> Optional[GdsLayer]:
        """Accessor by raw gf180mcu layer NAME (e.g. ``"Nplus"``, ``"Via1"``).

        Returns the GdsLayer for any named layer present in the parsed .lyp /
        verified table, or None if that name is absent. Useful for the implant
        and upper-metal/via layers that have no LogicalLayer enum member.
        """
        v = self._lyp_sources.get(gf_name)
        return GdsLayer(*v) if v is not None else None
