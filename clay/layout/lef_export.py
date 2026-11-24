"""Minimal LEF (Library Exchange Format) text exporter for ParameterizedLibrary.

Renders ``NamedCell`` instances from a :class:`ParameterizedLibrary` as LEF
MACRO blocks with SIZE derived from the bounding box of all
:class:`LayerPolygon` geometry plus optional PIN blocks sourced from a
:class:`~clay.layout.cell_ports.CellPortSet`.

Coordinate units: ``LayerPolygon`` vertices are stored as integer nanometers
(see ``ParameterizedLibrary.unit_db_meters = 1e-9`` and ``_nm_to_dbu`` in
``clay.layout.backend``). LEF SIZE/ORIGIN are expressed in micrometers, so
:func:`cell_bounding_box_um` divides the internal nm values by 1000.0.

This is a deliberately minimal LEF subset: header, units block, one MACRO
per cell with CLASS / ORIGIN / SIZE / SYMMETRY / SITE / PIN / END LIBRARY.
It is enough for tools that only need cell footprint + port direction
metadata (e.g. abstract floorplanners, place-and-route bootstrap, LEF
diffing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .backend import NamedCell, ParameterizedLibrary
from .cell_ports import CellPort, CellPortSet


__all__ = [
    "LefExportConfig",
    "cell_bounding_box_um",
    "export_cell_to_lef",
    "export_library_to_lef",
    "write_library_to_lef",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LefExportConfig:
    """Knobs for LEF rendering."""

    site_name: str = "CoreSite"
    database_units: int = 1000  # microns precision: 1000 = 1 nm per DBU
    include_pins: bool = True


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


# Sanity check: backend ``Point`` stores integer nanometers (see
# ``ParameterizedLibrary.unit_db_meters = 1e-9`` and ``_nm_to_dbu``). We
# divide by this scale to produce microns for LEF output.
_NM_PER_UM = 1000.0


def cell_bounding_box_um(cell: NamedCell) -> Tuple[float, float, float, float]:
    """Return ``(min_x, min_y, max_x, max_y)`` of the union of all
    :class:`LayerPolygon` geometry in ``cell`` expressed in micrometers.

    The backend stores vertex coordinates as integer nanometers, so this
    helper divides by 1000.0 on the way out. An empty cell (no polygons or
    only empty polygons) returns ``(0.0, 0.0, 0.0, 0.0)``.
    """
    xs: List[int] = []
    ys: List[int] = []
    for lp in cell.polygons:
        for v in lp.polygon.vertices:
            xs.append(v.x)
            ys.append(v.y)
    if not xs or not ys:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(xs) / _NM_PER_UM,
        min(ys) / _NM_PER_UM,
        max(xs) / _NM_PER_UM,
        max(ys) / _NM_PER_UM,
    )


# ---------------------------------------------------------------------------
# Port direction mapping
# ---------------------------------------------------------------------------


def _lef_direction(port: CellPort) -> str:
    """Map :class:`CellPort.direction` to a LEF DIRECTION keyword.

    ``input``  -> ``INPUT``,
    ``output`` -> ``OUTPUT``,
    ``power``  -> ``INOUT`` (LEF lacks a POWER direction in this subset),
    ``ground`` -> ``INOUT``.
    """
    return {
        "input": "INPUT",
        "output": "OUTPUT",
        "power": "INOUT",
        "ground": "INOUT",
    }[port.direction]


def _format_um(value: float) -> str:
    """Format a micrometer value with up to 6 decimals, trimming zeros."""
    # ``g`` strips trailing zeros and renders ``0`` as ``0`` (not ``0.000000``)
    return format(round(value, 6), "g")


# ---------------------------------------------------------------------------
# Cell + library rendering
# ---------------------------------------------------------------------------


def _render_pin_block(port: CellPort) -> List[str]:
    name = port.name
    return [
        f"  PIN {name}",
        f"    DIRECTION {_lef_direction(port)} ;",
        "    USE SIGNAL ;",
        f"  END {name}",
    ]


def export_cell_to_lef(
    cell: NamedCell,
    port_set: Optional[CellPortSet] = None,
    config: LefExportConfig = LefExportConfig(),
) -> str:
    """Render a single :class:`NamedCell` as a LEF ``MACRO`` block string.

    Includes ``PIN`` blocks for every port in ``port_set`` when
    ``config.include_pins`` is true and ``port_set`` is not ``None``. The
    returned string ends with a trailing newline.
    """
    min_x, min_y, max_x, max_y = cell_bounding_box_um(cell)
    w = max_x - min_x
    h = max_y - min_y

    lines: List[str] = [
        f"MACRO {cell.name}",
        "  CLASS CORE ;",
        "  ORIGIN 0 0 ;",
        f"  SIZE {_format_um(w)} BY {_format_um(h)} ;",
        "  SYMMETRY X Y ;",
        f"  SITE {config.site_name} ;",
    ]

    if config.include_pins and port_set is not None:
        for port in port_set.ports:
            lines.extend(_render_pin_block(port))

    lines.append(f"END {cell.name}")
    return "\n".join(lines) + "\n"


def _render_header(config: LefExportConfig) -> List[str]:
    return [
        "VERSION 5.7 ;",
        "NAMESCASESENSITIVE ON ;",
        'BUSBITCHARS "[]" ;',
        'DIVIDERCHAR "/" ;',
        "UNITS",
        f"  DATABASE MICRONS {config.database_units} ;",
        "END UNITS",
        "",
    ]


def export_library_to_lef(
    lib: ParameterizedLibrary,
    port_sets: Optional[Dict[str, CellPortSet]] = None,
    config: LefExportConfig = LefExportConfig(),
) -> str:
    """Render a full :class:`ParameterizedLibrary` as LEF text.

    ``port_sets`` maps ``cell.name`` to a :class:`CellPortSet`. Cells absent
    from the dict (or all cells when ``port_sets`` is ``None``) render with
    no ``PIN`` blocks. The output is terminated with ``END LIBRARY``.
    """
    port_sets = port_sets or {}
    lines: List[str] = list(_render_header(config))
    for cell in lib.cells:
        block = export_cell_to_lef(cell, port_sets.get(cell.name), config)
        # ``export_cell_to_lef`` returns a block ending in '\n'; split it
        # back into lines and append a blank separator between MACRO blocks
        # for readability.
        lines.extend(block.rstrip("\n").splitlines())
        lines.append("")
    lines.append("END LIBRARY")
    return "\n".join(lines) + "\n"


def write_library_to_lef(
    lib: ParameterizedLibrary,
    output_path: str,
    port_sets: Optional[Dict[str, CellPortSet]] = None,
    config: LefExportConfig = LefExportConfig(),
) -> None:
    """Write the LEF text for ``lib`` to ``output_path`` (UTF-8)."""
    text = export_library_to_lef(lib, port_sets, config)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
