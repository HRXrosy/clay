"""GDSII metadata reader.

Lightweight parser that extracts top-level metadata (library name, cell
names, layer/datatype set) from a binary GDSII file by scanning only the
record headers and the LIBNAME/STRNAME/LAYER/DATATYPE record bodies. It
deliberately does NOT decode polygon coordinates (XY records), so it is
much cheaper than ``parse_binary_gds`` on large layouts.

The GDSII stream format (Calma) is a sequence of records:

    +------+------+------+------+----------------+
    | len  | len  | rtyp | dtyp |   data ...     |
    | hi   | lo   |      |      |                |
    +------+------+------+------+----------------+

* ``len`` is a 16-bit big-endian total length (header + data, always even).
* ``rtyp`` is the record type byte (HEADER=0x00, BGNLIB=0x01, ...).
* ``dtyp`` is the data type byte (we ignore it here).
* ``data`` is the record body (``len - 4`` bytes).

Records of interest for metadata extraction:

* HEADER   (0x00) — file must start with this
* BGNLIB   (0x01) — library begin
* LIBNAME  (0x02) — ASCII library name
* BGNSTR   (0x05) — structure (cell) begin
* STRNAME  (0x06) — ASCII cell name
* LAYER    (0x0D) — INT16 layer number (in BOUNDARY context)
* DATATYPE (0x0E) — INT16 datatype number
* ENDLIB   (0x04) — library end

All other records are skipped by length.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set, Tuple, Union


# --- GDSII record type constants ---
_REC_HEADER = 0x00
_REC_BGNLIB = 0x01
_REC_LIBNAME = 0x02
_REC_ENDLIB = 0x04
_REC_BGNSTR = 0x05
_REC_STRNAME = 0x06
_REC_LAYER = 0x0D
_REC_DATATYPE = 0x0E


class GdsMetadataError(Exception):
    """Raised when GDS metadata scanning fails (truncation, bad record, ...)."""


@dataclass
class GdsMetadata:
    """Top-level GDS metadata extracted without full polygon decode."""

    library_name: str
    cell_names: List[str] = field(default_factory=list)
    layer_set: Set[Tuple[int, int]] = field(default_factory=set)
    cell_count: int = 0
    file_size_bytes: int = 0


def _decode_ascii(data: bytes) -> str:
    """Decode a GDS ASCII record body: strip trailing NULs + padding."""
    # GDS ASCII records pad odd-length names with a trailing 0x00. There can
    # also be embedded NULs at the end when the writer pads to even length.
    s = data.rstrip(b"\x00").decode("ascii", errors="replace")
    return s


def read_gds_metadata(path: Union[str, Path]) -> GdsMetadata:
    """Parse binary GDS header + scan STRNAME, LAYER, DATATYPE records.

    Args:
        path: filesystem path to a binary ``.gds`` file.

    Returns:
        GdsMetadata containing library name, cell names, layer/datatype
        set, cell count, and file size in bytes.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        GdsMetadataError: on truncation, bad record length, or missing
            mandatory records (HEADER, BGNLIB, LIBNAME).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GDS file not found: {p}")

    raw = p.read_bytes()
    size = len(raw)
    # Minimum viable GDS: HEADER(6) + BGNLIB(28) + LIBNAME(>=4) + ... + ENDLIB(4)
    # Realistically anything under ~40 bytes can't be a real GDS. Enforce a
    # generous lower bound; the structured checks below will catch truncation
    # at finer granularity.
    if size < 40:
        raise GdsMetadataError(
            f"file too small to be a valid GDS ({size} bytes < 40 byte floor)"
        )

    pos = 0

    def _read_record() -> Tuple[int, int, bytes]:
        """Read one record at ``pos``; advance ``pos``; return (rtype, dtype, data)."""
        nonlocal pos
        if pos + 4 > size:
            raise GdsMetadataError(
                f"truncated record header at byte {pos} (need 4, have {size - pos})"
            )
        length = (raw[pos] << 8) | raw[pos + 1]
        if length < 4 or (length % 2) != 0:
            raise GdsMetadataError(
                f"bad record length {length} at byte {pos} (must be even, >=4)"
            )
        if pos + length > size:
            raise GdsMetadataError(
                f"truncated record body at byte {pos} "
                f"(length={length}, file remainder={size - pos})"
            )
        rtype = raw[pos + 2]
        dtype = raw[pos + 3]
        data = raw[pos + 4: pos + length]
        pos += length
        return rtype, dtype, data

    # --- HEADER ---
    rtype, _, _ = _read_record()
    if rtype != _REC_HEADER:
        raise GdsMetadataError(
            f"expected HEADER (0x00) as first record, got 0x{rtype:02X}"
        )

    # --- BGNLIB ---
    rtype, _, _ = _read_record()
    if rtype != _REC_BGNLIB:
        raise GdsMetadataError(
            f"expected BGNLIB (0x01) after HEADER, got 0x{rtype:02X}"
        )

    # --- LIBNAME ---
    rtype, _, libname_data = _read_record()
    if rtype != _REC_LIBNAME:
        raise GdsMetadataError(
            f"expected LIBNAME (0x02) after BGNLIB, got 0x{rtype:02X}"
        )
    library_name = _decode_ascii(libname_data)

    cell_names: List[str] = []
    layer_set: Set[Tuple[int, int]] = set()
    current_layer: int | None = None

    # --- Scan remaining records until ENDLIB ---
    while pos < size:
        rtype, _, data = _read_record()
        if rtype == _REC_ENDLIB:
            break
        if rtype == _REC_STRNAME:
            cell_names.append(_decode_ascii(data))
            current_layer = None
            continue
        if rtype == _REC_LAYER:
            if len(data) >= 2:
                # INT16 big-endian, signed
                val = int.from_bytes(data[0:2], byteorder="big", signed=True)
                current_layer = val
            continue
        if rtype == _REC_DATATYPE:
            if len(data) >= 2 and current_layer is not None:
                dt = int.from_bytes(data[0:2], byteorder="big", signed=True)
                layer_set.add((current_layer, dt))
                current_layer = None  # pair consumed
            continue
        # all other records are skipped (we already advanced pos via _read_record)

    return GdsMetadata(
        library_name=library_name,
        cell_names=cell_names,
        layer_set=layer_set,
        cell_count=len(cell_names),
        file_size_bytes=size,
    )
