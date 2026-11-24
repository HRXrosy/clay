"""Binary GDSII v6.0 emitter backed by Clay's pure C++20 engine (`_engine_cpp` / `clay_engine_cli`).

Primary in-process path (`emit_binary_gds_native`) calls `_engine_cpp.gds_write_library_cpp`
with zero subprocess overhead and zero KLayout/Rust dependency.
Subprocess path (`emit_binary_gds_via_subprocess`) prefers the standalone C++20
`clay_engine_cli gds-emit-json` binary and falls back to legacy Rust `clay_layout_gds_bin`.
"""
from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

from .backend import ParameterizedLibrary


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEBUG_BIN = REPO_ROOT / "crates" / "clay_layout" / "target" / "debug" / "clay_layout_gds_bin"
DEFAULT_RELEASE_BIN = REPO_ROOT / "crates" / "clay_layout" / "target" / "release" / "clay_layout_gds_bin"
DEFAULT_CPP_CLI_BIN = REPO_ROOT / "cpp" / "clay_pnr" / "build" / "modular" / "bin" / "clay_engine_cli"


class BinaryGdsSubprocessError(RuntimeError):
    pass


def _serialize_polygon(poly) -> dict:
    return {"vertices": [{"x": v.x, "y": v.y} for v in poly.vertices]}


def _serialize_lib(lib: ParameterizedLibrary) -> str:
    """Serialize ParameterizedLibrary to JSON matching the Rust serde format."""
    cells = []
    for cell in lib.cells:
        polygons = []
        for lp in cell.polygons:
            polygons.append({
                "layer": lp.layer,
                "datatype": lp.datatype,
                "polygon": _serialize_polygon(lp.polygon),
            })
        cells.append({"name": cell.name, "polygons": polygons})
    payload = {
        "name": lib.name,
        "unit_user": lib.unit_user,
        "unit_db_meters": lib.unit_db_meters,
        "cells": cells,
    }
    return json.dumps(payload)


def _find_binary() -> Optional[Path]:
    """Locate the Rust CLI binary; check both .exe (Windows) and bare names."""
    candidates = [
        DEFAULT_RELEASE_BIN.with_suffix(".exe"),
        DEFAULT_RELEASE_BIN,
        DEFAULT_DEBUG_BIN.with_suffix(".exe"),
        DEFAULT_DEBUG_BIN,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_cpp_cli_binary() -> Optional[Path]:
    """Locate the standalone pure C++20 `clay_engine_cli` binary."""
    for candidate in (DEFAULT_CPP_CLI_BIN.with_suffix(".exe"), DEFAULT_CPP_CLI_BIN):
        if candidate.exists():
            return candidate
    return None


def emit_binary_gds_native(
    lib: ParameterizedLibrary,
    output_path: os.PathLike,
) -> Path:
    """Emit a binary `.gds` file directly in-process via pure C++20 `_engine_cpp`.

    Supports empty, single-cell, and multi-cell libraries with zero KLayout,
    zero Rust, and zero subprocess spawn overhead.
    """
    from clay._native import _engine_cpp

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cells_payload = [
        (
            str(cell.name),
            [
                (
                    int(lp.layer),
                    int(lp.datatype),
                    [(int(v.x), int(v.y)) for v in lp.polygon.vertices],
                )
                for lp in cell.polygons
            ],
        )
        for cell in lib.cells
    ]
    _engine_cpp.gds_write_library_cpp(
        str(output_path),
        str(lib.name),
        cells_payload,
        float(lib.unit_user),
        float(lib.unit_db_meters),
    )
    return output_path


def emit_binary_gds_via_subprocess(
    lib: ParameterizedLibrary,
    output_path: os.PathLike,
    *,
    binary_path: Optional[Path] = None,
    timeout_sec: float = 60.0,
) -> Path:
    """Emit a binary `.gds` file via CLI subprocess (pure C++20 `clay_engine_cli` or legacy Rust `clay_layout_gds_bin`).

    Args:
        lib: ParameterizedLibrary to encode
        output_path: where to write the .gds file
        binary_path: explicit override for the CLI binary path. If None,
            prefers pure C++20 `clay_engine_cli` and falls back to legacy Rust `clay_layout_gds_bin`.
        timeout_sec: subprocess timeout (default 60 s)

    Returns: Path to the written `.gds` file.
    """
    output_path = Path(output_path)
    bin_path = binary_path if binary_path is not None else (_find_cpp_cli_binary() or _find_binary())
    if bin_path is None or not bin_path.exists():
        raise BinaryGdsSubprocessError(
            "GDS CLI binary not found (checked C++20 `clay_engine_cli` and Rust `clay_layout_gds_bin`). Build it first:\n"
            "  python cpp/clay_pnr/build.py --target cli"
        )

    json_payload = _serialize_lib(lib)
    if bin_path.stem == "clay_engine_cli":
        cmd = [str(bin_path), "gds-emit-json", "-", "-o", str(output_path)]
    else:
        cmd = [str(bin_path), "-o", str(output_path)]
    bin_label = bin_path.name

    try:
        proc = subprocess.run(
            cmd,
            input=json_payload,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise BinaryGdsSubprocessError(f"{bin_label} timed out after {timeout_sec}s") from e
    except OSError as e:
        raise BinaryGdsSubprocessError(f"failed to spawn {bin_label}: {e}") from e

    if proc.returncode != 0:
        raise BinaryGdsSubprocessError(
            f"{bin_label} exited {proc.returncode}\n"
            f"stderr: {proc.stderr}\n"
            f"stdout: {proc.stdout}"
        )

    if not output_path.exists():
        raise BinaryGdsSubprocessError(
            f"{bin_label} reported success but no file produced at {output_path}"
        )

    return output_path


def cargo_binary_available() -> bool:
    """Return True if the Rust CLI binary is present on disk (existence only; not version/freshness/executability verified)."""
    return _find_binary() is not None
