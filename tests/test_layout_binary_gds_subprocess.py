import json
import os
import pytest
from pathlib import Path

from clay.layout import (
    ParameterizedLibrary, NamedCell, LayerPolygon, Polygon, Point,
    emit_binary_gds_via_subprocess, BinaryGdsSubprocessError,
    cargo_binary_available,
)


def _sample_lib() -> ParameterizedLibrary:
    lib = ParameterizedLibrary(name="subproc_smoke")
    lib.add_cell(NamedCell(name="cell_a", polygons=[
        LayerPolygon(layer=66, datatype=20, polygon=Polygon.rectangle(0, 0, 500, 1000))
    ]))
    return lib


def test_subprocess_wrapper_produces_file(tmp_path):
    if not cargo_binary_available():
        pytest.skip("Rust CLI binary clay_layout_gds_bin not built; run cargo build first")
    out = tmp_path / "test.gds"
    written = emit_binary_gds_via_subprocess(_sample_lib(), out)
    assert written.exists()
    assert written.stat().st_size > 100
    assert written == out


def test_subprocess_wrapper_file_first_bytes_valid_record(tmp_path):
    if not cargo_binary_available():
        pytest.skip("Rust CLI binary clay_layout_gds_bin not built")
    out = tmp_path / "test.gds"
    emit_binary_gds_via_subprocess(_sample_lib(), out)
    raw = out.read_bytes()
    # First 2 bytes: record length (big-endian). Should be even and >= 4.
    length = (raw[0] << 8) | raw[1]
    assert length >= 4
    assert length % 2 == 0
    # Third byte: record type. For HEADER it's 0x00.
    assert raw[2] == 0x00


def test_subprocess_wrapper_empty_library(tmp_path):
    if not cargo_binary_available():
        pytest.skip("Rust CLI binary clay_layout_gds_bin not built")
    out = tmp_path / "empty.gds"
    empty = ParameterizedLibrary(name="empty")
    emit_binary_gds_via_subprocess(empty, out)
    assert out.exists()
    # Empty library still has HEADER + BGNLIB + LIBNAME + UNITS + ENDLIB records, > 30 bytes
    assert out.stat().st_size > 30


def test_subprocess_wrapper_explicit_binary_path_missing(tmp_path):
    fake_bin = tmp_path / "does_not_exist.exe"
    out = tmp_path / "no.gds"
    with pytest.raises(BinaryGdsSubprocessError) as exc:
        emit_binary_gds_via_subprocess(_sample_lib(), out, binary_path=fake_bin)
    assert "not found" in str(exc.value).lower() or "Build it first" in str(exc.value)


def test_cargo_binary_available_returns_bool():
    # Always returns a bool (True or False); doesn't crash
    result = cargo_binary_available()
    assert isinstance(result, bool)


def test_subprocess_wrapper_cpp20_cli_emits_valid_gds(tmp_path):
    from clay.layout.binary_gds_subprocess import _find_cpp_cli_binary, emit_binary_gds_native

    cpp_cli = _find_cpp_cli_binary()
    if cpp_cli is None:
        pytest.skip("C++20 CLI binary clay_engine_cli not built")
    out_cli = tmp_path / "cli.gds"
    out_native = tmp_path / "native.gds"
    emit_binary_gds_via_subprocess(_sample_lib(), out_cli, binary_path=cpp_cli)
    emit_binary_gds_native(_sample_lib(), out_native)
    assert out_cli.exists() and out_cli.stat().st_size > 100
    assert out_cli.read_bytes() == out_native.read_bytes()
