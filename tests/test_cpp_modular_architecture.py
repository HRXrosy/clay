"""Architectural & hygiene regression tests for Clay's modern C++20 modular PnR stack
and permanent deletion of the retired legacy compatibility layer.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
CPP_ROOT = REPO_ROOT / "cpp" / "clay_pnr"

PURE_CPP_FILES: tuple[str, ...] = (
    "include/clay_pnr/core/geometry.hpp",
    "include/clay_pnr/core/status.hpp",
    "include/clay_pnr/core/hash.hpp",
    "include/clay_pnr/core/polygon_set.hpp",
    "include/clay_pnr/core/symbol_table.hpp",
    "include/clay_pnr/gds/gds_stream.hpp",
    "include/clay_pnr/layout/layout_db.hpp",
    "include/clay_pnr/pcell/mos_pcell.hpp",
    "include/clay_pnr/ir/clay_ir.hpp",
    "include/clay_pnr/engine/standalone_synth.hpp",
    "include/clay_pnr/place/sweep_union.hpp",
    "include/clay_pnr/place/bvh_index.hpp",
    "include/clay_pnr/place/seqpair.hpp",
    "include/clay_pnr/place/compactor.hpp",
    "include/clay_pnr/place/envelope_placer.hpp",
    "include/clay_pnr/place/banded_placer.hpp",
    "include/clay_pnr/route/pathfinder_core.hpp",
    "include/clay_pnr/route/gridless_core.hpp",
    "include/clay_pnr/route/gridless_spatial.hpp",
    "include/clay_pnr/route/gridless_search.hpp",
    "src/core/polygon_set.cpp",
    "src/core/symbol_table.cpp",
    "src/gds/gds_stream.cpp",
    "src/layout/layout_db.cpp",
    "src/pcell/mos_pcell.cpp",
    "src/ir/clay_parser.cpp",
    "src/engine/standalone_synth.cpp",
    "src/cli/clay_engine_main.cpp",
    "src/place/sweep_union.cpp",
    "src/place/bvh_index.cpp",
    "src/place/seqpair.cpp",
    "src/place/compactor.cpp",
    "src/place/envelope_placer.cpp",
    "src/place/banded_placer.cpp",
    "src/route/pathfinder_core.cpp",
    "src/route/gridless_core.cpp",
    "src/route/gridless_spatial.cpp",
    "src/route/gridless_search.cpp",
    "tests/test_framework.hpp",
    "tests/test_main.cpp",
    "tests/test_core_geometry.cpp",
    "tests/test_core_polygon_gds.cpp",
    "tests/test_place_kernels.cpp",
    "tests/test_pathfinder_core.cpp",
    "tests/test_gridless_core.cpp",
    "tests/test_engine_phase23.cpp",
    "tests/test_ir_and_synth.cpp",
)

PY_BINDING_FILES: tuple[str, ...] = (
    "include/clay_pnr/core/py_raii.hpp",
    "include/clay_pnr/route/gridless_py_internal.hpp",
    "src/engine/py_engine_module.cpp",
    "src/place/py_place_module.cpp",
    "src/route/py_pathfinder_module.cpp",
    "src/route/gridless_spatial_py.cpp",
    "src/route/gridless_search_py.cpp",
    "src/route/py_gridless_module.cpp",
)

RETIRED_MONOLITHS: tuple[str, ...] = (
    "src/placement_kernels.cpp",
    "src/pathfinder_kernels.cpp",
    "src/gridless_kernels.cpp",
)


def test_retired_layer_deleted_and_zero_active_imports() -> None:
    assert not (REPO_ROOT / "clay" / "_retired").exists(), "clay/_retired must remain deleted"
    assert not (REPO_ROOT / "clay" / "_retired_compat.py").exists(), "clay/_retired_compat.py must remain deleted"

    import clay.testing.pdk_isolation as pdk_iso

    assert callable(getattr(pdk_iso, "snapshot_clay_state", None))
    assert callable(getattr(pdk_iso, "restore_clay_state", None))
    assert callable(getattr(pdk_iso, "reap_stray_ngspice", None))

    offenders: list[str] = []
    for root_name in ("clay", "tests", "scripts", "circuits"):
        root = REPO_ROOT / root_name
        if not root.is_dir():
            continue
        for py_path in root.rglob("*.py"):
            if "__pycache__" in py_path.parts:
                continue
            try:
                tree = ast.parse(py_path.read_text(encoding="utf-8-sig"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith(("clay._retired", "clay._retired_compat")):
                            offenders.append(f"{py_path.relative_to(REPO_ROOT)}:{node.lineno}:{alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if mod.startswith(("clay._retired", "clay._retired_compat")):
                        offenders.append(f"{py_path.relative_to(REPO_ROOT)}:{node.lineno}:{mod}")
    assert not offenders, f"Unexpected imports of deleted retired layer: {offenders}"


def test_old_cpp_monoliths_removed_and_modular_layout_present() -> None:
    for rel in RETIRED_MONOLITHS:
        assert not (CPP_ROOT / rel).exists(), f"Superseded monolithic C++ file still exists: {rel}"

    for rel in (*PURE_CPP_FILES, *PY_BINDING_FILES):
        assert (CPP_ROOT / rel).is_file(), f"Missing modular C++20 file: {rel}"


def test_pure_cpp_core_has_zero_python_h_dependency() -> None:
    inc_re = re.compile(r'^\s*#\s*include\s*[<"]Python\.h[">]', re.MULTILINE)
    def_re = re.compile(r'^\s*#\s*define\s+PY_SSIZE_T_CLEAN\b', re.MULTILINE)
    leaks: list[str] = []
    for rel in PURE_CPP_FILES:
        text = (CPP_ROOT / rel).read_text(encoding="utf-8")
        if inc_re.search(text) or def_re.search(text):
            leaks.append(rel)
    assert not leaks, f"Pure C++20 core/test files must never include <Python.h>: {leaks}"


def test_cmake_and_presets_declare_layered_targets() -> None:
    cmake = (CPP_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    for token in (
        "add_library(clay_pnr_core STATIC",
        "add_executable(clay_pnr_cpp_tests",
        "add_clay_py_ext(_place_cpp",
        "add_clay_py_ext(_pathfinder_cpp",
        "add_clay_py_ext(_gridless_cpp",
        "/W4 /permissive-",
    ):
        assert token in cmake, f"CMakeLists.txt missing expected modular target token: {token}"

    presets = json.loads((CPP_ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    assert presets.get("version", 0) >= 6
    assert any(p.get("name") == "win-msvc-release" for p in presets.get("configurePresets", []))


def test_native_modules_import_and_cpp_cli_entrypoints_work() -> None:
    from clay._native import _gridless_cpp, _pathfinder_cpp, _place_cpp

    assert hasattr(_place_cpp, "NativeRectangleIndex")
    assert hasattr(_pathfinder_cpp, "NativePathFinderGrid")
    assert hasattr(_gridless_cpp, "NativeSpatialIndex")

    r_build = subprocess.run(
        [sys.executable, "-m", "clay", "build", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert r_build.returncode == 0, r_build.stderr
    assert "--werror" in r_build.stdout and "--test" in r_build.stdout

    r_test = subprocess.run(
        [sys.executable, "-m", "clay", "test", "--tier", "cpp", "--list"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert r_test.returncode == 0, r_test.stderr
    assert "CPP_NATIVE:" in r_test.stdout
    assert "tests/test_cpp_modular_architecture.py" in r_test.stdout


def test_pure_cpp20_native_unit_test_suite_passes() -> None:
    exe_name = "clay_pnr_cpp_tests.exe" if platform.system() == "Windows" else "clay_pnr_cpp_tests"
    exe_path = CPP_ROOT / "build" / "modular" / "bin" / exe_name
    if not exe_path.is_file():
        rc = subprocess.run(
            [sys.executable, str(CPP_ROOT / "build.py"), "--target", "tests"],
            cwd=str(REPO_ROOT),
        ).returncode
        assert rc == 0, "Failed to build pure C++20 native unit tests"
    assert exe_path.is_file()
    res = subprocess.run([str(exe_path)], cwd=str(CPP_ROOT), capture_output=True, text=True)
    assert res.returncode == 0, f"Pure C++20 tests failed:\n{res.stdout}\n{res.stderr}"
    assert "(0 failed)" in res.stdout
    passed_lines = [ln for ln in res.stdout.splitlines() if ln.startswith("[  PASS  ]")]
    assert len(passed_lines) >= 30, f"Expected >= 30 pure C++20 unit tests, got {len(passed_lines)}"

