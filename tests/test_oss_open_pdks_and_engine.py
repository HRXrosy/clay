"""End-to-end OSS tests for SKY130 & GF180MCU PDKs, benchmarks, and C++20 Main Engine."""
from __future__ import annotations

import json
import math
import platform
import subprocess
from pathlib import Path

import pytest

from clay._native import _engine_cpp
from clay.parser import parse_clay_file
from clay.pdk import get_pdk
from clay.pdk_dims import PdkDims

ROOT = Path(__file__).resolve().parents[1]
CLI_EXE = (
    ROOT / "cpp" / "clay_pnr" / "build" / "modular" / "bin"
    / ("clay_engine_cli.exe" if platform.system() == "Windows" else "clay_engine_cli")
)

ALL_BENCHMARKS = sorted(ROOT.glob("benchmarks/*/*/source/design.clay")) + sorted(
    ROOT.glob("benchmarks/*/*/design.clay")
)


def test_open_pdk_adapters_and_luts_present() -> None:
    gf = get_pdk("gf180mcu")
    sky = get_pdk("sky130")
    assert gf.device_bindings["nmos"] == "nmos_3p3"
    assert sky.device_bindings["nmos"] == "sky130_fd_pr__nfet_01v8"
    assert PdkDims.gf180mcu().min_L > 0
    assert PdkDims.sky130().min_L > 0
    assert (ROOT / "data/pdk_luts/gf180mcu_gm_id.json").is_file()
    assert (ROOT / "data/pdk_luts/sky130_gm_id.json").is_file()


def test_benchmark_catalog_covers_both_sky130_and_gf180mcu() -> None:
    gf_bms = [p for p in ALL_BENCHMARKS if "gf180mcu" in p.parts]
    sky_bms = [p for p in ALL_BENCHMARKS if "sky130" in p.parts]
    assert len(gf_bms) >= 6
    assert len(sky_bms) >= 8


@pytest.mark.parametrize("clay_path", ALL_BENCHMARKS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_cpp20_and_lark_parsers_agree_on_all_open_pdk_benchmarks(clay_path: Path) -> None:
    py_cf = parse_clay_file(clay_path)
    cpp_cf = _engine_cpp.parse_clay_ir_cpp(clay_path.read_text(encoding="utf-8"))
    assert cpp_cf["pdk"] in ("gf180mcu", "sky130")
    assert cpp_cf["pdk"] == py_cf.pdk_id
    assert len(cpp_cf["cells"]) == len(py_cf.cell_defs)
    for py_cell, cpp_cell in zip(py_cf.cell_defs, cpp_cf["cells"]):
        assert cpp_cell["name"] == py_cell.local_name
        assert len(cpp_cell["devices"]) == len(py_cell.devices)
        for py_dev, cpp_dev in zip(py_cell.devices, cpp_cell["devices"]):
            assert cpp_dev["name"] == py_dev.instance_name
            for k, py_v in py_dev.sizing.items():
                if isinstance(py_v, (int, float)):
                    assert math.isclose(cpp_dev["params"][k], float(py_v), rel_tol=1e-9, abs_tol=1e-18)


def test_standalone_cpp20_cli_synthesizes_gf180mcu_and_sky130_gds(tmp_path: Path) -> None:
    assert CLI_EXE.is_file(), f"Build C++20 CLI first: {CLI_EXE}"

    fc_clay = ROOT / "benchmarks/gf180mcu/folded_cascode_ota/source/design.clay"
    out_gf = tmp_path / "fc_ota_gf180.gds"
    r1 = subprocess.run([str(CLI_EXE), "synth", str(fc_clay), "-o", str(out_gf)], capture_output=True, text=True, check=True)
    d1 = json.loads(r1.stdout)
    assert d1["status"] == "PASS" and d1["pdk"] == "gf180mcu" and d1["devices"] == 20
    assert len(_engine_cpp.gds_inspect_layers_cpp(str(out_gf), d1["cell"])) >= 8

    ota_clay = ROOT / "benchmarks/gf180mcu/two_stage_ota/source/design.clay"
    out_sky = tmp_path / "two_stage_ota_sky130.gds"
    r2 = subprocess.run(
        [str(CLI_EXE), "synth", str(ota_clay), "--pdk", "sky130", "-o", str(out_sky)],
        capture_output=True, text=True, check=True,
    )
    d2 = json.loads(r2.stdout)
    assert d2["status"] == "PASS" and d2["pdk"] == "sky130"
    assert len(_engine_cpp.gds_inspect_layers_cpp(str(out_sky), d2["cell"])) >= 6


def test_lvs_gate_and_passive_cell_load_cleanly_without_commercial_pdks() -> None:
    from clay.layout import passive_cell
    from clay.layout.route import lvs_gate

    assert "gf180mcu" in lvs_gate._PDK_LVS
    assert set(passive_cell._PASSIVE_FACTS.keys()) == {"gf180mcu"}


@pytest.mark.parametrize(
    "case_dir",
    sorted(p.parent for p in ROOT.glob("benchmarks/*/*/case.yaml")),
    ids=lambda p: p.relative_to(ROOT).as_posix(),
)
def test_all_open_pdk_benchmarks_pass_case_validate(case_dir: Path) -> None:
    from clay.casepkg import validate_case_package

    report = validate_case_package(case_dir)
    failed = [c for c in report.checks if c.status != "PASS"]
    assert report.passed, f"Benchmark {case_dir.relative_to(ROOT)} failed validation: {failed}"
