#!/usr/bin/env python3
"""End-to-end self-verification of the Clay Open-Source Package (SKY130 + GF180MCU).

Checks:
  1. Zero-tolerance NDA / proprietary-foundry token scan across the entire tree.
  2. C++20 Main Engine build & 51 pure C++20 native unit tests (`--werror --test`).
  3. Standalone zero-Python `clay_engine_cli` selftest, parse, and GDSII synthesis
     on both GF180MCU and SKY130 benchmarks.
  4. Python pytest suite (`tests/`).
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PATTERNS = (
    re.compile(r"\btsmc\w*\b", re.I),
    re.compile(r"\bsmic\w*\b", re.I),
    re.compile(r"\basap7\w*\b", re.I),
    re.compile(r"\bcalibre\b", re.I),
    re.compile(r"\bassura\b", re.I),
    re.compile(r"\bcrn28\w*\b", re.I),
    re.compile(r"\bms018\w*\b", re.I),
    re.compile(r"\brf018\w*\b", re.I),
    re.compile(r"\brnhpoly\b", re.I),
    re.compile(r"\brnpo1rpo\b", re.I),
    re.compile(r"\brppo1rpo\b", re.I),
    re.compile(r"\brphpoly\b", re.I),
    re.compile(r"\bn18_ckt\b", re.I),
    re.compile(r"\bp18_ckt\b", re.I),
    re.compile(r"\brnposab\b", re.I),
    re.compile(r"\brpposab\b", re.I),
    re.compile(r"C:\\Users\\Administrator", re.I),
    re.compile(r"C:/Users/Administrator", re.I),
)

TEXT_SUFFIXES = {
    ".py", ".cpp", ".hpp", ".h", ".c", ".clay", ".yaml", ".yml",
    ".json", ".toml", ".md", ".txt", ".cmake", ".lark", ".sh", ".cir", ".sp", ".spice",
}


def check_nda_clean() -> list[str]:
    violations: list[str] = []
    skip_dirs = {"__pycache__", "build", ".git", ".pytest_cache", "pdks", "_flow_run"}
    # Exempt the hygiene verifier files themselves (which hold the regex patterns)
    exempt = {"scripts/verify_oss_package.py", "tests/test_oss_nda_hygiene.py"}
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in skip_dirs]
        for fn in fns:
            p = Path(dp) / fn
            rel = p.relative_to(ROOT).as_posix()
            if rel in exempt:
                continue
            if p.suffix.lower() not in TEXT_SUFFIXES and fn not in ("CMakeLists.txt", "LICENSE"):
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore")
            for pat in FORBIDDEN_PATTERNS:
                m = pat.search(txt)
                if m:
                    violations.append(f"{rel}: forbidden token {m.group(0)!r}")
                    break
    return violations


def main() -> int:
    print("[1/4] Running zero-tolerance NDA hygiene audit ...")
    violations = check_nda_clean()
    if violations:
        print(f"FAIL: {len(violations)} NDA violation(s) found:")
        for v in violations[:20]:
            print("  -", v)
        return 1
    print("  PASS: 0 proprietary foundry tokens found across entire OSS package.")

    print("[2/4] Building & testing pure C++20 Main Engine (/WX) ...")
    subprocess.run(
        [sys.executable, str(ROOT / "cpp" / "clay_pnr" / "build.py"), "--target", "all", "--werror", "--test"],
        cwd=ROOT,
        check=True,
    )

    print("[3/4] Running standalone zero-Python CLI synthesis (GF180MCU + SKY130) ...")
    cli = ROOT / "cpp" / "clay_pnr" / "build" / "modular" / "bin" / (
        "clay_engine_cli.exe" if platform.system() == "Windows" else "clay_engine_cli"
    )
    subprocess.run([str(cli), "selftest"], check=True)
    with tempfile.TemporaryDirectory() as td:
        out_gf = Path(td) / "fc_ota_gf180.gds"
        r1 = subprocess.run(
            [str(cli), "synth", str(ROOT / "benchmarks/gf180mcu/folded_cascode_ota/source/design.clay"), "-o", str(out_gf)],
            capture_output=True, text=True, check=True,
        )
        d1 = json.loads(r1.stdout)
        assert d1["status"] == "PASS" and d1["pdk"] == "gf180mcu" and out_gf.stat().st_size > 1000

        out_sky = Path(td) / "two_stage_ota_sky130.gds"
        r2 = subprocess.run(
            [str(cli), "synth", str(ROOT / "benchmarks/gf180mcu/two_stage_ota/source/design.clay"), "--pdk", "sky130", "-o", str(out_sky)],
            capture_output=True, text=True, check=True,
        )
        d2 = json.loads(r2.stdout)
        assert d2["status"] == "PASS" and d2["pdk"] == "sky130" and out_sky.stat().st_size > 1000
        print(f"  PASS: GF180MCU ({d1['elapsed_ms']} ms) & SKY130 ({d2['elapsed_ms']} ms) standalone GDSII synth.")

    print("[4/4] Running OSS pytest suite ...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], cwd=ROOT, env=env, check=True)
    print("ALL OSS RELEASE CHECKS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
