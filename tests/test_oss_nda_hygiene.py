"""Zero-tolerance NDA hygiene contract for the Clay Open-Source Release."""
from __future__ import annotations

import os
import re
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


def test_oss_tree_contains_zero_commercial_foundry_tokens() -> None:
    skip_dirs = {"__pycache__", "build", ".git", ".pytest_cache", "pdks", "_flow_run"}
    exempt = {"scripts/verify_oss_package.py", "tests/test_oss_nda_hygiene.py"}
    violations: list[str] = []
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
                    violations.append(f"{rel}: {m.group(0)!r}")
                    break
    assert violations == [], f"NDA violations found: {violations[:10]}"


def test_only_open_pdks_are_registered() -> None:
    from clay.pdk import all_adapters, get_pdk

    adapters = {a.pdk_id for a in all_adapters()}
    assert adapters == {"gf180mcu", "sky130"}
    assert get_pdk("gf180mcu").pdk_id == "gf180mcu"
    assert get_pdk("sky130").pdk_id == "sky130"


def test_copyleft_agpl3_license_present() -> None:
    lic = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in lic
    assert "Version 3, 19 November 2007" in lic
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "AGPL-3.0-only" in pyproject
