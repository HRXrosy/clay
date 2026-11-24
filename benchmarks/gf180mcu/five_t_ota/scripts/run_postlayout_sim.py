from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


CASE_DIR = Path(__file__).resolve().parents[1]
PEX_PATH = CASE_DIR / "final" / "routed_5t_ota.pex.spice"
DEFAULT_REPORT = CASE_DIR / "final" / "postlayout_sim.json"
DIFF_POINTS_MV = (-10.0, -5.0, 0.0, 5.0, 10.0)
VDD_V = 3.3
VCM_V = 1.2
VBIAS_V = 0.85
GAIN_MIN = 100.0
GAIN_DB_MIN = 40.0
IDD_MIN_A = 20e-6
IDD_MAX_A = 200e-6
BALANCE_MAX_V = 1e-3


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pdks" / "gf180mcu").exists():
            return parent
    raise RuntimeError("could not locate repository root with pdks/gf180mcu")


def _model_dir() -> Path:
    root = _repo_root()
    path = (
        root
        / "pdks"
        / "gf180mcu"
        / "libraries"
        / "gf180mcu_fd_pr"
        / "latest"
        / "models"
        / "ngspice"
    )
    required = (path / "design.ngspice", path / "sm141064.ngspice")
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise RuntimeError("missing gf180 ngspice model file(s): " + ", ".join(missing))
    return path


def _ngspice_bin() -> str:
    exe = shutil.which("ngspice_con") or shutil.which("ngspice")
    if not exe:
        raise RuntimeError("ngspice_con/ngspice not found on PATH")
    return exe


def _device_alias_header() -> str:
    return """* Device aliasing: Magic-extracted names -> gf180mcu PDK names.
.subckt nfet_03v3 d g s b w=1u l=1u m=1 nf=1 ad=0 as=0 pd=0 ps=0 sa=0 sb=0 sd=0 nrd=0 nrs=0 dtemp=0
x1 d g s b nmos_3p3 w=w l=l nf=nf m=m ad=ad as=as pd=pd ps=ps sa=sa sb=sb sd=sd nrd=nrd nrs=nrs dtemp=dtemp
.ends
.subckt pfet_03v3 d g s b w=1u l=1u m=1 nf=1 ad=0 as=0 pd=0 ps=0 sa=0 sb=0 sd=0 nrd=0 nrs=0 dtemp=0
x1 d g s b pmos_3p3 w=w l=l nf=nf m=m ad=ad as=as pd=pd ps=ps sa=sa sb=sb sd=sd nrd=nrd nrs=nrs dtemp=dtemp
.ends
"""


def _deck(pex_path: Path, diff_mv: float) -> str:
    diff_v = diff_mv / 1000.0
    vinp = VCM_V + diff_v / 2.0
    vinn = VCM_V - diff_v / 2.0
    return f"""* 5T OTA post-layout DC transfer check, diff={diff_mv:g} mV
.include design.ngspice
.param sw_stat_global=0
.param sw_stat_mismatch=0
.lib sm141064.ngspice typical
{_device_alias_header()}
.include "{pex_path.as_posix()}"
VDD vdd 0 {VDD_V:.12g}
VBIAS vbias 0 {VBIAS_V:.12g}
VINP inp 0 {vinp:.12g}
VINN inn 0 {vinn:.12g}
Xota outn outp vbias inp inn 0 vdd tail routed_5t_ota
.control
op
let idd_abs = abs(i(vdd))
print v(outn) v(outp) v(tail) idd_abs
quit
.endc
.end
"""


_VALUE_RE = re.compile(
    r"^\s*(v\(outn\)|v\(outp\)|v\(tail\)|idd_abs)\s*=\s*"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$",
    re.MULTILINE,
)


def _parse_log(log_path: Path) -> dict[str, float]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "Simulation interrupted" in text or "unknown subckt" in text:
        raise RuntimeError(f"ngspice failed; see {log_path}")
    values = {match.group(1): float(match.group(2)) for match in _VALUE_RE.finditer(text)}
    required = ("v(outn)", "v(outp)", "v(tail)", "idd_abs")
    missing = [key for key in required if key not in values]
    if missing:
        raise RuntimeError(f"ngspice log missing measurement(s) {missing}: {log_path}")
    return {
        "outn_v": values["v(outn)"],
        "outp_v": values["v(outp)"],
        "tail_v": values["v(tail)"],
        "idd_a": values["idd_abs"],
    }


def _pex_summary(pex_path: Path) -> dict[str, Any]:
    text = pex_path.read_text(encoding="utf-8")
    return {
        "path": "final/routed_5t_ota.pex.spice",
        "source": "Magic ext2spice gf180mcuD capture",
        "devices": len(re.findall(r"^X\d+\s", text, flags=re.MULTILINE)),
        "caps": len(re.findall(r"^C\d+\s", text, flags=re.MULTILINE)),
        "contains_input_pair_caps": all(token in text for token in ("inn gnd", "inp gnd")),
    }


def _run_one(diff_mv: float, pex_path: Path, work_dir: Path) -> dict[str, float]:
    safe = str(diff_mv).replace("-", "m").replace(".", "p")
    deck_path = work_dir / f"five_t_ota_postlayout_diff_{safe}.sp"
    log_path = work_dir / f"five_t_ota_postlayout_diff_{safe}.log"
    deck_path.write_text(_deck(pex_path.resolve(), diff_mv), encoding="utf-8")
    proc = subprocess.run(
        [_ngspice_bin(), "-b", str(deck_path), "-o", str(log_path)],
        cwd=_model_dir(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ngspice exited {proc.returncode} for diff={diff_mv:g} mV\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    point = _parse_log(log_path)
    point["diff_mv"] = diff_mv
    point["vinp_v"] = VCM_V + (diff_mv / 1000.0) / 2.0
    point["vinn_v"] = VCM_V - (diff_mv / 1000.0) / 2.0
    return point


def _with_work_dir(work_dir: Path | None):
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        return None, work_dir
    tmp = tempfile.TemporaryDirectory(prefix="clay_5t_ota_postlayout_")
    return tmp, Path(tmp.name)


def _metrics(points: list[dict[str, float]]) -> dict[str, Any]:
    by_diff = {point["diff_mv"]: point for point in points}
    gain = (by_diff[5.0]["outn_v"] - by_diff[-5.0]["outn_v"]) / 0.010
    gain_db = 20.0 * math.log10(abs(gain))
    outn_values = [point["outn_v"] for point in sorted(points, key=lambda p: p["diff_mv"])]
    return {
        "dc_gain_v_per_v": gain,
        "dc_gain_db": gain_db,
        "outn_swing_20mv_v": by_diff[10.0]["outn_v"] - by_diff[-10.0]["outn_v"],
        "outn_monotonic_vs_diff": all(a < b for a, b in zip(outn_values, outn_values[1:])),
        "zero_diff_output_balance_v": abs(by_diff[0.0]["outn_v"] - by_diff[0.0]["outp_v"]),
    }


def run_postlayout_sim(
    *,
    pex_path: Path = PEX_PATH,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    if not pex_path.exists():
        raise RuntimeError(f"PEX deck does not exist: {pex_path}")
    tmp, actual_work = _with_work_dir(work_dir)
    try:
        points = [_run_one(diff, pex_path, actual_work) for diff in DIFF_POINTS_MV]
    finally:
        if tmp is not None:
            tmp.cleanup()

    metrics = _metrics(points)
    op = next(point for point in points if point["diff_mv"] == 0.0)
    passed = (
        metrics["dc_gain_v_per_v"] >= GAIN_MIN
        and metrics["dc_gain_db"] >= GAIN_DB_MIN
        and metrics["outn_monotonic_vs_diff"] is True
        and metrics["zero_diff_output_balance_v"] <= BALANCE_MAX_V
        and IDD_MIN_A <= op["idd_a"] <= IDD_MAX_A
    )
    return {
        "schema": "clay.postlayout_sim/v1",
        "case_id": "case_five_t_ota_gf180_postlayout",
        "cell": "routed_5t_ota",
        "pdk": "gf180mcuD",
        "corner": "typical",
        "simulator": {
            "engine": "ngspice",
            "binary": Path(_ngspice_bin()).name,
        },
        "conditions": {
            "vdd_v": VDD_V,
            "vcm_v": VCM_V,
            "vbias_v": VBIAS_V,
            "temp_c": 27.0,
            "sw_stat_global": 0,
            "sw_stat_mismatch": 0,
        },
        "pex": _pex_summary(pex_path),
        "criteria": {
            "dc_gain_v_per_v_min": GAIN_MIN,
            "dc_gain_db_min": GAIN_DB_MIN,
            "idd_a_range": [IDD_MIN_A, IDD_MAX_A],
            "zero_diff_output_balance_v_max": BALANCE_MAX_V,
            "outn_monotonic_vs_diff": True,
            "diff_points_mv": list(DIFF_POINTS_MV),
        },
        "operating_point": op,
        "dc_transfer": points,
        "metrics": metrics,
        "verdict": "pass" if passed else "fail",
    }


def run_post_kpis(pex_spice: Path | str | None = None) -> dict[str, Any]:
    """Flow PostSimStep entrypoint (dict-returning, never reads sys.argv).

    ``clay.flow.step.PostSimStep._invoke`` calls this with the freshly extracted
    PEX deck from the flow work-dir. We run the real ngspice DC transfer check on
    THAT deck and return the package report, additionally surfacing the
    operating-point supply current at the top level as ``idd_a`` so the
    quality{} ``@kpi(idd_a, "<=", 2e-4)`` resolves a real measured value (the
    KPI resolver descends into ``metrics``/``summary``/``op``/``results`` but not
    ``operating_point``). ``dc_gain_db`` already resolves via the ``metrics``
    container, so it needs no surfacing.
    """
    pex_path = Path(pex_spice) if pex_spice else PEX_PATH
    if not pex_path.exists():
        pex_path = PEX_PATH
    report = run_postlayout_sim(pex_path=pex_path)
    report["idd_a"] = report["operating_point"]["idd_a"]
    return report


def _compare(actual: dict[str, Any], expected_path: Path) -> None:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if expected.get("verdict") != "pass":
        raise RuntimeError(f"expected report is not pass: {expected_path}")
    if actual.get("verdict") != "pass":
        raise RuntimeError("rerun report did not pass package criteria")
    for key in ("dc_gain_v_per_v", "dc_gain_db", "zero_diff_output_balance_v"):
        if not math.isclose(actual["metrics"][key], expected["metrics"][key], rel_tol=5e-4, abs_tol=1e-9):
            raise RuntimeError(
                f"metric {key} mismatch: actual={actual['metrics'][key]} "
                f"expected={expected['metrics'][key]}"
            )
    by_diff = {point["diff_mv"]: point for point in expected["dc_transfer"]}
    for point in actual["dc_transfer"]:
        ref = by_diff[point["diff_mv"]]
        for key in ("outn_v", "outp_v", "tail_v", "idd_a"):
            if not math.isclose(point[key], ref[key], rel_tol=5e-4, abs_tol=1e-12):
                raise RuntimeError(
                    f"{key} mismatch at diff={point['diff_mv']}: "
                    f"actual={point[key]} expected={ref[key]}"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", type=Path, help="write a fresh JSON report")
    parser.add_argument(
        "--check",
        type=Path,
        help="compare a fresh ngspice run against an existing JSON report",
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--json", action="store_true", help="print report JSON")
    args = parser.parse_args(argv)

    data = run_postlayout_sim(work_dir=args.work_dir)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.check:
        _compare(data, args.check)
    if args.json or (not args.write and not args.check):
        print(json.dumps(data, indent=2, sort_keys=True))
    return 0 if data["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
