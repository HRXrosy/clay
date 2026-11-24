"""Generic SPICE testbench adapter for zero-glue Clay pre-sim & post-sim.

Consumes a SPICE circuit (or post-layout PEX ``.spice`` netlist) + one or more
``.tb`` / ``.sp`` testbenches containing ``.meas`` or ``echo key=val`` directives,
runs ``ngspice_con`` in batch mode, and parses standardized KPI dictionaries.

Also renders self-contained Python ``presim`` / ``postsim`` driver wrappers for
``clay.frontend.onboard.scaffold_case`` compatible with:
  * ``make_presim_evaluate`` (``SOURCE_CIR`` + ``run()``)
  * ``CompileStep`` / ``SizingStep`` (``run_candidate(...)``)
  * ``PostSimStep._invoke`` & ``PostSimStep._invoke_declared`` (``run_post_kpis(...)``)
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "TbRunResult",
    "parse_ngspice_metrics",
    "normalize_kpis",
    "run_tb_kpis",
    "render_sim_driver_module",
]

_FLOAT_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_UNIT_VAL_RE = re.compile(
    rf"^\s*({_FLOAT_RE})\s*(meg|t|g|k|m|u|n|p|f|a)?\s*$", re.IGNORECASE
)
_UNIT_MULT: Dict[str, float] = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "a": 1e-18,
}

# Canonical key aliases from common SPICE .meas/.echo names to Clay KPI metrics
_METRIC_ALIASES: Dict[str, str] = {
    "ugf_hz": "gbw_hz",
    "dc_gain": "dc_gain_db",
    "gain_db": "dc_gain_db",
    "phase_margin_deg": "pm_deg",
    "iq_a": "idd_a",
    "vout_27c": "vref_v",
    "vref_27c": "vref_v",
    "vref_val": "vref_v",
    "idd_peak_a_val": "idd_peak_a",
    "idd_avg_eval_a_val": "idd_avg_eval_a",
    "vout_m_final_v": "vout_m_final",
    "vout_p_final_v": "vout_p_final",
    "vout_diff_final_v": "vout_diff_final",
    "tempco": "tempco_ppm",
}


@dataclass(frozen=True)
class TbRunResult:
    tb_path: str
    success: bool
    metrics: Dict[str, float]
    raw_output: str
    returncode: int
    error: Optional[str] = None


def _to_float(tok: str) -> Optional[float]:
    m = _UNIT_VAL_RE.match(str(tok))
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    suf = (m.group(2) or "").lower()
    if suf in _UNIT_MULT:
        v *= _UNIT_MULT[suf]
    return v if math.isfinite(v) else None


_NOISE_METRIC_KEYS = frozenset({
    "no", "reference", "temp", "tnom", "tref", "level", "version",
    "binflag", "scale", "date", "corner", "pdk", "bjt", "dio",
    "res", "at", "from", "to", "rise", "fall", "step", "steps",
})


def parse_ngspice_metrics(output: str) -> Dict[str, float]:
    """Parse ``.meas`` lines and ``echo key=val`` tokens from ngspice stdout/stderr."""
    metrics: Dict[str, float] = {}
    if not output:
        return metrics

    # 1. Standard ngspice `.meas` batch line: `dc_gain_db          =  6.51234e+01`
    for m in re.finditer(
        rf"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*({_FLOAT_RE})\b",
        output,
        re.M,
    ):
        name = m.group(1).strip()
        if name.lower() in _NOISE_METRIC_KEYS:
            continue
        val = _to_float(m.group(2))
        if val is not None:
            metrics[name] = val

    # 2. Key=value tokens inside echoed lines (`echo idd_peak_a_val=$&... tpre_s=20n`)
    for m in re.finditer(
        rf"(?<![A-Za-z0-9_.])([A-Za-z_][A-Za-z0-9_]*)=({_FLOAT_RE}(?:meg|t|g|k|m|u|n|p|f|a)?)\b",
        output,
        re.I,
    ):
        name = m.group(1).strip()
        if name.lower() in _NOISE_METRIC_KEYS:
            continue
        val = _to_float(m.group(2))
        if val is not None:
            metrics.setdefault(name, val)

    return metrics


def normalize_kpis(
    raw_metrics: Mapping[str, float],
    *,
    vdd_v: Optional[float] = None,
    kpi_specs: Sequence[object] = (),
    is_post: bool = False,
) -> Dict[str, float]:
    """Normalize raw SPICE measurement names into canonical Clay KPI dicts."""
    out: Dict[str, float] = {}
    lower_map: Dict[str, float] = {}
    for k, v in raw_metrics.items():
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            fv = float(v)
            out[k] = fv
            lower_map[k.lower()] = fv

    # Derived common analog KPIs
    for src_key, dst_key in _METRIC_ALIASES.items():
        if src_key in lower_map and dst_key not in out:
            out[dst_key] = lower_map[src_key]

    if "iq_ua" in lower_map and "iq_a" not in out:
        out["iq_a"] = abs(lower_map["iq_ua"]) * 1e-6
        out.setdefault("idd_a", out["iq_a"])

    if "t_cross_half_s" in lower_map and "tpre_s" in lower_map and "t_decide_s" not in out:
        dt = lower_map["t_cross_half_s"] - lower_map["tpre_s"]
        if dt > 0:
            out["t_decide_s"] = dt

    if vdd_v is not None and math.isfinite(float(vdd_v)):
        idd = out.get("idd_a") or out.get("iq_a") or out.get("idd_op_a")
        if idd is not None and "power_w" not in out:
            out["power_w"] = abs(float(idd)) * float(vdd_v)
            out.setdefault("power_uw", out["power_w"] * 1e6)

    # Map case-insensitively to requested quality @kpi metrics when available
    for spec in kpi_specs or ():
        metric = str(getattr(spec, "metric", "") or "")
        if not metric or metric in out:
            continue
        mlow = metric.lower()
        if mlow in lower_map:
            out[metric] = lower_map[mlow]

    if is_post:
        post_copy = dict(out)
        for k, v in list(out.items()):
            if not k.endswith("_post"):
                post_copy.setdefault(f"{k}_post", v)
        return post_copy
    return out


def _discover_ngspice() -> str:
    try:
        from clay._ir_bridge import _discover_ngspice_con

        p = _discover_ngspice_con()
        if p and Path(p).exists():
            return str(p)
    except Exception:  # noqa: BLE001
        pass
    try:
        from circuits._harness import ngspice_bin

        p = ngspice_bin()
        if p and Path(p).exists():
            return str(p)
    except Exception:  # noqa: BLE001
        pass
    raise FileNotFoundError("ngspice_con binary not found in environment")


def _subckt_signature(spice_text: str, preferred_name: Optional[str] = None) -> Optional[Tuple[str, Tuple[str, ...]]]:
    matches = list(
        re.finditer(
            r"^\s*\.subckt\s+([A-Za-z_]\w*)\s+([^\n*$;]+)",
            spice_text,
            re.M | re.I,
        )
    )
    if not matches:
        return None
    chosen = matches[-1]
    if preferred_name:
        for m in matches:
            if m.group(1).lower() == preferred_name.lower():
                chosen = m
                break
    ports = tuple(p for p in chosen.group(2).strip().split() if "=" not in p)
    return chosen.group(1), ports


def _rewrite_param_lines(spice_text: str, overrides: Mapping[str, object]) -> str:
    """Return a copy of SPICE text with matching ``.param`` definitions updated."""
    if not overrides:
        return spice_text
    low_map = {str(k).lower(): (str(k), str(v)) for k, v in overrides.items()}
    used: set[str] = set()
    out_lines: List[str] = []
    for ln in spice_text.splitlines():
        s = ln.strip()
        if s.lower().startswith(".param"):
            # Replace each k=v inside this .param line
            def _sub(m: re.Match) -> str:
                k = m.group(1)
                if k.lower() in low_map:
                    canon_k, new_v = low_map[k.lower()]
                    used.add(k.lower())
                    return f"{k}={new_v}"
                return m.group(0)

            ln = re.sub(r"([A-Za-z_]\w*)\s*=\s*([^\s$]+)", _sub, ln)
        out_lines.append(ln)
    missing = [f".param {orig_k}={val}" for lk, (orig_k, val) in low_map.items() if lk not in used]
    if missing:
        return "\n".join(missing) + "\n" + "\n".join(out_lines) + "\n"
    return "\n".join(out_lines) + "\n"


def _strip_embedded_subckt(text: str, target_name: str) -> Tuple[str, bool]:
    """Remove an embedded ``.subckt <target_name> ... .ends`` block when swapping PEX."""
    pattern = re.compile(
        rf"^\s*\.subckt\s+{re.escape(target_name)}\b.*?^\s*\.ends\b[^\n]*",
        re.M | re.S | re.I,
    )
    new_text, count = pattern.subn("", text, count=1)
    return new_text, count > 0


def _build_runnable_deck(
    tb_path: Path,
    dut_path: Path,
    *,
    pdk_id: str,
    corner: str,
    temp_c: float,
    sizing: Optional[Mapping[str, object]],
    testbench_params: Optional[Mapping[str, object]],
    work_dir: Path,
    subckt_name: Optional[str],
    is_post: bool,
) -> str:
    from circuits._harness import _preamble_for, _resolve_marker, _strip_old_pdk_lines

    tb_text = tb_path.read_text(encoding="utf-8")
    dut_text = dut_path.read_text(encoding="utf-8", errors="ignore")

    # Apply sizing overrides to a materialized copy of the DUT if needed
    effective_dut = dut_path.resolve()
    if sizing:
        patched = _rewrite_param_lines(dut_text, sizing)
        if patched != dut_text or not is_post:
            patched_path = work_dir / f"_dut_{dut_path.name}"
            patched_path.write_text(patched, encoding="utf-8")
            effective_dut = patched_path.resolve()

    # Check if PEX subckt name differs from what the testbench instantiates
    alias_block = ""
    if is_post and subckt_name:
        sig = _subckt_signature(dut_text, preferred_name=subckt_name)
        if sig is not None and sig[0].lower() != subckt_name.lower():
            pex_name, pex_ports = sig
            # Emit shim subckt so X1 <ports> <subckt_name> works cleanly
            alias_block = (
                f".subckt {subckt_name} {' '.join(pex_ports)}\n"
                f"X_pex {' '.join(pex_ports)} {pex_name}\n"
                f".ends {subckt_name}\n"
            )

    # Resolve PDK preamble (marker-aware, respecting corner argument)
    marker_pdk, marker_args = None, ""
    for ln in tb_text.splitlines():
        pid, args = _resolve_marker(ln)
        if pid is not None:
            marker_pdk, marker_args = pid, args or ""
            break
    eff_pdk = pdk_id or marker_pdk or "gf180mcu"
    args_eff = re.sub(r"corner\s*=\s*\w+", f"corner={corner}", marker_args) if "corner=" in marker_args else (
        f"{marker_args} corner={corner}".strip()
    )
    if ("bjt=1" not in args_eff) and ("pnp" in dut_text.lower() or "npn" in dut_text.lower()):
        args_eff = f"{args_eff} bjt=1".strip()
    preamble = _preamble_for(eff_pdk, args_eff)

    tb_cleaned = _strip_old_pdk_lines(tb_text)
    stripped_embedded = False
    if subckt_name:
        tb_cleaned, stripped_embedded = _strip_embedded_subckt(tb_cleaned, subckt_name)
    elif is_post:
        sig_tb = _subckt_signature(tb_cleaned)
        if sig_tb is not None:
            tb_cleaned, stripped_embedded = _strip_embedded_subckt(tb_cleaned, sig_tb[0])

    if testbench_params:
        tb_cleaned = _rewrite_param_lines(tb_cleaned, testbench_params)

    # Build final deck lines
    out_lines: List[str] = []
    replaced_marker = False
    swapped_dut = False
    has_temp = False

    for ln in tb_cleaned.splitlines():
        ls = ln.strip().lower()
        pid, _ = _resolve_marker(ln)
        if pid is not None and not replaced_marker:
            out_lines.append(preamble)
            replaced_marker = True
            continue
        if re.match(r"^\.option\s+temp\s*=", ls):
            out_lines.append(f".option temp = {temp_c:g}")
            has_temp = True
            continue
        if ls.startswith(".include") and (".cir" in ls or ".spice" in ls) and "design.ngspice" not in ls:
            out_lines.append(f'.include "{effective_dut.as_posix()}"')
            if alias_block:
                out_lines.append(alias_block)
            swapped_dut = True
            continue
        out_lines.append(ln)

    if not replaced_marker:
        out_lines.insert(0, preamble)
    if not swapped_dut and dut_path.resolve() != tb_path.resolve():
        # Standalone .tb (e.g. tb_bandgap.sp) didn't originally .include a .cir,
        # while effective_dut may itself carry standalone VVDD/.control lines.
        # Extract only .param/.option scale/.subckt ... .ends so we never collide.
        if stripped_embedded or _subckt_signature(dut_text, subckt_name) is not None:
            subckt_only: List[str] = []
            in_sub = False
            for dln in dut_text.splitlines():
                dls = dln.strip().lower()
                if dls.startswith(".subckt"):
                    in_sub = True
                    subckt_only.append(dln)
                elif dls.startswith(".ends"):
                    subckt_only.append(dln)
                    in_sub = False
                elif in_sub or dls.startswith((".param", ".option scale")):
                    subckt_only.append(dln)
            if subckt_only:
                sub_file = work_dir / f"_subckt_{dut_path.name}"
                sub_file.write_text("\n".join(subckt_only) + "\n", encoding="utf-8")
                effective_dut = sub_file.resolve()
        out_lines.insert(1, f'.include "{effective_dut.as_posix()}"')
        if alias_block:
            out_lines.insert(2, alias_block)
    if not has_temp and abs(float(temp_c) - 27.0) > 1e-6:
        out_lines.insert(1, f".option temp = {temp_c:g}")
    if (
        not is_post
        and re.search(r"^\s*\.option\b[^\n]*\bscale\s*=\s*1(\.0+)?u\b", dut_text, re.I | re.M)
        and not re.search(r"^\s*\.option\b[^\n]*\bscale\s*=", tb_cleaned, re.I | re.M)
    ):
        out_lines.insert(1, ".option scale=1.0u")

    return "\n".join(out_lines) + "\n"


def run_tb_kpis(
    cir_or_pex_path: Union[str, Path],
    tb_paths: Sequence[Union[str, Path]],
    *,
    pdk_id: str = "gf180mcu",
    corner: str = "tt",
    temp_c: float = 27.0,
    sizing: Optional[Mapping[str, object]] = None,
    testbench_params: Optional[Mapping[str, object]] = None,
    vdd_v: Optional[float] = None,
    kpi_specs: Sequence[object] = (),
    is_post: bool = False,
    subckt_name: Optional[str] = None,
    timeout_s: float = 120.0,
) -> Dict[str, float]:
    """Run one or more ``.tb``/``.sp`` testbenches on a SPICE netlist and return KPIs."""
    dut = Path(cir_or_pex_path).resolve()
    if not dut.is_file():
        raise FileNotFoundError(f"DUT SPICE netlist not found: {dut}")

    tbs = [Path(p).resolve() for p in tb_paths if p and Path(p).is_file()]
    if not tbs:
        raise FileNotFoundError("No valid testbench paths provided to run_tb_kpis")

    ngspice = _discover_ngspice()
    raw_all: Dict[str, float] = {}
    errors: List[str] = []

    for tb in tbs:
        with tempfile.TemporaryDirectory(prefix="clay_tb_") as td:
            work_dir = Path(td)
            deck = _build_runnable_deck(
                tb,
                dut,
                pdk_id=pdk_id,
                corner=corner,
                temp_c=float(temp_c),
                sizing=sizing,
                testbench_params=testbench_params,
                work_dir=work_dir,
                subckt_name=subckt_name,
                is_post=is_post,
            )
            deck_path = work_dir / f"_run_{tb.name}"
            deck_path.write_text(deck, encoding="utf-8")
            proc = subprocess.run(
                [ngspice, "-b", str(deck_path)],
                cwd=str(tb.parent),
                capture_output=True,
                text=True,
                timeout=float(timeout_s),
            )
            stdout_text = proc.stdout or ""
            stderr_text = proc.stderr or ""
            if proc.returncode != 0:
                diag = (stderr_text or stdout_text).strip()[-500:]
                errors.append(f"{tb.name} (rc={proc.returncode}): {diag}")
                continue
            parsed = parse_ngspice_metrics(stdout_text)
            if not parsed:
                diag = (stderr_text or stdout_text).strip()[-400:]
                errors.append(f"{tb.name} (rc=0, no metrics in stdout): {diag}")
                continue
            raw_all.update(parsed)

    if errors and (not raw_all or any("(rc=" in err for err in errors)):
        raise RuntimeError(
            "ngspice testbench execution failed:\n"
            + "\n".join(errors)
        )

    return normalize_kpis(
        raw_all,
        vdd_v=vdd_v,
        kpi_specs=kpi_specs,
        is_post=is_post,
    )


def render_sim_driver_module(
    *,
    mode: str,
    cir_relpath: str,
    tb_relpaths: Sequence[str],
    pdk_id: str,
    cell_name: str,
    vdd_v: float = 3.3,
) -> str:
    """Render a zero-glue presim or postsim Python module powered by ``tb_adapter``."""
    tb_tuple = ", ".join(repr(p) for p in tb_relpaths)
    if len(tb_relpaths) == 1:
        tb_tuple += ","
    header = (
        f'"""Auto-generated generic {mode} driver for {cell_name} via clay.frontend.tb_adapter."""\n'
        "from __future__ import annotations\n\n"
        "from pathlib import Path\n"
        "from typing import Any, Dict, Optional\n\n"
        "from clay.frontend.tb_adapter import run_tb_kpis\n\n"
        "CASE_DIR = Path(__file__).resolve().parents[1]\n"
        f"SOURCE_CIR = (CASE_DIR / {cir_relpath!r}).resolve()\n"
        f"TB_PATHS = tuple((CASE_DIR / rel).resolve() for rel in ({tb_tuple}))\n"
        f"DEFAULT_PDK = {pdk_id!r}\n"
        f"CELL_NAME = {cell_name!r}\n"
        f"DEFAULT_VDD = {float(vdd_v)!r}\n\n"
    )
    if mode == "presim":
        return header + (
            "def run() -> Dict[str, float]:\n"
            "    return run_tb_kpis(\n"
            "        SOURCE_CIR,\n"
            "        TB_PATHS,\n"
            "        pdk_id=DEFAULT_PDK,\n"
            "        vdd_v=DEFAULT_VDD,\n"
            "        is_post=False,\n"
            "        subckt_name=CELL_NAME,\n"
            "    )\n\n"
            "def run_candidate(\n"
            "    source_cir: str | Path,\n"
            "    *,\n"
            "    testbench_params: Optional[Dict[str, Any]] = None,\n"
            "    sizing: Optional[Dict[str, Any]] = None,\n"
            "    work_dir: Optional[str | Path] = None,\n"
            "    pdk: str = DEFAULT_PDK,\n"
            "    vdd: float = DEFAULT_VDD,\n"
            "    corner: str = 'tt',\n"
            "    temp_c: float = 27.0,\n"
            "    **extra: Any,\n"
            ") -> Dict[str, float]:\n"
            "    merged_sizing = dict(sizing or {})\n"
            "    tb_params = dict(testbench_params or {})\n"
            "    # Route W/L/M/C/R parameters into DUT sizing if matching cir .params\n"
            "    for k, v in list(tb_params.items()):\n"
            "        if k.lower() not in ('vdd', 'vcm', 'ibias', 'cld', 'vin_cm', 'vin_diff'):\n"
            "            merged_sizing.setdefault(k, v)\n"
            "    return run_tb_kpis(\n"
            "        Path(source_cir),\n"
            "        TB_PATHS,\n"
            "        pdk_id=pdk,\n"
            "        corner=corner,\n"
            "        temp_c=float(temp_c),\n"
            "        sizing=merged_sizing or None,\n"
            "        testbench_params=tb_params or None,\n"
            "        vdd_v=float(vdd),\n"
            "        is_post=False,\n"
            "        subckt_name=CELL_NAME,\n"
            "    )\n"
        )
    # Post-sim driver: avoids bare `sizing` in explicit signature so PostSimStep._invoke_declared
    # works seamlessly even when candidate_sizing is None, while **extra still accepts sizing=...,
    # and attaches `__pvt__` evidence required by JointSizingProblem.promote_acceptance().
    return header + (
        "def run_post_kpis(\n"
        "    pex_spice: Path,\n"
        "    *,\n"
        "    pdk: str = DEFAULT_PDK,\n"
        "    corners: tuple[str, ...] = ('tt',),\n"
        "    temps_c: tuple[float, ...] = (27.0,),\n"
        "    pvt_mode: str = 'nominal',\n"
        "    kpi_bounds: Optional[Dict[str, Any]] = None,\n"
        "    simulation_seed: Optional[int] = None,\n"
        "    **extra: Any,\n"
        ") -> Dict[str, Any]:\n"
        "    c_list = tuple(str(c) for c in (corners or ('tt',)))\n"
        "    t_list = tuple(float(t) for t in (temps_c or (27.0,)))\n"
        "    nom_c = 'tt' if 'tt' in c_list else c_list[0]\n"
        "    nom_t = 27.0 if 27.0 in t_list else t_list[0]\n"
        "    nominal = run_tb_kpis(\n"
        "        Path(pex_spice),\n"
        "        TB_PATHS,\n"
        "        pdk_id=pdk,\n"
        "        corner=nom_c,\n"
        "        temp_c=nom_t,\n"
        "        vdd_v=DEFAULT_VDD,\n"
        "        is_post=True,\n"
        "        subckt_name=CELL_NAME,\n"
        "    )\n"
        "    if pvt_mode != 'acceptance' or (len(c_list) == 1 and len(t_list) == 1):\n"
        "        out: Dict[str, Any] = dict(nominal)\n"
        "        if pvt_mode == 'acceptance':\n"
        "            out['__pvt__'] = {\n"
        "                'mode': 'acceptance',\n"
        "                'corners': list(c_list),\n"
        "                'temps_c': list(t_list),\n"
        "                'nominal_post': dict(nominal),\n"
        "            }\n"
        "        return out\n"
        "    samples = [(nom_c, nom_t, nominal)]\n"
        "    for c in c_list:\n"
        "        for t in t_list:\n"
        "            if c == nom_c and t == nom_t:\n"
        "                continue\n"
        "            samples.append((c, t, run_tb_kpis(\n"
        "                Path(pex_spice), TB_PATHS, pdk_id=pdk, corner=c, temp_c=t,\n"
        "                vdd_v=DEFAULT_VDD, is_post=True, subckt_name=CELL_NAME,\n"
        "            )))\n"
        "    agg: Dict[str, Any] = dict(nominal)\n"
        "    bounds = dict(kpi_bounds or {})\n"
        "    for key in list(nominal.keys()):\n"
        "        vals = [float(s[2][key]) for s in samples if key in s[2]]\n"
        "        if not vals:\n"
        "            continue\n"
        "        bnd = bounds.get(key)\n"
        "        op = bnd[0] if isinstance(bnd, (tuple, list)) and bnd else None\n"
        "        if op == 'ge':\n"
        "            agg[key] = min(vals)\n"
        "        elif op == 'le':\n"
        "            agg[key] = max(vals)\n"
        "        elif op == 'near' and len(bnd) >= 2:\n"
        "            tgt = float(bnd[1][0])\n"
        "            agg[key] = max(vals, key=lambda v: abs(v - tgt))\n"
        "    agg['__pvt__'] = {\n"
        "        'mode': 'acceptance',\n"
        "        'corners': list(c_list),\n"
        "        'temps_c': list(t_list),\n"
        "        'nominal_post': dict(nominal),\n"
        "    }\n"
        "    return agg\n"
    )
