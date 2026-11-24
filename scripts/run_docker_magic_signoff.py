"""Reusable GF180MCU Docker silicon-level signoff runner.

Runs real foundry-grade verification inside ``hpretl/iic-osic-tools:latest --skip``
using Magic (``drc(full)``, ``ext2spice lvs``, ``ext2spice extresist on``),
Netgen LVS (``gf180mcuD_setup.tcl``), and host ``ngspice_con`` simulation
(via ``clay.frontend.tb_adapter.run_tb_kpis``), producing a structured JSON
receipt alongside all intermediate logs and extracted SPICE netlists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import klayout.db as kdb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clay.frontend.tb_adapter import (  # noqa: E402
    _build_runnable_deck,
    _discover_ngspice,
    normalize_kpis,
    parse_ngspice_metrics,
)
from clay.layout.pex.joint_extraction import (  # noqa: E402
    _bjt_alias,
    _mos_alias,
    _resolve_bjt_magic_aliases,
    _spice_logical_cards,
    _subckt_ports,
    magic_device_models_for_pdk,
    write_joint_pex,
)
from clay.layout.pex.magic_parser import _parse_spice_value  # noqa: E402
from clay.layout.pex.magic_runner import (  # noqa: E402
    MAGIC_DOCKER_IMAGE,
    MagicTechnologyMismatchError,
    inspect_gf180_mim_stack,
    magic_available,
    validate_gds_technology,
)

PDK_TECH_ROOT = "/foss/pdks/gf180mcuD/libs.tech"
MAGIC_RC = f"{PDK_TECH_ROOT}/magic/gf180mcuD.magicrc"
NETGEN_SETUP = f"{PDK_TECH_ROOT}/netgen/gf180mcuD_setup.tcl"
DOCKER_NGSPICE_DIR = f"{PDK_TECH_ROOT}/ngspice"
DOCKER_MIM_LIB = f"{DOCKER_NGSPICE_DIR}/sm141064_mim.ngspice"

_GF180_RES_MODELS: frozenset[str] = frozenset({
    "nplus_u",
    "pplus_u",
    "nplus_s",
    "pplus_s",
    "nwell",
    "npolyf_u",
    "ppolyf_u",
    "npolyf_s",
    "ppolyf_s",
    "ppolyf_u_1k",
    "ppolyf_u_2k",
    "ppolyf_u_1k_6p0",
    "ppolyf_u_2k_6p0",
    "ppolyf_u_3k",
    "rm1",
    "rm2",
    "rm3",
    "rm4",
    "tm6k",
    "tm9k",
    "tm11k",
    "tm30k",
})

_GF180_RES_CORNER_SECTION: dict[str, str] = {
    "tt": "res_typical",
    "typical": "res_typical",
    "ff": "res_ff",
    "ss": "res_ss",
    "fs": "res_typical",
    "sf": "res_typical",
}

_CAP_VALUE_RE = re.compile(
    r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[a-zA-Z]*$"
)

# Schematic GF180 model -> Netgen gf180mcuD extracted class
_GF180_MOS_TO_NETGEN: dict[str, str] = {
    "nmos_3p3": "nfet_03v3",
    "pmos_3p3": "pfet_03v3",
    "nmos_3p3_dw": "nfet_03v3",
    "nmos_6p0": "nfet_06v0",
    "pmos_6p0": "pfet_06v0",
    "nmos_6p0_nat": "nfet_06v0_nvt",
    "nfet_03v3": "nfet_03v3",
    "pfet_03v3": "pfet_03v3",
    "nfet_06v0": "nfet_06v0",
    "pfet_06v0": "pfet_06v0",
    "nfet_06v0_nvt": "nfet_06v0_nvt",
}

_GF180_BJT_TO_NETGEN: dict[str, str] = {
    "vpnp_0p42x10": "pnp_10p00x00p42",
    "vpnp_0p42x5": "pnp_05p00x00p42",
    "vpnp_10x10": "pnp_10p00x10p00",
    "vpnp_5x5": "pnp_05p00x05p00",
    "pnp_10p00x00p42": "pnp_10p00x00p42",
    "pnp_05p00x00p42": "pnp_05p00x00p42",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _docker_mount_path(path: Path) -> str:
    return path.resolve().as_posix()


def _docker_run(
    inputs_dir: Path,
    work_dir: Path,
    stage: str,
    cmd: Sequence[str],
    *,
    timeout_s: int = 600,
    docker_image: str = MAGIC_DOCKER_IMAGE,
) -> dict[str, Any]:
    argv = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{_docker_mount_path(inputs_dir)}:/inputs:ro",
        "-v",
        f"{_docker_mount_path(work_dir)}:/work",
        docker_image,
        "--skip",
        *cmd,
    ]
    t0 = time.monotonic()
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    elapsed = time.monotonic() - t0
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path = work_dir / f"{stage}.log"
    log_path.write_text(log, encoding="utf-8")
    return {
        "stage": stage,
        "exit_code": int(proc.returncode),
        "seconds": float(elapsed),
        "log": log,
        "log_path": str(log_path),
    }


def _infer_subckt_info(
    cir_path: Path, cell_name: str | None = None
) -> tuple[str, tuple[str, ...]]:
    """Return ``(subckt_name, ordered_ports)`` from a SPICE source file."""
    text = cir_path.read_text(encoding="utf-8", errors="ignore")
    subckts: list[tuple[str, tuple[str, ...]]] = []
    for card in _spice_logical_cards(text):
        toks = card.split()
        if len(toks) >= 2 and toks[0].casefold() == ".subckt":
            ports = tuple(tok for tok in toks[2:] if "=" not in tok)
            subckts.append((toks[1], ports))
    if not subckts:
        raise ValueError(f"No .subckt found in {cir_path}")
    if cell_name:
        for name, ports in subckts:
            if name.casefold() == cell_name.casefold():
                return name, ports
        raise ValueError(f"Declared cell {cell_name!r} not found in {cir_path}")
    return subckts[-1]


def _gds_metrics(gds_path: Path, cell_name: str) -> dict[str, Any]:
    layout = kdb.Layout()
    layout.read(str(gds_path))
    cell = layout.cell(cell_name)
    if cell is None:
        raise ValueError(f"Top cell {cell_name!r} not found in {gds_path}")
    bbox = cell.dbbox()
    bbox_nm = cell.bbox()
    layers = sorted(
        {(int(layout.get_info(idx).layer), int(layout.get_info(idx).datatype))
         for idx in layout.layer_indices()}
    )
    texts: dict[str, list[str]] = {}
    for idx in layout.layer_indices():
        info = layout.get_info(idx)
        key = f"{info.layer}/{info.datatype}"
        for shape in cell.shapes(idx).each():
            if shape.is_text():
                texts.setdefault(key, []).append(shape.text.string)
    for key in texts:
        texts[key] = sorted(set(texts[key]))
    width_um = float(bbox.width())
    height_um = float(bbox.height())
    return {
        "path": str(gds_path),
        "sha256": _sha256(gds_path),
        "bytes": int(gds_path.stat().st_size),
        "top_cell": cell.name,
        "dbu_um": float(layout.dbu),
        "bbox_nm": [
            int(bbox_nm.left),
            int(bbox_nm.bottom),
            int(bbox_nm.right),
            int(bbox_nm.top),
        ],
        "bbox_um": [
            float(bbox.left),
            float(bbox.bottom),
            float(bbox.right),
            float(bbox.top),
        ],
        "width_um": width_um,
        "height_um": height_um,
        "area_um2": width_um * height_um,
        "layers": [list(pair) for pair in layers],
        "text_labels_by_layer": texts,
    }


def _mim_side_um(
    cap_farads: float,
    *,
    density_f_per_um2: float = 2.0e-15,
    grid_nm: int = 5,
    min_area_nm2: float = 25_000_000.0,
    max_area_nm2: float = 10_000_000_000.0,
) -> float:
    area_nm2 = (float(cap_farads) / float(density_f_per_um2)) * 1_000_000.0
    area_nm2 = max(min_area_nm2, min(max_area_nm2, area_nm2))
    side_nm = max(1, int(round(math.sqrt(area_nm2))))
    snapped_nm = ((side_nm + grid_nm - 1) // grid_nm) * grid_nm
    return snapped_nm / 1000.0


_GF180_BJT_TO_SIM_FALLBACK: dict[str, str] = {
    "pnp_10p00x00p42": "vpnp_0p42x10",
    "pnp_05p00x00p42": "vpnp_0p42x5",
    "pnp_10p00x10p00": "vpnp_10x10",
    "pnp_05p00x05p00": "vpnp_5x5",
}


_BARE_NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _apply_scale_to_mos_params(
    params: Sequence[str], scale_factor: float
) -> list[str]:
    """Scale bare numeric W/L tokens by SPICE `.option scale=` for Netgen."""
    if abs(scale_factor - 1.0) <= 1e-18:
        return list(params)
    out: list[str] = []
    for p in params:
        k, sep, v = p.partition("=")
        if sep and k.strip().casefold() in {"w", "l"} and _BARE_NUM_RE.match(v.strip()):
            scaled_um = float(v.strip()) * scale_factor * 1e6
            out.append(f"{k}={scaled_um:g}u")
        else:
            out.append(p)
    return out


def _expand_bjt_card(
    inst: str,
    nodes: Sequence[str],
    bjt_mapped: str,
    params: Sequence[str],
) -> list[str]:
    """Emit Netgen-mergeable BJT subcircuit instances (expanding m=/par= > 1)."""
    mult = 1
    kept_params: list[str] = []
    for p in params:
        k, sep, v = p.partition("=")
        if sep and k.strip().casefold() in {"m", "par"}:
            try:
                mult = max(1, int(round(float(_parse_spice_value(v.strip())))))
            except ValueError:
                kept_params.append(p)
        else:
            kept_params.append(p)
    node_str = " ".join(nodes)
    param_str = (" " + " ".join(kept_params)) if kept_params else ""
    if mult == 1:
        return [f"{inst} {node_str} {bjt_mapped}{param_str}"]
    return [
        f"{inst}_m{idx} {node_str} {bjt_mapped}{param_str}"
        for idx in range(mult)
    ]


def _collect_spice_params(lines: Sequence[str]) -> dict[str, str]:
    """Collect `.param key=value` definitions (case-insensitive) from SPICE lines."""
    params: dict[str, str] = {}
    kv_re = re.compile(r"([A-Za-z_]\w*)\s*=\s*([^\s$]+)")
    for raw_ln in lines:
        s = raw_ln.strip()
        if s.casefold().startswith(".param"):
            body = s[len(".param") :].split("$", 1)[0]
            for m in kv_re.finditer(body):
                params[m.group(1).casefold()] = m.group(2)
    return params


def _resolve_spice_param_token(tok: str, params: Mapping[str, str]) -> str:
    """Substitute `{PARAM}` references in a token using collected `.param` values."""
    if "{" not in tok:
        return tok
    return re.sub(
        r"\{([A-Za-z_]\w*)\}",
        lambda m: params.get(m.group(1).casefold(), m.group(0)),
        tok,
    )


def _translate_cir_for_netgen(
    cir_path: Path,
    out_path: Path,
    *,
    cap_density_f_per_um2: float = 2.0e-15,
    mim_model: str = "cap_mim_2f0_m4m5_noshield",
    poly_res_sheet_ohm_sq: float = 350.0,
    poly_res_width_um: float = 1.0,
) -> dict[str, Any]:
    """Translate a GF180MCU source .cir to gf180mcuD Magic/Netgen device classes."""
    counts: dict[str, Any] = {"mos": 0, "cap": 0, "res": 0, "bjt": 0}
    raw_lines = cir_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    spice_params = _collect_spice_params(raw_lines)
    out_lines: list[str] = []
    scale_factor = 1.0

    for raw_ln in raw_lines:
        stripped = raw_ln.strip()
        if not stripped or stripped.startswith(("*", "+")):
            out_lines.append(raw_ln)
            continue
        if stripped.startswith("."):
            low = stripped.casefold()
            if low.startswith(".option"):
                for tok in stripped.split()[1:]:
                    k, sep, v = tok.partition("=")
                    if sep and k.strip().casefold() == "scale":
                        try:
                            scale_factor = float(_parse_spice_value(v.strip()))
                            counts["spice_scale"] = scale_factor
                        except ValueError:
                            pass
            out_lines.append(raw_ln)
            continue
        toks = [_resolve_spice_param_token(t, spice_params) for t in stripped.split()]
        head = toks[0]
        kind = head[:1].upper()
        # Handle 4-pin MOS (`M...` / `xM...`, >=6 tokens) and 3-pin BJT (`xQ...`, >=5 tokens)
        if kind in {"M", "X"} and len(toks) >= 5:
            pos = [t for t in toks[1:] if "=" not in t]
            params = [t for t in toks[1:] if "=" in t]
            if pos:
                model_tok = pos[-1]
                mapped = _GF180_MOS_TO_NETGEN.get(model_tok.casefold())
                if mapped and len(pos) >= 5:
                    inst = head if kind == "X" else f"X{head}"
                    scaled_params = _apply_scale_to_mos_params(params, scale_factor)
                    out_lines.append(
                        f"{inst} {' '.join(pos[:-1])} {mapped} {' '.join(scaled_params)}".strip()
                    )
                    counts["mos"] += 1
                    continue
                bjt_mapped = _GF180_BJT_TO_NETGEN.get(model_tok.casefold())
                if bjt_mapped and len(pos) >= 4:
                    inst = head if kind == "X" else f"X{head}"
                    expanded = _expand_bjt_card(inst, pos[:-1], bjt_mapped, params)
                    out_lines.extend(expanded)
                    counts["bjt"] += len(expanded)
                    continue
        if kind == "C" and len(toks) >= 4:
            # C<name> <n1> <n2> <value> [params...]
            n1, n2, val_tok = toks[1], toks[2], toks[3]
            cap_val: float | None = None
            try:
                cap_val = float(_parse_spice_value(val_tok))
            except ValueError:
                for p in toks[4:]:
                    if p.casefold().startswith("c="):
                        cap_val = float(_parse_spice_value(p.split("=", 1)[1]))
                        break
            if cap_val is not None:
                side_um = _mim_side_um(
                    cap_val, density_f_per_um2=cap_density_f_per_um2
                )
                # Clay's passive_device_cell places C<name> pin1 ('p') on the
                # bottom plate (*m4) and pin2 ('n') on the top plate (*mimcap).
                # Magic gf180mcuD.tech extracts cap_mim_2f0_m4m5_noshield with
                # pin1=*mimcap (top) and pin2=*m4 (bottom); ideal unpolarized
                # spice 'C' cards follow Clay's terminal-to-plate mapping here,
                # whereas explicit X* subcircuit calls preserve verbatim pins.
                out_lines.append(
                    f"X{head} {n2} {n1} {mim_model} "
                    f"c_width={side_um:g}u c_length={side_um:g}u"
                )
                counts["cap"] += 1
                counts["ideal_c_plate_convention"] = "pin1=n2(top/*mimcap),pin2=n1(bot/*m4)"
                continue
        if kind == "Q" and len(toks) >= 5:
            pos = [t for t in toks[1:] if "=" not in t]
            params = [t for t in toks[1:] if "=" in t]
            bjt_mapped = _GF180_BJT_TO_NETGEN.get(pos[-1].casefold())
            if bjt_mapped and len(pos) >= 4:
                expanded = _expand_bjt_card(f"X{head}", pos[:-1], bjt_mapped, params)
                out_lines.extend(expanded)
                counts["bjt"] += len(expanded)
                continue
        if kind == "R" and len(toks) >= 4:
            pos = [t for t in toks[1:] if "=" not in t]
            if len(pos) >= 3:
                try:
                    r_val = float(_parse_spice_value(pos[2]))
                    r_len = (r_val / poly_res_sheet_ohm_sq) * poly_res_width_um
                    bulk = pos[3] if len(pos) >= 5 else "vss"
                    model_name = pos[4] if len(pos) >= 5 else "ppolyf_u"
                    out_lines.append(
                        f"X{head} {pos[0]} {pos[1]} {bulk} {model_name} "
                        f"r_width={poly_res_width_um:g}u r_length={r_len:.6g}u"
                    )
                    counts["res"] += 1
                    continue
                except ValueError:
                    pass
        out_lines.append(raw_ln)

    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return counts


def _flatten_tcl(cell_name: str) -> str:
    flat = f"{cell_name}_flat_signoff"
    return (
        f"flatten {flat}\n"
        f"load {flat}\n"
        f"cellname delete {cell_name}\n"
        f"cellname rename {flat} {cell_name}\n"
        "select top cell\n"
    )


def _write_docker_tcl_scripts(
    inputs_dir: Path,
    cell_name: str,
    *,
    schematic_filename: str = "schematic_gf180d.cir",
    flatten: bool = True,
) -> None:
    drc_tcl = f"""crashbackups stop
gds read /inputs/layout.gds
load {cell_name}
select top cell
drc euclidean on
drc style drc(full)
drc check
drc catchup
set findings [drc listall why]
set fout [open /work/magic_drc_violations.txt w]
set oscale [cif scale out]
set count 0
foreach {{category boxes}} $findings {{
  puts $fout $category
  foreach box $boxes {{
    puts $fout [format " %.9gum %.9gum %.9gum %.9gum" [expr {{$oscale*[lindex $box 0]}}] [expr {{$oscale*[lindex $box 1]}}] [expr {{$oscale*[lindex $box 2]}}] [expr {{$oscale*[lindex $box 3]}}]]
    incr count
  }}
}}
puts $fout "COUNT: $count"
close $fout
puts "MAGIC_DRC_COUNT=$count"
quit -noprompt
"""
    flat_block = _flatten_tcl(cell_name) if flatten else ""
    lvs_tcl = (
        "crashbackups stop\n"
        "drc off\n"
        "gds read /inputs/layout.gds\n"
        f"load {cell_name}\n"
        "select top cell\n"
        + flat_block
        + "extract path /work/lvs_extract\n"
        "extract no capacitance\n"
        "extract no coupling\n"
        "extract no resistance\n"
        "extract no length\n"
        "extract all\n"
        "ext2spice lvs\n"
        f"ext2spice -p /work/lvs_extract -o /work/lvs_extract/{cell_name}.lvs.spice\n"
        'puts "MAGIC_LVS_EXTRACT_COMPLETE"\n'
        "quit -noprompt\n"
    )
    netgen_tcl = (
        f"lvs {{/work/lvs_extract/{cell_name}.lvs.spice {cell_name}}} "
        f" {{/inputs/{schematic_filename} {cell_name}}} "
        f"{{{NETGEN_SETUP}}} /work/netgen_lvs.out -json\n"
        "set unique [verify unique]\n"
        'puts "NETGEN_UNIQUE=$unique"\n'
        'if {$unique != 1} {error "Netgen LVS is not uniquely equivalent"}\n'
    )
    rc_tcl = (
        "crashbackups stop\n"
        "drc off\n"
        "gds read /inputs/layout.gds\n"
        f"load {cell_name}\n"
        "select top cell\n"
        + flat_block
        + "extract path /work/rc_extract\n"
        "extract all\n"
        "ext2spice lvs\n"
        "ext2spice cthresh 0\n"
        "ext2spice rthresh 0\n"
        "extresist all\n"
        "ext2spice extresist on\n"
        f"ext2spice -p /work/rc_extract -o /work/rc_extract/{cell_name}.rc.pex.spice\n"
        'puts "MAGIC_RC_PEX_COMPLETE"\n'
        "quit -noprompt\n"
    )
    (inputs_dir / "drc.tcl").write_text(drc_tcl, encoding="utf-8")
    (inputs_dir / "lvs.tcl").write_text(lvs_tcl, encoding="utf-8")
    (inputs_dir / "netgen.tcl").write_text(netgen_tcl, encoding="utf-8")
    (inputs_dir / "rc.tcl").write_text(rc_tcl, encoding="utf-8")


def _parse_drc_report(report_path: Path) -> tuple[int | None, dict[str, int]]:
    if not report_path.is_file():
        return None, {}
    by_cat: dict[str, int] = {}
    current_cat: str | None = None
    total_count: int | None = None
    for raw in report_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith("COUNT:"):
            try:
                total_count = int(raw.split(":", 1)[1].strip())
            except ValueError:
                pass
            continue
        if raw.startswith(" "):
            if current_cat is not None:
                by_cat[current_cat] = by_cat.get(current_cat, 0) + 1
        elif raw.strip():
            current_cat = raw.strip()
            by_cat.setdefault(current_cat, 0)
    return total_count, by_cat


def _parse_magic_rc_stats(pex_text: str) -> dict[str, Any]:
    cards = _spice_logical_cards(pex_text)
    r_cnt = 0
    c_vals: list[float] = []
    neg_c = 0
    neg_c_total_f = 0.0
    mos_cnt = 0
    mim_cnt = 0
    other_x = 0
    total_c_f = 0.0
    total_r_ohm = 0.0
    for card in cards:
        toks = card.split()
        if not toks or toks[0].startswith("."):
            continue
        kind = toks[0][:1].upper()
        if kind == "R" and len(toks) >= 4:
            r_cnt += 1
            try:
                total_r_ohm += float(_parse_spice_value(toks[3]))
            except ValueError:
                pass
        elif kind == "C" and len(toks) >= 4:
            val_tok = toks[3]
            if val_tok.startswith("-"):
                neg_c += 1
            try:
                cval = float(_parse_spice_value(val_tok))
                c_vals.append(cval)
                total_c_f += cval
                if cval < 0.0:
                    neg_c_total_f += cval
            except ValueError:
                pass
        elif kind in {"X", "M"}:
            pos = [t for t in toks[1:] if "=" not in t]
            model = pos[-1].casefold() if pos else ""
            if "mim" in model:
                mim_cnt += 1
            elif any(tok in model for tok in ("fet", "mos")):
                mos_cnt += 1
            else:
                other_x += 1
    total_c_ff = total_c_f * 1e15
    neg_c_ff = neg_c_total_f * 1e15
    neg_ratio = abs(neg_c_ff) / total_c_ff if total_c_ff > 0 else float("inf")
    return {
        "resistors": r_cnt,
        "total_wire_r_ohm": total_r_ohm,
        "capacitors": len(c_vals),
        "negative_capacitors": neg_c,
        "negative_cap_total_fF": neg_c_ff,
        "negative_cap_ratio": neg_ratio,
        "total_wire_c_fF": total_c_ff,
        "mos_devices": mos_cnt,
        "mim_devices": mim_cnt,
        "other_subckts": other_x,
        "devices": mos_cnt + mim_cnt + other_x,
    }


def _fetch_docker_mim_subckt(cache_path: Path, model_name: str = "cap_mim_2f0_m4m5_noshield") -> str:
    """Fetch exact MIM subcircuit definition from Docker gf180mcuD ngspice libs."""
    if cache_path.is_file():
        cached = cache_path.read_text(encoding="utf-8")
        if f".subckt {model_name}" in cached:
            return cached
    proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            MAGIC_DOCKER_IMAGE,
            "--skip",
            "bash",
            "-lc",
            f"sed -n '248,300p' {DOCKER_MIM_LIB}",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to fetch {model_name} from Docker PDK:\n{proc.stderr}")
    kept: list[str] = []
    started = False
    for ln in proc.stdout.splitlines():
        if ln.lstrip().startswith("[INFO]"):
            continue
        if ln.strip().startswith(f".subckt {model_name}"):
            started = True
        if started:
            kept.append(ln)
        if started and ln.strip().startswith(f".ends {model_name}"):
            break
    if len(kept) < 5:
        raise RuntimeError(f"Could not locate .subckt {model_name} in {DOCKER_MIM_LIB}")
    content = "\n".join([
        f"* Exact gf180mcuD MIM subcircuit from {DOCKER_MIM_LIB}",
        ".param mim_corner_2p0fF=1",
        ".param mc_c_cox_2p0fF=0",
        *kept,
        "",
    ])
    cache_path.write_text(content, encoding="utf-8")
    return content


def _prepare_simulatable_magic_pex(
    raw_pex_path: Path,
    out_sim_path: Path,
    *,
    cell_name: str,
    external_ports: Sequence[str],
    pdk: str = "gf180mcu",
    mim_cache_dir: Path,
) -> dict[str, Any]:
    """Wrap raw Magic ext2spice output with exact schematic port order & PDK aliases."""
    raw_text = raw_pex_path.read_text(encoding="utf-8")
    magic_ports = _subckt_ports(raw_text, cell_name)
    by_fold = {port.casefold(): port for port in magic_ports}
    missing = [p for p in external_ports if p.casefold() not in by_fold]
    if missing:
        raise ValueError(
            f"Magic PEX .subckt {cell_name} is missing external schematic port(s): {missing!r} "
            f"(extracted ports={magic_ports!r})"
        )

    raw_name = f"{cell_name}__magic_raw"
    header_re = re.compile(rf"(?im)^(\s*\.subckt\s+){re.escape(cell_name)}(?=\s)")
    renamed, count = header_re.subn(rf"\g<1>{raw_name}", raw_text, count=1)
    if count != 1:
        raise ValueError(f"Could not rename raw Magic .subckt {cell_name}")

    # Collect instantiated subckt model tokens to emit only needed wrappers
    instantiated: set[str] = set()
    for card in _spice_logical_cards(raw_text):
        toks = card.split()
        if toks and toks[0][:1].casefold() == "x":
            pos = [t for t in toks[1:] if "=" not in t]
            if pos:
                instantiated.add(pos[-1].casefold())

    aliases = magic_device_models_for_pdk(pdk, include_deep_nwell=True)
    header_blocks: list[str] = [
        "* Simulatable Magic R+C PEX wrapper (exact schematic port order + PDK aliases).",
        "",
    ]
    aliased_models: list[str] = []
    for raw_alias, sim_model in sorted(aliases.items()):
        if raw_alias.casefold() in instantiated:
            header_blocks.extend(_mos_alias(raw_alias, sim_model))
            aliased_models.append(f"{raw_alias}->{sim_model}")

    bjt_aliases = dict(_GF180_BJT_TO_SIM_FALLBACK) if pdk == "gf180mcu" else {}
    bjt_aliases.update(_resolve_bjt_magic_aliases(raw_text, pdk=pdk))
    bjt_aliased_models: list[str] = []
    for raw_alias, sim_model in sorted(bjt_aliases.items()):
        if raw_alias.casefold() in instantiated:
            header_blocks.extend(_bjt_alias(raw_alias, sim_model))
            bjt_aliased_models.append(f"{raw_alias}->{sim_model}")

    mim_models_included: list[str] = []
    docker_mim_blocks: list[str] = []
    if "cap_mim_2f0_m4m5_noshield" in instantiated:
        mim_text = _fetch_docker_mim_subckt(
            mim_cache_dir / "gf180mcuD_cap_mim_2f0_m4m5_noshield.spice",
            "cap_mim_2f0_m4m5_noshield",
        )
        header_blocks.append("")
        header_blocks.append(mim_text.rstrip())
        docker_mim_blocks.append(mim_text.rstrip())
        mim_models_included.append("cap_mim_2f0_m4m5_noshield")

    # Map magic ports in exact order to schematic port names (with int_ prefix if any extra)
    ext_fold = {p.casefold(): p for p in external_ports}
    xraw_ports = [
        ext_fold.get(mp.casefold(), f"int_{mp}") for mp in magic_ports
    ]

    wrapper_lines = [
        *header_blocks,
        "",
        renamed.rstrip(),
        "",
        f".subckt {cell_name} {' '.join(external_ports)}",
        f"XRAW {' '.join(xraw_ports)} {raw_name}",
        f".ends {cell_name}",
        "",
    ]
    out_sim_path.write_text("\n".join(wrapper_lines), encoding="utf-8")

    # Docker foundry PDK (gf180mcuD) natively defines nfet_03v3/pfet_03v3/pnp_*;
    # keep native device names intact so it does not collide with sm141064.ngspice.
    docker_sim_path = out_sim_path.with_name(f"{cell_name}.magic_rc.docker_sim.spice")
    docker_header_lines = [
        "* Docker gf180mcuD native Magic R+C PEX wrapper (exact schematic port order).",
        "",
    ]
    if docker_mim_blocks:
        docker_header_lines.extend(docker_mim_blocks)
        docker_header_lines.append("")
    docker_wrapper_lines = [
        *docker_header_lines,
        renamed.rstrip(),
        "",
        f".subckt {cell_name} {' '.join(external_ports)}",
        f"XRAW {' '.join(xraw_ports)} {raw_name}",
        f".ends {cell_name}",
        "",
    ]
    docker_sim_path.write_text("\n".join(docker_wrapper_lines), encoding="utf-8")

    return {
        "sim_pex_path": str(out_sim_path),
        "sim_pex_sha256": _sha256(out_sim_path),
        "docker_sim_pex_path": str(docker_sim_path),
        "docker_sim_pex_sha256": _sha256(docker_sim_path),
        "magic_ports": list(magic_ports),
        "schematic_ports": list(external_ports),
        "mos_aliases": aliased_models,
        "bjt_aliases": bjt_aliased_models,
        "mim_models": mim_models_included,
    }


_NGSPICE_ABORT_RE = re.compile(
    r"(?:run simulation\(s\) aborted|fatal error|unknown subckt|could not find library)",
    re.I,
)


def _spice_uses_gf180_resistors(spice_text: str) -> bool:
    """Return True when SPICE text instantiates any GF180 PDK resistor subckt."""
    for card in _spice_logical_cards(spice_text):
        toks = card.split()
        if toks and toks[0][:1].casefold() == "x":
            pos = [t for t in toks[1:] if "=" not in t]
            if pos and pos[-1].casefold() in _GF180_RES_MODELS:
                return True
    return False


def _inject_gf180_res_lib(
    deck: str,
    *,
    pdk_id: str,
    corner: str,
    uses_resistors: bool,
) -> str:
    """Ensure ``sm141064.ngspice`` resistor corner ``res_<corner>`` is loaded."""
    if pdk_id != "gf180mcu" or not uses_resistors:
        return deck
    if re.search(
        r"(?im)^\s*\.lib\s+.*sm141064\.ngspice.*\bres_(?:typical|ff|ss)\b",
        deck,
    ):
        return deck
    res_section = _GF180_RES_CORNER_SECTION.get(corner.lower(), "res_typical")
    m = re.search(
        r'(?im)^([ \t]*\.lib[ \t]+(["\']?)([^"\'\r\n]*sm141064\.ngspice)\2[ \t]+\S+[^\r\n]*)$',
        deck,
    )
    if not m:
        return deck
    return deck[: m.end()] + f'\n.lib "{m.group(3)}" {res_section}' + deck[m.end() :]


def _ensure_host_compat_pdk_dir(work_dir: Path, pdk_models_dir: Path) -> Path:
    """Materialize a runtime-patched GF180 ngspice PDK dir under ``work_dir``."""
    compat_dir = work_dir / "ngspice_pdk_compat"
    compat_dir.mkdir(parents=True, exist_ok=True)
    orig_design = pdk_models_dir / "design.ngspice"
    dst_design = compat_dir / "design.ngspice"
    if orig_design.is_file() and not dst_design.is_file():
        shutil.copy2(orig_design, dst_design)
    orig_lib = pdk_models_dir / "sm141064.ngspice"
    dst_lib = compat_dir / "sm141064.ngspice"
    if orig_lib.is_file() and not dst_lib.is_file():
        lib_text = orig_lib.read_text(encoding="utf-8", errors="ignore")
        patched = re.sub(
            r"(?m)^([ \t]*rb[ \t]+\S+[ \t]+\S+[ \t]+\S+)[ \t]+(l=r_l[ \t]+w=r_w)[ \t]*\r?\n(\+r=[^\r\n]+)",
            r"\1\n\3 \2",
            lib_text,
        )
        dst_lib.write_text(patched, encoding="utf-8")
    return compat_dir


def _strip_docker_info_lines(text: str) -> str:
    return "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("[INFO]")
    )


def _rewrite_deck_includes(
    deck: str,
    *,
    work_dir: Path,
    sim_dir: Path,
    tb_dir: Path,
    for_docker: bool,
    host_compat_dir: Path | None = None,
) -> str:
    """Rewrite PDK and include paths cleanly for Docker or host compat runs."""
    out_lines: list[str] = []
    staged_idx = 0
    inc_re = re.compile(
        r'^([ \t]*\.(include|lib)[ \t]+)(["\']?)([^"\'\s]+)\3(.*)$',
        re.I,
    )
    for ln in deck.splitlines():
        m = inc_re.match(ln)
        if not m:
            out_lines.append(ln)
            continue
        prefix, _, _, raw_target, suffix = m.groups()
        low_name = Path(raw_target.replace("\\", "/")).name.casefold()
        if for_docker:
            if low_name == "design.ngspice":
                out_lines.append(f'{prefix}"{DOCKER_NGSPICE_DIR}/design.ngspice"{suffix}')
                continue
            if low_name == "sm141064.ngspice":
                out_lines.append(f'{prefix}"{DOCKER_NGSPICE_DIR}/sm141064.ngspice"{suffix}')
                continue
            cand = Path(raw_target)
            if not cand.is_absolute():
                cand = (tb_dir / cand).resolve()
            else:
                cand = cand.resolve()
            mapped: str | None = None
            for root, mount in ((work_dir, "/work"), (tb_dir, "/tbdir")):
                try:
                    rel = cand.relative_to(root.resolve())
                    mapped = f"{mount}/{rel.as_posix()}"
                    break
                except ValueError:
                    continue
            if mapped is None and cand.is_file():
                staged_name = f"_staged_{staged_idx}_{cand.name}"
                staged_idx += 1
                staged_path = sim_dir / staged_name
                shutil.copy2(cand, staged_path)
                rel = staged_path.resolve().relative_to(work_dir.resolve())
                mapped = f"/work/{rel.as_posix()}"
            out_lines.append(
                f'{prefix}"{mapped}"{suffix}' if mapped is not None else ln
            )
            continue

        if host_compat_dir is not None:
            if low_name == "design.ngspice":
                out_lines.append(
                    f'{prefix}"{(host_compat_dir / "design.ngspice").as_posix()}"{suffix}'
                )
                continue
            if low_name == "sm141064.ngspice":
                out_lines.append(
                    f'{prefix}"{(host_compat_dir / "sm141064.ngspice").as_posix()}"{suffix}'
                )
                continue
        out_lines.append(ln)
    return "\n".join(out_lines) + "\n"


_HOST_TO_DOCKER_MOS: dict[str, str] = {
    "nmos_3p3": "nfet_03v3",
    "pmos_3p3": "pfet_03v3",
    "nmos_3p3_dw": "nfet_03v3",
    "nmos_6p0": "nfet_06v0",
    "pmos_6p0": "pfet_06v0",
    "nmos_6p0_nat": "nfet_06v0_nvt",
}

_HOST_TO_DOCKER_BJT: dict[str, str] = {
    "vpnp_0p42x10": "pnp_10p00x00p42",
    "vpnp_0p42x5": "pnp_05p00x00p42",
    "vpnp_10x10": "pnp_10p00x10p00",
    "vpnp_5x5": "pnp_05p00x05p00",
}


def _stage_docker_reverse_aliases_if_needed(
    dut_path: Path, sim_dir: Path, *, pdk_id: str
) -> Path:
    """Wrap host PDK model names (nmos_3p3/vpnp_*) to gf180mcuD native names when needed."""
    if pdk_id != "gf180mcu":
        return dut_path
    text = dut_path.read_text(encoding="utf-8", errors="ignore")
    instantiated: set[str] = set()
    defined: set[str] = set()
    for card in _spice_logical_cards(text):
        toks = card.split()
        if not toks:
            continue
        head = toks[0].casefold()
        if head == ".subckt" and len(toks) >= 2:
            defined.add(toks[1].casefold())
        elif head.startswith("x"):
            pos = [t for t in toks[1:] if "=" not in t]
            if pos:
                instantiated.add(pos[-1].casefold())
    header: list[str] = []
    for src_name, dst_name in sorted(_HOST_TO_DOCKER_MOS.items()):
        if src_name in instantiated and src_name not in defined:
            header.extend(_mos_alias(src_name, dst_name))
    for src_name, dst_name in sorted(_HOST_TO_DOCKER_BJT.items()):
        if src_name in instantiated and src_name not in defined:
            header.extend(_bjt_alias(src_name, dst_name))
    if not header:
        return dut_path
    staged = sim_dir / f"_docker_alias_{dut_path.name}"
    staged.write_text("\n".join(header) + "\n\n" + text, encoding="utf-8")
    return staged


def _run_verified_tb_kpis(
    dut_path: Path,
    tb_paths: Sequence[Path],
    *,
    work_dir: Path,
    stage_label: str,
    pdk_id: str = "gf180mcu",
    corner: str = "tt",
    temp_c: float = 27.0,
    vdd_v: float | None = 3.3,
    is_post: bool = False,
    subckt_name: str | None = None,
    docker_dut_path: Path | None = None,
    expected_kpis: Sequence[str] = (),
    ngspice_backend: str = "auto",
    timeout_s: int = 300,
) -> dict[str, Any]:
    """Execute ngspice testbenches strictly: exit code 0 + valid stdout KPIs."""
    work_dir = Path(work_dir).resolve()
    dut = Path(dut_path).resolve()
    dut_text = dut.read_text(encoding="utf-8", errors="ignore")
    dut_uses_res = _spice_uses_gf180_resistors(dut_text)
    raw_all: dict[str, float] = {}
    runs: list[dict[str, Any]] = []
    errors: list[str] = []
    all_passed = bool(tb_paths)
    expected_base = [k for k in expected_kpis if not k.endswith("_post")]

    for idx, tb in enumerate(tb_paths):
        tb_resolved = Path(tb).resolve()
        tb_text = tb_resolved.read_text(encoding="utf-8", errors="ignore")
        uses_res = dut_uses_res or _spice_uses_gf180_resistors(tb_text)

        backend = ngspice_backend
        if backend == "auto":
            backend = (
                "docker"
                if (pdk_id == "gf180mcu" and uses_res and magic_available())
                else "host"
            )

        run_tag = f"{stage_label}_{idx}_{tb_resolved.stem}"
        sim_dir = (work_dir / "ngspice_runs" / run_tag).resolve()
        sim_dir.mkdir(parents=True, exist_ok=True)
        effective_dut_src = (
            Path(docker_dut_path).resolve()
            if (backend == "docker" and docker_dut_path is not None)
            else dut
        )
        if backend == "docker" and docker_dut_path is None:
            effective_dut_src = _stage_docker_reverse_aliases_if_needed(
                effective_dut_src, sim_dir, pdk_id=pdk_id
            )

        deck = _build_runnable_deck(
            tb_resolved,
            effective_dut_src,
            pdk_id=pdk_id,
            corner=corner,
            temp_c=float(temp_c),
            sizing=None,
            testbench_params=None,
            work_dir=sim_dir,
            subckt_name=subckt_name,
            is_post=is_post,
        )
        deck = _inject_gf180_res_lib(
            deck,
            pdk_id=pdk_id,
            corner=corner,
            uses_resistors=uses_res,
        )

        host_compat_dir: Path | None = None
        if backend == "host" and pdk_id == "gf180mcu" and uses_res:
            from circuits._harness import gf180_models_dir

            host_compat_dir = _ensure_host_compat_pdk_dir(work_dir, gf180_models_dir())

        deck = _rewrite_deck_includes(
            deck,
            work_dir=work_dir,
            sim_dir=sim_dir,
            tb_dir=tb_resolved.parent,
            for_docker=(backend == "docker"),
            host_compat_dir=host_compat_dir,
        )
        deck_path = (sim_dir / f"_run_{tb_resolved.name}").resolve()
        deck_path.write_text(deck, encoding="utf-8")

        if backend == "docker":
            (sim_dir / ".spiceinit").write_text(
                "set num_threads=4\nset ngbehavior=hsa\nset ng_nomodcheck\n",
                encoding="utf-8",
            )
            rel_sim_dir = sim_dir.relative_to(work_dir)
            rel_deck = deck_path.relative_to(work_dir)
            argv = [
                "docker",
                "run",
                "--rm",
                "-w",
                f"/work/{rel_sim_dir.as_posix()}",
                "-v",
                f"{_docker_mount_path(work_dir)}:/work",
                "-v",
                f"{_docker_mount_path(tb_resolved.parent)}:/tbdir:ro",
                MAGIC_DOCKER_IMAGE,
                "--skip",
                "ngspice",
                "-b",
                f"/work/{rel_deck.as_posix()}",
            ]
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=float(timeout_s),
            )
            stdout_clean = _strip_docker_info_lines(proc.stdout or "")
        else:
            ngspice_exe = _discover_ngspice()
            proc = subprocess.run(
                [ngspice_exe, "-b", str(deck_path)],
                cwd=str(tb_resolved.parent),
                capture_output=True,
                text=True,
                timeout=float(timeout_s),
            )
            stdout_clean = proc.stdout or ""

        stderr_text = proc.stderr or ""
        log_path = work_dir / f"ngspice_{run_tag}.log"
        log_path.write_text(
            f"=== STDOUT (rc={proc.returncode}, backend={backend}) ===\n"
            + (proc.stdout or "")
            + "\n=== STDERR ===\n"
            + stderr_text,
            encoding="utf-8",
        )

        # Parse ONLY ngspice stdout (never stderr) and enforce returncode == 0.
        parsed_stdout = parse_ngspice_metrics(stdout_clean) if proc.returncode == 0 else {}
        norm_tb = (
            normalize_kpis(parsed_stdout, vdd_v=vdd_v, is_post=is_post)
            if parsed_stdout
            else {}
        )
        abort_match = _NGSPICE_ABORT_RE.search(stdout_clean + "\n" + stderr_text)
        missing_keys = (
            [k for k in expected_base if k not in norm_tb]
            if (len(tb_paths) == 1 and expected_base)
            else []
        )
        tb_ok = (
            proc.returncode == 0
            and abort_match is None
            and len(parsed_stdout) > 0
            and not missing_keys
        )

        err_msg: str | None = None
        if not tb_ok:
            reasons: list[str] = []
            if proc.returncode != 0:
                reasons.append(f"returncode={proc.returncode}")
            if abort_match is not None:
                reasons.append(f"abort={abort_match.group(0)!r}")
            if proc.returncode == 0 and len(parsed_stdout) == 0:
                reasons.append("no_stdout_metrics")
            if missing_keys:
                reasons.append(f"missing_expected_kpis={missing_keys}")
            tail = (stderr_text.strip() or stdout_clean.strip())[-400:]
            err_msg = f"{tb_resolved.name} ({', '.join(reasons)}): {tail}"
            errors.append(err_msg)
            all_passed = False
        else:
            raw_all.update(parsed_stdout)

        runs.append({
            "testbench": str(tb_resolved),
            "backend": backend,
            "uses_gf180_resistors": uses_res,
            "res_lib_injected": bool(pdk_id == "gf180mcu" and uses_res),
            "exit_code": int(proc.returncode),
            "stdout_metric_count": len(parsed_stdout),
            "missing_expected_kpis": missing_keys,
            "log_path": str(log_path),
            "pass": bool(tb_ok),
            "error": err_msg,
        })

    if not all_passed:
        return {
            "pass": False,
            "kpis": {},
            "runs": runs,
            "error": "; ".join(errors) or "ngspice run failed",
        }

    normalized = normalize_kpis(raw_all, vdd_v=vdd_v, is_post=is_post)
    missing_total = [k for k in expected_base if k not in normalized]
    if not normalized or missing_total:
        err = f"Missing expected KPI keys: {missing_total}" if missing_total else "No valid KPIs extracted"
        return {
            "pass": False,
            "kpis": {},
            "runs": runs,
            "error": err,
        }

    return {
        "pass": True,
        "kpis": normalized,
        "runs": runs,
        "error": None,
    }


def _relative_deltas(
    baseline: Mapping[str, float],
    target: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    deltas: dict[str, dict[str, float]] = {}
    for k, v_target in target.items():
        if k.endswith("_post"):
            continue
        v_base = baseline.get(k)
        if v_base is None or not (math.isfinite(v_base) and math.isfinite(v_target)):
            continue
        diff = v_target - v_base
        rel_pct = (diff / abs(v_base) * 100.0) if abs(v_base) > 1e-30 else 0.0
        deltas[k] = {
            "pre": float(v_base),
            "post": float(v_target),
            "delta": float(diff),
            "delta_pct": float(rel_pct),
        }
    return deltas


def run_docker_magic_signoff(
    *,
    gds_path: Path,
    cir_path: Path,
    out_dir: Path,
    cell_name: str | None = None,
    tb_paths: Sequence[Path] = (),
    pdk: str = "gf180mcu",
    corner: str = "tt",
    temp_c: float = 27.0,
    vdd_v: float = 3.3,
    cap_density_f_per_um2: float = 2.0e-15,
    poly_res_width_um: float = 1.0,
    lvsdb_path: Path | None = None,
    geo_pex_path: Path | None = None,
    probe_drc_on_mismatch: bool = False,
    strict_zero_neg_c: bool = False,
    ngspice_backend: str = "auto",
    timeout_s: int = 600,
) -> dict[str, Any]:
    gds_path = gds_path.resolve(strict=True)
    cir_path = cir_path.resolve(strict=True)
    out_dir = out_dir.resolve()
    inputs_dir = out_dir / "inputs"
    work_dir = out_dir / "work"
    (work_dir / "lvs_extract").mkdir(parents=True, exist_ok=True)
    (work_dir / "rc_extract").mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    resolved_cell, external_ports = _infer_subckt_info(cir_path, cell_name)
    gds_info = _gds_metrics(gds_path, resolved_cell)
    mim_stack = inspect_gf180_mim_stack(gds_path, resolved_cell) if pdk == "gf180mcu" else {"has_mim": False}

    receipt: dict[str, Any] = {
        "schema": "clay.gf180-docker-magic-signoff/v1",
        "docker_image": MAGIC_DOCKER_IMAGE,
        "docker_magic_available": bool(magic_available()),
        "pdk": pdk,
        "cell_name": resolved_cell,
        "external_ports": list(external_ports),
        "gds": gds_info,
        "schematic_cir": {
            "path": str(cir_path),
            "sha256": _sha256(cir_path),
        },
        "mim_stack_inspection": mim_stack,
        "technology_gate": {"pass": True, "error_type": None, "message": None},
        "status": "FAIL",
    }

    if not receipt["docker_magic_available"]:
        raise RuntimeError(f"Docker image {MAGIC_DOCKER_IMAGE!r} is unavailable")

    shutil.copy2(gds_path, inputs_dir / "layout.gds")
    shutil.copy2(cir_path, inputs_dir / cir_path.name)
    translation_counts = _translate_cir_for_netgen(
        cir_path,
        inputs_dir / "schematic_gf180d.cir",
        cap_density_f_per_um2=cap_density_f_per_um2,
        poly_res_width_um=poly_res_width_um,
    )
    receipt["netgen_schematic_translation"] = translation_counts
    _write_docker_tcl_scripts(inputs_dir, resolved_cell)

    try:
        validate_gds_technology(gds_path, resolved_cell, pdk)
    except MagicTechnologyMismatchError as exc:
        receipt["technology_gate"] = {
            "pass": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        if probe_drc_on_mismatch:
            drc_res = _docker_run(
                inputs_dir,
                work_dir,
                "magic_drc",
                ["magic", "-dnull", "-noconsole", "-rcfile", MAGIC_RC, "/inputs/drc.tcl"],
                timeout_s=timeout_s,
            )
            drc_count, by_cat = _parse_drc_report(work_dir / "magic_drc_violations.txt")
            receipt["magic_drc"] = {
                "pass": drc_res["exit_code"] == 0 and drc_count == 0,
                "exit_code": drc_res["exit_code"],
                "violations": drc_count,
                "by_category": by_cat,
                "seconds": drc_res["seconds"],
                "report_path": str(work_dir / "magic_drc_violations.txt"),
            }
        receipt_path = out_dir / "docker_signoff_summary.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt

    # 1. Magic full DRC
    drc_res = _docker_run(
        inputs_dir,
        work_dir,
        "magic_drc",
        ["magic", "-dnull", "-noconsole", "-rcfile", MAGIC_RC, "/inputs/drc.tcl"],
        timeout_s=timeout_s,
    )
    drc_count, by_cat = _parse_drc_report(work_dir / "magic_drc_violations.txt")
    receipt["magic_drc"] = {
        "pass": drc_res["exit_code"] == 0 and drc_count == 0,
        "exit_code": drc_res["exit_code"],
        "violations": drc_count,
        "by_category": by_cat,
        "seconds": drc_res["seconds"],
        "report_path": str(work_dir / "magic_drc_violations.txt"),
        "log_path": drc_res["log_path"],
    }

    # 2. Magic LVS extract + Netgen LVS
    lvs_ext = _docker_run(
        inputs_dir,
        work_dir,
        "magic_lvs_extract",
        ["magic", "-dnull", "-noconsole", "-rcfile", MAGIC_RC, "/inputs/lvs.tcl"],
        timeout_s=timeout_s,
    )
    netgen_res = _docker_run(
        inputs_dir,
        work_dir,
        "netgen_lvs",
        ["bash", "-c", "cd /work && netgen -batch source /inputs/netgen.tcl"],
        timeout_s=timeout_s,
    )
    netgen_out = (
        (work_dir / "netgen_lvs.out").read_text(encoding="utf-8", errors="replace")
        if (work_dir / "netgen_lvs.out").is_file()
        else ""
    )
    prop_err = bool(
        re.search(
            r"(?im)^(?:Property errors|Netlists do not match|Circuits do not match|The following cells had property errors)",
            netgen_out,
        )
    )
    circuits_unique = "Circuits match uniquely." in netgen_out
    netlists_unique = (
        circuits_unique or "Netlists match uniquely" in netgen_out
    )
    unique_ok = "NETGEN_UNIQUE=1" in netgen_res["log"] and circuits_unique and not prop_err

    lvs_json_path = work_dir / "netgen_lvs.json"
    badnets_cnt = None
    badelems_cnt = None
    extra_gds_labels: list[str] = []
    ext_pins_matched = False
    if lvs_json_path.is_file():
        try:
            nj = json.loads(lvs_json_path.read_text(encoding="utf-8"))
            top_nj = next(
                (
                    entry
                    for entry in nj
                    if isinstance(entry, dict)
                    and entry.get("name") == [resolved_cell, resolved_cell]
                ),
                nj[-1] if nj else {},
            )
            badnets_cnt = len(top_nj.get("badnets") or [])
            badelems_cnt = len(top_nj.get("badelements") or [])
            pins_pair = top_nj.get("pins") or []
            if len(pins_pair) == 2 and len(pins_pair[0]) == len(pins_pair[1]):
                matched_pairs = dict(zip(pins_pair[0], pins_pair[1]))
                extra_gds_labels = [
                    p1 for p1, p2 in matched_pairs.items() if p2 == "(no matching pin)"
                ]
                ext_pins_matched = all(
                    matched_pairs.get(p) == p for p in receipt["external_ports"]
                )
        except Exception:  # noqa: BLE001
            pass

    receipt["netgen_lvs"] = {
        "pass": lvs_ext["exit_code"] == 0 and netgen_res["exit_code"] == 0 and unique_ok,
        "magic_extract_exit_code": lvs_ext["exit_code"],
        "netgen_exit_code": netgen_res["exit_code"],
        "magic_extract_seconds": lvs_ext["seconds"],
        "netgen_seconds": netgen_res["seconds"],
        "circuits_match_uniquely": circuits_unique,
        "netlists_match_uniquely": netlists_unique,
        "topology_and_external_pins_match": bool(
            netlists_unique
            and not prop_err
            and badnets_cnt == 0
            and badelems_cnt == 0
            and ext_pins_matched
        ),
        "badnets_count": badnets_cnt,
        "badelements_count": badelems_cnt,
        "external_pins_matched": ext_pins_matched,
        "extra_gds_port_labels": extra_gds_labels,
        "property_errors": prop_err,
        "report_path": str(work_dir / "netgen_lvs.out"),
        "json_path": str(lvs_json_path) if lvs_json_path.is_file() else None,
    }

    # 3. Magic R+C PEX (extresist)
    rc_res = _docker_run(
        inputs_dir,
        work_dir,
        "magic_rc_pex",
        ["magic", "-dnull", "-noconsole", "-rcfile", MAGIC_RC, "/inputs/rc.tcl"],
        timeout_s=timeout_s,
    )
    pex_file = work_dir / "rc_extract" / f"{resolved_cell}.rc.pex.spice"
    pex_text = pex_file.read_text(encoding="utf-8") if pex_file.is_file() else ""
    rc_stats = _parse_magic_rc_stats(pex_text)
    shorts_flag = bool(re.search(r"Ports .*electrically shorted", rc_res["log"], re.I))
    neg_ok = (
        rc_stats["negative_capacitors"] == 0
        if strict_zero_neg_c
        else rc_stats["negative_cap_ratio"] <= 0.02
    )
    magic_rc_ok = (
        rc_res["exit_code"] == 0
        and pex_file.is_file()
        and not shorts_flag
        and rc_stats["resistors"] > 0
        and rc_stats["capacitors"] > 0
        and rc_stats["total_wire_c_fF"] > 0
        and neg_ok
        and rc_stats["devices"] > 0
    )
    receipt["magic_rc_pex"] = {
        "pass": magic_rc_ok,
        "exit_code": rc_res["exit_code"],
        "port_shorts": shorts_flag,
        **rc_stats,
        "pex_path": str(pex_file),
        "pex_sha256": _sha256(pex_file) if pex_file.is_file() else None,
        "seconds": rc_res["seconds"],
        "log_path": rc_res["log_path"],
    }

    # 4. Prepare simulatable Magic R+C wrapper + optional joint PEX check
    sim_pex_info: dict[str, Any] | None = None
    if pex_file.is_file() and not shorts_flag:
        sim_pex_path = work_dir / f"{resolved_cell}.magic_rc.sim.spice"
        sim_pex_info = _prepare_simulatable_magic_pex(
            pex_file,
            sim_pex_path,
            cell_name=resolved_cell,
            external_ports=external_ports,
            pdk=pdk,
            mim_cache_dir=work_dir,
        )
        receipt["magic_rc_pex"]["simulatable_wrapper"] = sim_pex_info

        if lvsdb_path and lvsdb_path.is_file():
            try:
                magic_ports = _subckt_ports(pex_text, resolved_cell)
                joint_path, passives = write_joint_pex(
                    pex_file,
                    work_dir / f"{resolved_cell}.joint_rc.pex.spice",
                    lvsdb_path=lvsdb_path,
                    cell_name=resolved_cell,
                    external_ports=external_ports,
                    required_layout_nets=magic_ports,
                    magic_device_models=magic_device_models_for_pdk(pdk, include_deep_nwell=True),
                    pdk=pdk,
                )
                receipt["joint_pex_integrity"] = {
                    "pass": True,
                    "joint_pex_path": str(joint_path),
                    "joint_pex_sha256": _sha256(joint_path),
                    "lvsdb_caps": len(passives.caps),
                    "lvsdb_resistors": len(passives.resistors),
                }
            except Exception as exc:  # noqa: BLE001
                receipt["joint_pex_integrity"] = {
                    "pass": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }

    # 5. ngspice pre-sim, geometric PEX (if supplied), and real Magic R+C post-sim
    valid_tbs = [Path(p).resolve() for p in tb_paths if p]
    if valid_tbs:
        sim_summary: dict[str, Any] = {
            "corner": corner,
            "temp_c": float(temp_c),
            "ngspice_backend": ngspice_backend,
            "testbenches": [str(p) for p in valid_tbs],
        }
        pre_res = _run_verified_tb_kpis(
            cir_path,
            valid_tbs,
            work_dir=work_dir,
            stage_label="presim",
            pdk_id=pdk,
            corner=corner,
            temp_c=temp_c,
            vdd_v=vdd_v,
            is_post=False,
            subckt_name=resolved_cell,
            expected_kpis=(),
            ngspice_backend=ngspice_backend,
            timeout_s=timeout_s,
        )
        pre_kpis = pre_res["kpis"]
        sim_summary["presim"] = pre_kpis
        sim_summary["presim_runs"] = pre_res["runs"]
        if not pre_res["pass"]:
            sim_summary["presim_error"] = pre_res["error"]

        if geo_pex_path and geo_pex_path.is_file():
            geo_res = _run_verified_tb_kpis(
                geo_pex_path,
                valid_tbs,
                work_dir=work_dir,
                stage_label="geo_postsim",
                pdk_id=pdk,
                corner=corner,
                temp_c=temp_c,
                vdd_v=vdd_v,
                is_post=True,
                subckt_name=resolved_cell,
                expected_kpis=list(pre_kpis.keys()) if pre_res["pass"] else (),
                ngspice_backend=ngspice_backend,
                timeout_s=timeout_s,
            )
            sim_summary["geometric_pex_postsim"] = {
                k: v for k, v in geo_res["kpis"].items() if not k.endswith("_post")
            }
            sim_summary["geometric_pex_runs"] = geo_res["runs"]
            if not geo_res["pass"]:
                sim_summary["geometric_pex_error"] = geo_res["error"]

        if sim_pex_info is not None:
            t0 = time.monotonic()
            docker_sim_path_raw = sim_pex_info.get("docker_sim_pex_path")
            rc_res = _run_verified_tb_kpis(
                Path(sim_pex_info["sim_pex_path"]),
                valid_tbs,
                work_dir=work_dir,
                stage_label="magic_rc_postsim",
                pdk_id=pdk,
                corner=corner,
                temp_c=temp_c,
                vdd_v=vdd_v,
                is_post=True,
                subckt_name=resolved_cell,
                docker_dut_path=Path(docker_sim_path_raw) if docker_sim_path_raw else None,
                expected_kpis=list(pre_kpis.keys()) if pre_res["pass"] else (),
                ngspice_backend=ngspice_backend,
                timeout_s=timeout_s,
            )
            rc_sim_s = time.monotonic() - t0
            rc_clean = {
                k: v for k, v in rc_res["kpis"].items() if not k.endswith("_post")
            }
            sim_summary["magic_rc_postsim"] = rc_clean
            sim_summary["magic_rc_postsim_runs"] = rc_res["runs"]
            sim_summary["magic_rc_postsim_seconds"] = rc_sim_s
            sim_summary["magic_rc_vs_presim_deltas"] = (
                _relative_deltas(pre_kpis, rc_clean) if rc_res["pass"] else {}
            )
            if not rc_res["pass"]:
                sim_summary["error"] = rc_res["error"]
            sim_summary["pass"] = bool(
                pre_res["pass"] and rc_res["pass"] and len(rc_clean) > 0
            )
        else:
            sim_summary["pass"] = False
            sim_summary["error"] = "No simulatable Magic PEX netlist produced"
        receipt["ngspice_simulation"] = sim_summary

    sim_ok = receipt.get("ngspice_simulation", {}).get("pass", True)
    receipt["status"] = (
        "PASS"
        if (
            receipt["technology_gate"]["pass"]
            and receipt["magic_drc"]["pass"]
            and receipt["netgen_lvs"]["pass"]
            and receipt["magic_rc_pex"]["pass"]
            and sim_ok
        )
        else "FAIL"
    )
    receipt_path = out_dir / "docker_signoff_summary.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run GF180MCU Docker Magic DRC + extresist R+C PEX + Netgen LVS + ngspice postsim."
    )
    parser.add_argument("--gds", required=True, type=Path)
    parser.add_argument("--cir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cell", default=None)
    parser.add_argument("--tb", action="append", default=[], type=Path)
    parser.add_argument("--pdk", default="gf180mcu")
    parser.add_argument("--corner", default="tt")
    parser.add_argument("--temp-c", type=float, default=27.0)
    parser.add_argument("--vdd", type=float, default=3.3)
    parser.add_argument("--cap-density-ff-um2", type=float, default=2.0)
    parser.add_argument("--poly-res-width-um", type=float, default=1.0)
    parser.add_argument("--lvsdb", type=Path, default=None)
    parser.add_argument("--geo-pex", type=Path, default=None)
    parser.add_argument("--probe-drc-on-mismatch", action="store_true")
    parser.add_argument("--strict-zero-neg-c", action="store_true")
    parser.add_argument(
        "--ngspice-backend",
        choices=("auto", "docker", "host"),
        default="auto",
        help="Run ngspice via Docker gf180mcuD, host ngspice_con, or auto (docker for GF180 resistors).",
    )
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)

    receipt = run_docker_magic_signoff(
        gds_path=args.gds,
        cir_path=args.cir,
        out_dir=args.out_dir,
        cell_name=args.cell,
        tb_paths=args.tb,
        pdk=args.pdk,
        corner=args.corner,
        temp_c=args.temp_c,
        vdd_v=args.vdd,
        cap_density_f_per_um2=args.cap_density_ff_um2 * 1e-15,
        poly_res_width_um=args.poly_res_width_um,
        lvsdb_path=args.lvsdb,
        geo_pex_path=args.geo_pex,
        probe_drc_on_mismatch=args.probe_drc_on_mismatch,
        strict_zero_neg_c=args.strict_zero_neg_c,
        ngspice_backend=args.ngspice_backend,
        timeout_s=args.timeout,
    )
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
