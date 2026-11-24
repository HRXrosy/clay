"""Lightweight DC operating-point extraction & analytic fallback for Clay auto-craft.

Enriches a topology decomposition (or SPICE netlist / Clay CellDef) with per-device
operating points ``(Id, gm, Vov, Vdsat, Vth, Vds, region)`` and per-net DC currents
``|I_net|`` without hand-written circuit-specific harnesses.

Resolution order:
  1. When ``run_ngspice=True`` and ``ngspice_con`` + PDK models are available:
     run a fast 1-shot ``.op`` analysis (<= ``timeout_s`` seconds) either using a
     supplied ``tb_paths`` testbench or an auto-synthesized minimal DC-bias harness.
     Parse ``print allv`` and ``show m : id gm vth vds vdsat vgs`` directly.
  2. When ngspice is disabled, unavailable, or fails to converge:
     fall back cleanly to topology + W/L analytic current propagation so downstream
     auto-craft synthesis still runs deterministically offline.
"""
from __future__ import annotations

import math
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from clay.frontend.role_infer import infer_roles
from clay.frontend.role_model import DeviceCard, RoleDecomposition

__all__ = [
    "DeviceOpPoint",
    "CellOpProfile",
    "extract_op_profile",
    "resolve_netlist_params",
]

# Nominal PDK defaults when not explicitly declared in testbench/netlist.
_PDK_NOMINAL_VDD: Dict[str, float] = {
    "gf180mcu": 3.3,
    "sky130": 1.8,
    "vpdk55nm": 1.2,
    "vpdk180nm": 1.8,
}

# Approximate mu*Cox (A/V^2) for coarse analytic fallback Vov/gm estimation.
_DEFAULT_K_P = {"nmos": 200e-6, "pmos": 70e-6}

_SI_MULTIPLIERS: Dict[str, float] = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "x": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "a": 1e-18,
}

_NUM_UNIT_RE = re.compile(
    r"^\s*\{?\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([a-zA-Z]*)\s*\}?\s*$"
)
_BRACED_RE = re.compile(r"^\s*\{([^}]+)\}\s*$")
_PARAM_LINE_RE = re.compile(r"([A-Za-z_]\w*)\s*=\s*([^\s$]+)")


@dataclass(frozen=True)
class DeviceOpPoint:
    """Single-device operating-point summary (DC current magnitude in Amps)."""

    name: str
    kind: str
    id_a: float
    gm_s: float = 0.0
    vov_v: float = 0.0
    vdsat_v: float = 0.0
    vth_v: float = 0.0
    vds_v: float = 0.0
    region: str = "unknown"
    source: str = "analytic"


@dataclass(frozen=True)
class CellOpProfile:
    """Cell-wide DC operating-point profile for downstream auto-craft synthesis."""

    cell: str
    pdk_id: str
    vdd_v: float
    source: str
    device_ops: Dict[str, DeviceOpPoint]
    net_currents_a: Dict[str, float]
    high_current_nets: Tuple[str, ...]
    high_z_nets: Tuple[str, ...]
    bias_nets: Tuple[str, ...]
    node_voltages_v: Dict[str, float] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()


def _parse_spice_number(tok: str) -> Optional[float]:
    if tok is None:
        return None
    s = str(tok).strip()
    if not s:
        return None
    m = _NUM_UNIT_RE.match(s)
    if not m:
        return None
    mant, suf = m.group(1), m.group(2).lower()
    try:
        val = float(mant)
    except ValueError:
        return None
    if not suf:
        return val
    if suf in ("meg",):
        return val * 1e6
    mult = _SI_MULTIPLIERS.get(suf[0])
    if mult is not None:
        return val * mult
    return val


def _format_si_token(value: float, unit_hint: str = "") -> str:
    """Format a positive SI float into a Clay-grammar-friendly token."""
    if not math.isfinite(value):
        return "1u"
    if value == 0.0:
        return "0"
    av = abs(value)
    if unit_hint == "um":
        # Convert already-in-um values or raw meters? unit_hint="um" when raw is um.
        um = av
        if um >= 1000.0:
            return f"{um * 1e-3:.6g}m"
        return f"{um:.6g}u"
    if av >= 1e6:
        return f"{value / 1e6:.6g}meg"
    if av >= 1e3:
        return f"{value / 1e3:.6g}k"
    if av >= 1.0:
        return f"{value:.6g}"
    if av >= 1e-3:
        return f"{value * 1e3:.6g}m"
    if av >= 1e-6:
        return f"{value * 1e6:.6g}u"
    if av >= 1e-9:
        return f"{value * 1e9:.6g}n"
    if av >= 1e-12:
        return f"{value * 1e12:.6g}p"
    return f"{value * 1e15:.6g}f"


def _extract_cir_params(cir_text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in cir_text.splitlines():
        s = line.strip()
        if s.lower().startswith(".param"):
            body = s[len(".param") :].split("$", 1)[0]
            for m in _PARAM_LINE_RE.finditer(body):
                out[m.group(1)] = m.group(2)
    return out


def resolve_netlist_params(
    cir_text: str,
    overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Extract and recursively evaluate ``.param`` definitions from SPICE text."""
    raw = _extract_cir_params(cir_text)
    # Also parse .param lines inside .subckt or multi-line continuations handled by regex
    if overrides:
        for k, v in overrides.items():
            raw[str(k)] = str(v)

    resolved: Dict[str, str] = {}
    num_env: Dict[str, float] = {}

    for _pass in range(6):
        progressed = False
        for key, expr in raw.items():
            if key in num_env:
                continue
            cleaned = expr.strip()
            if cleaned.startswith("{") and cleaned.endswith("}"):
                cleaned = cleaned[1:-1].strip()
            v = _parse_spice_number(cleaned)
            if v is not None:
                num_env[key] = v
                num_env[key.lower()] = v
                resolved[key] = cleaned
                progressed = True
                continue
            # Evaluate simple arithmetic over resolved params + SPICE suffixes
            sub_expr = re.sub(
                r"\b([A-Za-z_]\w*)\b",
                lambda m: repr(num_env[m.group(1)])
                if m.group(1) in num_env
                else (
                    repr(num_env[m.group(1).lower()])
                    if m.group(1).lower() in num_env
                    else m.group(0)
                ),
                cleaned,
            )
            sub_expr = re.sub(
                r"(\d+(?:\.\d+)?)\s*(meg|u|n|p|f|k|m)\b",
                lambda m: f"({m.group(1)}*{_SI_MULTIPLIERS[m.group(2).lower()]})",
                sub_expr,
                flags=re.I,
            )
            if re.fullmatch(r"[0-9eE.+\-*/()\s]+", sub_expr):
                try:
                    val = float(eval(sub_expr, {"__builtins__": {}}, {}))  # noqa: S307
                    if math.isfinite(val):
                        num_env[key] = val
                        num_env[key.lower()] = val
                        resolved[key] = _format_si_token(val)
                        progressed = True
                except Exception:  # noqa: BLE001
                    pass
        if not progressed:
            break

    # Include unresolved raw strings as-is if not already evaluated
    for key, expr in raw.items():
        if key not in resolved:
            resolved[key] = expr.strip().strip("{}")
    return resolved


def _resolve_token_value(
    tok: str,
    params: Mapping[str, str],
    *,
    default_scale_um: bool = False,
) -> Optional[float]:
    """Resolve a SPICE size/value token via ``params`` into SI float (or um->m)."""
    if not tok:
        return None
    s = str(tok).strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    if s in params:
        s = str(params[s]).strip()
    else:
        for pk, pv in params.items():
            if pk.lower() == s.lower():
                s = str(pv).strip()
                break
    val = _parse_spice_number(s)
    if val is None:
        # Try simple arithmetic expression like 2*W_UNIT
        sub = resolve_netlist_params(f".param _tmp = {s}\n", params).get("_tmp")
        if sub:
            val = _parse_spice_number(sub)
    if val is None:
        return None
    # Handle unitless bare micron MOS sizes (e.g. W=20 under .option scale=1.0u)
    if default_scale_um and re.fullmatch(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", s):
        if 0.05 <= abs(val) <= 100000.0:
            val *= 1e-6
    return val


def _canon_kind(card: DeviceCard) -> str:
    k = (card.kind or "").strip().lower()
    if k.startswith("n"):
        return "nmos"
    if k.startswith("p") and k != "pnp":
        return "pmos"
    if k in ("res", "resistor", "r"):
        return "res"
    if k in ("cap", "capacitor", "c", "mim"):
        return "cap"
    if "pnp" in k or "pnp" in (card.source_model or "").lower():
        return "pnp"
    if "npn" in k or "npn" in (card.source_model or "").lower():
        return "npn"
    if k == "bjt":
        return "bjt"
    return k


def _has_scale_u_option(text: str) -> bool:
    return bool(re.search(r"^\s*\.option[^\n]*\bscale\s*=\s*1(\.0+)?u\b", text, re.I | re.M))


def _is_bare_micron_mos_cir(
    decomp: RoleDecomposition,
    cir_text: str,
    params: Optional[Mapping[str, str]] = None,
) -> bool:
    if _has_scale_u_option(cir_text):
        return True
    pmap = {str(k).lower(): str(v).strip() for k, v in (params or {}).items()}
    for d in decomp.devices:
        k = _canon_kind(d)
        if k in ("nmos", "pmos"):
            for tok in (d.w, d.l):
                s = str(tok or "").strip().strip("{}")
                if not s:
                    continue
                resolved_tok = pmap.get(s.lower(), s)
                if re.fullmatch(r"\d+(?:\.\d+)?", resolved_tok):
                    v = float(resolved_tok)
                    if 0.08 <= v <= 10000.0:
                        return True
    return False


def _detect_cir_pdk(decomp: RoleDecomposition, requested_pdk: str, cir_text: str) -> str:
    models = " ".join((d.source_model or "").lower() for d in decomp.devices)
    low = cir_text.lower()
    if "nmos_3p3" in models or "pmos_3p3" in models or "nfet_0" in models or "pfet_0" in models or "@gf180_pdk" in low:
        return "gf180mcu"
    if "sky130_fd_pr" in models:
        return "sky130"
    if re.search(r"\b(nch|pch)\b", models):
        if requested_pdk in ("open_pdk", "open_pdk", "open_pdk", "vpdk180nm", "vpdk55nm"):
            return requested_pdk
        return "vpdk180nm"
    return requested_pdk


def _infer_vdd_nominal(
    pdk_id: str,
    cir_text: str,
    tb_texts: Sequence[str],
    decomp: RoleDecomposition,
) -> float:
    for txt in [*tb_texts, cir_text]:
        m = re.search(r"\.param\s+VDD\s*=\s*([0-9.]+)", txt, re.I)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        m = re.search(
            r"^\s*V(?:dd|in|avdd)\s+\S+\s+\S+(?:\s+DC)?\s+([0-9.]+)",
            txt,
            re.I | re.M,
        )
        if m:
            try:
                v = float(m.group(1))
                if 0.5 <= v <= 6.0:
                    return v
            except ValueError:
                pass
    models = " ".join((d.source_model or "").lower() for d in decomp.devices)
    if "3p3" in models or "3v" in models or "03v3" in models:
        return 3.3
    try:
        from clay.pdk import pdk_spec

        spec = pdk_spec(pdk_id)
        if spec is not None and getattr(spec, "vdd_nominal", None):
            return float(spec.vdd_nominal)
    except Exception:  # noqa: BLE001
        pass
    return _PDK_NOMINAL_VDD.get(pdk_id, 1.8)


def _infer_port_kind(net: str) -> str:
    from clay.frontend.clay_emit import infer_port_kind

    return infer_port_kind(net)


def _is_power_or_ground_net(net: str, role: str = "") -> bool:
    if role in ("power_p", "power_n", "ground"):
        return True
    low = net.lower()
    return low in {"vdd", "avdd", "dvdd", "vdda", "vss", "avss", "dvss", "gnd", "agnd", "dgnd", "0"}


def _build_synthetic_op_tb(
    decomp: RoleDecomposition,
    cir_text: str,
    cir_path: Optional[Path],
    *,
    pdk_id: str,
    vdd_v: float,
    params: Mapping[str, str],
) -> Tuple[str, Optional[Path]]:
    """Build a standalone 1-shot ngspice .op wrapper for an un-wrapped .subckt."""
    ports = list(decomp.source_ports or ())
    if not ports:
        # Derive ordered candidate ports from non-inout inferred roles
        seen: set[str] = set()
        for dev in decomp.devices:
            for net in dev.terminals.values():
                if net not in seen:
                    seen.add(net)
                    if _infer_port_kind(net) != "inout":
                        ports.append(net)

    uses_bjt = any(_canon_kind(d) in ("pnp", "npn", "bjt") for d in decomp.devices)
    preamble = _pdk_preamble_for(pdk_id, corner="tt", bjt=uses_bjt)
    if not preamble:
        raise RuntimeError(f"No SPICE model preamble available for PDK {pdk_id!r}")

    lines: List[str] = [
        f"* Auto-generated 1-shot OP harness for {decomp.cell}",
        ".option numthreads=1",
        ".option itl1=500 itl6=200 gminsteps=20 srcsteps=20",
    ]
    if _is_bare_micron_mos_cir(decomp, cir_text, params=params):
        lines.append(".option scale=1.0u")
    lines.append(preamble)

    # Inject parameter overrides if any
    for pk, pv in params.items():
        if pk.lower() not in ("mc_mm_switch", "mc_pr_switch"):
            lines.append(f".param {pk}={pv}")

    # Detect existing subckt or append cir_text
    if cir_path is not None and cir_path.is_file():
        lines.append(f'.include "{cir_path.resolve().as_posix()}"')
    else:
        # Strip any embedded .control/.end or conflicting model includes
        cleaned = []
        skip_ctrl = False
        for ln in cir_text.splitlines():
            ls = ln.strip().lower()
            if ls.startswith(".control"):
                skip_ctrl = True
                continue
            if ls.startswith(".endc"):
                skip_ctrl = False
                continue
            if skip_ctrl or ls == ".end":
                continue
            if ls.startswith((".lib", ".include")) and any(
                tok in ls for tok in ("sm141064", "design.ngspice", "clay_wrappers", "gf180mcu", "sky130")
            ):
                continue
            cleaned.append(ln)
        lines.append("\n".join(cleaned))

    # Build voltage/current stimuli from port roles
    p_roles = {p: _infer_port_kind(p) for p in ports}
    vdd_nets = [p for p in ports if p_roles[p] == "power_p" or p.lower() in ("vin", "avdd", "vdd")]
    gnd_nets = [p for p in ports if p_roles[p] in ("power_n", "ground") or p.lower() in ("vss", "gnd", "agnd")]
    inp_nets = [p for p in ports if p_roles[p] == "input"]
    out_nets = [p for p in ports if p_roles[p] == "output"]
    bias_nets = [p for p in ports if p_roles[p] == "bias"]
    clk_nets = [p for p in ports if p_roles[p] == "clock"]

    x_ports = ports
    driven: set[str] = set()

    for p in gnd_nets:
        if p.lower() != "0" and p not in driven:
            # ngspice aliases 'gnd' to 0, avoid shorting node 'gnd' with a 0V source
            if p.lower() == "gnd":
                x_ports = ["0" if xp == p else xp for xp in x_ports]
            else:
                lines.append(f"V_{p} {p} 0 DC 0")
            driven.add(p)

    vref_port = next((p for p in ports if "vref" in p.lower() or p.lower() == "ref"), None)
    for idx, p in enumerate(vdd_nets):
        if p not in driven:
            lines.append(f"V_{p} {p} 0 DC {vdd_v}")
            driven.add(p)

    if not vdd_nets and ports:
        # Check for vin if not classified as power_p
        vin_cand = next((p for p in ports if p.lower().startswith("vin")), None)
        if vin_cand and vin_cand not in driven:
            lines.append(f"V_{vin_cand} {vin_cand} 0 DC {vdd_v}")
            driven.add(vin_cand)

    if vref_port and vref_port not in driven:
        vref_val = 1.25 if vdd_v >= 2.2 else 0.5 * vdd_v
        lines.append(f"V_{vref_port} {vref_port} 0 DC {vref_val}")
        driven.add(vref_port)

    # Check if LDO feedback pair (vfb & vfbs) exists -> short them closed-loop
    lower_port_map = {p.lower(): p for p in ports}
    if "vfb" in lower_port_map and "vfbs" in lower_port_map:
        vfb_p, vfbs_p = lower_port_map["vfb"], lower_port_map["vfbs"]
        lines.append(f"Vshort_fb {vfb_p} {vfbs_p} DC 0")
        driven.update((vfb_p, vfbs_p))

    # Differential / OTA feedback closure: keep vout from railing in open loop
    vcm = 0.5 * vdd_v
    vinp_p = next((p for p in inp_nets if p.lower().endswith(("p", "plus")) or "inp" in p.lower()), None)
    vinn_p = next((p for p in inp_nets if p.lower().endswith(("n", "m", "minus")) or "inn" in p.lower()), None)
    vout_p = out_nets[0] if out_nets else None

    if vinp_p and vinn_p and vout_p and vinp_p not in driven and vinn_p not in driven:
        lines.append(f"V_{vinp_p} {vinp_p} 0 DC {vcm:.4g}")
        lines.append(f"Rfb_op {vout_p} {vinn_p} 1G")
        driven.update((vinp_p, vinn_p))

    for p in inp_nets:
        if p not in driven:
            lines.append(f"V_{p} {p} 0 DC {vcm:.4g}")
            driven.add(p)

    for p in clk_nets:
        if p not in driven:
            lines.append(f"V_{p} {p} 0 DC {vdd_v}")
            driven.add(p)

    # Bias current inference: check whether diode-connected NMOS or PMOS is on bias port
    vdd_set = {p.lower() for p in vdd_nets} | {"vdd", "avdd", "vin"}
    gnd_set = {p.lower() for p in gnd_nets} | {"vss", "gnd", "agnd", "0"}
    for p in bias_nets:
        if p in driven:
            continue
        p_low = p.lower()
        # Look for diode connected to this port
        sink_from_vdd = False
        pull_to_gnd = False
        for d in decomp.devices:
            dk = _canon_kind(d)
            nets = {k: v.lower() for k, v in d.terminals.items()}
            if dk == "nmos" and nets.get("g") == p_low and nets.get("d") == p_low:
                sink_from_vdd = True
            elif dk == "pmos" and nets.get("g") == p_low and nets.get("d") == p_low:
                pull_to_gnd = True
        if p_low.startswith("i"):
            if pull_to_gnd and not sink_from_vdd:
                lines.append(f"I_{p} {p} 0 DC 10u")
            else:
                src_node = vdd_nets[0] if vdd_nets else "0"
                lines.append(f"I_{p} {src_node} {p} DC 10u")
        else:
            lines.append(f"V_{p} {p} 0 DC {0.8 if vdd_v >= 2.0 else 0.55}")
        driven.add(p)

    for p in out_nets:
        if p not in driven:
            # Small nominal output load if LDO-like pass FET exists, else 1G high-Z
            has_wide_pass = any(
                _canon_kind(d) in ("nmos", "pmos")
                and (d.terminals.get("d") == p or d.terminals.get("s") == p)
                and (_resolve_token_value(d.w, params, default_scale_um=True) or 0.0)
                * (_resolve_token_value(d.m, params) or 1.0)
                >= 200e-6
                for d in decomp.devices
            )
            r_load = "1.25k" if has_wide_pass else "1G"
            lines.append(f"Rload_{p} {p} 0 {r_load}")
            driven.add(p)

    # Tie any remaining floating port gently to Vcm via 100M
    for p in ports:
        if p not in driven and p.lower() != "gnd":
            lines.append(f"Rtie_{p} {p} 0 100meg")

    lines.append(f"X1 {' '.join(x_ports)} {decomp.cell}")
    lines.extend(
        [
            ".control",
            "op",
            "print allv",
            "show m : id gm vth vds vdsat vgs",
            "quit",
            ".endc",
            ".end",
        ]
    )
    return "\n".join(lines) + "\n", cir_path.parent if cir_path else None


def _pdk_preamble_for(pdk_id: str, *, corner: str = "tt", bjt: bool = False) -> Optional[str]:
    try:
        from circuits._harness import _preamble_for

        args = f"corner={corner}" + (" bjt=1" if bjt else "")
        return _preamble_for(pdk_id, args)
    except Exception:  # noqa: BLE001
        pass
    try:
        from clay.pdk import pdk_spec

        spec = pdk_spec(pdk_id)
        if spec is None:
            return None
        lib = spec.spice_lib()
        if lib is None or not Path(lib).exists():
            return None
        c = spec.corner_section(corner) or corner
        return f'.lib "{Path(lib).as_posix()}" {c}'
    except Exception:  # noqa: BLE001
        return None


def _prepare_existing_tb_op_deck(
    tb_path: Path,
    cir_path: Optional[Path],
    *,
    pdk_id: str,
    params: Mapping[str, str],
) -> Optional[str]:
    try:
        text = tb_path.read_text(encoding="utf-8")
    except OSError:
        return None

    # Use circuits._harness if available for marker expansion
    try:
        from circuits._harness import _resolve_marker, _strip_old_pdk_lines

        marker_pdk, marker_args = None, None
        for ln in text.splitlines():
            pid, args = _resolve_marker(ln)
            if pid is not None:
                marker_pdk, marker_args = pid, args
                break
        cleaned = _strip_old_pdk_lines(text)
        eff_pdk = pdk_id if pdk_id else (marker_pdk or "gf180mcu")
        bjt = bool(marker_args and "bjt=1" in marker_args)
        preamble = _pdk_preamble_for(eff_pdk, corner="tt", bjt=bjt)
        if not preamble:
            return None
        out_lines = []
        replaced = False
        for ln in cleaned.splitlines():
            pid, _ = _resolve_marker(ln)
            if pid is not None and not replaced:
                out_lines.append(preamble)
                replaced = True
            else:
                # Rewrite relative .cir include if cir_path was passed explicitly
                if cir_path is not None and cir_path.is_file():
                    m_inc = re.match(r"^\s*\.include\s+[\"']?([^\"'\s]+)", ln, re.I)
                    if m_inc and ".cir" in m_inc.group(1).lower():
                        out_lines.append(f'.include "{cir_path.resolve().as_posix()}"')
                        continue
                out_lines.append(ln)
        if not replaced:
            out_lines.insert(0, preamble)
        deck = "\n".join(out_lines) + "\n"
    except Exception:  # noqa: BLE001
        return None

    # Inject param overrides near top (after preamble) if requested
    if params:
        param_block = "\n".join(
            f".param {k}={v}"
            for k, v in params.items()
            if k.lower() not in ("mc_mm_switch", "mc_pr_switch")
        )
        if param_block:
            deck = re.sub(
                r"(\.option\b[^\n]*\n)",
                r"\1" + param_block + "\n",
                deck,
                count=1,
                flags=re.I,
            )

    # Drop standalone .tran/.ac/.dc cards outside .control so batch op runs cleanly
    stripped_lines: List[str] = []
    in_ctrl = False
    for ln in deck.splitlines():
        ls = ln.strip().lower()
        if ls.startswith(".control"):
            in_ctrl = True
        elif ls.startswith(".endc"):
            in_ctrl = False
        elif not in_ctrl and re.match(r"^\.(tran|ac|dc|noise|tf)\b", ls):
            continue
        stripped_lines.append(ln)
    deck = "\n".join(stripped_lines) + "\n"

    ctrl_body = (
        ".control\n"
        "op\n"
        "print allv\n"
        "show m : id gm vth vds vdsat vgs\n"
        "quit\n"
        ".endc"
    )
    if re.search(r"\.control\b.*?\.endc", deck, re.S | re.I):
        deck = re.sub(r"\.control\b.*?\.endc", ctrl_body, deck, count=1, flags=re.S | re.I)
    elif re.search(r"^\s*\.end\s*$", deck, re.M | re.I):
        deck = re.sub(r"^\s*\.end\s*$", ctrl_body + "\n.end", deck, count=1, flags=re.M | re.I)
    else:
        deck += "\n" + ctrl_body + "\n.end\n"
    return deck


def _select_op_tb(tb_paths: Sequence[str]) -> List[Path]:
    existing = [Path(p) for p in tb_paths if p and Path(p).is_file()]
    if not existing:
        return []

    def _score(p: Path) -> Tuple[int, str]:
        name = p.name.lower()
        if "op" in name:
            return (0, name)
        if "dc" in name or "bandgap" in name:
            return (1, name)
        if "ac" in name:
            return (2, name)
        if "tran" in name:
            return (4, name)
        return (3, name)

    return sorted(existing, key=_score)


def _discover_ngspice_bin() -> Optional[str]:
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
    return None


def _parse_ngspice_op_output(
    stdout: str,
    decomp: RoleDecomposition,
    params: Mapping[str, str],
) -> Tuple[Dict[str, float], Dict[str, DeviceOpPoint]]:
    """Parse ``print allv`` and ``show m : id gm vth vds vdsat vgs`` output."""
    node_voltages: Dict[str, float] = {}
    for m in re.finditer(
        r"^\s*([A-Za-z0-9_.]+)\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$",
        stdout,
        re.M,
    ):
        raw_node = m.group(1)
        val = float(m.group(2))
        if not math.isfinite(val):
            continue
        node_voltages[raw_node] = val
        # Normalize x1.ntail / xdut.ntail -> ntail as fallback lookup
        low = raw_node.lower()
        for prefix in ("x1.", "xdut."):
            if low.startswith(prefix):
                node_voltages.setdefault(raw_node[len(prefix) :], val)

    # Parse BSIM table columns from `show m : id gm vth vds vdsat vgs`
    parsed_mos: Dict[str, Dict[str, float]] = {}
    current_devs: List[str] = []
    for raw_ln in stdout.splitlines():
        s = raw_ln.strip()
        if not s:
            continue
        parts = s.split()
        if not parts:
            continue
        head = parts[0].lower()
        if head == "device" and len(parts) >= 2:
            current_devs = [p.lower() for p in parts[1:]]
            for d in current_devs:
                parsed_mos.setdefault(d, {})
        elif head in ("id", "gm", "vth", "vds", "vdsat", "vgs") and current_devs:
            vals = parts[1 : 1 + len(current_devs)]
            for dev_id, tok in zip(current_devs, vals):
                if "?" in tok:
                    continue
                try:
                    fv = float(tok)
                    if math.isfinite(fv):
                        parsed_mos[dev_id][head] = fv
                except ValueError:
                    pass

    # Match ngspice hierarchical MOS names back to RoleDecomposition card names
    dev_by_low = {d.name.lower(): d for d in decomp.devices}
    raw_by_short: Dict[str, Dict[str, float]] = {}
    for full_name, vals in parsed_mos.items():
        # Typical shapes: m.x1.xmnbias.m0, m.x1.xmn1.m0, m.x1.m1, m.xdut.xm1.m0
        tokens = full_name.split(".")
        cands: List[str] = [full_name]
        if len(tokens) >= 3 and tokens[0] == "m":
            inner = tokens[2:] if tokens[1].startswith("x") else tokens[1:]
            base = inner[0]
            cands.append(base)
            if base.startswith("x") and len(base) > 1:
                cands.append(base[1:])
        for c in cands:
            if c in dev_by_low:
                # Aggregate parallel instances if multi-instance exists, else take
                prev = raw_by_short.get(c)
                if prev is None:
                    raw_by_short[c] = dict(vals)
                else:
                    prev["id"] = abs(prev.get("id", 0.0)) + abs(vals.get("id", 0.0))
                    prev["gm"] = abs(prev.get("gm", 0.0)) + abs(vals.get("gm", 0.0))
                break

    def _node_v(net: str) -> Optional[float]:
        if net in node_voltages:
            return node_voltages[net]
        low = net.lower()
        if low in ("0", "gnd", "vss"):
            return node_voltages.get(low, node_voltages.get("vss", 0.0))
        for k, v in node_voltages.items():
            if k.lower() == low:
                return v
        return None

    scale_um = _is_bare_micron_mos_cir(decomp, "")
    ops: Dict[str, DeviceOpPoint] = {}
    for card in decomp.devices:
        k = _canon_kind(card)
        low_name = card.name.lower()
        if k in ("nmos", "pmos") and low_name in raw_by_short:
            m = raw_by_short[low_name]
            id_a = abs(m.get("id", 0.0))
            gm_s = abs(m.get("gm", 0.0))
            vth = abs(m.get("vth", 0.7))
            vds = abs(m.get("vds", 0.0))
            vdsat = abs(m.get("vdsat", 0.15))
            vgs = abs(m.get("vgs", vth + 0.15))
            vov = max(0.0, vgs - vth)
            if id_a < 1e-12:
                region = "cutoff"
            elif vgs < vth:
                region = "subthreshold"
            elif vds >= 0.85 * vdsat:
                region = "sat"
            else:
                region = "triode"
            ops[card.name] = DeviceOpPoint(
                name=card.name,
                kind=k,
                id_a=id_a,
                gm_s=gm_s,
                vov_v=vov,
                vdsat_v=vdsat,
                vth_v=vth,
                vds_v=vds,
                region=region,
                source="ngspice_op",
            )
        elif k == "res":
            r_val = _resolve_token_value(card.w, params)
            m_val = _resolve_token_value(card.m, params) or 1.0
            vp = _node_v(card.terminals.get("p", ""))
            vn = _node_v(card.terminals.get("n", ""))
            if r_val and r_val > 0 and vp is not None and vn is not None:
                r_eff = r_val / max(1.0, m_val)
                i_r = abs(vp - vn) / r_eff
            else:
                i_r = 0.0
            ops[card.name] = DeviceOpPoint(
                name=card.name,
                kind="res",
                id_a=i_r,
                region="passive",
                source="ngspice_op",
            )
        elif k == "cap":
            ops[card.name] = DeviceOpPoint(
                name=card.name,
                kind="cap",
                id_a=0.0,
                region="passive",
                source="ngspice_op",
            )

    # Infer BJT currents from series devices sharing their collector/emitter net
    for card in decomp.devices:
        k = _canon_kind(card)
        if k in ("pnp", "npn", "bjt") and card.name not in ops:
            em = card.terminals.get("e", "")
            col = card.terminals.get("c", "")
            branch_i = 0.0
            for probe_net in (em, col):
                if not probe_net or _is_power_or_ground_net(probe_net):
                    continue
                branch_i = max(
                    branch_i,
                    sum(
                        ops[o.name].id_a
                        for o in decomp.devices
                        if o.name in ops
                        and _canon_kind(o) in ("nmos", "pmos", "res")
                        and probe_net in (
                            o.terminals.get("d"),
                            o.terminals.get("s"),
                            o.terminals.get("p"),
                            o.terminals.get("n"),
                        )
                    ),
                )
            ops[card.name] = DeviceOpPoint(
                name=card.name,
                kind=k,
                id_a=branch_i,
                region="active" if branch_i > 1e-11 else "unknown",
                source="ngspice_op",
            )
    return node_voltages, ops


def _run_ngspice_op(
    decomp: RoleDecomposition,
    cir_text: str,
    cir_path: Optional[Path],
    tb_paths: Sequence[str],
    *,
    pdk_id: str,
    vdd_v: float,
    params: Mapping[str, str],
    timeout_s: float,
) -> Optional[Tuple[Dict[str, float], Dict[str, DeviceOpPoint], str]]:
    ngspice = _discover_ngspice_bin()
    if ngspice is None:
        return None

    candidate_decks: List[Tuple[str, Optional[Path], str]] = []
    for tb in _select_op_tb(tb_paths):
        deck = _prepare_existing_tb_op_deck(tb, cir_path, pdk_id=pdk_id, params=params)
        if deck:
            candidate_decks.append((deck, tb.parent, f"tb:{tb.name}"))

    try:
        syn_deck, syn_cwd = _build_synthetic_op_tb(
            decomp, cir_text, cir_path, pdk_id=pdk_id, vdd_v=vdd_v, params=params
        )
        candidate_decks.append((syn_deck, syn_cwd, "synthetic_op"))
    except Exception:  # noqa: BLE001
        pass

    mos_count = sum(1 for d in decomp.devices if _canon_kind(d) in ("nmos", "pmos"))
    for deck, cwd, label in candidate_decks:
        work_dir = cwd if (cwd is not None and cwd.is_dir()) else None
        with tempfile.TemporaryDirectory(prefix="clay_op_") as td:
            deck_path = Path(td) / "_clay_op.spice"
            deck_path.write_text(deck, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [ngspice, "-b", str(deck_path)],
                    cwd=str(work_dir or td),
                    capture_output=True,
                    text=True,
                    timeout=max(2.0, float(timeout_s)),
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
            node_v, ops = _parse_ngspice_op_output(combined, decomp, params)
            n_mos_ok = sum(
                1
                for op in ops.values()
                if op.kind in ("nmos", "pmos") and op.source == "ngspice_op"
            )
            has_nonzero = any(op.id_a > 1e-12 for op in ops.values())
            if mos_count == 0 or (n_mos_ok >= max(1, mos_count // 2) and has_nonzero):
                return node_v, ops, label
    return None


def _estimate_op_analytic(
    decomp: RoleDecomposition,
    params: Mapping[str, str],
    *,
    vdd_v: float,
    cir_text: str,
) -> Tuple[Dict[str, DeviceOpPoint], Tuple[str, ...]]:
    """Topology + W/L current propagation fallback when ngspice is unavailable."""
    scale_um = _is_bare_micron_mos_cir(decomp, cir_text, params=params)
    role_of: Dict[str, str] = {}
    for hit in decomp.roles:
        for dname in hit.devices:
            role_of.setdefault(dname, hit.role)

    # Compute effective (W/L) aspect for every device
    w_eff: Dict[str, float] = {}
    l_eff: Dict[str, float] = {}
    ratio: Dict[str, float] = {}
    for d in decomp.devices:
        k = _canon_kind(d)
        if k in ("nmos", "pmos"):
            w = _resolve_token_value(d.w, params, default_scale_um=scale_um) or 2e-6
            l = _resolve_token_value(d.l, params, default_scale_um=scale_um) or 0.5e-6
            m = _resolve_token_value(d.m, params, default_scale_um=False) or 1.0
            we = max(1e-8, w * max(1.0, m))
            le = max(1e-8, l)
            w_eff[d.name] = we
            l_eff[d.name] = le
            ratio[d.name] = we / le

    # Estimate base reference current from bias resistors or 10 uA default
    i_ref = 10e-6
    for d in decomp.devices:
        if _canon_kind(d) == "res":
            p = d.terminals.get("p", "")
            n = d.terminals.get("n", "")
            r_val = _resolve_token_value(d.w, params)
            m_val = _resolve_token_value(d.m, params) or 1.0
            if r_val and r_val >= 5e3:
                r_eff = r_val / max(1.0, m_val)
                if _is_power_or_ground_net(p) != _is_power_or_ground_net(n):
                    est = max(1e-6, min(100e-6, max(0.3, vdd_v - 0.8) / r_eff))
                    i_ref = est
                    break

    # Find diode-connected bias references (d == g and d != s)
    diode_devs = [
        d
        for d in decomp.devices
        if _canon_kind(d) in ("nmos", "pmos")
        and d.terminals.get("d") == d.terminals.get("g")
        and d.terminals.get("d") != d.terminals.get("s")
    ]
    currents: Dict[str, float] = {}
    gate_ref: Dict[Tuple[str, str], Tuple[float, float]] = {}  # (kind, gate_net) -> (I, W/L)

    for d in diode_devs:
        currents[d.name] = i_ref
        k = _canon_kind(d)
        g = d.terminals.get("g", "")
        if g:
            gate_ref[(k, g)] = (i_ref, ratio.get(d.name, 4.0))

    # Propagate current ratios along gate-connected mirrors (3 passes handle chains)
    for _ in range(3):
        for d in decomp.devices:
            k = _canon_kind(d)
            if k not in ("nmos", "pmos"):
                continue
            g = d.terminals.get("g", "")
            key = (k, g)
            if key in gate_ref and d.name not in currents:
                i_b, r_b = gate_ref[key]
                r_cur = ratio.get(d.name, r_b)
                est_i = i_b * (r_cur / max(0.1, r_b))
                currents[d.name] = est_i
            elif d.name in currents and g and key not in gate_ref:
                gate_ref[key] = (currents[d.name], ratio.get(d.name, 4.0))

    # Tail current & differential pair split (per distinct shared source node)
    tail_devs = [d for d in decomp.devices if role_of.get(d.name) == "tail_current"]
    for td in tail_devs:
        currents.setdefault(td.name, 2.0 * i_ref)

    # Propagate load, cascode & current-steering switch currents through intermediate nodes
    for _ in range(4):
        net_known: Dict[str, float] = {}
        for d in decomp.devices:
            if _canon_kind(d) not in ("nmos", "pmos"):
                continue
            dr = d.terminals.get("d", "")
            sr = d.terminals.get("s", "")
            if d.name in currents:
                for node in (dr, sr):
                    if node and not _is_power_or_ground_net(node):
                        net_known[node] = max(net_known.get(node, 0.0), currents[d.name])

        # Split each tail/cascode node cleanly among differential pairs sharing that source node
        src_pairs: Dict[str, List[DeviceCard]] = {}
        for d in decomp.devices:
            if d.name not in currents and role_of.get(d.name) == "input_pair":
                sr = d.terminals.get("s", "")
                if sr and not _is_power_or_ground_net(sr):
                    src_pairs.setdefault(sr, []).append(d)

        progressed = False
        for sr, legs in src_pairs.items():
            if sr in net_known:
                split_i = max(2e-6, net_known[sr] / max(1, len(legs)))
                for leg in legs:
                    currents[leg.name] = split_i
                    progressed = True

        for d in decomp.devices:
            if d.name in currents or _canon_kind(d) not in ("nmos", "pmos"):
                continue
            dr = d.terminals.get("d", "")
            sr = d.terminals.get("s", "")
            for probe in (sr, dr):
                if probe in net_known:
                    currents[d.name] = net_known[probe]
                    progressed = True
                    break
        if not progressed:
            break

    # Fallback for any remaining input_pair with no upstream tail current source
    for d in decomp.devices:
        if d.name not in currents and role_of.get(d.name) == "input_pair":
            currents[d.name] = i_ref

    # Assign wide pass/output devices a realistic high-current estimate if un-mirrored or huge
    for d in decomp.devices:
        k = _canon_kind(d)
        if k not in ("nmos", "pmos"):
            continue
        we_um = w_eff.get(d.name, 2e-6) * 1e6
        dr = d.terminals.get("d", "")
        sr = d.terminals.get("s", "")
        touches_rail_and_out = (
            _infer_port_kind(dr) == "output" and _is_power_or_ground_net(sr)
        ) or (_infer_port_kind(sr) == "output" and _is_power_or_ground_net(dr))
        if d.name not in currents:
            if touches_rail_and_out and we_um >= 200.0:
                currents[d.name] = max(2.0e-3, min(10.0e-3, we_um * 2.5e-6))
            elif role_of.get(d.name) == "second_stage" or we_um >= 50.0:
                currents[d.name] = max(50e-6, min(500e-6, we_um * 1.5e-6))
            else:
                currents[d.name] = i_ref
        elif touches_rail_and_out and we_um >= 200.0 and currents[d.name] < 1.0e-3:
            currents[d.name] = max(2.0e-3, currents[d.name])

    # Estimate resistors & BJTs
    ops: Dict[str, DeviceOpPoint] = {}
    for d in decomp.devices:
        k = _canon_kind(d)
        if k in ("nmos", "pmos"):
            id_a = max(1e-8, float(currents.get(d.name, i_ref)))
            kp = _DEFAULT_K_P.get(k, 120e-6)
            asp = ratio.get(d.name, 4.0)
            vov = max(0.08, min(0.50, math.sqrt(2.0 * id_a / max(1e-9, kp * asp))))
            gm = 2.0 * id_a / vov
            ops[d.name] = DeviceOpPoint(
                name=d.name,
                kind=k,
                id_a=id_a,
                gm_s=gm,
                vov_v=vov,
                vdsat_v=vov,
                vth_v=0.72,
                vds_v=max(0.3, 0.35 * vdd_v),
                region="sat",
                source="analytic",
            )
        elif k == "res":
            r_val = _resolve_token_value(d.w, params) or 10e3
            m_val = _resolve_token_value(d.m, params) or 1.0
            r_eff = max(1.0, r_val / max(1.0, m_val))
            i_r = max(1e-7, min(2e-3, (0.4 * vdd_v) / r_eff))
            ops[d.name] = DeviceOpPoint(
                name=d.name, kind="res", id_a=i_r, region="passive", source="analytic"
            )
        elif k == "cap":
            ops[d.name] = DeviceOpPoint(
                name=d.name, kind="cap", id_a=0.0, region="passive", source="analytic"
            )
        elif k in ("pnp", "npn", "bjt"):
            ops[d.name] = DeviceOpPoint(
                name=d.name, kind=k, id_a=i_ref, region="active", source="analytic"
            )

    notes = (
        f"analytic fallback OP estimation (Vdd={vdd_v:.2f}V, base_iref={i_ref*1e6:.1f}uA)",
    )
    return ops, notes


def _compute_net_profile(
    decomp: RoleDecomposition,
    device_ops: Mapping[str, DeviceOpPoint],
) -> Tuple[Dict[str, float], Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    """Aggregate per-net currents, high-current nets, high-Z nets, and bias nets."""
    port_role_map: Dict[str, str] = {}
    for card in decomp.devices:
        for net in card.terminals.values():
            if net and net not in port_role_map:
                port_role_map[net] = _infer_port_kind(net)

    branch_sums: Dict[str, float] = {}
    branch_pull_up: Dict[str, float] = {}
    branch_pull_dn: Dict[str, float] = {}
    branch_max: Dict[str, float] = {}
    net_gates: Dict[str, int] = {}
    net_drains: Dict[str, int] = {}
    bulk_nets: set[str] = set()
    diode_nets: set[str] = set()

    for card in decomp.devices:
        op = device_ops.get(card.name)
        id_a = abs(op.id_a) if op is not None else 0.0
        k = _canon_kind(card)
        pins = card.terminals
        if k in ("nmos", "pmos"):
            d, g, s, b = pins.get("d", ""), pins.get("g", ""), pins.get("s", ""), pins.get("b", "")
            if g:
                net_gates[g] = net_gates.get(g, 0) + 1
            if d:
                net_drains[d] = net_drains.get(d, 0) + 1
            if b:
                bulk_nets.add(b)
            if d and d == g and d != s:
                diode_nets.add(d)
            if d != s:
                for term in (d, s):
                    if term:
                        branch_sums[term] = branch_sums.get(term, 0.0) + id_a
                        branch_max[term] = max(branch_max.get(term, 0.0), id_a)
                        if k == "pmos":
                            branch_pull_up[term] = branch_pull_up.get(term, 0.0) + id_a
                        else:
                            branch_pull_dn[term] = branch_pull_dn.get(term, 0.0) + id_a
        elif k == "res":
            p, n = pins.get("p", ""), pins.get("n", "")
            if p != n:
                for term in (p, n):
                    if term:
                        branch_sums[term] = branch_sums.get(term, 0.0) + id_a
                        branch_max[term] = max(branch_max.get(term, 0.0), id_a)
        elif k in ("pnp", "npn", "bjt"):
            c, e = pins.get("c", ""), pins.get("e", "")
            if c != e:
                for term in (c, e):
                    if term:
                        branch_sums[term] = branch_sums.get(term, 0.0) + id_a
                        branch_max[term] = max(branch_max.get(term, 0.0), id_a)

    # Identify implicit power supplies such as `vin` when wired directly as body/rail source
    def _is_supply_like(net_name: str, p_role: str) -> bool:
        if _is_power_or_ground_net(net_name, p_role):
            return True
        low = net_name.lower()
        if net_name in bulk_nets and (low in ("vin", "vpwr", "vgnd", "vddh", "vddl") or low.startswith(("vdd", "vss", "avdd", "avss"))):
            return True
        return False

    net_currents: Dict[str, float] = {}
    for net in port_role_map:
        role = port_role_map[net]
        total = branch_sums.get(net, 0.0)
        peak = branch_max.get(net, 0.0)
        if _is_supply_like(net, role):
            # Supply rails aggregate every branch drawn from / returned to the rail
            net_currents[net] = max(total, peak)
        elif role == "output":
            # KCL: avoid double-counting push-pull/cascode PMOS + NMOS quiescent current
            net_currents[net] = max(
                peak,
                branch_pull_up.get(net, 0.0),
                branch_pull_dn.get(net, 0.0),
            )
        else:
            # Internal node: KCL inflow equals outflow (~total / 2)
            net_currents[net] = max(peak, 0.5 * total)

    # Bias nets
    bias_nets: List[str] = []
    for net in port_role_map:
        role = port_role_map[net]
        if _is_supply_like(net, role):
            continue
        low = net.lower()
        if (
            net in diode_nets
            or role == "bias"
            or any(tok in low for tok in ("bias", "vbias", "ibias", "vbc", "iref"))
        ):
            bias_nets.append(net)

    # High-Z sensitive nodes: diff-pair inputs, high-ro cascode/mirror outputs, stage2 gate
    bias_set = set(bias_nets)
    role_of_dev = {dn: h.role for h in decomp.roles for dn in h.devices}
    high_z_cands: List[str] = []
    seen_hz: set[str] = set()

    def _add_hz(net: str) -> None:
        if (
            net
            and net not in seen_hz
            and not _is_supply_like(net, port_role_map.get(net, ""))
            and net not in diode_nets
            and net_currents.get(net, 0.0) < 100e-6
            and port_role_map.get(net, "") != "output"
        ):
            seen_hz.add(net)
            high_z_cands.append(net)

    for card in decomp.devices:
        if _canon_kind(card) not in ("nmos", "pmos"):
            continue
        r = role_of_dev.get(card.name, "")
        if r == "input_pair":
            # Only treat diff-pair inputs/drains as high-Z when not driving a high-current output/switch load
            d_net = card.terminals.get("d", "")
            g_net = card.terminals.get("g", "")
            if net_currents.get(d_net, 0.0) < 200e-6:
                _add_hz(g_net)
                _add_hz(d_net)
        elif r == "second_stage":
            _add_hz(card.terminals.get("g", ""))
        elif r == "cascode":
            d_net = card.terminals.get("d", "")
            if net_currents.get(d_net, 0.0) < 200e-6:
                _add_hz(d_net)

    # Also capture explicit high-Z feedback/summing nodes (`vfb`, `vfbs`, `vm`)
    for net, role in port_role_map.items():
        low = net.lower()
        if low in ("vfb", "vfbs", "vm", "vref") or (role == "input" and net not in bias_set):
            _add_hz(net)

    # High-current nets: primary rails + nets carrying large current
    i_peak = max(net_currents.values(), default=0.0)
    thresh = max(100e-6, 0.12 * i_peak)
    high_curr: List[str] = []
    for net, ic in net_currents.items():
        role = port_role_map.get(net, "")
        is_sup = _is_supply_like(net, role)
        if is_sup and ic >= 20e-6:
            high_curr.append(net)
        elif ic >= thresh and (is_sup or role not in ("input", "clock")):
            high_curr.append(net)

    return (
        net_currents,
        tuple(high_curr),
        tuple(high_z_cands),
        tuple(bias_nets),
    )


def _decomp_from_cell_def(cell_def) -> RoleDecomposition:
    """Convert a parsed Clay CellDef into a lightweight RoleDecomposition."""
    cards: List[DeviceCard] = []
    for dev in getattr(cell_def, "devices", ()) or ():
        name = str(getattr(dev, "instance_name", ""))
        klass = str(getattr(dev, "device_class", ""))
        pins = {str(k): str(v) for k, v in (getattr(dev, "pins", {}) or {}).items()}
        sizing = dict(getattr(dev, "sizing", {}) or {})
        w_raw = sizing.get("W", sizing.get("value"))
        l_raw = sizing.get("L")
        m_raw = sizing.get("M")
        cards.append(
            DeviceCard(
                name=name,
                kind=klass,
                terminals=pins,
                w=_format_si_token(float(w_raw)) if w_raw is not None else "",
                l=_format_si_token(float(l_raw)) if l_raw is not None else "",
                m=str(int(m_raw)) if m_raw is not None else "",
                source_model=str(getattr(dev, "source_model", "") or ""),
            )
        )
    ports = tuple(str(p) for p in (getattr(cell_def, "ports", ()) or ()))
    return RoleDecomposition(
        cell=str(getattr(cell_def, "local_name", "") or "cell"),
        devices=tuple(cards),
        roles=(),
        matched_pairs=(),
        source_ports=ports,
    )


def extract_op_profile(
    cell_def_or_cir: Union[RoleDecomposition, str, Path, object],
    *,
    pdk_id: str = "gf180mcu",
    cir_path: Optional[Union[str, Path]] = None,
    tb_paths: Sequence[Union[str, Path]] = (),
    run_ngspice: bool = True,
    timeout_s: float = 15.0,
    param_overrides: Optional[Mapping[str, str]] = None,
) -> CellOpProfile:
    """Extract per-device DC operating points and per-net currents.

    Tries a fast 1-shot ``ngspice_con`` ``.op`` run when ``run_ngspice=True``
    and falls back cleanly to topology + W/L analytic current propagation.
    """
    cir_file: Optional[Path] = Path(cir_path) if cir_path else None
    cir_text = ""

    if isinstance(cell_def_or_cir, RoleDecomposition):
        decomp = cell_def_or_cir
        if cir_file and cir_file.is_file():
            cir_text = cir_file.read_text(encoding="utf-8", errors="ignore")
    elif isinstance(cell_def_or_cir, (str, Path)):
        raw_str = str(cell_def_or_cir)
        p_cand = Path(raw_str) if "\n" not in raw_str else None
        if p_cand is not None and p_cand.is_file():
            cir_file = cir_file or p_cand
            cir_text = p_cand.read_text(encoding="utf-8", errors="ignore")
            decomp = infer_roles(str(p_cand))
        else:
            cir_text = raw_str
            decomp = infer_roles(cir_text)
    else:
        decomp = _decomp_from_cell_def(cell_def_or_cir)
        if cir_file and cir_file.is_file():
            cir_text = cir_file.read_text(encoding="utf-8", errors="ignore")

    tb_str_paths = [str(p) for p in tb_paths if p]
    tb_texts: List[str] = []
    for p in tb_str_paths:
        try:
            if Path(p).is_file():
                tb_texts.append(Path(p).read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            pass

    params = resolve_netlist_params(cir_text, overrides=param_overrides)
    sim_pdk = _detect_cir_pdk(decomp, pdk_id, cir_text)
    vdd_v = _infer_vdd_nominal(sim_pdk, cir_text, tb_texts, decomp)

    node_voltages: Dict[str, float] = {}
    device_ops: Dict[str, DeviceOpPoint] = {}
    notes: List[str] = []
    source = "analytic_fallback"

    if run_ngspice and decomp.devices:
        res = _run_ngspice_op(
            decomp,
            cir_text,
            cir_file,
            tb_str_paths,
            pdk_id=sim_pdk,
            vdd_v=vdd_v,
            params=params,
            timeout_s=timeout_s,
        )
        if res is not None:
            node_voltages, device_ops, how = res
            source = "ngspice_op"
            notes.append(f"ngspice_op succeeded via {how} (pdk={sim_pdk}, vdd={vdd_v:.2f}V)")

    if not device_ops:
        device_ops, fb_notes = _estimate_op_analytic(
            decomp, params, vdd_v=vdd_v, cir_text=cir_text
        )
        notes.extend(fb_notes)

    net_currents, high_curr, high_z, bias_nets = _compute_net_profile(
        decomp, device_ops
    )
    return CellOpProfile(
        cell=decomp.cell,
        pdk_id=pdk_id,
        vdd_v=vdd_v,
        source=source,
        device_ops=device_ops,
        net_currents_a=net_currents,
        high_current_nets=high_curr,
        high_z_nets=high_z,
        bias_nets=bias_nets,
        node_voltages_v=node_voltages,
        notes=tuple(notes),
    )
