"""Automatic post-layout R+C parasitic diagnosis and targeted layout-intent repair.

Reads an actual :class:`~clay.layout.pex.base.PexReport` (or raw ``.pex.spice``
text/file), joins optional DC operating-point context (``op_profile``),
``intent_verdict``, and pre/post KPIs, attributes physical degradations to three
root causes, and emits idempotent ``layout_intent`` patches plus pre-sim
parasitic back-annotation without mutating circuit topology:

  1. **IR-drop / series resistance** (high $I_{\\text{net}} \\cdot R_{\\text{net}}$
     or power/output droop): computes physically widened ``@wire_width``,
     lower-$R_\\square$ upper-layer ``@trunk``, and ``@kelvin`` sense routing.
  2. **Coupling imbalance / kickback** (``matched_cc_balance``, differential
     aggressor asymmetry, comparator clock kickback): locks Top Aggressor(s)
     from the extracted $C_{ij}$ matrix and emits symmetric ``@shield`` and
     ``@net_spacing`` constraints.
  3. **High-Z / dominant-pole capacitive loading** (GBW/PM drop on high-Z
     nodes): emits upper-metal low-C ``@trunk`` / ``@net_spacing`` patches and
     ``(C_total_fF, R_series_ohm)`` back-annotation for pre-sim warm-start retuning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .base import PexCap, PexReport, PexRes
from .magic_parser import parse_ext2spice

_GROUND_STEMS = frozenset({"gnd", "vss", "vssa", "vssd", "agnd", "dgnd", "vgnd", "sub", "0"})
_POWER_STEMS = frozenset({"vdd", "vdda", "vddd", "avdd", "dvdd", "vpwr", "vcc"})
_CLK_TOKENS = ("clk", "phi", "strobe", "latch", "rst", "samp", "ck")
_FEEDBACK_TOKENS = ("vfb", "fb", "sense", "vsense")
_HIGH_Z_TOKENS = ("comp", "miller", "cc", "vctrl", "vo1", "stg1", "gm1", "cas", "highz")


@dataclass(frozen=True)
class RcRepairAction:
    """One targeted layout-intent repair or pre-sim parasitic back-annotation."""

    kind: str                 # wire_width | trunk | kelvin | shield | net_spacing | presim_backannotate
    category: str             # ir_drop | coupling_imbalance | high_z_cap
    net: str
    target_net: str = ""
    target_device: str = ""
    directive: str = ""
    directive_key: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "category": self.category,
            "net": self.net,
            "target_net": self.target_net,
            "target_device": self.target_device,
            "directive": self.directive,
            "directive_key": self.directive_key,
            "params": dict(self.params),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RcRepairPlan:
    """Complete post-sim R+C diagnosis and idempotent Clay repair plan."""

    cell_name: str
    pdk_id: str
    degradation_detected: bool
    root_causes: Tuple[str, ...]
    actions: Tuple[RcRepairAction, ...]
    net_series_resistance_ohm: Dict[str, float] = field(default_factory=dict)
    net_ir_drop_mv: Dict[str, float] = field(default_factory=dict)
    net_total_cap_fF: Dict[str, float] = field(default_factory=dict)
    coupling_pairs_fF: Dict[str, float] = field(default_factory=dict)
    presim_backannotation: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    @property
    def clay_directives(self) -> Tuple[str, ...]:
        out: List[str] = []
        seen: Set[str] = set()
        for act in self.actions:
            if act.directive and act.directive not in seen:
                seen.add(act.directive)
                out.append(act.directive)
        return tuple(out)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_name": self.cell_name,
            "pdk_id": self.pdk_id,
            "degradation_detected": bool(self.degradation_detected),
            "root_causes": list(self.root_causes),
            "clay_directives": list(self.clay_directives),
            "actions": [a.to_dict() for a in self.actions],
            "net_series_resistance_ohm": dict(self.net_series_resistance_ohm),
            "net_ir_drop_mv": dict(self.net_ir_drop_mv),
            "net_total_cap_fF": dict(self.net_total_cap_fF),
            "coupling_pairs_fF": dict(self.coupling_pairs_fF),
            "presim_backannotation": dict(self.presim_backannotation),
            "metrics": dict(self.metrics),
            "summary": self.summary,
        }


# --------------------------------------------------------------------------- #
# Parsing PexReport / .pex.spice & Topology Context
# --------------------------------------------------------------------------- #
def _coerce_pex_report(pex_report: Any, cell_name: str, pdk_id: str) -> PexReport:
    if isinstance(pex_report, PexReport):
        return pex_report
    if pex_report is None:
        return PexReport(
            cell_name=cell_name,
            caps=(),
            resistors=(),
            backend="none",
            connectivity_source="empty",
            constants_source="none",
            calibrated=False,
            field_solver=False,
        )
    if isinstance(pex_report, (str, Path)):
        text: Optional[str] = None
        if isinstance(pex_report, Path):
            text = pex_report.read_text(encoding="utf-8")
        else:
            stripped = pex_report.strip()
            if "\n" not in stripped and len(stripped) < 260:
                cand = Path(stripped)
                if cand.is_file():
                    text = cand.read_text(encoding="utf-8")
            if text is None:
                text = pex_report
        return parse_ext2spice(text, cell_name, pdk_id)
    if isinstance(pex_report, Mapping):
        caps = tuple(
            c if isinstance(c, PexCap) else PexCap(str(c["a"]), str(c["b"]), float(c["cap_aF"]))
            for c in (pex_report.get("caps") or ())
        )
        res = tuple(
            r if isinstance(r, PexRes) else PexRes(str(r["a"]), str(r["b"]), float(r["res_ohm"]))
            for r in (pex_report.get("resistors") or ())
        )
        return PexReport(
            cell_name=str(pex_report.get("cell_name") or cell_name),
            caps=caps,
            resistors=res,
            backend=str(pex_report.get("backend") or "custom"),
            connectivity_source=str(pex_report.get("connectivity_source") or "mapping"),
            constants_source=str(pex_report.get("constants_source") or pdk_id),
            calibrated=bool(pex_report.get("calibrated", False)),
            field_solver=bool(pex_report.get("field_solver", False)),
        )
    return PexReport(
        cell_name=cell_name,
        caps=tuple(getattr(pex_report, "caps", ()) or ()),
        resistors=tuple(getattr(pex_report, "resistors", ()) or ()),
        backend=str(getattr(pex_report, "backend", "unknown")),
        connectivity_source=str(getattr(pex_report, "connectivity_source", "unknown")),
        constants_source=str(getattr(pex_report, "constants_source", pdk_id)),
        calibrated=bool(getattr(pex_report, "calibrated", False)),
        field_solver=bool(getattr(pex_report, "field_solver", False)),
    )


def _known_topology_info(cell_def: Any) -> Dict[str, Any]:
    """Extract nets, devices, device pins, diff pairs, and port roles from ``cell_def``."""
    nets: Set[str] = set()
    device_pins: Dict[str, Dict[str, str]] = {}
    passive_pins: Dict[str, Dict[str, str]] = {}
    diff_pairs: List[Tuple[str, str]] = []
    existing_widths: Dict[str, float] = {}
    cell_name = "Cell"
    sub_net = "vss"
    pos_net = "vdd"

    if cell_def is None:
        return {
            "cell_name": cell_name,
            "nets": nets,
            "device_pins": device_pins,
            "passive_pins": passive_pins,
            "diff_pairs": diff_pairs,
            "existing_widths": existing_widths,
            "sub_net": sub_net,
            "pos_net": pos_net,
            "device_universe_known": False,
        }

    cell_name = str(getattr(cell_def, "local_name", None) or getattr(cell_def, "name", "Cell"))
    for p in getattr(cell_def, "ports", ()) or ():
        pname = str(getattr(p, "name", None) or p)
        if pname:
            nets.add(pname)

    for dev in getattr(cell_def, "devices", ()) or ():
        dname = str(getattr(dev, "instance_name", None) or getattr(dev, "name", ""))
        pins = {str(k).lower(): str(v) for k, v in (getattr(dev, "pins", {}) or {}).items()}
        for v in pins.values():
            if v:
                nets.add(v)
        if dname:
            device_pins[dname] = pins

    for pas in getattr(cell_def, "passives", ()) or ():
        pname = str(getattr(pas, "instance_name", None) or getattr(pas, "name", ""))
        pins = {
            str(k).lower(): str(v)
            for k, v in (getattr(pas, "pins", None) or getattr(pas, "nets", {}) or {}).items()
        }
        for v in pins.values():
            if v:
                nets.add(v)
        if pname:
            passive_pins[pname] = pins

    try:
        from clay.port_roles import resolve_positive_supply_net, resolve_substrate_net

        sub_fb = next((n for n in sorted(nets) if n.lower().rstrip("!") in _GROUND_STEMS), "vss")
        pos_fb = next((n for n in sorted(nets) if n.lower().rstrip("!") in _POWER_STEMS), "vdd")
        sub_net = resolve_substrate_net(cell_def, fallback=sub_fb)
        pos_net = resolve_positive_supply_net(cell_def, fallback=pos_fb)
    except Exception:
        sub_net = next((n for n in sorted(nets) if n.lower().rstrip("!") in _GROUND_STEMS), "vss")
        pos_net = next((n for n in sorted(nets) if n.lower().rstrip("!") in _POWER_STEMS), "vdd")

    for decl in getattr(cell_def, "layout_intents", ()) or ():
        kind = getattr(decl, "kind", "")
        if kind in {"symmetric", "match", "mirror"}:
            d_nets = [str(n) for n in (getattr(decl, "nets", ()) or ())]
            if len(d_nets) == 2 and d_nets[0] != d_nets[1]:
                diff_pairs.append((d_nets[0], d_nets[1]))
        elif kind == "wire_width":
            d_nets = [str(n) for n in (getattr(decl, "nets", ()) or ())]
            attrs = getattr(decl, "attributes", None) or {}
            w = attrs.get("width_um") or attrs.get("width")
            if d_nets and w is not None:
                try:
                    existing_widths[d_nets[0]] = float(w)
                except Exception:
                    pass

    # Infer differential/matched pairs from standard symmetric suffix naming if none declared
    if not diff_pairs and nets:
        by_low = {n.lower(): n for n in nets}
        for n in sorted(nets):
            low = n.lower()
            for p_suf, n_suf in (("_p", "_n"), ("p", "n"), ("_plus", "_minus")):
                if low.endswith(p_suf) and len(low) > len(p_suf):
                    mate_low = low[: -len(p_suf)] + n_suf
                    if mate_low in by_low and by_low[mate_low] != n:
                        pair = (n, by_low[mate_low])
                        if pair not in diff_pairs and (pair[1], pair[0]) not in diff_pairs:
                            diff_pairs.append(pair)
                        break

    has_instances = bool(getattr(cell_def, "instances", None))
    has_style = bool((getattr(cell_def, "topology", None) or {}).get("style"))
    device_universe_known = bool(device_pins) and (not has_instances) and (not has_style)

    return {
        "cell_name": cell_name,
        "nets": nets,
        "device_pins": device_pins,
        "passive_pins": passive_pins,
        "diff_pairs": diff_pairs,
        "existing_widths": existing_widths,
        "sub_net": sub_net,
        "pos_net": pos_net,
        "device_universe_known": device_universe_known,
    }


def _normalize_spice_node(node: str, known_nets: Set[str]) -> str:
    """Map an extracted RC sub-node like ``vout_3`` or ``vout.n1`` back to ``vout``."""
    raw = str(node).strip()
    if not raw:
        return raw
    if raw in known_nets:
        return raw
    low_map = {n.lower(): n for n in known_nets}
    if raw.lower() in low_map:
        return low_map[raw.lower()]

    # Try stripping Magic/ext2spice RC segmentation suffixes while avoiding collisions
    # Longest known net match with delimiter [_.#/:] wins.
    best_match: Optional[str] = None
    best_len = -1
    raw_low = raw.lower()
    for net in known_nets:
        nl = net.lower()
        if raw_low.startswith(nl) and len(raw_low) > len(nl):
            sep = raw_low[len(nl)]
            if sep in "_.#/[:":
                if len(nl) > best_len:
                    best_match = net
                    best_len = len(nl)
    if best_match is not None:
        return best_match

    # Heuristic fallback when cell_def has partial netlist
    for pat in (r"^(.*?)(?:_n\d+|_\d+|\.n\d+|\.\d+|#\d+|/\d+)$",):
        m = re.match(pat, raw)
        if m and m.group(1):
            return m.group(1)
    return raw


def _aggregate_rc(
    report: PexReport,
    known_nets: Set[str],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[Tuple[str, str], float]]:
    """Aggregate per-net series R (ohm), total node C (fF), and pair coupling C (fF)."""
    r_by_net: Dict[str, float] = {}
    for res in report.resistors:
        if not math.isfinite(res.res_ohm) or res.res_ohm <= 0.0 or res.res_ohm >= 1e9:
            continue
        na = _normalize_spice_node(res.a, known_nets)
        nb = _normalize_spice_node(res.b, known_nets)
        if na == nb:
            r_by_net[na] = r_by_net.get(na, 0.0) + float(res.res_ohm)
        else:
            # Explicit branch/lumped series resistor attached to a net port (e.g. vout <-> vout_load)
            stem_a = re.split(r"[_.:/]", na)[0]
            stem_b = re.split(r"[_.:/]", nb)[0]
            if stem_a == stem_b and stem_a:
                target = stem_a if stem_a in known_nets else na
                r_by_net[target] = r_by_net.get(target, 0.0) + float(res.res_ohm)
            else:
                for cand in (na, nb):
                    if cand in known_nets or not known_nets:
                        r_by_net[cand] = r_by_net.get(cand, 0.0) + 0.5 * float(res.res_ohm)

    total_cap_fF: Dict[str, float] = {}
    pair_cc_fF: Dict[Tuple[str, str], float] = {}
    for cap in report.caps:
        if not math.isfinite(cap.cap_aF) or cap.cap_aF <= 0.0:
            continue
        c_ff = float(cap.cap_aF) * 1e-3  # aF -> fF
        na = _normalize_spice_node(cap.a, known_nets)
        nb = _normalize_spice_node(cap.b, known_nets)
        total_cap_fF[na] = total_cap_fF.get(na, 0.0) + c_ff
        if nb != na:
            total_cap_fF[nb] = total_cap_fF.get(nb, 0.0) + c_ff
            pair = (na, nb) if na <= nb else (nb, na)
            pair_cc_fF[pair] = pair_cc_fF.get(pair, 0.0) + c_ff

    return r_by_net, total_cap_fF, pair_cc_fF


def _extract_op_context(
    op_profile: Any,
    topo: Mapping[str, Any],
) -> Dict[str, Any]:
    """Read branch currents (A), node impedances (ohm), and roles from ``op_profile``."""
    currents_a: Dict[str, float] = {}
    impedance_ohm: Dict[str, float] = {}
    high_z_nets: Set[str] = set()
    clock_nets: Set[str] = set()
    diff_pairs: List[Tuple[str, str]] = list(topo["diff_pairs"])

    def _get(obj: Any, *keys: str, default: Any = None) -> Any:
        for k in keys:
            if isinstance(obj, Mapping) and k in obj:
                return obj[k]
            if hasattr(obj, k):
                return getattr(obj, k)
        return default

    if op_profile is not None:
        raw_cur = _get(op_profile, "net_currents_a", "branch_currents_a", "currents_a") or {}
        if isinstance(raw_cur, Mapping):
            for k, v in raw_cur.items():
                try:
                    currents_a[str(k)] = max(currents_a.get(str(k), 0.0), abs(float(v)))
                except Exception:
                    pass

        raw_imp = _get(op_profile, "net_impedance_ohm", "node_resistance_ohm", "impedances_ohm") or {}
        if isinstance(raw_imp, Mapping):
            for k, v in raw_imp.items():
                try:
                    impedance_ohm[str(k)] = max(impedance_ohm.get(str(k), 0.0), abs(float(v)))
                except Exception:
                    pass

        raw_nets = _get(op_profile, "nets", "node_ops")
        if isinstance(raw_nets, Mapping):
            for net, info in raw_nets.items():
                n = str(net)
                cur = _get(info, "dc_current_a", "current_a", "i_a")
                imp = _get(info, "r_out_ohm", "impedance_ohm", "r_thev_ohm")
                is_hz = _get(info, "high_z", default=False)
                is_clk = _get(info, "is_clock", "clock", default=False)
                if cur is not None:
                    currents_a[n] = max(currents_a.get(n, 0.0), abs(float(cur)))
                if imp is not None:
                    impedance_ohm[n] = max(impedance_ohm.get(n, 0.0), abs(float(imp)))
                if is_hz:
                    high_z_nets.add(n)
                if is_clk:
                    clock_nets.add(n)

        raw_devs = _get(op_profile, "device_ops", "devices")
        if isinstance(raw_devs, Mapping):
            dev_pins = topo["device_pins"]
            for dev_name, dop in raw_devs.items():
                pins = dev_pins.get(str(dev_name), {})
                id_val = _get(dop, "id_a", "ids_a", "i_dc_a", "current_a")
                ro_val = _get(dop, "ro_ohm", "r_out_ohm")
                if id_val is not None:
                    i_abs = abs(float(id_val))
                    for role in ("d", "drain", "s", "source"):
                        net = pins.get(role)
                        if net:
                            currents_a[net] = max(currents_a.get(net, 0.0), i_abs)
                if ro_val is not None and pins.get("d"):
                    dnet = pins["d"]
                    impedance_ohm[dnet] = max(impedance_ohm.get(dnet, 0.0), abs(float(ro_val)))

        for n in _get(op_profile, "high_z_nets") or ():
            high_z_nets.add(str(n))
        for n in _get(op_profile, "clock_nets") or ():
            clock_nets.add(str(n))
        for pair in _get(op_profile, "diff_pairs") or ():
            if isinstance(pair, (tuple, list)) and len(pair) == 2:
                t = (str(pair[0]), str(pair[1]))
                if t not in diff_pairs and (t[1], t[0]) not in diff_pairs:
                    diff_pairs.append(t)

    # Augment from impedance & net naming heuristics
    for net, imp in impedance_ohm.items():
        if imp >= 50e3:
            high_z_nets.add(net)
    for net in topo["nets"]:
        nl = net.lower()
        if any(tok in nl for tok in _CLK_TOKENS):
            clock_nets.add(net)
        if any(tok in nl for tok in _HIGH_Z_TOKENS):
            high_z_nets.add(net)

    return {
        "currents_a": currents_a,
        "impedance_ohm": impedance_ohm,
        "high_z_nets": high_z_nets,
        "clock_nets": clock_nets,
        "diff_pairs": diff_pairs,
    }


def _select_low_r_trunk_layer(tech: Any) -> Tuple[int, str, float, float]:
    """Return ``(layer_idx, orient, sheet_r_ohm_sq, min_width_um)`` for top routable metal."""
    from clay.layout.route.lower import conventional_axes_for

    axes = conventional_axes_for(tech)
    best_idx = 3
    best_rsq = 0.080
    candidates: List[Tuple[int, float, float]] = []
    for n in (5, 4, 3, 2, 1):
        try:
            tech.metal_key(n)
            mw_nm = tech.min_width(f"metal{n}")
        except Exception:
            continue
        if mw_nm is None:
            continue
        try:
            rsq = float(tech.calibrated_metal_sheet_resistance_ohm_sq(n) or 0.080)
        except Exception:
            rsq = 0.080 if n < 4 else 0.050
        candidates.append((n, rsq, float(mw_nm) / 1000.0))

    # Prefer metal4 / metal3 when sheet-R is minimal among standard metals
    routable = [c for c in candidates if c[0] in (3, 4)] or candidates
    if routable:
        # Pick highest metal among tied minimum sheet resistance
        best = min(routable, key=lambda c: (c[1], -c[0]))
        best_idx, best_rsq, min_w_um = best
    else:
        min_w_um = 0.280
    return best_idx, axes.get(best_idx, "H" if best_idx % 2 == 1 else "V"), best_rsq, min_w_um


def _kpi_val(kpis: Optional[Mapping[str, Any]], *keys: str) -> Optional[float]:
    if not isinstance(kpis, Mapping):
        return None
    for k in keys:
        if k in kpis and kpis[k] is not None:
            try:
                v = float(kpis[k])
                if math.isfinite(v):
                    return v
            except Exception:
                pass
    return None


def _kpi_indicates_ir_drop(
    pre_kpis: Optional[Mapping[str, Any]],
    post_kpis: Optional[Mapping[str, Any]],
) -> bool:
    for key in ("load_reg_mv", "dropout_mv", "vout_error_mv", "ir_drop_mv", "inl_max_lsb"):
        pre_v = _kpi_val(pre_kpis, key)
        post_v = _kpi_val(post_kpis, key)
        if post_v is not None and pre_v is not None and post_v > max(pre_v * 1.25, pre_v + 2.0):
            return True
    vout_pre = _kpi_val(pre_kpis, "vout_v", "vout")
    vout_post = _kpi_val(post_kpis, "vout_v", "vout")
    if vout_pre is not None and vout_post is not None and vout_pre > 0.1:
        if (vout_pre - vout_post) >= 0.015:
            return True
    return False


def _kpi_indicates_coupling_or_kickback(
    pre_kpis: Optional[Mapping[str, Any]],
    post_kpis: Optional[Mapping[str, Any]],
    intent_verdict: Any,
) -> bool:
    for key in ("kickback_mv", "offset_mv", "clk_regen_imbalance_fF", "diff_cc_imbalance_fF"):
        pre_v = _kpi_val(pre_kpis, key)
        post_v = _kpi_val(post_kpis, key)
        if post_v is not None and (
            (pre_v is not None and post_v > max(pre_v * 1.3, pre_v + 1.0))
            or (pre_v is None and post_v > 5.0)
        ):
            return True
    if intent_verdict is not None:
        text = str(intent_verdict).lower()
        if any(k in text for k in ("matched_cc_balance", "kickback", "shield", "clk_regen")):
            if "fail" in text or "violat" in text or "false" in text:
                return True
    return False


def _kpi_indicates_bandwidth_drop(
    pre_kpis: Optional[Mapping[str, Any]],
    post_kpis: Optional[Mapping[str, Any]],
) -> Tuple[bool, Dict[str, float]]:
    deltas: Dict[str, float] = {}
    degraded = False
    for key in ("gbw_hz", "ugf_hz", "bw_hz", "f3db_hz"):
        pre_v = _kpi_val(pre_kpis, key)
        post_v = _kpi_val(post_kpis, key)
        if pre_v is not None and post_v is not None and pre_v > 0:
            ratio = (post_v - pre_v) / pre_v
            deltas[f"{key}_rel_delta"] = round(ratio, 4)
            if ratio <= -0.10:
                degraded = True
    pm_pre = _kpi_val(pre_kpis, "pm_deg")
    pm_post = _kpi_val(post_kpis, "pm_deg")
    if pm_pre is not None and pm_post is not None:
        deltas["pm_deg_delta"] = round(pm_post - pm_pre, 3)
        if pm_post - pm_pre <= -5.0:
            degraded = True
    return degraded, deltas


# --------------------------------------------------------------------------- #
# Root Cause Diagnosis Synthesizers
# --------------------------------------------------------------------------- #
def _find_kelvin_sense_and_device(
    force_net: str,
    topo: Mapping[str, Any],
) -> Tuple[Optional[str], Optional[str], bool]:
    """Locate ``(sense_net, target_device, can_emit_directive)`` for @kelvin."""
    device_pins: Mapping[str, Mapping[str, str]] = topo["device_pins"]
    passive_pins: Mapping[str, Mapping[str, str]] = topo["passive_pins"]

    # Prefer explicit feedback net connected together with force_net on one device
    feedback_nets = [
        n
        for n in sorted(topo["nets"])
        if n != force_net and any(tok in n.lower() for tok in _FEEDBACK_TOKENS)
    ]

    for dev_name, pins in sorted(device_pins.items()):
        pvals = set(pins.values())
        if force_net not in pvals:
            continue
        for fb in feedback_nets:
            if fb in pvals and fb != force_net:
                return fb, dev_name, True

    # If topology devices don't directly connect force+sense (e.g., passive divider),
    # check passives for feedback sense pair, and check whether net_binding allows target
    for pas_name, pins in sorted(passive_pins.items()):
        pvals = set(pins.values())
        if force_net in pvals:
            for fb in feedback_nets:
                if fb in pvals and fb != force_net:
                    # Only emit `.clay` directive when universe is synthesis-driven
                    can_emit = not topo["device_universe_known"]
                    return fb, pas_name, can_emit

    # Otherwise find main force-carrying pass/mirror device
    for dev_name, pins in sorted(device_pins.items()):
        if force_net in pins.values():
            other_nets = [v for v in pins.values() if v and v != force_net]
            if feedback_nets:
                can_emit = not topo["device_universe_known"]
                return feedback_nets[0], dev_name, can_emit
            if other_nets:
                return other_nets[0], dev_name, not topo["device_universe_known"]
    return None, None, False


def _diagnose_ir_drop(
    *,
    tech: Any,
    topo: Mapping[str, Any],
    op_ctx: Mapping[str, Any],
    r_by_net: Mapping[str, float],
    pre_kpis: Optional[Mapping[str, Any]],
    post_kpis: Optional[Mapping[str, Any]],
    intent_verdict: Any,
) -> Tuple[List[RcRepairAction], Dict[str, float]]:
    actions: List[RcRepairAction] = []
    ir_drop_mv: Dict[str, float] = {}
    currents: Mapping[str, float] = op_ctx["currents_a"]

    top_idx, orient, rsq_top, min_w_um = _select_low_r_trunk_layer(tech)
    try:
        rsq_m1 = float(tech.calibrated_metal_sheet_resistance_ohm_sq(1) or 0.090)
    except Exception:
        rsq_m1 = 0.090

    ir_kpi_bad = _kpi_indicates_ir_drop(pre_kpis, post_kpis)
    verdict_ir_bad = bool(
        intent_verdict is not None
        and any(k in str(intent_verdict).lower() for k in ("power_budget", "ir_drop", "width"))
        and "fail" in str(intent_verdict).lower()
    )

    all_nets = sorted(set(r_by_net) | set(currents))
    for net in all_nets:
        r_net = float(r_by_net.get(net, 0.0))
        i_net = float(currents.get(net, 0.0))
        if r_net <= 0.0:
            continue
        v_drop_mv = i_net * r_net * 1000.0
        ir_drop_mv[net] = round(v_drop_mv, 3)

        is_supply_or_out = (
            net.lower().rstrip("!") in (_POWER_STEMS | _GROUND_STEMS)
            or any(tok in net.lower() for tok in ("vout", "out", "ldo", "iout"))
        )
        thresh_mv = 10.0 if not (ir_kpi_bad or verdict_ir_bad) else 4.0
        triggered = (
            (v_drop_mv >= thresh_mv)
            or (is_supply_or_out and r_net >= 5.0)
            or (ir_kpi_bad and is_supply_or_out and r_net >= 1.0)
        )
        if not triggered:
            continue

        cur_w = float(topo["existing_widths"].get(net, min_w_um))
        target_r = min(max(0.05, r_net * 0.2), 0.005 / max(i_net, 1e-4))
        scale_factor = max(2.0, r_net / max(target_r, 0.05))
        rsq_ratio = min(1.0, rsq_top / max(rsq_m1, 0.01))
        raw_new_w = cur_w * rsq_ratio * scale_factor
        # Snap upward to 0.05um steps and clamp cleanly to routable envelope range
        snapped_w = math.ceil(raw_new_w / 0.05) * 0.05
        new_w_um = round(max(2.0 * min_w_um, min(snapped_w, 12.0)), 3)

        reason = (
            f"Series R={r_net:.3f}ohm with I={i_net * 1000.0:.2f}mA gives "
            f"IR drop={v_drop_mv:.2f}mV on {net!r}; theoretical widened target={new_w_um:.3f}um "
            f"on metal{top_idx}"
        )
        actions.append(
            RcRepairAction(
                kind="wire_width",
                category="ir_drop",
                net=net,
                directive=f"@wire_width({net}, {new_w_um:.3f}u);",
                directive_key=f"wire_width:{net}",
                params={
                    "width_um": new_w_um,
                    "r_net_ohm": round(r_net, 4),
                    "i_net_a": round(i_net, 6),
                    "ir_drop_mv": round(v_drop_mv, 3),
                    "r_sheet_top_ohm_sq": rsq_top,
                },
                reason=reason,
            )
        )
        actions.append(
            RcRepairAction(
                kind="trunk",
                category="ir_drop",
                net=net,
                directive=f"@trunk({net}, layer=metal{top_idx}, orient={orient}, width={new_w_um:.3f});",
                directive_key=f"trunk:{net}",
                params={"layer": f"metal{top_idx}", "orient": orient, "width_um": new_w_um},
                reason=f"Promote high-current net {net!r} trunk to lower-resistance metal{top_idx}",
            )
        )

        # Kelvin sense recommendation on output / power regulation nets
        if any(tok in net.lower() for tok in ("vout", "ldo", "vreg")) or net.lower() == topo["pos_net"].lower():
            sense_net, target_dev, can_emit = _find_kelvin_sense_and_device(net, topo)
            if sense_net and target_dev:
                directive = (
                    f"@kelvin({net}, sense={sense_net}, target={target_dev});"
                    if can_emit
                    else ""
                )
                actions.append(
                    RcRepairAction(
                        kind="kelvin",
                        category="ir_drop",
                        net=net,
                        target_net=sense_net,
                        target_device=target_dev,
                        directive=directive,
                        directive_key=f"kelvin:{net}:{sense_net}" if directive else "",
                        params={
                            "force_net": net,
                            "sense_net": sense_net,
                            "target_device": target_dev,
                            "emittable_to_clay": can_emit,
                        },
                        reason=(
                            f"Use Kelvin force/sense tap separating current on {net!r} "
                            f"from feedback sense {sense_net!r} at {target_dev!r}"
                        ),
                    )
                )

    return actions, ir_drop_mv


def _diagnose_coupling_imbalance(
    *,
    tech: Any,
    topo: Mapping[str, Any],
    op_ctx: Mapping[str, Any],
    pair_cc_fF: Mapping[Tuple[str, str], float],
    pre_kpis: Optional[Mapping[str, Any]],
    post_kpis: Optional[Mapping[str, Any]],
    intent_verdict: Any,
) -> List[RcRepairAction]:
    actions: List[RcRepairAction] = []
    sub_net = topo["sub_net"]
    clock_nets: Set[str] = op_ctx["clock_nets"]
    diff_pairs: Sequence[Tuple[str, str]] = op_ctx["diff_pairs"]
    ignored_rails = {sub_net.lower(), topo["pos_net"].lower()} | _GROUND_STEMS | _POWER_STEMS

    try:
        min_sp_um = float(tech.min_space("metal2") or 280) / 1000.0
    except Exception:
        min_sp_um = 0.280
    target_sp_um = round(max(3.0 * min_sp_um, 0.84), 3)

    def _c_between(a: str, b: str) -> float:
        pair = (a, b) if a <= b else (b, a)
        return float(pair_cc_fF.get(pair, 0.0))

    kpi_cc_bad = _kpi_indicates_coupling_or_kickback(pre_kpis, post_kpis, intent_verdict)
    shielded_pairs: Set[Tuple[str, str]] = set()

    # 1. Differential / matched pair imbalance check -> find worst asymmetric aggressor
    candidate_nets = sorted(
        {n for p in pair_cc_fF for n in p} | set(topo["nets"])
    )
    for na, nb in diff_pairs:
        best_agg: Optional[str] = None
        best_score = 0.0
        ca_best = 0.0
        cb_best = 0.0
        for agg in candidate_nets:
            if agg in (na, nb) or agg.lower().rstrip("!") in ignored_rails:
                continue
            ca = _c_between(agg, na)
            cb = _c_between(agg, nb)
            diff = abs(ca - cb)
            tot = ca + cb
            rel = diff / max(ca, cb, 1e-6)
            is_clk = agg in clock_nets
            score = diff * (2.0 if is_clk else 1.0) + (0.25 * tot if is_clk else 0.0)
            if (diff >= 0.5 and rel >= 0.10) or (is_clk and max(ca, cb) >= 1.5 and (kpi_cc_bad or diff >= 0.3)):
                if score > best_score:
                    best_score = score
                    best_agg = agg
                    ca_best, cb_best = ca, cb

        if best_agg is not None:
            for vic, c_val in ((na, ca_best), (nb, cb_best)):
                pair_key = (best_agg, vic)
                if pair_key in shielded_pairs:
                    continue
                shielded_pairs.add(pair_key)
                sp_key = "|".join(sorted((best_agg, vic)))
                reason = (
                    f"Coupling imbalance on diff pair ({na}, {nb}): aggressor {best_agg!r} "
                    f"couples {ca_best:.2f}fF vs {cb_best:.2f}fF (|dC|={abs(ca_best - cb_best):.2f}fF)"
                )
                actions.append(
                    RcRepairAction(
                        kind="shield",
                        category="coupling_imbalance",
                        net=vic,
                        target_net=best_agg,
                        directive=f"@shield({best_agg} -> {vic}, {sub_net});",
                        directive_key=f"shield:{best_agg}->{vic}",
                        params={
                            "aggressor": best_agg,
                            "victim": vic,
                            "shield_net": sub_net,
                            "c_aggressor_fF": round(c_val, 3),
                            "delta_c_fF": round(abs(ca_best - cb_best), 3),
                        },
                        reason=reason,
                    )
                )
                actions.append(
                    RcRepairAction(
                        kind="net_spacing",
                        category="coupling_imbalance",
                        net=vic,
                        target_net=best_agg,
                        directive=f"@net_spacing({best_agg}, {vic}, {target_sp_um:.3f}u);",
                        directive_key=f"net_spacing:{sp_key}",
                        params={"net_a": best_agg, "net_b": vic, "spacing_um": target_sp_um},
                        reason=f"Increase spacing between {best_agg!r} and {vic!r} to {target_sp_um:.3f}um",
                    )
                )

    # 2. Direct clock / kickback coupling even when diff pair wasn't explicitly named
    if not actions or kpi_cc_bad:
        scored_cc = sorted(pair_cc_fF.items(), key=lambda kv: -kv[1])
        for (a, b), c_ff in scored_cc:
            if a.lower().rstrip("!") in ignored_rails or b.lower().rstrip("!") in ignored_rails:
                continue
            a_clk, b_clk = a in clock_nets, b in clock_nets
            if not (a_clk ^ b_clk) and not kpi_cc_bad:
                continue
            agg, vic = (a, b) if a_clk else (b, a)
            if c_ff < (1.5 if a_clk or b_clk else 3.0):
                continue
            if (agg, vic) in shielded_pairs:
                continue
            shielded_pairs.add((agg, vic))
            sp_key = "|".join(sorted((agg, vic)))
            actions.append(
                RcRepairAction(
                    kind="shield",
                    category="coupling_imbalance",
                    net=vic,
                    target_net=agg,
                    directive=f"@shield({agg} -> {vic}, {sub_net});",
                    directive_key=f"shield:{agg}->{vic}",
                    params={"aggressor": agg, "victim": vic, "shield_net": sub_net, "c_aggressor_fF": round(c_ff, 3)},
                    reason=f"Kickback / coupling from {agg!r} to {vic!r} is {c_ff:.2f}fF; insert grounded shield {sub_net!r}",
                )
            )
            actions.append(
                RcRepairAction(
                    kind="net_spacing",
                    category="coupling_imbalance",
                    net=vic,
                    target_net=agg,
                    directive=f"@net_spacing({agg}, {vic}, {target_sp_um:.3f}u);",
                    directive_key=f"net_spacing:{sp_key}",
                    params={"net_a": agg, "net_b": vic, "spacing_um": target_sp_um},
                    reason=f"Widen spacing {agg!r}<->{vic!r} to {target_sp_um:.3f}um to suppress kickback",
                )
            )
            break

    return actions


def _diagnose_high_z_cap(
    *,
    tech: Any,
    topo: Mapping[str, Any],
    op_ctx: Mapping[str, Any],
    r_by_net: Mapping[str, float],
    total_cap_fF: Mapping[str, float],
    pair_cc_fF: Mapping[Tuple[str, str], float],
    pre_kpis: Optional[Mapping[str, Any]],
    post_kpis: Optional[Mapping[str, Any]],
) -> Tuple[List[RcRepairAction], Dict[str, Dict[str, float]], Dict[str, float]]:
    actions: List[RcRepairAction] = []
    backann: Dict[str, Dict[str, float]] = {}
    bw_bad, bw_deltas = _kpi_indicates_bandwidth_drop(pre_kpis, post_kpis)

    high_z_nets: Set[str] = set(op_ctx["high_z_nets"])
    impedances: Mapping[str, float] = op_ctx["impedance_ohm"]
    ignored_rails = {topo["sub_net"].lower(), topo["pos_net"].lower()} | _GROUND_STEMS | _POWER_STEMS

    top_idx, orient, _rsq_top, _min_w = _select_low_r_trunk_layer(tech)
    try:
        min_sp_um = float(tech.min_space(f"metal{top_idx}") or 280) / 1000.0
    except Exception:
        min_sp_um = 0.280
    target_sp_um = round(max(2.5 * min_sp_um, 0.75), 3)

    # Rank candidate high-Z or heavily loaded signal nets
    ranked = sorted(
        (
            (net, total_cap_fF.get(net, 0.0), impedances.get(net, 0.0))
            for net in sorted(set(total_cap_fF) | high_z_nets)
            if net.lower().rstrip("!") not in ignored_rails
        ),
        key=lambda row: -(row[1] * max(row[2], 1e4)),
    )

    for net, c_tot, r_out in ranked:
        r_ser = float(r_by_net.get(net, 0.0))
        is_explicit_hz = net in high_z_nets or r_out >= 50e3
        cap_excess = c_tot >= (10.0 if (is_explicit_hz or bw_bad) else 30.0)
        if not cap_excess and not (bw_bad and c_tot >= 5.0 and is_explicit_hz):
            continue

        backann[net] = {
            "c_total_fF": round(c_tot, 3),
            "r_series_ohm": round(r_ser, 4),
            "r_out_ohm": round(r_out, 1) if r_out > 0 else 0.0,
        }
        actions.append(
            RcRepairAction(
                kind="trunk",
                category="high_z_cap",
                net=net,
                directive=f"@trunk({net}, layer=metal{top_idx}, orient={orient});",
                directive_key=f"trunk:{net}",
                params={"layer": f"metal{top_idx}", "orient": orient, "c_total_fF": round(c_tot, 3)},
                reason=(
                    f"High-Z / sensitive node {net!r} carries C_tot={c_tot:.2f}fF; "
                    f"promote routing to upper metal{top_idx} to reduce substrate capacitance"
                ),
            )
        )

        # Find worst non-rail neighbor coupling onto this high-Z node
        neighbors = []
        for (a, b), c_ff in pair_cc_fF.items():
            other = b if a == net else (a if b == net else None)
            if other and other.lower().rstrip("!") not in ignored_rails:
                neighbors.append((other, c_ff))
        neighbors.sort(key=lambda kv: -kv[1])
        if neighbors and neighbors[0][1] >= 1.5:
            nbr, nbr_c = neighbors[0]
            sp_key = "|".join(sorted((net, nbr)))
            actions.append(
                RcRepairAction(
                    kind="net_spacing",
                    category="high_z_cap",
                    net=net,
                    target_net=nbr,
                    directive=f"@net_spacing({net}, {nbr}, {target_sp_um:.3f}u);",
                    directive_key=f"net_spacing:{sp_key}",
                    params={"net_a": net, "net_b": nbr, "spacing_um": target_sp_um, "c_coupling_fF": round(nbr_c, 3)},
                    reason=(
                        f"Widen spacing between high-Z node {net!r} and neighbor {nbr!r} "
                        f"(C_ij={nbr_c:.2f}fF) to cut sidewall Miller/fringe capacitance"
                    ),
                )
            )

        actions.append(
            RcRepairAction(
                kind="presim_backannotate",
                category="high_z_cap",
                net=net,
                directive="",
                directive_key="",
                params={
                    "c_total_fF": round(c_tot, 3),
                    "r_series_ohm": round(r_ser, 4),
                    **bw_deltas,
                },
                reason=(
                    f"Back-annotate C={c_tot:.2f}fF, R={r_ser:.3f}ohm on {net!r} "
                    "for pre-sim warm-start retuning of compensation/bias"
                ),
            )
        )

    return actions, backann, bw_deltas


def diagnose_postsim_rc(
    cell_def: Any,
    *,
    op_profile: Any = None,
    pex_report: Any = None,
    intent_verdict: Any = None,
    pre_kpis: Optional[Mapping[str, Any]] = None,
    post_kpis: Optional[Mapping[str, Any]] = None,
    pdk_id: str = "gf180mcu",
) -> RcRepairPlan:
    """Attribute post-layout R+C degradations and synthesize an actionable Clay repair plan."""
    from clay.layout.tech import LayoutTech

    tech = LayoutTech.resolve(pdk_id)
    topo = _known_topology_info(cell_def)
    cell_name = topo["cell_name"]
    report = _coerce_pex_report(pex_report, cell_name, pdk_id)
    r_by_net, total_cap_fF, pair_cc_fF = _aggregate_rc(report, topo["nets"])
    op_ctx = _extract_op_context(op_profile, topo)

    ir_actions, ir_drop_mv = _diagnose_ir_drop(
        tech=tech,
        topo=topo,
        op_ctx=op_ctx,
        r_by_net=r_by_net,
        pre_kpis=pre_kpis,
        post_kpis=post_kpis,
        intent_verdict=intent_verdict,
    )
    cc_actions = _diagnose_coupling_imbalance(
        tech=tech,
        topo=topo,
        op_ctx=op_ctx,
        pair_cc_fF=pair_cc_fF,
        pre_kpis=pre_kpis,
        post_kpis=post_kpis,
        intent_verdict=intent_verdict,
    )
    hz_actions, backann, bw_deltas = _diagnose_high_z_cap(
        tech=tech,
        topo=topo,
        op_ctx=op_ctx,
        r_by_net=r_by_net,
        total_cap_fF=total_cap_fF,
        pair_cc_fF=pair_cc_fF,
        pre_kpis=pre_kpis,
        post_kpis=post_kpis,
    )

    # Deduplicate by directive_key preserving priority order (last wins if stronger)
    combined: List[RcRepairAction] = []
    seen_idx: Dict[str, int] = {}
    for act in [*ir_actions, *cc_actions, *hz_actions]:
        if act.directive_key:
            if act.directive_key in seen_idx:
                idx = seen_idx[act.directive_key]
                # Keep stronger constraint (e.g. trunk with explicit width overrides plain trunk)
                prev = combined[idx]
                if "width=" in act.directive and "width=" not in prev.directive:
                    combined[idx] = act
                continue
            seen_idx[act.directive_key] = len(combined)
        combined.append(act)

    root_causes = tuple(dict.fromkeys(a.category for a in combined))
    degraded = bool(combined)
    coupling_str_map = {
        f"{a}<->{b}": round(val, 3)
        for (a, b), val in sorted(pair_cc_fF.items(), key=lambda kv: -kv[1])
    }
    if degraded:
        summary = (
            f"RC diagnosis for {cell_name}: found {len(combined)} action(s) across "
            f"{', '.join(root_causes)}; {sum(1 for a in combined if a.directive)} layout_intent patch(es)"
        )
    else:
        summary = f"RC diagnosis for {cell_name}: no parasitic R+C degradation threshold exceeded"

    return RcRepairPlan(
        cell_name=cell_name,
        pdk_id=pdk_id,
        degradation_detected=degraded,
        root_causes=root_causes,
        actions=tuple(combined),
        net_series_resistance_ohm={k: round(v, 4) for k, v in sorted(r_by_net.items())},
        net_ir_drop_mv=ir_drop_mv,
        net_total_cap_fF={k: round(v, 3) for k, v in sorted(total_cap_fF.items())},
        coupling_pairs_fF=coupling_str_map,
        presim_backannotation=backann,
        metrics={
            "n_caps": len(report.caps),
            "n_resistors": len(report.resistors),
            "backend": report.backend,
            **bw_deltas,
        },
        summary=summary,
    )


# --------------------------------------------------------------------------- #
# Idempotent .clay patcher
# --------------------------------------------------------------------------- #
def _directive_semantic_key(line: str) -> Optional[str]:
    """Identify one ``layout_intent`` statement semantically so updates replace idempotently."""
    s = line.strip()
    m = re.match(r"^@wire_width\(\s*([A-Za-z_]\w*)\s*,", s)
    if m:
        return f"wire_width:{m.group(1)}"

    m = re.match(r"^@trunk\(\s*([A-Za-z_]\w*)\b(.*)$", s)
    if m and "edge=" not in m.group(2):
        return f"trunk:{m.group(1)}"

    m = re.match(r"^@kelvin\(\s*([A-Za-z_]\w*)\s*,\s*sense\s*=\s*([A-Za-z_]\w*)", s)
    if m:
        return f"kelvin:{m.group(1)}:{m.group(2)}"

    m = re.match(r"^@shield\(\s*([A-Za-z_]\w*)\s*->\s*([A-Za-z_]\w*)", s)
    if m:
        return f"shield:{m.group(1)}->{m.group(2)}"

    m = re.match(r"^@net_spacing\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*,", s)
    if m:
        pair = "|".join(sorted((m.group(1), m.group(2))))
        return f"net_spacing:{pair}"
    return None


def _find_matching_brace(text: str, open_idx: int) -> int:
    depth = 0
    in_line_comment = False
    in_str: Optional[str] = None
    i = open_idx
    n = len(text)
    while i < n:
        ch = text[i]
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_str is not None:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == in_str:
                in_str = None
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            in_line_comment = True
            i += 2
            continue
        if ch in ("'", '"'):
            in_str = ch
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("Unclosed '{' in Clay layout_intent block")


def apply_rc_repair_plan(clay_text: str, plan: RcRepairPlan) -> str:
    """Idempotently inject or update ``plan``'s directives in a ``.clay`` source string."""
    patches: List[Tuple[str, str]] = []
    seen_keys: Set[str] = set()
    for act in plan.actions:
        if not act.directive:
            continue
        key = act.directive_key or _directive_semantic_key(act.directive) or act.directive
        if key in seen_keys:
            continue
        seen_keys.add(key)
        patches.append((key, act.directive.strip()))

    if not patches:
        return clay_text

    # Find target `layout_intent <cell> { ... }`
    cell_pat = re.compile(
        rf"\blayout_intent\s+{re.escape(plan.cell_name)}\s*\{{",
        re.MULTILINE,
    )
    match = cell_pat.search(clay_text)
    if match is None:
        generic_pat = re.compile(r"\blayout_intent\s+[A-Za-z_]\w*\s*\{", re.MULTILINE)
        match = generic_pat.search(clay_text)

    if match is None:
        # Append a new layout_intent <cell_name> block cleanly at end of file
        sep = "" if (not clay_text or clay_text.endswith("\n")) else "\n"
        extra_nl = "\n" if clay_text.strip() and not clay_text.endswith("\n\n") else ""
        body_lines = "\n".join(f"  {d}" for _, d in patches)
        block = f"layout_intent {plan.cell_name} {{\n{body_lines}\n}}\n"
        return f"{clay_text}{sep}{extra_nl}{block}"

    open_idx = match.end() - 1
    close_idx = _find_matching_brace(clay_text, open_idx)
    body = clay_text[open_idx + 1 : close_idx]

    patch_map = dict(patches)
    remaining = dict(patches)
    out_lines: List[str] = []
    default_indent = "  "

    for raw_line in body.splitlines():
        stripped = raw_line.strip()
        if stripped:
            lead = raw_line[: len(raw_line) - len(raw_line.lstrip())]
            if lead:
                default_indent = lead
        key = _directive_semantic_key(stripped) if stripped else None
        if key is not None and key in patch_map:
            new_stmt = patch_map[key]
            remaining.pop(key, None)
            indent = raw_line[: len(raw_line) - len(raw_line.lstrip())] or default_indent
            out_lines.append(f"{indent}{new_stmt}")
        else:
            out_lines.append(raw_line)

    # Preserve order for newly appended patches
    for key, stmt in patches:
        if key in remaining:
            if out_lines and out_lines[-1].strip() == "" and len(out_lines) > 1:
                out_lines.insert(-1, f"{default_indent}{stmt}")
            else:
                out_lines.append(f"{default_indent}{stmt}")

    # Clean leading/trailing blank-line formatting identically to input
    while out_lines and out_lines[0].strip() == "":
        out_lines.pop(0)
    while out_lines and out_lines[-1].strip() == "":
        out_lines.pop()

    new_body = "\n" + "\n".join(out_lines) + "\n" if out_lines else "\n"
    return clay_text[: open_idx + 1] + new_body + clay_text[close_idx:]


__all__ = [
    "RcRepairAction",
    "RcRepairPlan",
    "apply_rc_repair_plan",
    "diagnose_postsim_rc",
]
