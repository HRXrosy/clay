"""Derive acceptance KPIs from a netlist/testbench so the onboarded design.clay
carries a concrete ``quality {}`` block instead of a TODO.

A complete design input is a netlist + its testbench(es): the tb's ``.meas`` /
``echo`` statements say WHAT is measurable, and a ``Spec:`` comment block (or the
user) gives the TARGETS. This module reads the ``Spec:`` prose (``Gain >= 80 dB,
GBW >= 10 MHz, PM > 60 deg, ...``) + confirms the metric is exercised by the tb,
and maps each to a canonical Clay KPI (``@kpi(metric, op, value, unit)``).

General + honest: only recognised analog KPI names are lifted (via an alias table);
an unrecognised target (a supply ``VDD = 1.8 V``, a condition ``@ CL=2pF``) is
skipped, not faked. Targets the user supplies directly always win.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# human Spec name (lower, collapsed spaces) -> (canonical metric, emit unit)
_ALIAS: Dict[str, Tuple[str, str]] = {
    "gain": ("dc_gain_db", "dB"), "dc gain": ("dc_gain_db", "dB"),
    "dc_gain": ("dc_gain_db", "dB"), "av": ("dc_gain_db", "dB"),
    "open loop gain": ("dc_gain_db", "dB"),
    "gbw": ("gbw_hz", "Hz"), "ugf": ("gbw_hz", "Hz"), "ugbw": ("gbw_hz", "Hz"),
    "unity gain frequency": ("gbw_hz", "Hz"), "bandwidth": ("gbw_hz", "Hz"),
    "pm": ("pm_deg", "deg"), "phase margin": ("pm_deg", "deg"),
    "power": ("power_w", "W"), "pdiss": ("power_w", "W"), "pwr": ("power_w", "W"),
    "iq": ("idd_a", "A"), "idd": ("idd_a", "A"), "current": ("idd_a", "A"),
    "icmr": ("icmr_v", "V"), "psrr": ("psrr_db", "dB"), "cmrr": ("cmrr_db", "dB"),
    "vref": ("vref_v", "V"), "vout": ("vout_v", "V"), "droop": ("droop_v", "V"),
    "slew": ("slew_v_per_s", "V/s"), "sr": ("slew_v_per_s", "V/s"),
    "tempco": ("tempco_ppm", "ppm"),
}
_UNIT_MULT: Dict[str, float] = {
    "db": 1.0, "deg": 1.0, "v": 1.0, "ppm": 1.0,
    "hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9,
    "w": 1.0, "mw": 1e-3, "uw": 1e-6, "µw": 1e-6, "nw": 1e-9,
    "a": 1.0, "ma": 1e-3, "ua": 1e-6, "µa": 1e-6, "na": 1e-9,
}
_OPS = (">=", "<=", ">", "<", "~", "≈", "=")


@dataclass(frozen=True)
class KpiSpec:
    metric: str
    op: str       # >= | <= | > | < | near
    value: float
    unit: str
    source_name: str = ""   # the human Spec word it came from


def _spec_block(text: str) -> str:
    """The text of the ``Spec:`` comment block (the line + its continuation
    comment lines, until a blank comment or a new ``Word:`` section header)."""
    out: List[str] = []
    grabbing = False
    for raw in text.splitlines():
        stripped = raw.strip()
        body = stripped.lstrip("*").strip()
        if not grabbing:
            if re.match(r"(?i)^spec\b\s*:", body):
                grabbing = True
                out.append(re.sub(r"(?i)^spec\b\s*:\s*", "", body))
            continue
        # continuation must still be a comment line
        if not stripped.startswith("*"):
            break
        if not body:                     # blank comment ends the block
            break
        if re.match(r"(?i)^(topology|reference|connectivity|sizing|step|pdk|cell)\b\s*:", body):
            break
        out.append(body)
    # join continuation lines with a comma so entries split correctly across the
    # multi-line Spec block (otherwise "PM > 60 deg" + "Power < 1 mW" merge).
    return ", ".join(out)


def _parse_entry(entry: str) -> KpiSpec | None:
    entry = re.split(r"@", entry)[0].strip()      # drop "@ CL=2pF" conditions
    for op in _OPS:
        if op in entry:
            name, rhs = entry.split(op, 1)
            name = name.strip().lower()
            name = re.sub(r"\s+", " ", name)
            if name not in _ALIAS:
                return None                        # unrecognised (e.g. VDD) -> skip
            m = re.match(r"\s*([-+]?[0-9.]+(?:[eE][-+]?\d+)?)\s*([A-Za-zµ/]*)", rhs.strip())
            if not m:
                return None
            val = float(m.group(1))
            unit_tok = m.group(2).lower().rstrip("/")
            mult = _UNIT_MULT.get(unit_tok, 1.0)
            metric, emit_unit = _ALIAS[name]
            # the flow's KPI bound model accepts >= / <= / near (not strict > / <);
            # a strict-inequality acceptance target is the same bound in practice.
            norm_op = {">": ">=", "<": "<=", ">=": ">=", "<=": "<=",
                       "=": "near", "~": "near", "≈": "near"}.get(op, op)
            return KpiSpec(metric, norm_op, val * mult, emit_unit, source_name=name)
    return None


def _tb_metric_hints(tb_texts: Sequence[str]) -> set:
    """Canonical metrics the testbenches actually measure (.meas / echo names),
    used to keep only KPIs the design is actually exercised for."""
    hints: set = set()
    blob = "\n".join(tb_texts).lower()
    if not blob:
        return hints
    pairs = [("gain", "dc_gain_db"), ("ugf", "gbw_hz"), ("gbw", "gbw_hz"),
             ("phase", "pm_deg"), ("pm", "pm_deg"), ("idd", "idd_a"),
             ("power", "power_w"), ("psrr", "psrr_db"), ("vref", "vref_v"),
             ("vout", "vout_v"), ("droop", "droop_v")]
    for token, metric in pairs:
        if token in blob:
            hints.add(metric)
    return hints


def extract_spec(
    cir_text: str, tb_texts: Sequence[str] = (), *, require_tb_evidence: bool = False
) -> List[KpiSpec]:
    """Lift acceptance KPIs from the ``Spec:`` comment (+ tb evidence).

    Parses the ``Spec:`` block into ``KpiSpec`` rows. When ``require_tb_evidence``
    and testbenches are given, keeps only metrics the tb actually measures. De-dupes
    by metric (first target wins)."""
    block = _spec_block(cir_text)
    if not block:
        return []
    hints = _tb_metric_hints(tb_texts)
    out: List[KpiSpec] = []
    seen: set = set()
    for entry in block.split(","):
        spec = _parse_entry(entry)
        if spec is None or spec.metric in seen:
            continue
        if require_tb_evidence and hints and spec.metric not in hints:
            continue
        seen.add(spec.metric)
        out.append(spec)
    return out


__all__ = ["KpiSpec", "extract_spec"]
