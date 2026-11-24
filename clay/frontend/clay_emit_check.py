"""Completeness + size-fidelity obligations for an emitted ``design.clay``.

These two checkers name the contract the front-end emitter
(:func:`clay.frontend.clay_emit.emit_design_clay`) must satisfy and that the
buggy role-group-iterating emitter VIOLATES today:

    1. ``assert_design_clay_complete`` -- LVS-grade schematic equivalence. The
       emitted design.clay's DECLARED device netlist must be a lossless
       projection of the source netlist: every source device appears as exactly
       one declared device with matching kind + terminal->net map; no device is
       dropped or invented; every net is preserved.
    2. ``assert_size_fidelity`` -- every declared device's W/L/M equals the
       source netlist's (within unit tolerance); RED on any default / zero /
       missing size.

Both are INDEPENDENT oracles: they re-derive the ground truth from the SOURCE
netlist (via :func:`clay.frontend.role_infer.infer_roles`, which yields the full
device list with kind + terminals + W/L/M) and read the emitted clay through the
canonical parser (:func:`clay.parser.parse_clay_file` -> ``CellDef.devices``),
never trusting the emitter's own bookkeeping. The verdict is a Clay 5-state
verdict and goes RED on any violation -- never a silent pass.

The terminal-net comparison is canonicalised so that a layout-side body/source
swap or pin-label drift does not mask a real connectivity change: each device is
reduced to its kind + the SET of {pin_label: net} it declares, mapped through a
short pin-alias table (drain/gate/source/body -> d/g/s/b, plus/minus/p/n for
2-terminal devices).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from clay.frontend.role_infer import infer_roles
from clay.frontend.role_model import DeviceCard
from clay.layout.extract.netlist_intent import model_oxide
from clay.parser import parse_clay_source


__all__ = [
    "Verdict",
    "CompletenessReport",
    "SizeFidelityReport",
    "VoltageDomainFidelityReport",
    "assert_design_clay_complete",
    "assert_size_fidelity",
    "assert_voltage_domain_fidelity",
]


# Clay 5-state verdict. Never collapse to a boolean -- an unmeasurable or
# unsupported emit must NOT read as a silent pass.
PASS = "pass"
FAIL = "fail"
UNMEASURABLE = "unmeasurable"
UNSUPPORTED = "unsupported"
ADVISORY = "advisory"


@dataclass(frozen=True)
class Verdict:
    """A 5-state verdict carrying its measured evidence."""

    state: str  # one of PASS / FAIL / UNMEASURABLE / UNSUPPORTED / ADVISORY
    summary: str

    @property
    def is_red(self) -> bool:
        return self.state in (FAIL, UNMEASURABLE, UNSUPPORTED)

    def __bool__(self) -> bool:  # pragma: no cover - guard against truthiness misuse
        raise TypeError(
            "Verdict is 5-state; test .state explicitly, do not use it as a bool"
        )


@dataclass
class CompletenessReport:
    verdict: Verdict
    source_count: int = 0
    declared_count: int = 0
    dropped: List[str] = field(default_factory=list)  # source devices with no decl
    extra: List[str] = field(default_factory=list)  # declared but not in source
    kind_mismatch: List[Tuple[str, str, str]] = field(default_factory=list)
    net_mismatch: List[Tuple[str, str, str]] = field(default_factory=list)
    missing_nets: List[str] = field(default_factory=list)


@dataclass
class SizeFidelityReport:
    verdict: Verdict
    checked: int = 0
    size_mismatch: List[Tuple[str, str, str, str]] = field(default_factory=list)
    missing_or_default: List[Tuple[str, str]] = field(default_factory=list)
    undeclared: List[str] = field(default_factory=list)


@dataclass
class VoltageDomainFidelityReport:
    """Independent source-model versus emitted-primitive evidence."""

    verdict: Verdict
    checked: int = 0
    # name, source model, expected domain, emitted domain
    domain_mismatch: List[Tuple[str, str, str, str]] = field(default_factory=list)
    undeclared: List[str] = field(default_factory=list)


# --- SPICE unit parsing (independent of the parser's Clay-unit table) --------

_SPICE_SUFFIX: Dict[str, float] = {
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


def _spice_to_float(tok: str) -> Optional[float]:
    """Parse a SPICE numeric token (``200u``, ``0.3u``, ``40``, ``54.2122K``).

    Returns ``None`` when the token is empty / symbolic / unparseable so the
    caller can record a missing size rather than crash.
    """
    if tok is None:
        return None
    s = str(tok).strip()
    if not s:
        return None
    m = re.fullmatch(r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([a-zA-Z]*)", s)
    if not m:
        return None
    mant, suf = m.group(1), m.group(2).lower()
    try:
        val = float(mant)
    except ValueError:
        return None
    if not suf:
        return val
    # trailing 'm' on a um/meter token is metres; bare suffixes use the SPICE map.
    if suf in ("meg",):
        return val * 1e6
    first = suf[0]
    if first in _SPICE_SUFFIX:
        return val * _SPICE_SUFFIX[first]
    return val  # unknown unit letter -> treat mantissa as-is


# --- pin canonicalisation ----------------------------------------------------

_PIN_ALIAS: Dict[str, str] = {
    "drain": "d",
    "gate": "g",
    "source": "s",
    "body": "b",
    "bulk": "b",
    "collector": "c",
    "base": "b",
    "emitter": "e",
    "d": "d",
    "g": "g",
    "s": "s",
    "b": "b",
    "plus": "p",
    "minus": "n",
    "pos": "p",
    "neg": "n",
    "p": "p",
    "n": "n",
    "c": "c",
    "e": "e",
}


def _canon_pins(pins: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in pins.items():
        kk = _PIN_ALIAS.get(str(k).strip().lower(), str(k).strip().lower())
        out[kk] = str(v).strip()
    return out


def _canon_kind(kind: str) -> str:
    k = str(kind).strip().lower()
    if k.startswith("nmos") or k in ("nch", "n"):
        return "nmos"
    if k.startswith("pmos") or k in ("pch", "p"):
        return "pmos"
    if k in ("res", "resistor", "r"):
        return "res"
    if k in ("cap", "capacitor", "c", "mim"):
        return "cap"
    if "pnp" in k:
        return "pnp"
    if "npn" in k:
        return "npn"
    return k


# --- source / emitted device extraction --------------------------------------

@dataclass(frozen=True)
class _Dev:
    name: str
    kind: str
    pins: Dict[str, str]
    w: str = ""
    l: str = ""
    m: str = ""


def _source_devices(source_netlist: str, cell_name: str) -> List[_Dev]:
    decomp = infer_roles(source_netlist, cell_name)
    out: List[_Dev] = []
    for d in decomp.devices:  # type: DeviceCard
        source_kind = _canon_kind(d.kind)
        if source_kind == "bjt":
            source_kind = _canon_kind(d.source_model)
        out.append(
            _Dev(
                name=d.name,
                kind=source_kind,
                pins=_canon_pins(d.terminals),
                w=d.w,
                l=d.l,
                m=d.m,
            )
        )
    return out


def _emitted_devices(design_clay: str) -> Optional[List[_Dev]]:
    """Parse the emitted design.clay and read its DECLARED device netlist.

    Returns ``None`` if the clay does not parse (UNMEASURABLE) -- a draft that
    will not even parse cannot be asserted complete.
    """
    cf = parse_clay_source(design_clay)
    cell_defs = getattr(cf, "cell_defs", None)
    if not cell_defs:
        return []
    devs: List[_Dev] = []
    for cd in cell_defs:
        for dev in getattr(cd, "devices", ()) or ():
            sizing = dict(getattr(dev, "sizing", {}) or {})
            kind = _canon_kind(getattr(dev, "device_class", ""))
            # A 2-terminal passive carries its magnitude under the `value` size
            # key (Ohms / Farads); the source-side `_Dev.w` holds the same
            # magnitude (the role model puts a passive's value in its `w` slot),
            # so map `value` -> w to compare like-for-like.
            if kind in ("res", "cap") and sizing.get("W") is None:
                w_val = sizing.get("value")
            else:
                w_val = sizing.get("W")
            devs.append(
                _Dev(
                    name=getattr(dev, "instance_name", ""),
                    kind=kind,
                    pins=_canon_pins(dict(getattr(dev, "pins", {}) or {})),
                    # sizing is already metres (float) from the parser
                    w="" if w_val is None else repr(w_val),
                    l="" if sizing.get("L") is None else repr(sizing.get("L")),
                    m="" if sizing.get("M") is None else repr(sizing.get("M")),
                )
            )
    return devs


# --- the two obligations ------------------------------------------------------

def assert_design_clay_complete(
    design_clay: str,
    source_netlist: str,
    *,
    cell_name: str = "",
) -> CompletenessReport:
    """LVS-grade completeness: the emitted clay's declared device netlist must be
    a lossless projection of ``source_netlist``.

    RED (FAIL) on any dropped / extra / kind-mismatched / net-mismatched device,
    or any source net absent from the declared netlist. UNMEASURABLE if the clay
    does not parse.
    """
    src = _source_devices(source_netlist, cell_name)
    try:
        emitted = _emitted_devices(design_clay)
    except Exception as exc:  # parse failure -> cannot measure completeness
        return CompletenessReport(
            verdict=Verdict(
                UNMEASURABLE,
                f"emitted design.clay did not parse: {type(exc).__name__}: {exc}",
            )
        )

    src_by_name = {d.name: d for d in src}
    emit_by_name = {d.name: d for d in emitted}

    dropped = sorted(set(src_by_name) - set(emit_by_name))
    extra = sorted(set(emit_by_name) - set(src_by_name))

    kind_mismatch: List[Tuple[str, str, str]] = []
    net_mismatch: List[Tuple[str, str, str]] = []
    for name in sorted(set(src_by_name) & set(emit_by_name)):
        s, e = src_by_name[name], emit_by_name[name]
        if s.kind != e.kind:
            kind_mismatch.append((name, s.kind, e.kind))
        if s.pins != e.pins:
            net_mismatch.append((name, str(s.pins), str(e.pins)))

    src_nets = {n for d in src for n in d.pins.values()}
    emit_nets = {n for d in emitted for n in d.pins.values()}
    missing_nets = sorted(src_nets - emit_nets)

    n_problems = (
        len(dropped)
        + len(extra)
        + len(kind_mismatch)
        + len(net_mismatch)
        + len(missing_nets)
    )
    if n_problems == 0:
        verdict = Verdict(
            PASS,
            f"complete: all {len(src)} source devices declared; "
            f"{len(src_nets)} nets preserved",
        )
    else:
        verdict = Verdict(
            FAIL,
            f"INCOMPLETE: {len(dropped)} dropped, {len(extra)} extra, "
            f"{len(kind_mismatch)} kind-mismatch, {len(net_mismatch)} net-mismatch, "
            f"{len(missing_nets)} missing nets "
            f"(source={len(src)}, declared={len(emitted)})",
        )

    return CompletenessReport(
        verdict=verdict,
        source_count=len(src),
        declared_count=len(emitted),
        dropped=dropped,
        extra=extra,
        kind_mismatch=kind_mismatch,
        net_mismatch=net_mismatch,
        missing_nets=missing_nets,
    )


def _sizes_match(src_tok: str, emit_metres: str, *, rel_tol: float = 1e-3) -> bool:
    sv = _spice_to_float(src_tok)
    # emitted sizing is already metres-float repr (or "" / dimensionless M)
    ev = _spice_to_float(emit_metres) if emit_metres else None
    if sv is None and ev is None:
        return True
    if sv is None or ev is None:
        return False
    if sv == 0.0:
        return abs(ev) < 1e-18
    return abs(sv - ev) <= rel_tol * abs(sv)


def assert_size_fidelity(
    design_clay: str,
    source_netlist: str,
    *,
    cell_name: str = "",
) -> SizeFidelityReport:
    """Every declared device's W/L/M equals the source's (unit tolerance).

    RED (FAIL) on any size mismatch, or any device emitted with a default /
    zero / missing size. UNMEASURABLE if the clay does not parse.
    """
    src = _source_devices(source_netlist, cell_name)
    try:
        emitted = _emitted_devices(design_clay)
    except Exception as exc:
        return SizeFidelityReport(
            verdict=Verdict(
                UNMEASURABLE,
                f"emitted design.clay did not parse: {type(exc).__name__}: {exc}",
            )
        )

    emit_by_name = {d.name: d for d in emitted}

    size_mismatch: List[Tuple[str, str, str, str]] = []
    missing_or_default: List[Tuple[str, str]] = []
    undeclared: List[str] = []
    checked = 0

    for s in src:
        e = emit_by_name.get(s.name)
        if e is None:
            undeclared.append(s.name)
            continue
        checked += 1
        for axis, src_tok, emit_tok in (
            ("W", s.w, e.w),
            ("L", s.l, e.l),
            ("M", s.m, e.m),
        ):
            # M defaults to 1 for both source and emit when unspecified.
            if axis == "M":
                src_eff = src_tok if str(src_tok).strip() else "1"
                emit_eff = emit_tok if str(emit_tok).strip() else "1"
            else:
                src_eff, emit_eff = src_tok, emit_tok
            # A W/L source value that exists but is absent (or zero) in the emit
            # is a dropped size (the default-sizing failure the spec flags).
            if axis in ("W", "L"):
                if str(src_eff).strip() and not str(emit_eff).strip():
                    missing_or_default.append((s.name, axis))
                    continue
            if not _sizes_match(src_eff, emit_eff):
                size_mismatch.append((s.name, axis, str(src_tok), str(emit_tok)))

    n_problems = len(size_mismatch) + len(missing_or_default) + len(undeclared)
    if n_problems == 0:
        verdict = Verdict(
            PASS, f"size-fidelity: all {checked} declared devices match W/L/M"
        )
    else:
        verdict = Verdict(
            FAIL,
            f"SIZE-INFIDELITY: {len(size_mismatch)} mismatched, "
            f"{len(missing_or_default)} missing/default size, "
            f"{len(undeclared)} undeclared (checked={checked})",
        )

    return SizeFidelityReport(
        verdict=verdict,
        checked=checked,
        size_mismatch=size_mismatch,
        missing_or_default=missing_or_default,
        undeclared=undeclared,
    )


def _emitted_mos_domains(design_clay: str) -> Dict[str, str]:
    parsed = parse_clay_source(design_clay)
    out: Dict[str, str] = {}
    for cell in getattr(parsed, "cell_defs", ()) or ():
        for device in getattr(cell, "devices", ()) or ():
            name = str(getattr(device, "instance_name", ""))
            primitive = str(getattr(device, "device_class", "")).casefold()
            if primitive.startswith(("nmos3v", "pmos3v")):
                out[name] = "thick"
            elif primitive.startswith(("nmos", "pmos")):
                out[name] = "thin"
    return out


def assert_voltage_domain_fidelity(
    design_clay: str,
    source_netlist: str,
    *,
    cell_name: str = "",
) -> VoltageDomainFidelityReport:
    """Require every source MOS oxide/domain to survive into emitted Clay.

    Completeness deliberately canonicalizes ``nmos3v`` to ``nmos`` because it
    checks topology.  This sibling obligation checks the orthogonal device-family
    fact from the source model and therefore catches a polarity-correct but
    voltage-domain-wrong emission.
    """

    decomp = infer_roles(source_netlist, cell_name)
    source = {
        card.name: (card.source_model, model_oxide(card.source_model))
        for card in decomp.devices
        if _canon_kind(card.kind) in {"nmos", "pmos"}
        and str(card.source_model).strip()
    }
    if not source:
        return VoltageDomainFidelityReport(
            verdict=Verdict(
                ADVISORY,
                "no source MOS model tokens were available for domain comparison",
            )
        )
    try:
        emitted = _emitted_mos_domains(design_clay)
    except Exception as exc:
        return VoltageDomainFidelityReport(
            verdict=Verdict(
                UNMEASURABLE,
                f"emitted design.clay did not parse: {type(exc).__name__}: {exc}",
            )
        )

    mismatch: List[Tuple[str, str, str, str]] = []
    undeclared: List[str] = []
    for name, (source_model, expected) in sorted(source.items()):
        actual = emitted.get(name)
        if actual is None:
            undeclared.append(name)
        elif actual != expected:
            mismatch.append((name, source_model, expected, actual))

    checked = len(source)
    if mismatch or undeclared:
        verdict = Verdict(
            FAIL,
            f"VOLTAGE-DOMAIN-INFIDELITY: {len(mismatch)} mismatched, "
            f"{len(undeclared)} undeclared (checked={checked})",
        )
    else:
        verdict = Verdict(
            PASS,
            f"voltage-domain fidelity: all {checked} source MOS models preserved",
        )
    return VoltageDomainFidelityReport(
        verdict=verdict,
        checked=checked,
        domain_mismatch=mismatch,
        undeclared=undeclared,
    )
