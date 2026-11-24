"""PDK characterizer: read a `pdks/<name>` directory and DERIVE the PDK-dependent
parameters Clay's porting needs — Vdd, Lmin, Vth, KP — so a new PDK is portable to
without hand-coding a preset. The general fix for "I added many PDKs": measure the
PDK, don't hand-write its parameters.

Targets the REAL commercial PDKs (external_pdk / external_pdk), which ship plaintext HSPICE BSIM4
models (not ngspice). Format handled:
  * `.model <name>[.bin] nmos|pmos ( level = 54  +vth0 = 'nom + deltas'  +u0 =
    '((nom + ...))'  +toxe = {nom*mult}  +lmin = 'nom - dxl' )`
  * models split across `.l` / `.lib` / `.inc`'d `.mdl` files, in `.lib <corner>`
    sections, L-binned (.1/.2/.3), with device flavors (core nch/n18 vs
    rf/hvt/lvt/native/25/33).
The CORE logic device is selected (skip rf/hvt/.../cap/esd/res/bjt) and, across its
L-bins, the smallest-Lmin bin (the process min channel). Each parameter is the
LEADING numeric of its (mismatch-laden) expression -- the nominal value.

Output is a `clay.frontend.pdk_retarget.PdkParams` (+ provenance), feeding
retarget_params (sizing port) and clay tune (sim-in-loop refine on the target PDK).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_EPS_OX = 3.9 * 8.854e-12          # oxide permittivity F/m
_NUM = r"\d*\.?\d+(?:[eE][-+]?\d+)?"
_MODEL_EXT = (".l", ".lib", ".mdl", ".model", ".mod", ".ngspice", ".spice", ".pm3", ".scs")
_MAX_FILE = 64_000_000             # commercial model files reach ~20 MB
# device-name tokens that are NOT the core logic FET.
_SKIP = ("rf", "cap", "esd", "res", "bjt", "dnw", "native", "_nat", "mim", "diode",
         "var", "_dn", "_iso", "hvt", "lvt", "mvt", "_25", "_33", "33", "_5", "_6",
         "tl", "fmom", "mom", "jvar")


@dataclass(frozen=True)
class PdkCharacterization:
    pdk: str
    params: object                      # clay.frontend.pdk_retarget.PdkParams
    mode: str
    provenance: Dict[str, object] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()


def _model_files(pdk_dir: Path) -> List[Path]:
    out = []
    for p in pdk_dir.rglob("*"):
        if not (p.is_file() and p.suffix.lower() in _MODEL_EXT):
            continue
        n = p.name.lower()
        # skip obviously-irrelevant model files (passives / RF / shrink variants)
        if any(s in n for s in ("res", "_mim", "_bjt", "_rf", "shrink", "_usage", "external_lvs", "external_drc", "qrc")):
            continue
        try:
            if p.stat().st_size <= _MAX_FILE:
                out.append(p)
        except OSError:
            pass
    return out


# standard analog supply rails -- a parsed tag is only a Vdd if it is one of these
# (rejects version tokens like `v1d0_2p2` whose `2p2` is not a supply).
_STD_VDD = frozenset({0.9, 1.0, 1.1, 1.2, 1.5, 1.8, 2.5, 3.0, 3.3, 5.0})


def _vdd_from_token(s: str) -> Optional[float]:
    """Supply from a voltage tag: 3p3->3.3, 1d8->1.8, 01v8->1.8, n18/_18->1.8.
    Collects ALL candidate tags and returns the first that is a STANDARD supply,
    so a version suffix (``v1d0_2p2`` -> 2.2) is not mistaken for the rail."""
    s = s.lower()
    cands: List[float] = []
    for pat in (r"(\d)d(\d)", r"(\d)p(\d)", r"0?(\d)v(\d)", r"(?:^|[_a-z])([1-9])(\d)(?:[^0-9]|$)"):
        for m in re.finditer(pat, s):
            cands.append(float(f"{m.group(1)}.{m.group(2)}"))
    for v in cands:
        if v in _STD_VDD:
            return v
    return None


def _grab(card: str, param: str) -> Optional[float]:
    """LEADING numeric of a BSIM param assignment in a model card -- the nominal
    value, skipping quotes/braces/parens, sign-aware. Handles `vth0 = '0.37+d...'`,
    `u0 = '((0.012+...))'`, `toxe = {4.1e-9*mult}`, `lmin = '9e-7-dxl'`."""
    m = re.search(rf"(?<![\w]){param}\s*=\s*[^\d+\-]*([-+]?{_NUM})", card, re.IGNORECASE)
    return float(m.group(1)) if m else None


_MODEL_HEAD = re.compile(r"^\s*\.model\s+(\S+)\s+(nmos|pmos)\b", re.IGNORECASE | re.MULTILINE)


def _cards(text: str) -> List[Tuple[str, str, str]]:
    """Every (model_name, kind, card_text) in a file. A card runs from its `.model`
    head to the next `.model` / `.ends` / `.lib` boundary (HSPICE + continuations)."""
    heads = list(_MODEL_HEAD.finditer(text))
    out = []
    for i, m in enumerate(heads):
        start = m.start()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        # also stop at a .ends/.lib boundary inside the span
        seg = text[start:end]
        b = re.search(r"\n\s*\.(ends|lib|endl)\b", seg, re.IGNORECASE)
        if b:
            seg = seg[:b.start()]
        out.append((m.group(1), m.group(2).lower(), seg))
    return out


def _base_name(name: str) -> str:
    return re.sub(r"\.\d+$", "", name).lower()        # nch.3 -> nch


def characterize_pdk(pdk_dir: str, *, name: Optional[str] = None) -> PdkCharacterization:
    """Derive a PdkParams for the PDK at ``pdk_dir`` from its HSPICE BSIM model cards."""
    from clay.frontend.pdk_retarget import PdkParams

    d = Path(pdk_dir)
    pdk_name = name or d.name

    # collect core nmos/pmos cards across all model files; remember the source file.
    cand: Dict[str, List[Tuple[float, dict]]] = {"nmos": [], "pmos": []}
    for f in _model_files(d):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if ".model" not in text.lower():
            continue
        for model_name, kind, card in _cards(text):
            base = _base_name(model_name)
            if any(s in base for s in _SKIP):
                continue
            lmin = _grab(card, "lmin")
            rec = {
                "file": f, "model": model_name, "base": base,
                "vth0": _grab(card, "vth0"), "u0": _grab(card, "u0"),
                "toxe": _grab(card, "toxe") or _grab(card, "tox") or _grab(card, "toxm"),
                "lmin": lmin,
            }
            # rank key: prefer a parsed Lmin (smallest = core min channel); cards
            # without Lmin sort last.
            cand[kind].append((lmin if lmin is not None else 1.0, rec))

    if not cand["nmos"] or not cand["pmos"]:
        raise RuntimeError(
            f"PDK {pdk_name}: no core NMOS/PMOS .model cards found under {d} "
            f"(nmos={len(cand['nmos'])}, pmos={len(cand['pmos'])}). "
            "Models may be encrypted or in an unsupported format."
        )

    nmos = min(cand["nmos"], key=lambda t: t[0])[1]
    pmos = min(cand["pmos"], key=lambda t: t[0])[1]
    notes: List[str] = []

    def _kp(rec: dict) -> Optional[float]:
        if rec["u0"] is None or not rec["toxe"]:
            return None
        return rec["u0"] * _EPS_OX / rec["toxe"] * 1e6   # uA/V^2 (ratio is what matters)

    kp_n, kp_p = _kp(nmos), _kp(pmos)
    lmin_um = (nmos["lmin"] or pmos["lmin"])
    lmin_um = lmin_um * 1e6 if lmin_um else None
    vdd = (_vdd_from_token(nmos["base"]) or _vdd_from_token(pmos["base"])
           or _vdd_from_token(Path(nmos["file"]).name) or _vdd_from_token(d.name))

    if vdd is None:
        vdd = 1.8
        notes.append("Vdd not in device/file name -> default 1.8 (override)")
    if lmin_um is None:
        lmin_um = 0.18
        notes.append("Lmin not parsed -> default 0.18um (override)")
    for k, rec in (("nmos", nmos), ("pmos", pmos)):
        if rec["vth0"] is None or _kp(rec) is None:
            notes.append(f"{k} vth0/KP partially unparsed ({rec['model']}) -> review")

    def rel(f):
        return str(f.relative_to(d)) if f.is_relative_to(d) else str(f)

    params = PdkParams(
        vdd=float(vdd),
        vth_n=float(nmos["vth0"]) if nmos["vth0"] is not None else 0.4,
        vth_p=float(abs(pmos["vth0"])) if pmos["vth0"] is not None else 0.45,
        kp_n=float(kp_n) if kp_n is not None else 250.0,
        kp_p=float(kp_p) if kp_p is not None else 60.0,
        lmin=float(lmin_um),
    )
    prov = {
        "nmos": {"model": nmos["model"], "file": rel(nmos["file"]),
                 "vth0": nmos["vth0"], "u0": nmos["u0"], "toxe": nmos["toxe"], "lmin": nmos["lmin"]},
        "pmos": {"model": pmos["model"], "file": rel(pmos["file"]),
                 "vth0": pmos["vth0"], "u0": pmos["u0"], "toxe": pmos["toxe"], "lmin": pmos["lmin"]},
    }
    return PdkCharacterization(pdk=pdk_name, params=params, mode="model_card",
                              provenance=prov, notes=tuple(notes))


__all__ = ["PdkCharacterization", "characterize_pdk"]
