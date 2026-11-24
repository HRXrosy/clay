"""Per-PDK geometric PEX constants, sourced from each PDK's OWN process data.

The lateral / vertical coupling estimate in :mod:`clay.layout.pex.coupling` needs
four real BEOL / device numbers per PDK:

  * ``metal_thickness_um``  -- metal1 conductor thickness (the side-wall plate
    height that faces a lateral neighbour, and the over/under plate of a vertical
    overlap). Sourced from the PDK's OWN interconnect-tech file.
  * ``ild_k`` / ``eps_ild_aF_per_um`` -- inter-metal dielectric relative
    permittivity (and the derived per-um permittivity). Sourced from the
    dielectric block of the same interconnect-tech file.
  * ``metal_vgap_um`` -- vertical separation between metal1 and the next routing
    layer (metal2), the plate separation of a metal-over-metal overlap cap.
    Derived from the two conductors' ``height`` entries in the tech file.
  * ``min_space_um``  -- metal1 minimum edge-to-edge spacing (floors the lateral
    1/s term at the densest legal pitch). Sourced from the PDK's DRC deck (the
    adapter's ``drc_rules()``), with the tech-file ``min_spacing`` as a second
    source.
  * ``cj_aF_um2`` / ``cjsw_aF_um`` -- junction area / sidewall cap of the core
    device (a diffusion-edge parasitic). Read from the PDK's BSIM model card.

WHERE A PDK SOURCE FILE: each interconnect-tech format Clay can read is parsed
from the PDK root --

  * Cadence/external_lvs ICT (``*.ict``): ``conductor metal1 { thickness .. height .. }``
    + ``dielectric <IMD> { dielectric_constant .. }`` (open_pdk, open_pdk).
  * external_lvs RCX ``procfile``: ``step <n> metal metal1 <thickness> ..`` +
    ``step <n> dielectric .. <name> <thickness> <diel_constant>`` (open_pdk).

HONESTY: every value carries provenance saying whether it is REAL-from-PDK (with
the source file + the verbatim datum) or a documented NODE-SCALED DEFAULT used
only where the PDK ships no such datum. This is a PDK-SPECIFIC GEOMETRIC estimate
(``calibrated=False``); it is NOT a field-solver / StarRC / foundry-ITF
extraction. The tech files are read for their geometric stack numbers ONLY; the
coupling model that consumes them stays the same parallel-plate + fringe
approximation, never a 3-D solve.

Default fallbacks (used ONLY where a PDK is silent), node-scaled from the
adapter's ``node_nm`` / ``lmin``:

  * metal_thickness_um : ~3x the node min metal width (Al/Cu BEOL aspect), floor
    0.12 um -- a process-typical metal1 thickness order of magnitude.
  * ild_k              : 4.0 (SiO2-class inter-metal dielectric).
  * min_space_um       : node min (lmin) when the deck/tech file is silent.
  * metal_vgap_um      : ~= metal_thickness_um (a one-thickness inter-layer gap).
  * cj / cjsw          : process-typical 1e-3 F/m^2 area, 2e-10 F/m sidewall
    (180 nm-class junctions); flagged default.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# eps0: 8.8541878128e-12 F/m * 1e-6 m/um * 1e18 aF/F.
EPS0_aF_PER_UM: float = 8.8541878128

# F/m^2 -> aF/um^2 : 1 F/m^2 = 1e18 aF / 1e12 um^2 = 1e6 aF/um^2.
_F_PER_M2_TO_aF_PER_UM2: float = 1.0e6
# F/m -> aF/um : 1 F/m = 1e18 aF / 1e6 um = 1e12 aF/um.
_F_PER_M_TO_aF_PER_UM: float = 1.0e12

# Node-scaled documented defaults (used ONLY where the PDK ships no datum).
_DEFAULT_ILD_K: float = 4.0          # SiO2-class inter-metal dielectric
_DEFAULT_CJ_F_PER_M2: float = 1.0e-3   # 180nm-class junction area cap
_DEFAULT_CJSW_F_PER_M: float = 2.0e-10  # 180nm-class junction sidewall cap
_DEFAULT_THICKNESS_FLOOR_UM: float = 0.12

_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


@dataclass(frozen=True)
class PdkPexConstants:
    """Per-PDK geometric PEX constants + per-field real-vs-default provenance.

    All-real fields trace to the named source file; defaulted fields are
    node-scaled documented values. ``calibrated`` is ALWAYS False.
    """

    pdk_id: str
    metal_thickness_um: float
    ild_k: float
    eps_ild_aF_per_um: float
    metal_vgap_um: float
    min_space_um: float
    cj_aF_um2: float
    cjsw_aF_um: float
    provenance: Dict[str, Dict[str, object]] = field(default_factory=dict)
    calibrated: bool = False

    @property
    def real_fields(self) -> List[str]:
        return sorted(
            k for k, v in self.provenance.items() if v.get("origin") == "pdk"
        )

    @property
    def defaulted_fields(self) -> List[str]:
        return sorted(
            k for k, v in self.provenance.items() if v.get("origin") == "default"
        )

    def summary(self) -> str:
        real = ", ".join(self.real_fields) or "none"
        dflt = ", ".join(self.defaulted_fields) or "none"
        return (
            f"{self.pdk_id}: real-from-PDK [{real}]; defaulted [{dflt}] "
            f"(PDK-specific geometric, calibrated=False, NOT field-solver/StarRC)"
        )


# ---------------------------------------------------------------------------
# PDK root resolution (adapter, else the per-PDK module's resolver, else prefix)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PDKS_DIR = _REPO_ROOT / "pdks"


def _resolve_root(pdk_id: str, root: Optional[str]) -> Optional[Path]:
    if root:
        p = Path(root)
        return p if p.is_dir() else None
    # 1) registered adapter
    try:
        from clay.pdk import get_pdk

        a = get_pdk(pdk_id)
        r = a.root()
        if r is not None:
            return Path(r)
    except Exception:  # noqa: BLE001  (KeyError for adapter-less PDKs, etc.)
        pass
    # 2) the per-PDK module's own resolver (e.g. open_pdk.resolve_pdk_root)
    try:
        import importlib

        mod = importlib.import_module(f"clay.pdk.{pdk_id}")
        fn = getattr(mod, "resolve_pdk_root", None)
        if callable(fn):
            r = fn()
            if r is not None and Path(r).is_dir():
                return Path(r)
    except Exception:  # noqa: BLE001
        pass
    # 3) directory-prefix scan under pdks/
    if _PDKS_DIR.is_dir():
        for d in sorted(_PDKS_DIR.iterdir()):
            if d.is_dir() and d.name.lower().startswith(pdk_id.lower()[:6]):
                return d
    return None


def _node_nm(pdk_id: str) -> Optional[int]:
    try:
        from clay.pdk import get_pdk

        a = get_pdk(pdk_id)
        if a.node_nm:
            return int(a.node_nm)
        if a.lmin_um:
            return int(round(a.lmin_um * 1000.0))
    except Exception:  # noqa: BLE001
        pass
    # crude fall-back from the id (e.g. open_pdk -> 180, open_pdk -> 180)
    m = re.search(r"(\d{2,3})", pdk_id)
    if m:
        n = int(m.group(1))
        return n if n >= 28 else n * 10   # "18" -> 180
    return None


# ---------------------------------------------------------------------------
# Interconnect-tech parsers (geometric stack only; never the cap LUTs)
# ---------------------------------------------------------------------------


def _find_tech_files(root: Path) -> Dict[str, List[Path]]:
    """Group candidate interconnect-tech files under the PDK root by kind."""
    out: Dict[str, List[Path]] = {"ict": [], "procfile": []}
    try:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            name = p.name.lower()
            if name.endswith(".ict"):
                # prefer a "typical"/"typ"/"tt" corner stack
                out["ict"].append(p)
            elif name == "procfile":
                out["procfile"].append(p)
    except OSError:
        pass

    def _rank(p: Path) -> tuple:
        n = p.name.lower()
        typ = 0 if ("typ" in n or "tt" in n or "nominal" in n) else 1
        return (typ, len(str(p)), str(p))

    out["ict"].sort(key=_rank)
    out["procfile"].sort(key=lambda p: (len(str(p)), str(p)))
    return out


def _parse_ict_metal1(text: str) -> Dict[str, float]:
    """Extract metal1 thickness/height/min_spacing + nearest ILD k from an ICT.

    Handles both quoted (``conductor "metal1" {``) and unquoted
    (``conductor metal1 {``) block names. Returns the keys it found.
    """
    out: Dict[str, float] = {}

    def _block(kind: str, name: str) -> Optional[str]:
        m = re.search(
            rf'{kind}\s+"?{re.escape(name)}"?\s*\{{', text, re.IGNORECASE
        )
        if not m:
            return None
        # capture from the open brace to a balanced-ish end (next "conductor"/
        # "dielectric"/"via" header or EOF) -- enough for the scalar fields.
        start = m.end()
        nxt = re.search(
            r"\n\s*(conductor|dielectric|via)\b", text[start:], re.IGNORECASE
        )
        return text[start : start + (nxt.start() if nxt else len(text) - start)]

    m1 = _block("conductor", "metal1")
    if m1:
        for key, pat in (
            ("thickness", r"\bthickness\s+(" + _NUM + r")"),
            ("height", r"\bheight\s+(" + _NUM + r")"),
            ("min_spacing", r"\bmin_spacing\s+(" + _NUM + r")"),
        ):
            mm = re.search(pat, m1, re.IGNORECASE)
            if mm:
                out[key] = float(mm.group(1))
    m2 = _block("conductor", "metal2")
    if m2:
        mm = re.search(r"\bheight\s+(" + _NUM + r")", m2, re.IGNORECASE)
        if mm:
            out["m2_height"] = float(mm.group(1))

    # ILD k: the dielectric sitting between/around metal1. Prefer an IMD1*/ILD*
    # block; else the median of all dielectric_constant entries (background 1.0
    # excluded). This is the inter-metal lateral dielectric.
    diel_ks: List[float] = []
    for mm in re.finditer(
        r'dielectric\s+"?(\w+)"?\s*\{([^}]*)\}', text, re.IGNORECASE | re.DOTALL
    ):
        name, body = mm.group(1), mm.group(2)
        km = re.search(r"\bdielectric_constant\s+(" + _NUM + r")", body, re.IGNORECASE)
        if not km:
            continue
        k = float(km.group(1))
        if re.search(r"imd1|ild_a|ild1|imd_1", name, re.IGNORECASE):
            out.setdefault("ild_k_imd1", k)
        diel_ks.append(k)
    if "ild_k_imd1" in out:
        out["ild_k"] = out.pop("ild_k_imd1")
    elif diel_ks:
        diel_ks.sort()
        out["ild_k"] = diel_ks[len(diel_ks) // 2]
    return out


def _parse_procfile(text: str) -> Dict[str, float]:
    """Extract metal1 thickness + adjacent ILD k from an external_lvs RCX procfile.

    Lines look like::

        step 4  metal       metal1   0.53   0      -0,0
        step 5  dielectric p IMD1a    0.6499 3.7
    """
    out: Dict[str, float] = {}
    # the metal layer line (and the dielectric immediately ABOVE it = the
    # inter-metal dielectric for metal1's sidewall).
    rows = []
    for line in text.splitlines():
        # step <n> <metal|dielectric> [<p|n> ] <layer_name> <thickness> <val> ...
        # the optional p/n type column is a SINGLE token; the layer name follows.
        m = re.match(
            r"\s*step\s+\d+\s+(metal|dielectric)\s+(?:[pn]\s+)?"
            r"(\w+)\s+(" + _NUM + r")\s+(" + _NUM + r")",
            line, re.IGNORECASE,
        )
        if m:
            rows.append((m.group(1).lower(), m.group(2), float(m.group(3)),
                         float(m.group(4))))
    for i, (kind, name, thick, val) in enumerate(rows):
        if kind == "metal" and name.lower() in ("metal1", "met1", "m1", "mt1"):
            out["thickness"] = thick
            # the NEXT dielectric step is the inter-metal (sidewall) dielectric.
            for j in range(i + 1, len(rows)):
                if rows[j][0] == "dielectric":
                    out["ild_k"] = rows[j][3]
                    break
            break
    return out


def _read_stack(root: Path) -> Tuple[Dict[str, float], Optional[str]]:
    """Return ({thickness/ild_k/min_spacing/m2_height}, source_rel) from the
    first parseable interconnect-tech file under ``root``; ({}, None) if none."""
    tf = _find_tech_files(root)
    for p in tf["ict"]:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        got = _parse_ict_metal1(txt)
        if "thickness" in got or "ild_k" in got:
            return got, _rel(p, root)
    for p in tf["procfile"]:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        got = _parse_procfile(txt)
        if "thickness" in got or "ild_k" in got:
            return got, _rel(p, root)
    return {}, None


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# Junction cap (cj / cjsw) from the BSIM model card
# ---------------------------------------------------------------------------

# core device model heads (skip rf/hvt/native/etc the same way the characterizer
# does), but we still ACCEPT an rf core when that is the only flavour present
# (open_pdk ships only nch_rf as the logic FET).
_CORE_SKIP = ("cap", "esd", "res", "bjt", "dnw", "native", "_nat", "mim",
              "diode", "var", "_dn", "_iso", "_25", "_33", "tl", "mom", "jvar")


def _grab_card_param(card: str, names: Tuple[str, ...]) -> Optional[float]:
    """LEADING numeric of the first of ``names`` assigned in a BSIM card.

    Handles ``cj = '0.00012*cj_d'``, ``cjs = 0.00098714``, ``CJ = 0`` etc.
    Returns the value (may be 0.0 -- caller decides whether 0 means 'use param').
    """
    for nm in names:
        m = re.search(
            rf"(?<![\w]){nm}\s*=\s*[^\d+\-.']*'?\s*([-+]?{_NUM})",
            card, re.IGNORECASE,
        )
        if m:
            return float(m.group(1))
    return None


def _grab_param_def(text: str, names: Tuple[str, ...]) -> Optional[float]:
    """First ``<name> = <number>`` param definition anywhere in the file
    (for cards that parameterise CJ to a top-level ``.param cjn = ...``)."""
    for nm in names:
        m = re.search(
            rf"(?<![\w]){nm}\s*=\s*([-+]?{_NUM})", text, re.IGNORECASE
        )
        if m:
            v = float(m.group(1))
            if v != 0.0:
                return v
    return None


_MODEL_HEAD = re.compile(
    r"^\s*\.model\s+(\S+)\s+(nmos|pmos)\b", re.IGNORECASE | re.MULTILINE
)
_MODEL_EXT = (".l", ".lib", ".mdl", ".model", ".mod", ".spice", ".inc")


def _junction_caps(root: Path) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """(cj_F_per_m2, cjsw_F_per_m, source_rel) of the core device, or Nones.

    Reads the first core NMOS ``.model`` card and pulls the junction area cap
    (cj/cjs/cjd) + sidewall cap (cjsw/cjsws/cjswd). When the literal card value
    is 0 (the value lives in a parameter), falls back to the matching top-level
    param definition (cjn/cjp / cjswn/cjswp).
    """
    files: List[Path] = []
    try:
        for p in root.rglob("*"):
            if not (p.is_file() and p.suffix.lower() in _MODEL_EXT):
                continue
            n = p.name.lower()
            if any(s in n for s in ("_bjt", "bjt", "_res", "_mim", "shrink",
                                    "_usage", "external_lvs", "external_drc", "qrc")):
                continue
            try:
                if p.stat().st_size <= 64_000_000:
                    files.append(p)
            except OSError:
                pass
    except OSError:
        pass
    # prefer hspice model files, smaller first
    files.sort(key=lambda p: (0 if "hspice" in str(p).lower() else 1,
                              p.stat().st_size if p.exists() else 0, str(p)))
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if ".model" not in text.lower():
            continue
        heads = list(_MODEL_HEAD.finditer(text))
        for i, m in enumerate(heads):
            name = m.group(1).lower()
            if m.group(2).lower() != "nmos":
                continue
            base = re.sub(r"\.\d+$", "", name)
            if any(s in base for s in _CORE_SKIP):
                continue
            start = m.start()
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            seg = text[start:end]
            b = re.search(r"\n\s*\.(ends|lib|endl)\b", seg, re.IGNORECASE)
            if b:
                seg = seg[: b.start()]
            cj = _grab_card_param(seg, ("cj", "cjs", "cjd"))
            cjsw = _grab_card_param(seg, ("cjsw", "cjsws", "cjswd"))
            # value(s) may be 0 in the literal card -> read the param definition.
            if cj in (None, 0.0):
                cj = _grab_param_def(text, ("cjn", "cjp", "cj_n", "cj_p"))
            if cjsw in (None, 0.0):
                cjsw = _grab_param_def(text, ("cjswn", "cjswp", "cjsw_n", "cjsw_p"))
            if cj or cjsw:
                return cj, cjsw, _rel(f, root)
    return None, None, None


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------


def stack_dielectric_integral(profile, lower_um: float, upper_um: float) -> float:
    """Integral dz/k for an unshielded one-dimensional stack approximation.

    Overlapping/conformal declarations use the most local layer. Gaps in a
    procfile's sparse deposition stack use its nearest dielectric, not another
    process's constants. This is geometric interpolation, not a field solve.
    """
    if upper_um <= lower_um:
        raise ValueError("PDK stack dielectric interval is not positive")
    dielectrics = tuple(d for d in profile.dielectrics
                        if d.thickness_um > 0 and d.dielectric_constant > 0)
    if not dielectrics:
        raise ValueError("PDK stack has no dielectric constants")
    boundaries = {lower_um, upper_um}
    for d in dielectrics:
        boundaries.update(z for z in (d.height_um, d.height_um + d.thickness_um)
                          if lower_um < z < upper_um)
    points = sorted(boundaries)
    total = 0.
    for left, right in zip(points, points[1:]):
        mid = (left + right) / 2.
        covering = [d for d in dielectrics
                    if d.height_um <= mid <= d.height_um + d.thickness_um]
        if covering:
            selected = max(covering, key=lambda d: (d.height_um, d.thickness_um))
        else:
            selected = min(dielectrics, key=lambda d: min(abs(mid - d.height_um),
                           abs(mid - d.height_um - d.thickness_um)))
        total += (right - left) / selected.dielectric_constant
    return total


def resolve_pdk_pex_constants(
    pdk_id: str, *, root: Optional[str] = None
) -> PdkPexConstants:
    """Resolve per-PDK geometric PEX constants from the PDK's OWN process data.

    BEOL stack (metal1 thickness, inter-metal dielectric k, metal1->metal2 gap)
    comes from the PDK's interconnect-tech file (Cadence ICT or external_lvs RCX
    procfile); min spacing from the adapter's DRC deck (else the tech-file
    min_spacing); junction cj/cjsw from the BSIM model card. Every datum the PDK
    is silent on falls back to a documented NODE-SCALED default and is flagged as
    such in ``provenance``. ``calibrated`` is always False.
    """
    prov: Dict[str, Dict[str, object]] = {}
    root_path = _resolve_root(pdk_id, root)
    node_nm = _node_nm(pdk_id)
    f_um = (node_nm / 1000.0) if node_nm else 0.18    # node feature size (um)

    stack: Dict[str, float] = {}
    stack_src: Optional[str] = None
    if root_path is not None:
        stack, stack_src = _read_stack(root_path)
    if root is None:
        from clay.pdk.pex_profile import get_profile

        profile = get_profile(pdk_id)
        if profile is not None:
            metals = {c.name.lower().replace("mt", "metal", 1): c
                      for c in profile.conductors}
            m1, m2 = metals.get("metal1"), metals.get("metal2")
            if m1 is not None and m2 is not None:
                lower = m1.height_um + m1.thickness_um
                upper = m2.height_um
                stack.update(thickness=m1.thickness_um, height=m1.height_um,
                             m2_height=m2.height_um,
                             ild_k=(upper - lower) / stack_dielectric_integral(profile, lower, upper))
                stack_src = "; ".join(profile.source_files)

    # --- metal thickness -----------------------------------------------------
    if "thickness" in stack:
        thickness = float(stack["thickness"])
        prov["metal_thickness_um"] = {
            "origin": "pdk", "value": thickness, "source": stack_src,
            "datum": "conductor metal1 thickness",
        }
    else:
        thickness = max(3.0 * f_um, _DEFAULT_THICKNESS_FLOOR_UM)
        prov["metal_thickness_um"] = {
            "origin": "default", "value": thickness,
            "note": f"node-scaled (~3x {f_um:g}um node min width, floor "
                    f"{_DEFAULT_THICKNESS_FLOOR_UM}um); PDK ships no tech-file stack",
        }

    # --- ILD k ---------------------------------------------------------------
    if "ild_k" in stack:
        ild_k = float(stack["ild_k"])
        prov["ild_k"] = {
            "origin": "pdk", "value": ild_k, "source": stack_src,
            "datum": "inter-metal dielectric_constant",
        }
    else:
        ild_k = _DEFAULT_ILD_K
        prov["ild_k"] = {
            "origin": "default", "value": ild_k,
            "note": f"SiO2-class k={_DEFAULT_ILD_K}; PDK ships no dielectric block",
        }
    eps_ild = EPS0_aF_PER_UM * ild_k

    # --- metal1->metal2 vertical gap (overlap plate separation) --------------
    if "height" in stack and "m2_height" in stack:
        # gap = (metal2 bottom) - (metal1 top) = m2_height - (m1_height + m1_thick)
        m1_top = stack["height"] + stack.get("thickness", thickness)
        vgap = float(stack["m2_height"]) - m1_top
        if vgap > 0:
            prov["metal_vgap_um"] = {
                "origin": "pdk", "value": vgap, "source": stack_src,
                "datum": "metal2.height - (metal1.height + metal1.thickness)",
            }
        else:
            vgap = None  # bad/negative -> fall through to default
    else:
        vgap = None
    if vgap is None:
        vgap = thickness
        prov["metal_vgap_um"] = {
            "origin": "default", "value": vgap,
            "note": "one-thickness inter-layer gap; PDK tech file gives no "
                    "metal1/metal2 height pair",
        }

    # --- min spacing (DRC deck first, then tech-file min_spacing) ------------
    min_space = None
    drc_src = None
    try:
        from clay.pdk import get_pdk

        a = get_pdk(pdk_id)
        rules = a.drc_rules()
        e = rules.get("metal1_space")
        if e and e.get("value_um"):
            min_space = float(e["value_um"])
            drc_src = "drc_rules()[metal1_space]"
    except Exception:  # noqa: BLE001
        pass
    if min_space is None and "min_spacing" in stack:
        min_space = float(stack["min_spacing"])
        drc_src = f"{stack_src} (conductor metal1 min_spacing)"
    if min_space is not None:
        prov["min_space_um"] = {
            "origin": "pdk", "value": min_space, "source": drc_src,
        }
    else:
        min_space = f_um
        prov["min_space_um"] = {
            "origin": "default", "value": min_space,
            "note": f"node min ({f_um:g}um); no DRC deck / tech-file min_spacing",
        }

    # --- junction caps cj / cjsw --------------------------------------------
    cj_F_m2 = cjsw_F_m = None
    cap_src = None
    if root_path is not None:
        cj_F_m2, cjsw_F_m, cap_src = _junction_caps(root_path)
    if cj_F_m2 and cj_F_m2 > 0:
        cj_aF_um2 = cj_F_m2 * _F_PER_M2_TO_aF_PER_UM2
        prov["cj_aF_um2"] = {
            "origin": "pdk", "value": cj_aF_um2, "source": cap_src,
            "datum": f"BSIM cj={cj_F_m2:g} F/m^2",
        }
    else:
        cj_aF_um2 = _DEFAULT_CJ_F_PER_M2 * _F_PER_M2_TO_aF_PER_UM2
        prov["cj_aF_um2"] = {
            "origin": "default", "value": cj_aF_um2,
            "note": "180nm-class junction area cap; no parseable BSIM cj",
        }
    if cjsw_F_m and cjsw_F_m > 0:
        cjsw_aF_um = cjsw_F_m * _F_PER_M_TO_aF_PER_UM
        prov["cjsw_aF_um"] = {
            "origin": "pdk", "value": cjsw_aF_um, "source": cap_src,
            "datum": f"BSIM cjsw={cjsw_F_m:g} F/m",
        }
    else:
        cjsw_aF_um = _DEFAULT_CJSW_F_PER_M * _F_PER_M_TO_aF_PER_UM
        prov["cjsw_aF_um"] = {
            "origin": "default", "value": cjsw_aF_um,
            "note": "180nm-class junction sidewall cap; no parseable BSIM cjsw",
        }

    return PdkPexConstants(
        pdk_id=pdk_id,
        metal_thickness_um=thickness,
        ild_k=ild_k,
        eps_ild_aF_per_um=eps_ild,
        metal_vgap_um=vgap,
        min_space_um=min_space,
        cj_aF_um2=cj_aF_um2,
        cjsw_aF_um=cjsw_aF_um,
        provenance=prov,
    )


__all__ = [
    "PdkPexConstants",
    "resolve_pdk_pex_constants",
    "EPS0_aF_PER_UM",
]
