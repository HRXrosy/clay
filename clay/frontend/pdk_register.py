"""Auto-registration of a real PDK into Clay's layout + signoff flow.

A new commercial PDK dropped under ``pdks/<name>/`` ships two things Clay needs to
ingest it without hand-coding a preset:

  1. plaintext HSPICE BSIM device models in ``.lib <corner>`` sections (read by the
     ngspice harness via a ``.lib "<file>" <corner>`` line), and
  2. a foundry DRC deck (external_drc/SVRF or Mentor TVF) stating the geometric design
     rules (metal width/space, poly width/space, contact size/space).

This module turns those two artifacts into the two objects the rest of Clay binds
against:

  * :func:`discover_pdk_adapter` -> a marker-adapter descriptor (the same key shape
    ``circuits/_harness.py`` already uses) so the simulation harness can resolve the
    model include + corner for the new PDK.
  * :func:`pdk_rules_from_deck` -> a real :class:`clay.layout.pdk_rules.PdkRules`
    populated with the rules parsed from the deck (anything unparseable is left at the
    class default and reported, never fabricated).

  * :func:`register_pdk` bundles the device characterization (from
    :func:`clay.frontend.pdk_characterize.characterize_pdk`), the adapter descriptor,
    the PdkRules and the parse report into a :class:`PdkRegistration` -- the one call a
    ``clay pdk add`` CLI wires up.

Honesty contract
================
Every rule we emit into the PdkRules carries a provenance: a value read from the deck
is ``PARSED_DECK``; a value we could not parse stays at the consolidated default and is
listed in ``report["defaulted"]`` with the reason. Parsed numbers are sanity-bounded
(metal width/space 0.02-2.0 um, etc.); an out-of-range parse is rejected and reported
rather than trusted. ``report["parsed"]`` records the source line for each accepted rule.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from clay.frontend.pdk_characterize import PdkCharacterization, characterize_pdk
from clay.layout.pdk_rules import (
    LayerRule,
    PdkData,
    PdkRules,
    Provenance,
    RuleValue,
    ViaRule,
)

# --- adapter descriptor ------------------------------------------------------

# Model-library file extensions that can carry ``.lib <corner>`` sections. The
# harness binds the file that HAS the corner sections (a ``.lib`` wrapper), not the
# raw ``.mdl`` model-card file that the wrapper includes.
_LIB_EXT = (".lib", ".l", ".inc", ".mdl", ".scs")

# corner-section name tokens that mark the typical / nominal process corner.
_TT_TOKENS = ("tt", "typical", "ttg", "nom", "nominal")
# corner-section qualifiers that mark a NON-core section (RF / device-flavour /
# passive) we must skip when picking the core typical corner.
_NONCORE = (
    "rf", "hvt", "lvt", "mvt", "dio", "res", "bjt", "bip", "mim", "var", "esd",
    "cap", "jvar", "fmom", "mom", "tl", "na", "_18", "_25", "_33", "_5",
)

_LIB_SECTION_RE = re.compile(r"^\s*\.lib\s+'?\"?([A-Za-z0-9_]+)", re.IGNORECASE | re.MULTILINE)

_STD_VDD = (0.9, 1.0, 1.1, 1.2, 1.5, 1.8, 2.5, 3.0, 3.3, 5.0)


def _short_pdk_id(name: str) -> str:
    """Collapse a long install-dir name to a stable short id.

    ``external_pdk..._v1.11_2`` -> ``open_pdk``; ``open_pdk`` -> ``open_pdk``.
    Takes the leading ``<vendor><node>[suffix]`` token before the first ``_``-joined
    descriptor block.
    """
    n = name.lower()
    m = re.match(r"((?:external_pdk|external_pdk|gf|sky|ihp|x-?fab|umc)[a-z]*\d+[a-z]*(?:mse|mmrf|rf|lp|mse)?)", n)
    if m:
        return m.group(1)
    return n.split("_")[0]


def _corner_sections(model_text: str) -> List[str]:
    """All ``.lib <name>`` section names declared in a model library file."""
    return _LIB_SECTION_RE.findall(model_text)


def _pick_tt_corner(sections: List[str]) -> Optional[str]:
    """Pick the core typical/tt corner section from a list of section names.

    Prefers an exact ``tt`` / ``typical``; otherwise the first section whose name
    starts with a tt-token and carries no non-core qualifier (rf / hvt / dio / ...).
    """
    low = [(s, s.lower()) for s in sections]
    # exact core typical first
    for s, l in low:
        if l in ("tt", "typical"):
            return s
    # tt-prefixed, core only (no rf/hvt/dio/... qualifier)
    for s, l in low:
        if any(l.startswith(t) for t in _TT_TOKENS) and not any(q in l for q in _NONCORE):
            return s
    # any tt-prefixed as a last resort
    for s, l in low:
        if any(l.startswith(t) for t in _TT_TOKENS):
            return s
    return None


def _find_model_lib(pdk_dir: Path, char: PdkCharacterization) -> Tuple[Optional[Path], Optional[str]]:
    """Locate the device model *library* file (the one the harness ``.lib``s) and its
    tt corner.

    The characterizer records the file holding the BSIM cards
    (``char.provenance['nmos']['file']``). The harness, however, includes the file that
    declares the ``.lib <corner>`` sections -- often a thin ``.lib`` wrapper alongside
    the ``.mdl`` card file. We therefore:

      1. start from the characterized card file's directory,
      2. scan candidate library files in that dir for ``.lib <corner>`` sections,
      3. return the file with a core typical corner (prefer one whose name lacks a
         device-flavour suffix), and that corner name.

    Falls back to the characterized file itself if it already carries corner sections.
    """
    rel = char.provenance.get("nmos", {}).get("file") if isinstance(char.provenance, dict) else None
    card_file = (pdk_dir / rel) if rel else None
    search_dirs: List[Path] = []
    if card_file is not None and card_file.parent.is_dir():
        search_dirs.append(card_file.parent)

    best: Tuple[Optional[Path], Optional[str], int] = (None, None, -1)
    seen = set()
    for d in search_dirs:
        for f in sorted(d.iterdir()):
            if not (f.is_file() and f.suffix.lower() in _LIB_EXT):
                continue
            if f in seen:
                continue
            seen.add(f)
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            secs = _corner_sections(text)
            tt = _pick_tt_corner(secs)
            if tt is None:
                continue
            # score: prefer the characterized card file's stem, then a ``.lib`` ext,
            # then fewer flavour tokens in the file name.
            score = 0
            if card_file is not None and f.stem == card_file.stem:
                score += 4
            if f.suffix.lower() == ".lib":
                score += 3
            if not any(q in f.name.lower() for q in ("rf", "hv", "shrink", "embedded", "toplevel")):
                score += 1
            if score > best[2]:
                best = (f, tt, score)
    return best[0], best[1]


def discover_pdk_adapter(pdk_dir: str) -> Dict[str, object]:
    """Build the marker-adapter descriptor binding a real PDK to the sim harness.

    Returns a dict mirroring the keys ``circuits/_harness.py`` resolves per PDK::

        {
          "pdk_id":    <short stable id, e.g. "open_pdk" / "open_pdk">,
          "model_lib": <abs path to the device model .lib the harness includes>,
          "corner_tt": <the typical/tt .lib section name>,
          "marker":    <"* @CLAY_PDK pdk=<id> corner=tt@" generic-dispatcher marker>,
          "vdd":       <core supply, V>,
          "devices":   {"nmos": <model name>, "pmos": <model name>},
        }

    ``model_lib`` + the device names are derived from the characterizer's provenance;
    ``corner_tt`` is found by scanning the model file's ``.lib <corner>`` sections for
    the core typical section.
    """
    d = Path(pdk_dir)
    char = characterize_pdk(str(d))
    pdk_id = _short_pdk_id(char.pdk)

    model_lib, corner_tt = _find_model_lib(d, char)
    if model_lib is None:
        # fall back to the characterized card file (it may itself carry sections,
        # or be all the harness has); corner falls back to "tt".
        rel = char.provenance.get("nmos", {}).get("file")
        model_lib = (d / rel) if rel else None
        corner_tt = "tt"
    if corner_tt is None:
        corner_tt = "tt"

    vdd = float(char.params.vdd)
    if vdd not in _STD_VDD:
        # keep the characterized value but it is informational only here.
        pass

    devices = {
        "nmos": char.provenance.get("nmos", {}).get("model"),
        "pmos": char.provenance.get("pmos", {}).get("model"),
    }
    return {
        "pdk_id": pdk_id,
        "model_lib": str(model_lib) if model_lib is not None else None,
        "corner_tt": corner_tt,
        "marker": f"* @CLAY_PDK pdk={pdk_id} corner=tt@",
        "vdd": vdd,
        "devices": devices,
    }


# --- DRC deck parsing --------------------------------------------------------

# sanity bounds (um) for parsed length rules. A parse outside the band is rejected
# (a deck literal that large/small is almost certainly a mis-parse, not a real rule).
#
# Enclosure rules are a narrow overlap dimension (0..0.5 um per the honesty
# contract); a min-enclosure of 0 is a real, legitimate value in modern decks
# (e.g. open_pdk M1_EN_1 = 0), so the lower bound is 0 inclusive for enclosures.
# Area rules are bounded 0..5 um^2 (a min-area literal outside that band is a
# mis-parse, not a real core-layer rule).
_BOUND_UM = {
    "metal1_width": (0.02, 2.0),
    "metal1_space": (0.02, 2.0),
    "poly_width": (0.02, 2.0),
    "poly_space": (0.02, 2.0),
    "diff_width": (0.02, 2.0),
    "diff_space": (0.02, 2.0),
    "contact_size": (0.02, 1.0),
    "contact_space": (0.02, 2.0),
    # enclosures: a thin overlap; 0 is a valid minimum.
    "contact_enclosure_metal1": (0.0, 0.5),
    "contact_enclosure_diff": (0.0, 0.5),
    "contact_enclosure_poly": (0.0, 0.5),
    "via1_enclosure_metal1": (0.0, 0.5),
    "via1_enclosure_metal2": (0.0, 0.5),
    "poly_extension": (0.0, 0.5),
}

# sanity bounds (um^2) for parsed area rules.
_BOUND_UM2 = {
    "metal1_min_area": (0.0, 5.0),
    "poly_min_area": (0.0, 5.0),
}

# the set of rule keys whose value is an AREA (um^2), not a length (um).
_AREA_KEYS = frozenset(_BOUND_UM2)

# --- new rule families (density / antenna / notch / line-end / wide-metal) ---
#
# Honesty bounds per the contract:
#   density is a dimensionless area-coverage FRACTION (0..1); a value > 1 (e.g. a
#     deck that states "30" meaning 30%) is out-of-band and rejected, not /100'd.
#   antenna is a dimensionless gate-area RATIO (1..200).
#   notch / line-end are short lengths (0..0.5 um).
#   wide-metal space + its width threshold are lengths (0..5 um).
#
# These extend the existing width/space/enclosure/area coverage. Each remains
# PARTIAL (not full external_drc sign-off); a rule absent from a deck stays unparsed.

# length-valued new keys (um), bounded.
_BOUND_UM.update({
    "metal1_notch": (0.0, 0.5),
    "metal1_line_end": (0.0, 0.5),
    "metal1_wide_space": (0.0, 5.0),
    # The wide-metal SPACE check value is the small (0..5 um) honesty-banded
    # number. The WIDTH THRESHOLD it triggers above is a *wide* feature size:
    # real foundry "wide-metal" thresholds are commonly 4-10 um (open_pdk uses a
    # tight 0.19 um spacing-by-width step; open_pdk / open_pdk trigger at 10 um),
    # so the threshold carries a wider sane band than the spacing itself. A
    # threshold outside 0..20 um is a mis-parse and is rejected.
    "metal1_wide_width": (0.0, 20.0),
})

# fraction-valued keys (dimensionless, 0..1).
_BOUND_FRAC = {
    "metal1_density_min": (0.0, 1.0),
    "metal1_density_max": (0.0, 1.0),
}
_FRAC_KEYS = frozenset(_BOUND_FRAC)

# ratio-valued keys (dimensionless antenna gate-area ratio). Real foundry poly
# antenna ratios are 200-250 (open_pdk/65 area antenna = 250); the band is 1..400 so a
# genuine 250 is accepted while a mis-parse (e.g. a 350000 VIA1.R.13 limit, or a 0
# gate threshold) is rejected.
_BOUND_RATIO = {
    "metal1_antenna_ratio": (1.0, 400.0),
    "poly_antenna_ratio": (1.0, 400.0),
}
_RATIO_KEYS = frozenset(_BOUND_RATIO)

_NUM = r"\d*\.?\d+"


@dataclass
class _Hit:
    value_um: float           # for area keys: um^2; for frac/ratio: the raw scalar
    line: str
    is_area: bool = False
    # rule family for downstream bounding + unit handling. "length" (um) is the
    # default; "area" mirrors ``is_area`` (um^2); "fraction" is a 0..1 density
    # coverage; "ratio" is a dimensionless antenna ratio. Only the new families
    # set this explicitly; existing length/area hits keep their behaviour.
    kind: str = "length"


def _scan_calibre_variables(text: str) -> Dict[str, float]:
    """Parse ``VARIABLE  <NAME>  <value>`` lines (external_pdk external_drc SVRF) -> {name: um}."""
    out: Dict[str, float] = {}
    for m in re.finditer(rf"^\s*VARIABLE\s+([A-Za-z0-9_]+)\s+({_NUM})", text, re.MULTILINE):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            continue
    return out


# external_pdk external_drc rule-id -> (VARIABLE name, our rule key). The VARIABLE table is the
# authoritative source; we also keep the rule-id for provenance.
_TSMC_RULES = {
    "metal1_width": ("M1.W.1", "M1_W_1"),
    "metal1_space": ("M1.S.1", "M1_S_1"),
    "poly_width": ("PO.W.1", "PO_W_1"),
    "poly_space": ("PO.S.1", "PO_S_1"),
    "contact_size": ("CO.W.1", "CO_W_1"),
    "contact_space": ("CO.S.1", "CO_S_1"),
}

# Enclosure + min-area rule-id -> (VARIABLE name, our rule key) for the modern external_pdk
# external_drc VARIABLE-table dialect. Each variable is the single scalar the rule body
# references; the rule-id is the deck's own rule label, kept for provenance.
#   CO_EN_2 / M1_EN_2 / VIA1_EN_2 / M2_EN_2 are the "two-opposite-sides" minimum
#   enclosures (the genuine min); CO_EN_1 is contact enclosure by OD (diff);
#   CO_EN_3 is contact enclosure by POLY; PO_EX_1 is the gate poly endcap over OD;
#   M1_A_1 / PO_A_1 are min-area variables (um^2).
_TSMC_ENCLOSURE_RULES = {
    "contact_enclosure_metal1": ("M1.EN.2", "M1_EN_2"),  # M1 over contact
    "contact_enclosure_diff": ("CO.EN.1", "CO_EN_1"),    # contact by OD/diff
    "contact_enclosure_poly": ("CO.EN.3", "CO_EN_3"),    # contact by poly
    "via1_enclosure_metal1": ("VIA1.EN.2", "VIA1_EN_2"),  # via1 by M1 (below)
    "via1_enclosure_metal2": ("M2.EN.2", "M2_EN_2"),     # via1 by M2 (above)
    "poly_extension": ("PO.EX.1", "PO_EX_1"),            # gate poly endcap on OD
}
_TSMC_AREA_RULES = {
    "metal1_min_area": ("M1.A.1", "M1_A_1"),
    "poly_min_area": ("PO.A.1", "PO_A_1"),
}

# Density (local area-coverage fraction) VARIABLEs. The rule bodies read e.g.
#   M1.DN.1   : ``DENSITY M1_CHECK CHIP_CHECK < M1_DN_1 ...``   (minimum coverage)
#   M1.DN.1.1 : ``DENSITY M1_CHECK CHIP_CHECK > M1_DN_1_1 ...`` (maximum coverage)
# so M1_DN_1 is the floor fraction (0.1) and M1_DN_1_1 the ceiling fraction (0.85).
_TSMC_DENSITY_RULES = {  # rule_key -> (rule_id, VARIABLE name)
    "metal1_density_min": ("M1.DN.1", "M1_DN_1"),
    "metal1_density_max": ("M1.DN.1.1", "M1_DN_1_1"),
}
# Wide-metal spacing-by-width: ``M1.S.2`` flags wide M1 closer than ``M1_S_2`` when
# the metal width exceeds ``M1_S_2_W`` (the threshold). Both are VARIABLEs.
_TSMC_WIDE_SPACE = ("M1.S.2", "M1_S_2", "M1_S_2_W")  # (rule_id, space var, width var)
# external_pdk external_drc online DRC decks state antenna rules in a SEPARATE deck (no antenna
# VARIABLE here); notch/line-end have no clean M1 scalar VARIABLE. Left unparsed.


def _parse_tsmc_calibre(text: str) -> Dict[str, _Hit]:
    """Parse a external_pdk ``external_drc.drc`` deck. Width/space/size come from the ``VARIABLE``
    table (e.g. ``VARIABLE M1_W_1 0.07``), which the rule bodies reference by name.

    Enclosure (CO/M1/VIA1/M2 EN, PO endcap) and min-area (M1/PO A) rules come from
    the same VARIABLE table -- each is the lone scalar the rule body references."""
    variables = _scan_calibre_variables(text)
    out: Dict[str, _Hit] = {}
    for key, (rule_id, var_name) in _TSMC_RULES.items():
        v = variables.get(var_name)
        if v is None:
            continue
        out[key] = _Hit(value_um=v, line=f"VARIABLE {var_name} {v}  ({rule_id})")
    for key, (rule_id, var_name) in _TSMC_ENCLOSURE_RULES.items():
        v = variables.get(var_name)
        if v is None:
            continue
        out[key] = _Hit(value_um=v, line=f"VARIABLE {var_name} {v}  ({rule_id})")
    for key, (rule_id, var_name) in _TSMC_AREA_RULES.items():
        v = variables.get(var_name)
        if v is None:
            continue
        out[key] = _Hit(value_um=v, line=f"VARIABLE {var_name} {v}  ({rule_id})", is_area=True)
    # density (fraction) rules from the M1_DN VARIABLEs.
    for key, (rule_id, var_name) in _TSMC_DENSITY_RULES.items():
        v = variables.get(var_name)
        if v is None:
            continue
        out[key] = _Hit(
            value_um=v, line=f"VARIABLE {var_name} {v}  ({rule_id})", kind="fraction"
        )
    # wide-metal space + its width threshold (M1.S.2: M1_S_2 above width M1_S_2_W).
    rule_id, space_var, width_var = _TSMC_WIDE_SPACE
    sv = variables.get(space_var)
    wv = variables.get(width_var)
    if sv is not None and wv is not None:
        out["metal1_wide_space"] = _Hit(
            value_um=sv, line=f"VARIABLE {space_var} {sv}  ({rule_id}, W>{width_var})"
        )
        out["metal1_wide_width"] = _Hit(
            value_um=wv, line=f"VARIABLE {width_var} {wv}  ({rule_id} width threshold)"
        )
    return out


# external_pdk TVF/SVRF rule blocks state the literal inline:
#   M1_1 { @ M1 width is >= 0.23um\n  INT M1_all < 0.23 ... }   -> metal1 width
#   M1_2 { @ ... >= 0.23um\n  EXT M1_all < 0.23 ... }           -> metal1 space
#   GT_1 { @ GT width ... >= 0.18um\n INT GT_all < 0.18 ... }   -> poly width
#   GT_3 { @ Space between two GT ... >= 0.25um\n EXT (...) < 0.25 } -> poly space
#   CT_2 { @ Space between two contacts ... >= 0.25um\n EXT CT < 0.25 } -> contact space
#   CT_1 { ... RECTANGLE ... == 0.22 BY == 0.22 }               -> contact size
_INT_RE = re.compile(rf"\bINT\b[^\n<]*<\s*({_NUM})", re.IGNORECASE)
_EXT_RE = re.compile(rf"\bEXT\b[^\n<]*<\s*({_NUM})", re.IGNORECASE)
# enclosure idiom shared by external_pdk TVF + older-external_pdk external_drc: ``ENC <inner> <outer> < V``
# (the value is the minimum overlap of <inner> by <outer>). We do NOT match the
# bracketed ``ENC [X] ...`` recompute form (a derived intermediate, not the rule
# minimum). ``AREA <layer> < V`` / ``<layer> AREA < V`` is the min-area idiom.
_ENC_RE = re.compile(rf"\bENC\s+[A-Za-z0-9_()]+\s+[A-Za-z0-9_()]+\s*<\s*({_NUM})", re.IGNORECASE)
# AREA is the SVRF operator (always uppercase in a rule body); the comment prose
# "Min M1 area" is lowercase, so a case-SENSITIVE match avoids picking the comment.
_AREA_RE = re.compile(rf"\bAREA\b[^\n<]*<\s*({_NUM})")
# DENSITY <layer> < V  -> minimum coverage fraction; DENSITY <layer> > V -> maximum.
# (case-sensitive: DENSITY is the SVRF operator; the prose "density" is lowercase.)
_DENSITY_MIN_RE = re.compile(rf"\bDENSITY\b[^\n<>]*<\s*({_NUM})")
_DENSITY_MAX_RE = re.compile(rf"\bDENSITY\b[^\n<>]*>\s*({_NUM})")
# wide-metal: a width-selected layer (``WITH WIDTH > T``) spaced (``EXT ... < S``).
_WIDTH_THRESH_RE = re.compile(rf"\bWITH\s+WIDTH\s*>\s*({_NUM})", re.IGNORECASE)
# line-end (end-of-line) metal extension past a via/contact: the bracketed
# recompute form ``ENC [CO] M1 < 0.06`` / ``ENC [VIA1] M1 < 0.06`` / ``ENC [x] M1
# < 0.06``. This is the EOL-specific minimum -- the metal must overhang the cut by
# >= V on the narrow (line-end) side. The bracket ``[..]`` distinguishes it from the
# plain min-enclosure ``ENC CO M1 < V`` (matched by _ENC_RE), which is the all-sides
# minimum, not the line-end one.
_ENC_BRACKET_RE = re.compile(
    rf"\bENC\s+\[[A-Za-z0-9_]+\]\s+[A-Za-z0-9_]+\s*<\s*({_NUM})", re.IGNORECASE
)
# antenna: connectivity NET-AREA-RATIO ``NET AREA RATIO <metal> GATE > <ratio>`` --
# the clean 2-operand AREA ratio (metal area / gate area). KLayout's antenna_check
# computes exactly this AREA ratio, so only the AREA-metric poly antenna rule (the
# "top area / gate area" form) is emittable; the perimeter-metric variant
# ("perim*thickness / gate area") is a different physical ratio KLayout does not
# compute, and is left honestly unparsed.
_NET_AREA_RATIO_RE = re.compile(
    rf"\bNET\s+AREA\s+RATIO\s+([A-Za-z0-9_]+)\s+GATE\s*>\s*({_NUM})", re.IGNORECASE
)


def _is_comment(line: str) -> bool:
    s = line.lstrip()
    return s.startswith("@") or s.startswith("//")


def _smic_block(text: str, rule_id: str) -> Optional[str]:
    """Extract the body of a ``<rule_id> { ... }`` block (brace-matched, first level)."""
    m = re.search(rf"(?m)^\s*{re.escape(rule_id)}\s*\{{", text)
    if not m:
        return None
    i = m.end() - 1  # at the opening brace
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def _smic_block_any(text: str, rule_id: str) -> Optional[str]:
    """Like :func:`_smic_block`, but also matches a block introduced by a leading
    ``RULECHECK`` keyword (``RULECHECK M1_5 { ... }``), which external_pdk uses for its
    conditional density / wide-metal checks. Falls back to the bare form."""
    body = _smic_block(text, rule_id)
    if body is not None:
        return body
    m = re.search(rf"(?m)^\s*RULECHECK\s+{re.escape(rule_id)}\s*\{{", text)
    if not m:
        return None
    i = text.index("{", m.start())
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def _first_line(block: str) -> str:
    for ln in block.splitlines():
        s = ln.strip()
        if s:
            return s
    return block.strip()[:80]


def _parse_smic_tvf(text: str) -> Dict[str, _Hit]:
    """Parse a external_pdk external_drc TVF/SVRF ``.drc`` deck (literal numbers embedded in the
    rule bodies as ``INT layer < W`` / ``EXT layer < S``)."""
    out: Dict[str, _Hit] = {}

    def _active_aa(rule_id: str, operator: str, key: str) -> None:
        # MSE AA_1_2/AA_3 retain commented legacy AA commands immediately above
        # the live AA_all commands. Bind only the exact live scalar constraint;
        # a comment, another operand, or an ambiguous rule is not authority.
        active = re.sub(
            r"/\*.*?(?:\*/|\Z)|//[^\n]*|(?m:^[ \t]*@[^\n]*)",
            lambda match: "\n" * match.group(0).count("\n"),
            text, flags=re.DOTALL,
        )
        # MMRF's installed DR8P7P deck has AA, derived from physical AAi
        # after exclusion/OPC masks. Do not treat an arbitrary old AA command
        # as authority: require the complete live foundry derivation and no
        # competing MSE AA_all layer. Comments cannot satisfy this proof.
        mmrf_aa = not re.search(r"\bAA_all\b", active) and all(
            len(re.findall(pattern, active, re.MULTILINE)) == 1
            for pattern in (
                r"^[ \t]*LAYER[ \t]+AAi[ \t]+10[ \t]+12[ \t]+13[ \t]*$",
                r"^[ \t]*AAO[ \t]*=[ \t]*AAi[ \t]+NOT[ \t]+EXCLU[ \t]*$",
                r"^[ \t]*AA[ \t]*=[ \t]*NOT[ \t]+AAO[ \t]+OPCBA[ \t]*$",
            )
        )
        operand = "AA" if mmrf_aa else "AA_all"
        if len(re.findall(rf"(?m)^\s*{re.escape(rule_id)}\s*\{{", active)) != 1:
            return
        body = _smic_block(active, rule_id)
        if not body:
            return
        commands = []
        for line in body.splitlines():
            command = line.strip()
            match = re.fullmatch(
                rf"{operator}\s+{operand}\s*<\s*({_NUM})\s+ABUT\s*<\s*90\s+SINGULAR\s+REGION",
                command, re.IGNORECASE,
            )
            if match:
                commands.append((match.group(1), command))
        if len(commands) == 1:
            literal, command = commands[0]
            out[key] = _Hit(value_um=float(literal), line=f"{rule_id}: {command}")

    def _width(rule_id: str, key: str) -> None:
        body = _smic_block(text, rule_id)
        if not body:
            return
        m = _INT_RE.search(body)
        if m:
            out[key] = _Hit(value_um=float(m.group(1)), line=f"{rule_id}: {_first_line(body)}")

    def _space(rule_id: str, key: str) -> None:
        body = _smic_block(text, rule_id)
        if not body:
            return
        m = _EXT_RE.search(body)
        if m:
            out[key] = _Hit(value_um=float(m.group(1)), line=f"{rule_id}: {_first_line(body)}")

    def _enc(rule_id: str, key: str, outer: Optional[str] = None) -> None:
        """Parse the min enclosure ``ENC <inner> <outer> < V`` from a block. When
        ``outer`` is given, pick the ENC line whose OUTER (second) operand is that
        layer (a block may state several, e.g. CT_6_7 has both GT and AA)."""
        body = _smic_block(text, rule_id)
        if not body:
            return
        for ln in body.splitlines():
            m = _ENC_RE.search(ln)
            if not m:
                continue
            if outer is not None and not re.search(rf"\bENC\s+\S+\s+{re.escape(outer)}\b", ln, re.IGNORECASE):
                continue
            out[key] = _Hit(value_um=float(m.group(1)), line=f"{rule_id}: {ln.strip()}")
            return

    def _area(rule_id: str, key: str) -> None:
        body = _smic_block(text, rule_id)
        if not body:
            return
        for ln in body.splitlines():
            m = _AREA_RE.search(ln)
            if m:
                out[key] = _Hit(
                    value_um=float(m.group(1)), line=f"{rule_id}: {ln.strip()}", is_area=True
                )
                return

    _active_aa("AA_1_2", "INT", "diff_width")
    _active_aa("AA_3", "EXT", "diff_space")
    _width("M1_1", "metal1_width")
    _space("M1_2", "metal1_space")
    _width("GT_1", "poly_width")
    _space("GT_3", "poly_space")
    _space("CT_2", "contact_space")
    # contact size: "NOT RECTANGLE (...) == 0.22 BY == 0.22"
    ct1 = _smic_block(text, "CT_1")
    if ct1:
        m = re.search(rf"==\s*({_NUM})\s+BY\s+==\s*({_NUM})", ct1, re.IGNORECASE)
        if m:
            out["contact_size"] = _Hit(value_um=float(m.group(1)), line=f"CT_1: {_first_line(ct1)}")
    # enclosures: M1 over CT (M1_3_4 "ENC X M1 < 0.005"); CT over AA/GT (CT_6_7
    # states both -- AA is the diff overlap, GT the poly overlap); poly endcap on
    # AA (GT_5 "ENC AA_GATE_EDGE GT < 0.22").
    _enc("M1_3_4", "contact_enclosure_metal1", outer="M1")
    _enc("CT_6_7", "contact_enclosure_diff", outer="AA")
    _enc("CT_6_7", "contact_enclosure_poly", outer="GT")
    _enc("GT_5", "poly_extension", outer="GT")
    # M1 line-end (end-of-line) extension past CT: M1_3_4 also states the EOL min as
    # the bracketed recompute ``Y = ENC [x] M1 < 0.06`` (the deck comment: "Enclosure
    # of M1 line end region beyond CT region is >= 0.06um"). The bracketed form is the
    # line-end-specific minimum (distinct from the all-sides 0.005 enclosure above).
    m134 = _smic_block(text, "M1_3_4")
    if m134:
        mle = _ENC_BRACKET_RE.search(m134)
        if mle:
            out["metal1_line_end"] = _Hit(
                value_um=float(mle.group(1)),
                line=f"M1_3_4: {mle.group(0).strip()} (M1 EOL ext past CT)",
            )
    # min area: M1 (M1_6 "AREA M1 < 0.20"). external_pdk's deck states no scalar poly
    # min-area block, so poly_min_area is left unparsed (honest gap).
    _area("M1_6", "metal1_min_area")

    # density: M1_7 ("DENSITY M1i_all < 0.3", minimum coverage) + M1_7a_R
    # ("DENSITY M1i_all > 0.75", maximum coverage across full chip).
    m7 = _smic_block_any(text, "M1_7")
    if m7:
        m = _DENSITY_MIN_RE.search(m7)
        if m:
            out["metal1_density_min"] = _Hit(
                value_um=float(m.group(1)),
                line=f"M1_7: {_first_line(m7)}",
                kind="fraction",
            )
    m7a = _smic_block_any(text, "M1_7a_R")
    if m7a:
        m = _DENSITY_MAX_RE.search(m7a)
        if m:
            out["metal1_density_max"] = _Hit(
                value_um=float(m.group(1)),
                line=f"M1_7a_R: {_first_line(m7a)}",
                kind="fraction",
            )

    # wide-metal: M1_5 selects wide M1 (``M1 WITH WIDTH > 10``) and flags space
    # (``EXT M1 Y < 0.6``). The space is the wide-spacing; the WITH-WIDTH operand
    # is the width threshold it applies above.
    m5 = _smic_block_any(text, "M1_5")
    if m5:
        wt = _WIDTH_THRESH_RE.search(m5)
        sp = _EXT_RE.search(m5)
        if wt and sp:
            out["metal1_wide_space"] = _Hit(
                value_um=float(sp.group(1)),
                line=f"M1_5: EXT M1 < {sp.group(1)} (wide, W>{wt.group(1)})",
            )
            out["metal1_wide_width"] = _Hit(
                value_um=float(wt.group(1)),
                line=f"M1_5: M1 WITH WIDTH > {wt.group(1)} (wide-metal threshold)",
            )
    # external_pdk's M1 line-end is parsed above (M1_3_4 bracketed EOL). The M1 notch has no
    # standalone scalar block in this deck (unparsed). external_pdk's poly antenna is a
    # PERIMETER ratio (``perim*thickness/gate area``, ANT_GT_1), which KLayout's
    # AREA-ratio antenna_check does not compute -- it lives in the SEPARATE .ant deck
    # and is read (when present) by _parse_antenna_deck, which only accepts an AREA
    # ratio; the perimeter form is reported honestly absent there.
    return out


# Older external_pdk external_drc decks (e.g. the 0.18um CL18S3_5M runset shipped in open_pdk)
# do NOT carry a VARIABLE table. Instead each rule is a named brace block whose body
# states the literal inline, exactly like the external_pdk TVF idiom but with dotted external_pdk
# rule-ids:
#   M1.W.1 { @ M1 width < 0.23\n  INT M1 < 0.23 SINGULAR REGION ABUT < 90 }
#   M1.S.1 { @ M1 spacing < 0.23\n  EXT M1 < 0.23 ABUT < 90 SINGULAR REGION }
#   PO.W.3 { @ Min. POLY width for interconnect ... < 0.18\n  INT POLY < 0.18 ... }
#   PO.S.2_PO.S.3 { @ Min. POLY space ... < 0.25\n  EXT POLY < 0.25 ... }
#   CO.W.1 { @ contact width != 0.22\n  NOT RECTANGLE CO == 0.22 BY == 0.22 ... }
#   CO.S.1 { @ contact spacing < 0.25\n  EXT CO < 0.25 ABUT < 90 SINGULAR REGION }
# We reuse the brace-block extraction + INT/EXT/RECTANGLE matchers; the rule keys map
# to these specific block ids (the general interconnect width/space rule, NOT a
# device-gate-length or wide-metal variant).
# Interconnect width: M1.W.1 .. M5.W.1 (``INT Mn < W``) -- the full 6-metal stack the
# reference routes (M1=0.23, M2/M3/M4=0.28, M5=0.44; the top metal MD has no clean
# INT-width block, so it is left for the dedicated thick-metal path). A router that
# routes an upper metal at the node-scaled M1 width would trip Mn.W.1 -- so each metal
# the router can use MUST carry its own parsed width/space.
_TSMC_OLD_INT = {  # rule_key -> (block_id) parsed via INT layer < W
    "diff_width": "OD.W.1_OD.W.2",
    "metal1_width": "M1.W.1",
    "metal2_width": "M2.W.1",
    "metal3_width": "M3.W.1",
    "metal4_width": "M4.W.1",
    "metal5_width": "M5.W.1",
    "poly_width": "PO.W.3",
}
_TSMC_OLD_EXT = {  # rule_key -> (block_id) parsed via EXT layer < S
    "diff_space": "OD.S.1",
    "metal1_space": "M1.S.1",
    "metal2_space": "M2.S.1",
    "metal3_space": "M3.S.1",
    "metal4_space": "M4.S.1",
    "metal5_space": "M5.S.1",
    "poly_space": "PO.S.2_PO.S.3",
    "contact_space": "CO.S.1",
    "via1_space": "VIA1.S.1",   # EXT VIA1 < 0.26 ABUT < 90 SINGULAR REGION
    "via2_space": "VIA2.S.1",
    "via3_space": "VIA3.S.1",
    "via4_space": "VIA4.S.1",   # EXT VIA4R < 0.35 (top routable via, larger cut)
}
# Square-cut size blocks: ``<id> { ... NOT RECTANGLE <cut> == X BY == Y ... }`` states
# the EXACT mandated cut side inline. CO.W.1 is the contact (0.22); VIA1-3.W.1 the
# 0.26x0.26 cuts; VIA4.W.1 the larger 0.36x0.36 cut between M4 and M5. The router must
# draw each cut at exactly its mandated side or every via trips its VIAn.W.1. Parsed
# identically (the ``== X BY == Y`` matcher), keyed per cut layer.
_TSMC_OLD_CUT_SIZE = {  # rule_key -> block_id (NOT RECTANGLE <cut> == X BY == Y)
    "contact_size": "CO.W.1",   # NOT RECTANGLE CO == 0.22 BY == 0.22
    "via1_size": "VIA1.W.1",    # NOT RECTANGLE VIA1 == 0.26 BY == 0.26
    "via2_size": "VIA2.W.1",
    "via3_size": "VIA3.W.1",
    "via4_size": "VIA4.W.1",    # NOT RECTANGLE VIA4R == 0.36 BY == 0.36
}

# Older-external_pdk enclosure blocks: each ``<rule_id> { ... ENC <inner> <outer> < V ... }``
# states the min overlap inline. We key off the OUTER operand to disambiguate when a
# block has more than one ENC line.
#   M1.E.1   : ENC CO M1   < 0.005   -> M1 over contact
#   CO.E.1   : ENC CO_DIFF OD < 0.10 -> contact by diff (OD)
#   CO.E.2   : ENC CO_POLY POLY_ISO < 0.10 -> contact by poly
#   VIA1.E.1 : ENC VIA1 M1 < 0.01    -> via1 by M1
#   M2.E.1   : ENC VIA1 M2 < 0.01    -> via1 by M2
#   PO.O.1   : ENC OD POLY < 0.22    -> poly endcap over active (gate extension)
_TSMC_OLD_ENC = {  # rule_key -> (block_id, outer_layer_token)
    "contact_enclosure_metal1": ("M1.E.1", "M1"),
    "contact_enclosure_diff": ("CO.E.1", "OD"),
    "contact_enclosure_poly": ("CO.E.2", "POLY_ISO"),
    "via1_enclosure_metal1": ("VIA1.E.1", "M1"),
    "via1_enclosure_metal2": ("M2.E.1", "M2"),
    "poly_extension": ("PO.O.1", "POLY"),
}
# Older-external_pdk min-area blocks: ``<rule_id> { ... AREA <layer> < V ... }``.
_TSMC_OLD_AREA = {  # rule_key -> block_id
    "metal1_min_area": "M1.A.1",
}
# Older-external_pdk density block: ``M1.R.1 { ... DENSITY M1xd < 0.3 ... }`` (minimum
# area coverage). This runset states no M1 maximum-density nor a clean M1 notch
# scalar. Antenna rules live in a SEPARATE antenna deck (``external_drc/ant.drc``), read
# by _parse_antenna_deck when present.
_TSMC_OLD_DENSITY_MIN_ID = "M1.R.1"
# Older-external_pdk M1 line-end (end-of-line): ``M1.E.2 { @ Min extension of M1 end-of-line
# region beyond CO region is 0.06 um\n  X = ENC [CO] M1 < 0.06 ABUT < 90 OPPOSITE }``.
# The bracketed recompute ``ENC [CO] M1 < V`` is the line-end-specific minimum (the
# all-sides M1-over-CO min is the separate M1.E.1 0.005). Parse the EOL scalar here.
_TSMC_OLD_LINE_END_ID = "M1.E.2"
# Older-external_pdk wide-metal: ``M1.S.2 { @ Wide M1 (>10um) min. to M1 < 0.6 um ...
# EXT M1_WideC M1_ElseC < 0.6 ... }``. The space is the EXT scalar; the width
# threshold (10um) is stated in the rule's own comment prose, not as a clean
# operator (the body derives wide M1 via SHRINK/GROW BY 5), so we read the
# threshold from the block's ``>NNum`` comment when present.
_TSMC_OLD_WIDE_SPACE_ID = "M1.S.2"
_COMMENT_WIDTH_RE = re.compile(rf">\s*({_NUM})\s*um", re.IGNORECASE)


def _constraint_line(block: str, rule_id: str) -> str:
    """Provenance label: the first INT/EXT/RECTANGLE constraint line of a block, or
    the block's leading comment if no constraint line is found."""
    for ln in block.splitlines():
        s = ln.strip()
        if re.match(r"(?i)\b(INT|EXT|NOT\s+RECTANGLE|RECTANGLE)\b", s):
            return f"{rule_id}: {s}"
    return f"{rule_id}: {_first_line(block)}"


def _parse_tsmc_old_calibre(text: str) -> Dict[str, _Hit]:
    """Parse an older external_pdk external_drc deck (no VARIABLE table) where each named rule
    block embeds its literal inline (``INT M1 < 0.23`` / ``EXT CO < 0.25`` /
    ``NOT RECTANGLE CO == 0.22 BY == 0.22``)."""
    out: Dict[str, _Hit] = {}
    for key, rule_id in _TSMC_OLD_INT.items():
        body = _smic_block(text, rule_id)
        if not body:
            continue
        m = (
            re.search(rf"\bINT\s+OD\s*<\s*({_NUM})", body, re.IGNORECASE)
            if key == "diff_width"
            else _INT_RE.search(body)
        )
        if m:
            out[key] = _Hit(value_um=float(m.group(1)), line=_constraint_line(body, rule_id))
    for key, rule_id in _TSMC_OLD_EXT.items():
        body = _smic_block(text, rule_id)
        if not body:
            continue
        m = (
            re.search(rf"\bEXT\s+OD\s*<\s*({_NUM})", body, re.IGNORECASE)
            if key == "diff_space"
            else _EXT_RE.search(body)
        )
        if m:
            out[key] = _Hit(value_um=float(m.group(1)), line=_constraint_line(body, rule_id))
    for key, rule_id in _TSMC_OLD_CUT_SIZE.items():
        body = _smic_block(text, rule_id)
        if not body:
            continue
        m = re.search(rf"==\s*({_NUM})\s+BY\s+==\s*({_NUM})", body, re.IGNORECASE)
        if m:
            out[key] = _Hit(
                value_um=float(m.group(1)),
                line=_constraint_line(body, rule_id),
            )
    for key, (rule_id, outer) in _TSMC_OLD_ENC.items():
        body = _smic_block(text, rule_id)
        if not body:
            continue
        for ln in body.splitlines():
            m = _ENC_RE.search(ln)
            if not m:
                continue
            if not re.search(rf"\bENC\s+\S+\s+{re.escape(outer)}\b", ln, re.IGNORECASE):
                continue
            out[key] = _Hit(value_um=float(m.group(1)), line=f"{rule_id}: {ln.strip()}")
            break
    for key, rule_id in _TSMC_OLD_AREA.items():
        body = _smic_block(text, rule_id)
        if not body:
            continue
        for ln in body.splitlines():
            m = _AREA_RE.search(ln)
            if m:
                out[key] = _Hit(
                    value_um=float(m.group(1)),
                    line=f"{rule_id}: {ln.strip()}",
                    is_area=True,
                )
                break
    # density minimum: M1.R.1 ``DENSITY M1xd < 0.3``.
    dbody = _smic_block(text, _TSMC_OLD_DENSITY_MIN_ID)
    if dbody:
        for ln in dbody.splitlines():
            m = _DENSITY_MIN_RE.search(ln)
            if m:
                out["metal1_density_min"] = _Hit(
                    value_um=float(m.group(1)),
                    line=f"{_TSMC_OLD_DENSITY_MIN_ID}: {ln.strip()}",
                    kind="fraction",
                )
                break
    # wide-metal: M1.S.2 ``EXT M1_WideC M1_ElseC < 0.6``; the width threshold is
    # in the block's comment prose (``Wide M1 (>10um)``).
    wbody = _smic_block(text, _TSMC_OLD_WIDE_SPACE_ID)
    if wbody:
        sp = None
        for ln in wbody.splitlines():
            m = _EXT_RE.search(ln)
            if m:
                sp = (float(m.group(1)), ln.strip())
                break
        if sp is not None:
            out["metal1_wide_space"] = _Hit(
                value_um=sp[0],
                line=f"{_TSMC_OLD_WIDE_SPACE_ID}: {sp[1]}",
            )
            cm = _COMMENT_WIDTH_RE.search(wbody)
            if cm:
                out["metal1_wide_width"] = _Hit(
                    value_um=float(cm.group(1)),
                    line=f"{_TSMC_OLD_WIDE_SPACE_ID}: width threshold "
                    f">{cm.group(1)}um (comment)",
                )
    # M1 line-end: M1.E.2 bracketed EOL extension past CO (``ENC [CO] M1 < 0.06``).
    lebody = _smic_block(text, _TSMC_OLD_LINE_END_ID)
    if lebody:
        mle = _ENC_BRACKET_RE.search(lebody)
        if mle:
            out["metal1_line_end"] = _Hit(
                value_um=float(mle.group(1)),
                line=f"{_TSMC_OLD_LINE_END_ID}: {mle.group(0).strip()} "
                f"(M1 EOL ext past CO)",
            )
    return out


# --- antenna deck (SEPARATE file) parsing ------------------------------------
#
# Commercial PDKs ship antenna rules in a SEPARATE deck alongside the main DRC
# deck (external_pdk ``*.ANT.21a`` / ``ant.drc`` / an ``ANTENNA_DRC`` subdir; external_pdk
# ``*.ant``). The clean, KLayout-expressible rule is the 2-operand poly AREA
# antenna ratio:
#   A.R.1:POLY { @ (Poly top area / gate area) <= 250\n  NET AREA RATIO POLY GATE > 250 }
# KLayout's ``antenna_check(gate, metal, ratio)`` computes precisely this AREA
# ratio (metal_area / gate_area), so we accept ONLY the AREA-metric poly antenna
# (comment names "...top area / gate area" or "...area / gate area"). The
# PERIMETER-metric variant (A.R.2: "Poly perim*thickness / gate area", the only
# poly antenna external_pdk/open_pdk state) is a different physical ratio KLayout does not
# compute, so it is reported honestly absent -- never coerced into an area ratio.
#
# The metal1 antenna LIMIT in every deck we ship is a conditional perimeter/diode
# EQUATION buried in a TVF ``[ ... ]`` recompute block (e.g. open_pdk A.R.6__A.R.8:M1
# "(M1 area/gate area) <= (1000 in OD_18, 5000 not in OD_18)"), with the
# ``NET AREA RATIO M1 ... GATE > 0`` line only a ``> 0`` gate. There is no clean
# static M1-area-to-gate scalar, so metal1_antenna_ratio stays honestly absent.

def _is_antenna_file(p: Path) -> bool:
    """True if ``p`` looks like a gate-antenna deck (not a MIM-cap antenna deck).

    Matches the real foundry names: external_pdk ``*.ANT.21a`` / ``*.ANT_002.14a`` /
    ``ant.drc``; external_pdk ``*.ant``. Skips ``*MIM_ANT*`` (MIM-cap antenna, a different
    rule family) and the external_lvs runset / report side files (we only parse
    external_drc/TVF deck text).
    """
    if not p.is_file():
        return False
    name = p.name.lower()
    if "mim_ant" in name or "ant_mim" in name:
        return False
    if name.endswith((".rsf", ".rul", ".db", ".rep", ".log", ".gds")):
        return False
    return (
        name == "ant.drc"
        or name.endswith(".ant")
        or ".ant." in name        # *.ANT.21a
        or ".ant_" in name        # *.ANT_002.14a
        or "antenna" in name      # ANTENNA_DRC/*
    )


def find_antenna_deck(pdk_dir: str) -> Optional[Path]:
    """Locate the SEPARATE antenna deck in a real PDK, preferring one beside the
    main DRC deck (so the metal stack matches), else anywhere under the PDK.

    Returns the path, or None (antenna rules then stay honestly unparsed). MIM
    antenna decks (``*MIM_ANT*``) are skipped -- they check the MIM cap, not the
    poly/metal gate antenna we emit.
    """
    d = Path(pdk_dir)
    if not d.is_dir():
        return None
    main = find_drc_deck(pdk_dir)
    search_roots: List[Path] = []
    if main is not None:
        search_roots.append(main.parent)
    search_roots.append(d)

    seen = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        # Files directly beside the main deck are authoritative for that stack.
        # Return before starting a recursive install-root walk: commercial OA
        # trees can contain millions of unrelated files.
        for c in sorted(root.iterdir()):
            try:
                rc = c.resolve()
            except OSError:
                continue
            if rc in seen:
                continue
            seen.add(rc)
            if _is_antenna_file(c):
                return c
        # Some installs keep antenna decks in a nested ``ANTENNA_DRC`` folder.
        for c in sorted(root.glob("**/*")):
            try:
                rc = c.resolve()
            except OSError:
                continue
            if rc in seen:
                continue
            seen.add(rc)
            if _is_antenna_file(c):
                return c
    return None


def parse_antenna_deck(text: str) -> Dict[str, _Hit]:
    """Parse the poly AREA antenna ratio from a SEPARATE antenna deck's text.

    Accepts ONLY an AREA-metric poly-to-gate ratio (KLayout antenna_check computes
    an area ratio): the rule's own comment must name an *area* ratio ("top area /
    gate area" or "area / gate area"), not a perimeter one ("perim*thickness").
    Returns ``{"poly_antenna_ratio": _Hit(kind="ratio")}`` or ``{}``.
    """
    out: Dict[str, _Hit] = {}
    # scan each NET AREA RATIO <layer> GATE > <ratio>; keep the first AREA-metric
    # poly one (layer token starts with PO/POLY/GT -- the poly conductor).
    for m in _NET_AREA_RATIO_RE.finditer(text):
        layer = m.group(1)
        ratio = float(m.group(2))
        if not re.match(r"(?i)^(po|poly|gt)", layer):
            continue
        # find the enclosing rule block's leading comment to confirm AREA metric.
        start = text.rfind("{", 0, m.start())
        ctx = text[max(0, start - 200): m.start()]
        is_area = bool(re.search(r"(?i)top\s*area\s*/\s*gate\s*area", ctx)) or bool(
            re.search(r"(?i)(?<!perim[\s*])\barea\s*/\s*gate\s*area", ctx)
        )
        is_perim = bool(re.search(r"(?i)perim", ctx))
        if is_perim and not re.search(r"(?i)top\s*area\s*/\s*gate\s*area", ctx):
            # perimeter-metric ratio -> not an AREA ratio KLayout computes.
            continue
        if not is_area:
            continue
        ratio_line = m.group(0).strip()
        out["poly_antenna_ratio"] = _Hit(
            value_um=ratio,
            line=f"{ratio_line}  (poly area antenna, separate deck)",
            kind="ratio",
        )
        break
    return out


def parsed_rules_from_deck_bytes(
    deck_bytes: bytes,
    *,
    antenna_bytes: Optional[bytes] = None,
) -> Dict[str, object]:
    """Parse authoritative rule facts from already captured source bytes.

    This is the byte-oriented counterpart of :func:`pdk_rules_from_deck`.  A
    caller that has atomically selected and read a commercial PDK source can
    parse exactly those bytes without reopening a path or re-running discovery.
    """

    if type(deck_bytes) is not bytes or not deck_bytes:
        raise ValueError("DRC rule source bytes must be non-empty exact bytes")
    _rules_nm, report = parse_drc_rules(
        deck_bytes.decode("utf-8", errors="replace")
    )
    parsed = dict(report["parsed"])
    if antenna_bytes is not None:
        if type(antenna_bytes) is not bytes or not antenna_bytes:
            raise ValueError("antenna rule source bytes must be non-empty exact bytes")
        antenna_text = antenna_bytes.decode("utf-8", errors="replace")
        for key, hit in parse_antenna_deck(antenna_text).items():
            lo, hi = _BOUND_RATIO.get(key, (1.0, 400.0))
            if lo <= hit.value_um <= hi:
                parsed[key] = {"value_ratio": hit.value_um, "line": hit.line}
    return parsed


def find_drc_deck(pdk_dir: str) -> Optional[Path]:
    """Locate a parseable foundry DRC deck inside a real PDK directory.

    Prefers a external_drc SVRF/TVF deck (``external_drc.drc`` for external_pdk, ``*.drc`` under
    ``external_drc/DRC`` for external_pdk). Returns the path, or None if none is found.
    """
    d = Path(pdk_dir)
    if not d.is_dir():
        return None
    cands: List[Path] = []
    # explicit external_pdk online external_drc deck (canonical name)
    cands.extend(sorted(d.glob("external_drc/online/drc_online/*/external_drc.drc")))
    # Older/RF kits commonly put the canonical deck directly in external_drc/.
    cands.append(d / "external_drc" / "external_drc.drc")
    # external_pdk ships *.drc under external_drc/DRC
    cands.extend(sorted(d.glob("external_drc/DRC/*.drc")))

    # De-dup, keep priority order, and prefer a non-empty canonical candidate.
    seen = set()
    uniq: List[Path] = []
    for c in cands:
        try:
            rc = c.resolve()
        except OSError:
            continue
        if rc in seen or not c.is_file():
            continue
        seen.add(rc)
        uniq.append(c)
    if uniq:
        uniq.sort(key=lambda p: (p.stat().st_size == 0,))
        return uniq[0]

    # Only non-canonical installs pay the recursive discovery cost.
    cands = sorted(d.glob("**/external_drc.drc")) + sorted(d.glob("**/*.drc"))
    seen.clear()
    uniq.clear()
    for c in cands:
        try:
            rc = c.resolve()
        except OSError:
            continue
        if rc in seen or not c.is_file():
            continue
        seen.add(rc)
        uniq.append(c)
    if not uniq:
        return None
    uniq.sort(key=lambda p: (p.stat().st_size == 0,))
    return uniq[0]


def _parse_deck_text(text: str) -> Tuple[Dict[str, _Hit], str]:
    """Dispatch to the right deck dialect parser. Returns (hits, dialect)."""
    # external_pdk external_drc decks declare a VARIABLE table; external_pdk TVF decks embed literals.
    if re.search(r"^\s*VARIABLE\s+[A-Za-z0-9_]+\s+\d", text, re.MULTILINE):
        hits = _parse_tsmc_calibre(text)
        if hits:
            return hits, "external_pdk"
    # Older external_pdk external_drc runsets (no VARIABLE table) use dotted rule-id brace blocks
    # with the literal inline, e.g. ``M1.W.1 { ... INT M1 < 0.23 ... }``.
    if re.search(r"(?m)^\s*M1\.W\.\d+\s*\{", text) or re.search(r"(?m)^\s*CO\.W\.\d+\s*\{", text):
        hits = _parse_tsmc_old_calibre(text)
        if hits:
            return hits, "external_pdk"
    if re.search(r"#!\s*tvf", text) or re.search(r"(?m)^\s*M1_1\s*\{", text):
        return _parse_smic_tvf(text), "external_pdk"
    # last resort: try all known dialects, prefer whichever yields more rules.
    candidates = [
        (_parse_tsmc_calibre(text), "external_pdk"),
        (_parse_tsmc_old_calibre(text), "external_pdk"),
        (_parse_smic_tvf(text), "external_pdk"),
    ]
    candidates.sort(key=lambda c: len(c[0]), reverse=True)
    return candidates[0]


def parse_drc_rules(deck_text: str) -> Tuple[Dict[str, int], Dict[str, object]]:
    """Parse the key DRC rules from a deck's text. Pure function on text (so it is
    exercised by inline fixtures without a multi-GB PDK).

    Returns ``(rules_nm, report)`` where ``rules_nm`` maps a rule key
    (``metal1_width`` / ``metal1_space`` / ``poly_width`` / ``poly_space`` /
    ``contact_size`` / ``contact_space``) to a value in NM, and ``report`` records
    which keys were parsed (with source line) and which were rejected/out-of-range.
    """
    hits, dialect = _parse_deck_text(deck_text)
    rules_nm: Dict[str, int] = {}
    parsed: Dict[str, object] = {}
    rejected: Dict[str, object] = {}
    for key, hit in hits.items():
        if hit.kind == "fraction" or key in _FRAC_KEYS:
            # density coverage fraction (0..1, dimensionless). A deck stating a
            # percentage (e.g. 30 for 30%) lands out-of-band and is rejected --
            # never silently /100'd. Not added to rules_nm (PdkRules has no
            # density field); surfaced only in report["parsed"] for the deck gen.
            lo, hi = _BOUND_FRAC.get(key, (0.0, 1.0))
            if not (lo <= hit.value_um <= hi):
                rejected[key] = {
                    "value_frac": hit.value_um,
                    "bound_frac": [lo, hi],
                    "line": hit.line,
                    "reason": "out-of-range",
                }
                continue
            parsed[key] = {"value_frac": hit.value_um, "line": hit.line}
            continue
        if hit.kind == "ratio" or key in _RATIO_KEYS:
            # antenna gate-area ratio (1..200, dimensionless). Parsed-but-not-
            # added to rules_nm; the deck generator records antenna as
            # parsed-not-emitted (a pure-geometry KLayout check cannot express it).
            lo, hi = _BOUND_RATIO.get(key, (1.0, 400.0))
            if not (lo <= hit.value_um <= hi):
                rejected[key] = {
                    "value_ratio": hit.value_um,
                    "bound_ratio": [lo, hi],
                    "line": hit.line,
                    "reason": "out-of-range",
                }
                continue
            parsed[key] = {"value_ratio": hit.value_um, "line": hit.line}
            continue
        if hit.is_area or key in _AREA_KEYS:
            # area rule: bound in um^2, store value_um2 + value_nm (in nm^2).
            lo, hi = _BOUND_UM2.get(key, (0.0, 5.0))
            if not (lo <= hit.value_um <= hi):
                rejected[key] = {
                    "value_um2": hit.value_um,
                    "bound_um2": [lo, hi],
                    "line": hit.line,
                    "reason": "out-of-range",
                }
                continue
            value_nm2 = int(round(hit.value_um * 1_000_000.0))
            rules_nm[key] = value_nm2
            parsed[key] = {
                "value_um2": hit.value_um,
                "value_nm": value_nm2,
                "line": hit.line,
            }
            continue
        # length rule (width/space/size/enclosure): bound in um, store value_um + value_nm.
        lo, hi = _BOUND_UM.get(key, (0.0, 1e9))
        if not (lo <= hit.value_um <= hi):
            rejected[key] = {
                "value_um": hit.value_um,
                "bound_um": [lo, hi],
                "line": hit.line,
                "reason": "out-of-range",
            }
            continue
        rules_nm[key] = int(round(hit.value_um * 1000.0))
        parsed[key] = {"value_um": hit.value_um, "value_nm": rules_nm[key], "line": hit.line}
    report = {"dialect": dialect, "parsed": parsed, "rejected": rejected}
    return rules_nm, report


# --- PdkRules construction ---------------------------------------------------

# Consolidated defaults (nm) used when a rule is NOT parseable from the deck. These
# are conservative gf180-class numbers, only ever used as a fallback and ALWAYS
# reported as defaulted (never claimed to come from the deck).
_DEFAULT_NM = {
    "metal1_width": 230,
    "metal1_space": 230,
    "poly_width": 180,
    "poly_space": 240,
    "contact_size": 220,
    "contact_space": 250,
}

# canonical drawing-layer GDS keys for the registered PDK. Real per-PDK layermaps
# differ; this is a neutral placeholder keyed by the generic names the rules use, so
# the PdkRules container is constructed validly. The numeric (layer, datatype) is not
# claimed to be the foundry's GDS map -- only the rule VALUES are sourced.
_GENERIC_LAYERMAP: Dict[str, Tuple[int, int]] = {
    "metal1": (34, 0),
    "poly2": (30, 0),
    "comp": (22, 0),
    "contact": (33, 0),
    "via1": (35, 0),
    "metal2": (36, 0),
}


def _rv(rules_nm: Dict[str, int], key: str, rule_id: str) -> Tuple[RuleValue, bool]:
    """Build a RuleValue from a parsed rule, or the default. Returns (rv, defaulted)."""
    if key in rules_nm:
        return RuleValue(value=rules_nm[key], rule_id=rule_id, source=Provenance.PARSED_DECK), False
    return (
        RuleValue(value=_DEFAULT_NM[key], rule_id=rule_id, source=Provenance.CONSOLIDATED),
        True,
    )


def build_pdk_rules(pdk_id: str, rules_nm: Dict[str, int]) -> PdkRules:
    """Build a real :class:`PdkRules` from parsed rule values (NM).

    Layers populated: metal1 (min width+space), poly2 (min width+space), the contact
    via (size+space). Unparsed rules fall back to the consolidated default. Layer
    rules / via rules are interned per ``pdk_id``.
    """
    pid = pdk_id.lower()
    LayerRule.clear_cache()  # avoid cross-PDK identity collisions on the generic names
    ViaRule.clear_cache()

    def L(name: str) -> LayerRule:
        layer, dt = _GENERIC_LAYERMAP[name]
        return LayerRule.intern(pid, name, layer, dt)

    def _parsed_rv(key: str, rule_id: str) -> Optional[RuleValue]:
        """A PARSED_DECK RuleValue for an enclosure/area key, or None if it was not
        parsed from the deck (these have no consolidated fallback -- absence is an
        honest gap, never fabricated)."""
        if key in rules_nm:
            return RuleValue(value=rules_nm[key], rule_id=rule_id, source=Provenance.PARSED_DECK)
        return None

    def _parsed_diff_rv(key: str, rule_id: str) -> Optional[RuleValue]:
        if key not in rules_nm:
            return None
        value = rules_nm[key]
        if type(value) is not int:
            raise TypeError(f"{key} must be an exact integer number of nanometres")
        lo_um, hi_um = _BOUND_UM[key]
        if not int(round(lo_um * 1000.0)) <= value <= int(round(hi_um * 1000.0)):
            raise ValueError(f"{key} is outside the accepted foundry-rule range")
        return RuleValue(value=value, rule_id=rule_id, source=Provenance.PARSED_DECK)

    metal1 = L("metal1")
    metal1.set_min_width(_rv(rules_nm, "metal1_width", "M1.W")[0])
    metal1.set_min_spacing(_rv(rules_nm, "metal1_space", "M1.S")[0])
    if (rv := _parsed_rv("metal1_min_area", "M1.A")) is not None:
        metal1.set_min_area(rv)
    if (rv := _parsed_rv("contact_enclosure_metal1", "M1.EN")) is not None:
        metal1.set_enclosure("contact", rv)
    if (rv := _parsed_rv("via1_enclosure_metal1", "VIA1.EN")) is not None:
        metal1.set_enclosure("via1", rv)

    poly2 = L("poly2")
    poly2.set_min_width(_rv(rules_nm, "poly_width", "PO.W")[0])
    poly2.set_min_spacing(_rv(rules_nm, "poly_space", "PO.S")[0])
    if (rv := _parsed_rv("poly_min_area", "PO.A")) is not None:
        poly2.set_min_area(rv)
    if (rv := _parsed_rv("contact_enclosure_poly", "CO.EN")) is not None:
        poly2.set_enclosure("contact", rv)
    if (rv := _parsed_rv("poly_extension", "PO.EX")) is not None:
        poly2.set_enclosure("comp", rv)  # poly endcap over active/diff

    comp = L("comp")
    if (rv := _parsed_diff_rv("diff_width", "OD.W")) is not None:
        comp.set_min_width(rv)
    if (rv := _parsed_diff_rv("diff_space", "OD.S")) is not None:
        comp.set_min_spacing(rv)
    if (rv := _parsed_rv("contact_enclosure_diff", "CO.EN")) is not None:
        comp.set_enclosure("contact", rv)

    layer_rules = {"metal1": metal1, "poly2": poly2, "comp": comp}

    clayer, cdt = _GENERIC_LAYERMAP["contact"]
    contact = ViaRule.intern(pid, "contact", clayer, cdt, "comp", "metal1")
    contact.set_size(_rv(rules_nm, "contact_size", "CO.W")[0])
    contact.set_spacing(_rv(rules_nm, "contact_space", "CO.S")[0])
    if (rv := _parsed_rv("contact_enclosure_diff", "CO.EN")) is not None:
        contact.set_enclosure_below(rv)
    if (rv := _parsed_rv("contact_enclosure_metal1", "M1.EN")) is not None:
        contact.set_enclosure_above(rv)
    via_rules = {"contact": contact}

    # Via1 enclosures (by M1 below, M2 above) when parsed -- a real via1 rule object.
    enc_b = _parsed_rv("via1_enclosure_metal1", "VIA1.EN")
    enc_a = _parsed_rv("via1_enclosure_metal2", "M2.EN")
    if enc_b is not None or enc_a is not None:
        vlayer, vdt = _GENERIC_LAYERMAP.get("via1", (35, 0))
        via1 = ViaRule.intern(pid, "via1", vlayer, vdt, "metal1", "metal2")
        if enc_b is not None:
            via1.set_enclosure_below(enc_b)
        if enc_a is not None:
            via1.set_enclosure_above(enc_a)
        via_rules["via1"] = via1

    data = PdkData(
        pdk_id=pid,
        dbu_nm=1,
        layermap=dict(_GENERIC_LAYERMAP),
        layer_rules=layer_rules,
        via_rules=via_rules,
        deck_dir="",
    )
    return PdkRules(data)


def pdk_rules_from_deck(pdk_dir: str) -> Tuple[PdkRules, Dict[str, object]]:
    """Locate the DRC deck under ``pdk_dir``, parse the key rules, build a real
    :class:`PdkRules`.

    Returns ``(rules, report)``. ``report`` lists, per rule key, whether it came from
    the deck (``parsed`` with source line) or a default (``defaulted`` with reason),
    plus any ``rejected`` out-of-range parses and the deck path + dialect.
    """
    d = Path(pdk_dir)
    pdk_id = _short_pdk_id(d.name)
    deck = find_drc_deck(str(d))
    if deck is None:
        # no deck -> everything defaults; honest report.
        rules = build_pdk_rules(pdk_id, {})
        report = {
            "deck": None,
            "dialect": None,
            "parsed": {},
            "defaulted": {
                k: {"value_nm": _DEFAULT_NM[k], "reason": "no DRC deck found"}
                for k in _DEFAULT_NM
            },
            "rejected": {},
        }
        return rules, report

    text = deck.read_text(encoding="utf-8", errors="replace")
    rules_nm, parse_report = parse_drc_rules(text)
    rules = build_pdk_rules(pdk_id, rules_nm)

    parsed = dict(parse_report["parsed"])
    rejected = dict(parse_report["rejected"])

    # antenna lives in a SEPARATE deck (foundry ships it as a sibling ``*.ANT.*`` /
    # ``ant.drc`` / a ``*.ant`` TVF, not in the main DRC deck). Parse the poly AREA
    # antenna ratio from there when present; sanity-bound it the same way
    # parse_drc_rules bounds a ratio. Absent (or perimeter-only) -> honestly unparsed.
    ant_deck = find_antenna_deck(str(d))
    antenna_deck_path = None
    if ant_deck is not None:
        antenna_deck_path = str(ant_deck)
        ant_text = ant_deck.read_text(encoding="utf-8", errors="replace")
        for key, hit in parse_antenna_deck(ant_text).items():
            lo, hi = _BOUND_RATIO.get(key, (1.0, 400.0))
            if not (lo <= hit.value_um <= hi):
                rejected[key] = {
                    "value_ratio": hit.value_um,
                    "bound_ratio": [lo, hi],
                    "line": hit.line,
                    "reason": "out-of-range",
                }
                continue
            parsed[key] = {"value_ratio": hit.value_um, "line": hit.line}

    defaulted = {
        k: {"value_nm": _DEFAULT_NM[k], "reason": "not parseable from deck"}
        for k in _DEFAULT_NM
        if k not in rules_nm
    }
    report = {
        "deck": str(deck),
        "antenna_deck": antenna_deck_path,
        "dialect": parse_report["dialect"],
        "parsed": parsed,
        "defaulted": defaulted,
        "rejected": rejected,
    }
    return rules, report


# --- one-call registration ---------------------------------------------------


@dataclass(frozen=True)
class PdkRegistration:
    """Everything ``clay pdk add`` needs to admit a new real PDK to Clay.

    Attributes:
        pdk_id: short stable id (e.g. ``open_pdk``, ``open_pdk``).
        characterization: the device-level :class:`PdkCharacterization` (Vdd/Lmin/Vth/KP).
        adapter: the marker-adapter descriptor (harness binding) from
            :func:`discover_pdk_adapter`.
        rules: the layout :class:`PdkRules` parsed from the DRC deck.
        report: the parsed-vs-defaulted rule report from :func:`pdk_rules_from_deck`.
    """

    pdk_id: str
    characterization: PdkCharacterization
    adapter: Dict[str, object]
    rules: PdkRules
    report: Dict[str, object] = field(default_factory=dict)


def register_pdk(pdk_dir: str) -> PdkRegistration:
    """One-call registration of a real PDK directory into Clay's flow.

    Runs the device characterizer, builds the harness adapter descriptor, parses the
    DRC deck into a PdkRules, and bundles them with the honesty report.
    """
    d = Path(pdk_dir)
    char = characterize_pdk(str(d))
    pdk_id = _short_pdk_id(char.pdk)
    adapter = discover_pdk_adapter(str(d))
    rules, report = pdk_rules_from_deck(str(d))
    return PdkRegistration(
        pdk_id=pdk_id,
        characterization=char,
        adapter=adapter,
        rules=rules,
        report=report,
    )


__all__ = [
    "PdkRegistration",
    "discover_pdk_adapter",
    "pdk_rules_from_deck",
    "parse_drc_rules",
    "build_pdk_rules",
    "find_drc_deck",
    "find_antenna_deck",
    "parse_antenna_deck",
    "register_pdk",
]
