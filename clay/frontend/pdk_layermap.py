"""PDK layer-map parser: extract real GDS (layer, datatype) per canonical layer token.

The standalone ``*.map`` stream files shipped with these PDKs are either empty or
device-pin maps -- they do not carry the drawn-layer GDS numbers a DRC-deck generator
needs. The authoritative source is the foundry's own sign-off artifact:

* gf180mcu -- the KLayout layer-properties file
  ``.../tech/klayout/gf180mcu.lyp`` (``<source>NAME L/D@idx</source>``); this is
  ground truth for the open-source PDK.
* external_pdk / external_pdk -- the external_drc DRC deck's drawn-layer section. Two declaration styles
  appear and both are handled:

    - ``LAYER MAP <gds> DATATYPE <dt> <internal>`` followed by ``LAYER <NAME> <internal>``
      (gds number lives in the MAP line; the trailing number on the ``LAYER`` line is a
      synthesized internal id, NOT a GDS number). external_pdk uses this for every drawn layer;
      external_pdk uses it for OD/PO/M1/Vn.
    - ``LAYER <NAME> <gds>`` with no ``LAYER MAP ... DATATYPE 0`` redirect for that gds
      (the bare number IS the GDS layer, datatype 0 implied). external_pdk uses this for the
      contact layer (``LAYER COi 30``).
    - ``<name> = polygons(<gds> , <dt>)`` -- the gf180/TED external_drc-style python deck form
      (handled for completeness / inline fixtures).

HONESTY: a mapping is emitted only when it traces to a real line in one of these PDK
files. Nothing is guessed. Layers that cannot be resolved are omitted and the reason is
recorded in :func:`parse_layermap_detail` under ``unresolved``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

LayerKey = Tuple[int, int]

# ---------------------------------------------------------------------------
# Canonicalisation: vendor layer NAME -> canonical token.
#
# Canonical tokens are chosen to line up with the rule keys produced by
# ``clay.frontend.pdk_register`` (``metal1_*`` / ``poly_*`` / ``contact_*``), so a
# downstream generator can strip the ``_width`` / ``_space`` / ``_size`` suffix and join
# rule -> layer directly. ``diff`` / ``nwell`` / ``via1`` extend the set.
#
# Keys are matched case-insensitively against the *bare* vendor name with any trailing
# ``i`` external_drc-internal suffix removed (e.g. ``M1i`` -> ``m1``, ``GTi`` -> ``gt``).
# ---------------------------------------------------------------------------
_CANON: Dict[str, str] = {
    # metal 1
    "metal1": "metal1",
    "met1": "metal1",
    "me1": "metal1",
    "m1": "metal1",
    # metal 2 (gf180 .lyp Metal2 36/0; open_pdk M2 32/0; external_pdk M2 62/0)
    "metal2": "metal2",
    "met2": "metal2",
    "me2": "metal2",
    "m2": "metal2",
    # metal 3 (gf180 .lyp Metal3 42/0; open_pdk M3 33/0; external_pdk M3 63/0)
    "metal3": "metal3",
    "met3": "metal3",
    "me3": "metal3",
    "m3": "metal3",
    # poly (gate)
    "poly": "poly",
    "poly2": "poly",
    "po": "poly",
    "gat": "poly",
    "gt": "poly",   # external_pdk gate == poly (techfile labels GT "poly")
    # contact (M1 <-> diffusion/gate)
    "contact": "contact",
    "cont": "contact",
    "co": "contact",
    "ct": "contact",
    # diffusion / active area
    "diff": "diff",
    "comp": "diff",
    "active": "diff",
    "od": "diff",
    "aa": "diff",   # external_pdk active area == diff (techfile labels AA "diff")
    # nwell
    "nwell": "nwell",
    "nw": "nwell",
    # via1 (M1 <-> M2)
    "via1": "via1",
    "vi1": "via1",
    "v1": "via1",
    # via2 (M2 <-> M3) (gf180 .lyp Via2 38/0; open_pdk VIA2 52/0; external_pdk V2 71/0)
    "via2": "via2",
    "vi2": "via2",
    "v2": "via2",
    # N+ / P+ source-drain implants (gf180 nplus 32 / pplus 31; external_pdk SN 40 / SP 43;
    # external_pdk NIMP / PIMP). Needed for implant-based MOS recognition + bulk-tap LVS.
    # Exact full-name aliases only -- no 2-char keys (np/pp) and no key that a
    # external_drc trailing-'i' layer strips onto (the canonical_layer 'i'-strip).
    "nplus": "nplus",
    "sn": "nplus",
    "nimp": "nplus",
    "pplus": "pplus",
    "sp": "pplus",
    "pimp": "pplus",
}

# canonical tokens we surface (order = emission/priority order)
CANONICAL_TOKENS = (
    "metal1", "poly", "contact", "diff", "nwell", "via1", "metal2", "metal3",
    "via2", "nplus", "pplus",
)

# A vendor name that should NOT be canonicalised even though a prefix matches:
# we strip exactly one trailing "i" (external_drc internal marker) and lowercase. Anything
# that is not an exact key in _CANON after that is rejected (no fuzzy prefix matching --
# that would mis-grab e.g. "M10" -> metal1 or "V12" -> via1).


def canonical_layer(vendor_name: str) -> Optional[str]:
    """Map a vendor layer name to a canonical token, or ``None`` if unknown.

    Strips trailing external_drc ``i`` markers and lowercases before the exact lookup.
    Some decks double the marker (open_pdk: ``COii`` / ``VIA1ii`` / ``NWii``).
    Exact match only -- ``M10`` / ``V2`` / ``POFUSE`` deliberately do not resolve
    (no canonical token ends in ``i``, so stripping trailing ``i`` can't over-match).
    """
    if not vendor_name:
        return None
    name = vendor_name.strip()
    low = name.lower()
    if low in _CANON:
        return _CANON[low]
    # strip trailing 'i' marker(s) (external_drc internal layer: M1i, GTi, COi, COii, VIA1ii...)
    stripped = low.rstrip("i")
    if stripped != low and stripped in _CANON:
        return _CANON[stripped]
    return None


# ---------------------------------------------------------------------------
# Parsers, one per source form. Each returns {canonical_token: {"gds": (l, d),
# "source": "<verbatim line>", "vendor": "<name>"}}.
# ---------------------------------------------------------------------------

# gf180 .lyp <source> form:  COMP 22/0@1   /   Metal1 34/0@1
_LYP_SOURCE = re.compile(r"<source>\s*([A-Za-z][\w]*)\s+(\d+)\s*/\s*(\d+)\s*(?:@\d+)?\s*</source>")

# optional trailing whitespace and "// comment" the external_drc decls/maps frequently carry
_COMMENT = r"\s*(?://.*)?$"

# external_drc "LAYER MAP <gds> DATATYPE <spec> <internal>".  We only accept maps that select
# datatype 0: either an explicit "DATATYPE 0" or a full-range form whose lower bound is 0
# ("DATATYPE >=0 <=255" / "DATATYPE >= 0") -- both include the drawn datatype-0 polygons.
# Datatype maps that do not include 0 (e.g. "DATATYPE 11", "DATATYPE >=1 <=10") are NOT a
# datatype-0 declaration and are ignored for layer extraction.
_LAYER_MAP_DT = re.compile(
    r"^\s*LAYER\s+MAP\s+(\d+)\s+DATATYPE\s+"
    r"(?:(0)\b|>=\s*0\b(?:\s*<=\s*\d+)?)"   # datatype exactly 0, or >=0 [ <=N ]
    r"\s+(\d+)" + _COMMENT,
    re.IGNORECASE,
)

# external_drc "LAYER <NAME> <num> [extra ints...] [// comment]"  (declaration; num may be an
# internal id or a real gds number depending on whether a LAYER MAP redirects it).
_LAYER_DECL = re.compile(
    r"^\s*LAYER\s+([A-Za-z][\w]*)\s+(\d+)(?:\s+\d+)*" + _COMMENT,
    re.IGNORECASE,
)

# external_drc-style python form:  metal1 = polygons(34 , 0)
_POLYGONS = re.compile(
    r"^\s*([A-Za-z][\w]*)\s*=\s*polygons?\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)",
    re.IGNORECASE,
)


def _emit(out: Dict[str, dict], token: str, gds: LayerKey, source: str, vendor: str) -> None:
    """Record the first trace for a canonical token (first hit wins -> the primary
    drawn-layer declaration, before dummy/OPC/SRAM variants)."""
    if token not in out:
        out[token] = {"gds": gds, "source": source.strip(), "vendor": vendor}


def parse_lyp_text(text: str) -> Dict[str, dict]:
    """Parse a KLayout ``.lyp`` layer-properties body (gf180 ground truth)."""
    out: Dict[str, dict] = {}
    for raw in text.splitlines():
        m = _LYP_SOURCE.search(raw)
        if not m:
            continue
        vendor, l, d = m.group(1), int(m.group(2)), int(m.group(3))
        tok = canonical_layer(vendor)
        if tok:
            _emit(out, tok, (l, d), raw, vendor)
    return out


def parse_calibre_text(text: str) -> Dict[str, dict]:
    """Parse a external_drc DRC deck body (external_pdk / external_pdk), all three declaration forms.

    Strategy:
      1. Collect ``LAYER MAP <gds> DATATYPE 0 <internal>`` -> {internal: (gds,0,line)}.
      2. Walk ``LAYER <NAME> <num>`` declarations:
           - if <num> is a known internal id from step 1, the GDS is that map's gds.
           - else <num> is a bare GDS layer (datatype 0) iff no datatype-0 MAP redirects
             it; the contact layer (``LAYER COi 30``) is the canonical such case.
      3. Also accept inline ``name = polygons(N, D)`` lines.
    """
    out: Dict[str, dict] = {}

    # step 1: internal-id -> (gds, dt, line) from datatype-0 / full-range maps
    internal_gds: Dict[int, Tuple[LayerKey, str]] = {}
    # set of gds layer numbers that have a datatype-0 MAP redirect (so a bare
    # "LAYER NAME <gds>" for the same number is an internal id, not the drawn layer).
    redirected_gds: Dict[int, bool] = {}
    for raw in text.splitlines():
        mm = _LAYER_MAP_DT.match(raw)
        if not mm:
            continue
        gds = int(mm.group(1))
        internal = int(mm.group(3))
        # group(2) is the explicit datatype (None when the ">=0" full-range branch hit)
        dt = int(mm.group(2)) if mm.group(2) is not None else 0
        internal_gds.setdefault(internal, ((gds, dt), raw))
        redirected_gds[gds] = True

    # step 2: LAYER declarations
    for raw in text.splitlines():
        pm = _POLYGONS.match(raw)
        if pm:
            tok = canonical_layer(pm.group(1))
            if tok:
                _emit(out, tok, (int(pm.group(2)), int(pm.group(3))), raw, pm.group(1))
            continue
        dm = _LAYER_DECL.match(raw)
        if not dm:
            continue
        vendor, num = dm.group(1), int(dm.group(2))
        if vendor.upper() == "MAP":  # this is a "LAYER MAP ..." line, not a decl
            continue
        tok = canonical_layer(vendor)
        if not tok:
            continue
        if num in internal_gds:
            (gds, dt), mapline = internal_gds[num]
            # source cites both the decl and the authoritative MAP line
            src = f"{raw.strip()}    [<- {mapline.strip()}]"
            _emit(out, tok, (gds, dt), src, vendor)
        elif num not in redirected_gds:
            # bare GDS number, datatype 0 implied (e.g. external_pdk "LAYER COi 30")
            _emit(out, tok, (num, 0), raw, vendor)
        # else: num is a gds that is wholly redirected by datatype maps and the decl
        # carried no datatype-0 form -> cannot honestly assign; skip.
    return out


# ---------------------------------------------------------------------------
# PDK-directory dispatch.
# ---------------------------------------------------------------------------

def _find_gf180_lyp(pdk_dir: Path) -> Optional[Path]:
    cand = (
        pdk_dir
        / "libraries" / "gf180mcu_fd_pr" / "latest" / "tech" / "klayout" / "gf180mcu.lyp"
    )
    if cand.is_file():
        return cand
    hits = sorted(pdk_dir.rglob("gf180mcu.lyp"))
    return hits[0] if hits else (sorted(pdk_dir.rglob("*.lyp"))[:1] or [None])[0]


def _find_tsmc40_deck(pdk_dir: Path) -> Optional[Path]:
    direct = (
        pdk_dir / "external_drc" / "online" / "drc_online"
        / "1P10M_7X0Y1Z0R1U" / "external_drc.drc",
        pdk_dir / "external_drc" / "external_drc.drc",
        pdk_dir / "external_drc.drc",
    )
    for candidate in direct:
        if candidate.is_file():
            return candidate
    hits = sorted(pdk_dir.rglob("external_drc.drc"))
    return hits[0] if hits else None


def _find_smic_deck(pdk_dir: Path) -> Optional[Path]:
    hits = sorted(pdk_dir.glob("external_drc/DRC/external_pdk*.drc"))
    if hits:
        return hits[0]
    hits = sorted(pdk_dir.rglob("external_pdk*.drc"))
    return hits[0] if hits else None


def _detect_source(pdk_dir: Path) -> Tuple[str, Optional[Path]]:
    """Return ``(kind, path)`` where kind is 'lyp' | 'external_drc' | 'none'."""
    name = pdk_dir.name.lower()
    if "gf180" in name:
        p = _find_gf180_lyp(pdk_dir)
        if p:
            return "lyp", p
    if name.startswith("external_pdk") or "external_pdk" in name:
        p = _find_tsmc40_deck(pdk_dir)
        if p:
            return "external_drc", p
    if "external_pdk" in name:
        p = _find_smic_deck(pdk_dir)
        if p:
            return "external_drc", p
    # generic fallback: prefer a .lyp, then a external_drc deck
    p = _find_gf180_lyp(pdk_dir)
    if p and p.suffix == ".lyp":
        return "lyp", p
    p = _find_tsmc40_deck(pdk_dir) or _find_smic_deck(pdk_dir)
    if p:
        return "external_drc", p
    return "none", None


def parse_layermap_detail(pdk_dir) -> Dict[str, object]:
    """Full provenance view.

    Returns::

        {
          "pdk_dir": str,
          "source_kind": "lyp" | "external_drc" | "none",
          "source_file": str | None,
          "layers": { token: {"gds": (l, d), "source": "<line>", "vendor": "<name>"} },
          "unresolved": { token: "reason" },
        }

    Every ``gds`` value cites the verbatim PDK line it was read from in ``source``.
    """
    pdk_dir = Path(pdk_dir)
    kind, path = _detect_source(pdk_dir)
    detail: Dict[str, object] = {
        "pdk_dir": str(pdk_dir),
        "source_kind": kind,
        "source_file": str(path) if path else None,
        "layers": {},
        "unresolved": {},
    }
    if kind == "none" or path is None:
        for tok in CANONICAL_TOKENS:
            detail["unresolved"][tok] = "no .lyp or external_drc deck found under pdk_dir"
        return detail

    text = path.read_text(encoding="utf-8", errors="replace")
    layers = parse_lyp_text(text) if kind == "lyp" else parse_calibre_text(text)
    detail["layers"] = layers
    for tok in CANONICAL_TOKENS:
        if tok not in layers:
            detail["unresolved"][tok] = f"not found in {path.name}"
    return detail


def parse_layermap(pdk_dir) -> Dict[str, LayerKey]:
    """Map canonical layer token -> (gds_layer, gds_datatype) for a real PDK directory.

    Only tokens that trace to a real PDK file line are included. Use
    :func:`parse_layermap_detail` for the source line behind each entry.
    """
    detail = parse_layermap_detail(pdk_dir)
    return {tok: info["gds"] for tok, info in detail["layers"].items()}  # type: ignore[index]


def parse_layermap_bytes(kind: str, payload: bytes) -> Dict[str, LayerKey]:
    """Parse one already captured layermap source without reopening its path."""

    if kind not in {"lyp", "external_drc"}:
        raise ValueError(f"unsupported layermap source kind: {kind!r}")
    if type(payload) is not bytes or not payload:
        raise ValueError("layermap source bytes must be non-empty exact bytes")
    text = payload.decode("utf-8", errors="replace")
    layers = parse_lyp_text(text) if kind == "lyp" else parse_calibre_text(text)
    return {token: tuple(info["gds"]) for token, info in layers.items()}
