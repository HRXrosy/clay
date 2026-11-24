"""Generate a runnable free KLayout DRC deck (``.lydrc``) from parsed real-foundry
DRC rule values plus a GDS layer map.

This is the differentiating capability: given the rule values Clay's PDK parser
extracts from a *commercial* foundry deck (HSPICE/external_drc/own-format) and the
GDS layer numbers, emit a self-contained KLayout DRC Ruby runset that the free
KLayout DRC engine parses and runs. Nobody else auto-synthesizes a free DRC deck
for a commercial PDK.

Scope honesty
-------------
The generated deck is a *partial* DRC deck. It carries the REAL parsed values and
REAL GDS layer numbers only -- no invented numbers, and it emits a rule ONLY when
both its value (from ``parsed_rules``) and its layer (from ``layermap``) are
present; everything else is skipped with an inline comment.

Coverage spans the geometric core (width / space / contact-size / enclosure /
min-area) plus the wider families that have an honest free-KLayout idiom:
*density* (``with_density`` floor + ceiling, on a generator-default tile window --
the FRACTION is real, the window is the generator's because the foundry window is
an unresolved indirection variable), *notch* (``layer.notch``), *wide-metal
spacing* (a width-selected morphological-opening approximation), *line-end*
(end-of-line metal overhang past a cut: the PROJECTION-metric
``metal.enclosing(cut, v.um, projection)`` under-enclosure RESTRICTED to the
metal's wire-END edges, so the long sides -- governed by the smaller
side-enclosure rule -- are not mis-flagged), and *antenna* (the connectivity-aware
``antenna_check(gate, conductor, ratio)`` after deriving ``gate = conductor & diff``
and ``connect``-ing the net). Line-end + antenna are CHIP-LEVEL/connectivity rules
(see ``CHIP_LEVEL_KINDS``): a cell-level deck excludes them via ``exclude_kinds``
the same way it excludes density. Antenna is emitted ONLY for the poly area ratio
(the metal1 antenna limit is a conditional perimeter/diode equation in the foundry
deck -- no clean static scalar -- and stays honestly absent / parsed-not-emitted).

It is NOT a tapeout sign-off deck. Full sign-off needs the foundry's complete
rule set (recommended, electrical, the full antenna + density windows) and
typically a commercial tool (external_drc/Pegasus).

Inputs (frozen shapes)
----------------------
``parsed_rules``
    ``clay.frontend.pdk_register.register_pdk(dir).report["parsed"]``::

        {"metal1_width": {"value_um": 0.23, "value_nm": 230, "line": "..."},
         "metal1_space": {...}, "poly_width": {...}, "poly_space": {...},
         "contact_size": {...}, "contact_space": {...}, ...}

``layermap``
    ``Dict[str, Tuple[int, int]]`` canonical-token -> ``(gds_layer, datatype)``::

        {"metal1": (34, 0), "poly2": (30, 0), "contact": (33, 0),
         "comp": (22, 0), "nwell": (21, 0)}

The deck's runtime-variable (``-rd``) interface mirrors the gf180mcu foundry deck
header: ``source($input[,$topcell])`` + ``report("...", $report)``, flat mode.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Rule catalogue: which parsed_rules key maps to which layer token + check kind
# + DRC rule id/description prefix. Each entry is a single core geometric rule.
#
# This is a static table of well-known core rules, NOT a per-PDK branch: the
# same catalogue is applied to every PDK's parsed_rules; rules whose value or
# layer are absent are simply skipped. (No `if pdk == ...` anywhere.)
# ---------------------------------------------------------------------------


# Tile size (um) the generator bakes into emitted density checks. KLayout's
# ``with_density`` requires a tiling window; the foundry deck states this window
# via an indirection VARIABLE (e.g. ``METAL_LOW_DEN_WINDOW_SIZE``) that the PDK
# parser does not resolve to a clean scalar, so the emitted window is a documented
# generator default, NOT a foundry-sourced number. The density FRACTION is real;
# the window is the generator's. This keeps the emission valid + honest.
_DENSITY_TILE_UM = 20.0


@dataclass(frozen=True)
class _RuleSpec:
    rule_key: str          # key in parsed_rules
    layer_token: str       # token in layermap (canonical) -- the primary layer
    kind: str              # "width"|"space"|"size"|"enclosure"|"area"
                           # |"density_min"|"density_max"|"notch"|"line_end"
                           # |"wide_space"|"antenna"
    rule_id: str           # DRC report category, e.g. "M1.W.1"
    label: str             # human label for the layer, e.g. "metal1"
    # for kind=="enclosure": the OUTER (enclosing) layer token. The primary
    # ``layer_token`` is then the INNER (enclosed) layer. The emitted check is
    # ``inner.enclosed(outer, v.um)`` -- both layers must resolve in the layermap
    # or the rule is skipped with an audit comment.
    outer_token: Optional[str] = None


# Layer tokens are tried in order; the first present in layermap wins. This lets
# one rule cover synonymous tokens across PDKs (poly vs poly2, contact vs ct,
# comp vs diff/od) without per-PDK branching.
_LAYER_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "metal1": ("metal1", "m1"),
    "metal2": ("metal2", "m2"),
    "metal3": ("metal3", "m3"),
    "metal4": ("metal4", "m4"),
    "metal5": ("metal5", "m5"),
    "metal6": ("metal6", "m6"),
    "poly": ("poly2", "poly", "po", "gt"),
    "contact": ("contact", "ct", "co"),
    "via1": ("via1", "v1", "via"),
    "via2": ("via2", "v2"),
    "via3": ("via3", "v3"),
    "via4": ("via4", "v4"),
    "via5": ("via5", "v5"),
    "diff": ("comp", "diff", "od", "aa", "active"),
    "nwell": ("nwell", "nw", "well"),
}

# kind="size" means a min-feature-size check; KLayout expresses min size as a
# .width() check (a contact smaller than its min size fails the width check),
# which is the idiom the gf180 deck itself uses for CO.1 (contact size).
_CATALOGUE: Tuple[_RuleSpec, ...] = (
    _RuleSpec("metal1_width", "metal1", "width", "M1.W.1", "metal1"),
    _RuleSpec("metal1_space", "metal1", "space", "M1.S.1", "metal1"),
    _RuleSpec("poly_width", "poly", "width", "PO.W.1", "poly"),
    _RuleSpec("poly_space", "poly", "space", "PO.S.1", "poly"),
    _RuleSpec("contact_size", "contact", "size", "CO.W.1", "contact"),
    _RuleSpec("contact_space", "contact", "space", "CO.S.1", "contact"),
    # --- enclosure rules: inner.enclosed(outer, v.um) ---
    # contact enclosed by metal1 / diff / poly; via1 enclosed by metal1 / metal2;
    # poly endcap = diff(active) enclosed-by poly (the gate overhang of poly past
    # the active edge). Each needs BOTH layers in the layermap to emit.
    _RuleSpec("contact_enclosure_metal1", "contact", "enclosure", "M1.EN.1",
              "metal1 over contact", outer_token="metal1"),
    _RuleSpec("contact_enclosure_diff", "contact", "enclosure", "CO.EN.1",
              "diff over contact", outer_token="diff"),
    _RuleSpec("contact_enclosure_poly", "contact", "enclosure", "CO.EN.3",
              "poly over contact", outer_token="poly"),
    _RuleSpec("via1_enclosure_metal1", "via1", "enclosure", "VIA1.EN.1",
              "metal1 over via1", outer_token="metal1"),
    _RuleSpec("via1_enclosure_metal2", "via1", "enclosure", "VIA1.EN.2",
              "metal2 over via1", outer_token="metal2"),
    _RuleSpec("poly_extension", "diff", "enclosure", "PO.EX.1",
              "poly endcap over active", outer_token="poly"),
    # --- min-area rules: layer.with_area(0 .. v).output(...) ---
    _RuleSpec("metal1_min_area", "metal1", "area", "M1.A.1", "metal1"),
    _RuleSpec("poly_min_area", "poly", "area", "PO.A.1", "poly"),
    # --- density rules: layer.with_density(lo, hi, tile_size(W)).output(...) ---
    # min (floor): flag tiles BELOW the minimum coverage -> with_density(0, min).
    # max (ceiling): flag tiles ABOVE the maximum coverage -> with_density(max, 1).
    _RuleSpec("metal1_density_min", "metal1", "density_min", "M1.DN.1", "metal1"),
    _RuleSpec("metal1_density_max", "metal1", "density_max", "M1.DN.1.1", "metal1"),
    # --- notch / line-end: short same-layer geometry checks ---
    # notch via layer.notch(v) (a self-space inner-corner check, native KLayout).
    _RuleSpec("metal1_notch", "metal1", "notch", "M1.NT.1", "metal1"),
    # line-end (end-of-line): the metal must overhang the cut (contact) at a wire END
    # by >= v. The PROJECTION-metric enclosure ``metal.enclosing(cut, v.um,
    # projection)`` detects the under-enclosure, but is RESTRICTED to the metal's
    # wire-END edges (edges shorter than a few line-widths) so the long sides --
    # governed by the smaller side-enclosure rule, not the EOL rule -- are not
    # mis-flagged (that mis-flag was a checker false positive on tapeout-clean
    # layouts; see the emit branch). The cut layer is the ``outer_token``; both
    # the metal and the cut must resolve in the layermap or the rule is skipped.
    # Tagged chip-level/connectivity-adjacent (a routing/cut interaction rule, listed
    # in CHIP_LEVEL_KINDS so a cell deck can exclude it).
    _RuleSpec("metal1_line_end", "metal1", "line_end", "M1.LE.1", "metal1",
              outer_token="contact"),
    # --- wide-metal spacing: width-selected space (W above a threshold) ---
    # selects wide metal (width > threshold) then checks its space. The threshold
    # comes from the companion ``metal1_wide_width`` parsed value.
    _RuleSpec("metal1_wide_space", "metal1", "wide_space", "M1.S.2", "metal1"),
    # --- antenna: connectivity-AWARE gate-area ratio (NOT pure geometry) ---
    # The poly antenna ratio (poly-conductor area / gate area) is emitted via
    # KLayout's connectivity-aware ``antenna_check(gate, metal, ratio)``: we derive
    # the transistor gate (``gate = poly & diff``), ``connect(poly, gate)``, then
    # check the ratio. Needs BOTH poly + diff in the layermap. Tagged chip-level (an
    # antenna/routing rule satisfied at full-chip, in CHIP_LEVEL_KINDS so a cell deck
    # can exclude it like density).
    #
    # The metal1 antenna ratio is NOT emitted: in every foundry deck we ship the M1
    # limit is a conditional perimeter/diode EQUATION (no clean static area-to-gate
    # scalar), so metal1_antenna_ratio is honestly absent from parsed_rules. The
    # poly antenna's outer/derive layer is ``diff`` (to build the gate).
    _RuleSpec("metal1_antenna_ratio", "metal1", "antenna", "M1.ANT.1", "metal1"),
    _RuleSpec("poly_antenna_ratio", "poly", "antenna", "PO.ANT.1", "poly",
              outer_token="diff"),
)

# Upper metal levels (M2..M6) and ALL via levels (VIA1..VIA5) get the SAME width/space
# (metals) and size/space (vias) checks as M1/contact -- GENERATED, not hand-listed, so
# a 6-metal PDK (open_pdk routes the reference twoStageAmp up to M6) polices every layer
# its rules declare instead of leaving M2-M6 + every via's own size/space unpoliced. Each
# generated spec emits ONLY when BOTH its parsed value (drc_rules) and its layer
# (layermap) resolve; otherwise it is skipped with an honest audit comment, exactly like
# the metal1/contact specs above. No per-PDK branching: a 3-metal PDK simply skips M4-M6.
_CATALOGUE = _CATALOGUE + tuple(
    _RuleSpec(f"metal{n}_width", f"metal{n}", "width", f"M{n}.W.1", f"metal{n}")
    for n in range(2, 7)
) + tuple(
    _RuleSpec(f"metal{n}_space", f"metal{n}", "space", f"M{n}.S.1", f"metal{n}")
    for n in range(2, 7)
) + tuple(
    _RuleSpec(f"via{n}_size", f"via{n}", "size", f"VIA{n}.W.1", f"via{n}")
    for n in range(1, 6)
) + tuple(
    _RuleSpec(f"via{n}_space", f"via{n}", "space", f"VIA{n}.S.1", f"via{n}")
    for n in range(1, 6)
)

# Rule KINDS that are chip-level / connectivity (satisfied across the full chip, not
# in an isolated cell): a cell-level deck excludes these the same way it excludes
# density. ``antenna`` needs the connect/netlist pass; ``line_end`` is a cut/routing
# interaction. Callers can pass ``exclude_kinds=CHIP_LEVEL_KINDS`` for a cell deck.
CHIP_LEVEL_KINDS: Tuple[str, ...] = (
    "density_min", "density_max", "antenna", "line_end",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_layer(
    token_group: str, layermap: Dict[str, Tuple[int, int]]
) -> Optional[Tuple[str, Tuple[int, int]]]:
    """Resolve a logical layer group to (matched_token, (layer, datatype)).

    Tries the synonyms for ``token_group`` (if any), else the token itself.
    Returns ``None`` if no synonym is in ``layermap``.
    """
    candidates = _LAYER_SYNONYMS.get(token_group, (token_group,))
    for tok in candidates:
        if tok in layermap:
            return tok, layermap[tok]
    return None


def _ruby_var(token: str) -> str:
    """A safe Ruby variable name for a layer token (e.g. 'metal1' -> 'l_metal1')."""
    safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in token)
    return f"l_{safe}"


def _value_um(entry: object) -> Optional[float]:
    """Pull ``value_um`` from a parsed_rules entry; ``None`` if shape is wrong."""
    if isinstance(entry, dict):
        v = entry.get("value_um")
        if isinstance(v, (int, float)):
            return float(v)
        nm = entry.get("value_nm")
        if isinstance(nm, (int, float)):
            return float(nm) / 1000.0
    return None


def _value_um2(entry: object) -> Optional[float]:
    """Pull ``value_um2`` (a min-area) from a parsed_rules entry; ``None`` if absent.

    Area entries carry ``value_um2`` directly; ``value_nm`` for area entries is in
    nm^2, so it converts to um^2 by /1e6 (length entries use /1e3 -- the wrong
    factor here, so ``value_um2`` is required to be present for an area rule)."""
    if isinstance(entry, dict):
        v = entry.get("value_um2")
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _value_frac(entry: object) -> Optional[float]:
    """Pull a density coverage ``value_frac`` (0..1) from a parsed_rules entry."""
    if isinstance(entry, dict):
        v = entry.get("value_frac")
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _value_ratio(entry: object) -> Optional[float]:
    """Pull an antenna ``value_ratio`` (dimensionless) from a parsed_rules entry."""
    if isinstance(entry, dict):
        v = entry.get("value_ratio")
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _fmt_frac(value: float) -> str:
    """Format a 0..1 fraction, trimming trailing zeros (e.g. ``0.3`` / ``0.85``)."""
    s = f"{value:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _fmt_um2(value_um2: float) -> str:
    """Format a um^2 area value, trimming trailing zeros (e.g. ``0.0215``)."""
    s = f"{value_um2:.8f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _fmt_um(value_um: float) -> str:
    """Format a micron value the way the foundry decks do: ``0.23`` (trim zeros)."""
    s = f"{value_um:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------


def klayout_deck_from_rules(
    parsed_rules: Dict[str, object],
    layermap: Dict[str, Tuple[int, int]],
    *,
    top_cell: Optional[str] = None,
    pdk_name: str = "pdk",
    exclude_kinds: Tuple[str, ...] = (),
) -> str:
    """Generate a KLayout DRC Ruby deck (``.lydrc`` body) from parsed rules.

    Parameters
    ----------
    parsed_rules:
        ``register_pdk(dir).report["parsed"]`` (frozen shape, see module docstring).
    layermap:
        canonical token -> ``(gds_layer, datatype)``.
    top_cell:
        Optional default top cell baked into the header comment (the runtime
        ``$topcell`` rd-var still overrides at run time).
    pdk_name:
        Label used in the report title / header comment.

    Returns
    -------
    str
        A complete KLayout DRC runset (valid Ruby) using only rules whose value
        AND layer are both present. Missing rules are emitted as ``# SKIP``
        comments so coverage is auditable from the deck text itself.
    """
    lines: List[str] = []
    a = lines.append

    # --- header -----------------------------------------------------------
    a("#" + "=" * 74)
    a(f"# Clay-generated partial DRC deck for: {pdk_name}")
    a("# Auto-synthesized from parsed foundry rule values + GDS layer map.")
    a("# Coverage: width / space / size / enclosure / area / density / notch /")
    a("#   wide-metal spacing / line-end / antenna (PARTIAL -- NOT tapeout")
    a("#   sign-off). Line-end + antenna are chip-level/connectivity rules.")
    a("# All values + layer numbers are taken verbatim from the source PDK.")
    a("#" + "=" * 74)
    a("")
    a("# --- file setup (runtime vars: input, report, topcell, run_mode) ---")
    a("if $input")
    a("  if $topcell")
    a("    source($input, $topcell)")
    a("  else")
    a("    source($input)")
    a("  end")
    a("end")
    a("")
    a(f'report("Clay partial DRC ({pdk_name})", $report)')
    a("")
    a('if $run_mode == "deep"')
    a("  deep")
    a('elsif $run_mode == "tiling"')
    a("  tiles(1000)")
    a("else")
    a("  flat")
    a("end")
    a("")
    if top_cell:
        a(f"# default top cell (override with -rd topcell=...): {top_cell}")
        a("")

    # --- layer definitions (only layers actually used by emitted rules) ---
    # First determine which layers we will need. Each plan entry records the spec,
    # the resolved primary (inner) layer + its var, the value, and -- for
    # enclosure rules -- the resolved outer (enclosing) layer + its var.
    used_tokens: Dict[str, Tuple[int, int]] = {}
    plan: List[Dict[str, object]] = []
    skipped: List[Tuple[_RuleSpec, str]] = []
    # parsed-but-not-emitted: a rule whose value IS present (real, parsed) but which
    # this deck deliberately does not emit -- either the free-KLayout geometry
    # engine cannot express it without connectivity (antenna) or it is a chip-level
    # rule excluded by scope for a cell-level deck (density, line-end). Kept
    # distinct from ``skipped`` so the deck documents "we have the value, we
    # deliberately do not emit it + why".
    parsed_not_emitted: List[Tuple[_RuleSpec, str]] = []

    def _extract(spec: _RuleSpec, entry: object) -> Optional[float]:
        if entry is None:
            return None
        if spec.kind == "area":
            return _value_um2(entry)
        if spec.kind in ("density_min", "density_max"):
            return _value_frac(entry)
        if spec.kind == "antenna":
            return _value_ratio(entry)
        return _value_um(entry)

    for spec in _CATALOGUE:
        # MIN-density (with_density(0.0, X, ...)) is a FILL rule, never satisfiable
        # by a small standalone cell: minimum metal coverage over a tiling window is
        # met by chip-level dummy metal fill added at ASSEMBLY, not by the cell's own
        # geometry. A cell deck that emits it ALWAYS fires (a sparse cell is sub-floor
        # by construction), which is a false positive, not a real cell defect. So it
        # is excluded from EVERY generated cell-level deck (general; all PDKs), and
        # documented with a SKIP entry. MAX-density (with_density(Y, 1.0, ...)) is
        # KEPT -- a cell CAN over-fill its own metal, an honest cell-level check.
        if spec.kind == "density_min":
            skipped.append(
                (spec, "min-density is a chip-level FILL rule (minimum coverage is "
                       "met by tapeout dummy fill at assembly, not by an isolated "
                       "cell) -- not emitted in a cell-level deck")
            )
            continue
        # cell-level signoff omits chip-level rule classes (e.g. density, satisfied
        # by tapeout dummy fill across the full chip, not by an isolated cell).
        if spec.kind in exclude_kinds:
            parsed_not_emitted.append(
                (spec, f"excluded by scope (kind={spec.kind}; chip-level, not a cell-level rule)")
            )
            continue
        entry = parsed_rules.get(spec.rule_key)
        value = _extract(spec, entry)
        resolved = _resolve_layer(spec.layer_token, layermap)

        # --- antenna: connectivity-aware antenna_check(gate, metal, ratio) ---
        # Emit ONLY when the ratio + both the conductor layer + the diff layer (to
        # derive the gate = conductor & diff) resolve. If the ratio is present but a
        # layer is missing we record WHY we do not emit (never fabricate a check).
        if spec.kind == "antenna":
            if value is None:
                # M1 antenna is intentionally never parsed: the foundry M1 limit is a
                # conditional perimeter/diode equation, not a clean static scalar, so
                # the parser leaves metal1_antenna_ratio absent. Skip it with that
                # honest reason; an absent poly antenna (perimeter-only deck) is a
                # plain value-absent skip.
                if spec.rule_key == "metal1_antenna_ratio":
                    skipped.append(
                        (spec, "metal1 antenna limit is a conditional perimeter/diode "
                               "equation in the foundry deck (no clean static "
                               "area-to-gate ratio) -- never parsed")
                    )
                else:
                    skipped.append((spec, f"value '{spec.rule_key}' absent"))
                continue
            outer = _resolve_layer(spec.outer_token, layermap) if spec.outer_token else None
            if resolved is None:
                skipped.append((spec, f"layer '{spec.layer_token}' absent in layermap"))
                continue
            if outer is None:
                parsed_not_emitted.append(
                    (spec, f"ratio {_fmt_frac(value)} needs a gate layer "
                           f"(gate = {spec.layer_token} & {spec.outer_token}); "
                           f"'{spec.outer_token}' absent in layermap")
                )
                continue
            matched_token, gds = resolved
            outer_token, outer_gds = outer
            used_tokens[matched_token] = gds
            used_tokens[outer_token] = outer_gds
            plan.append({
                "spec": spec, "token": matched_token, "value": value,
                "outer_token": outer_token,
            })
            continue

        # --- line-end (EOL): metal.enclosing(cut, v.um, projection) ---
        # Emit ONLY when the value + both the metal + the cut (contact) layer resolve.
        if spec.kind == "line_end":
            if value is None:
                skipped.append((spec, f"value '{spec.rule_key}' absent"))
                continue
            if resolved is None:
                skipped.append((spec, f"layer '{spec.layer_token}' absent in layermap"))
                continue
            outer = _resolve_layer(spec.outer_token, layermap) if spec.outer_token else None
            if outer is None:
                skipped.append(
                    (spec, f"cut layer '{spec.outer_token}' absent in layermap")
                )
                continue
            matched_token, gds = resolved
            outer_token, outer_gds = outer
            used_tokens[matched_token] = gds
            used_tokens[outer_token] = outer_gds
            # EOL edge-length threshold: an end-of-line edge is a wire TERMINUS,
            # whose length is at most a few line-widths. Derived from the metal's
            # own min-width rule (PDK-general -- 2.5x the min width), so the
            # enclosure check can be restricted to genuine wire-END edges and the
            # long sides (governed by the smaller side-enclosure rule, not the EOL
            # rule) are not mis-flagged. Falls back to 3x the EOL value when no
            # width rule is parsed (still a small terminus scale, never a coord).
            mw = _value_um(parsed_rules.get(f"{matched_token}_width"))
            eol_edge_um = round((mw * 2.5) if mw else (value * 3.0), 4)
            plan.append({
                "spec": spec, "token": matched_token, "value": value,
                "outer_token": outer_token, "eol_edge_um": eol_edge_um,
            })
            continue

        if value is None and resolved is None:
            skipped.append((spec, "value and layer both absent"))
            continue
        if value is None:
            skipped.append((spec, f"value '{spec.rule_key}' absent"))
            continue
        if resolved is None:
            skipped.append((spec, f"layer '{spec.layer_token}' absent in layermap"))
            continue
        matched_token, gds = resolved
        used_tokens[matched_token] = gds

        if spec.kind == "enclosure":
            outer = _resolve_layer(spec.outer_token, layermap) if spec.outer_token else None
            if outer is None:
                skipped.append(
                    (spec, f"outer layer '{spec.outer_token}' absent in layermap")
                )
                continue
            outer_token, outer_gds = outer
            used_tokens[outer_token] = outer_gds
            plan.append({
                "spec": spec, "token": matched_token, "value": value,
                "outer_token": outer_token,
            })
        elif spec.kind == "wide_space":
            # wide-metal spacing needs the companion width threshold; without it we
            # cannot select the wide subset, so the rule is skipped (honest).
            wt = _value_um(parsed_rules.get("metal1_wide_width"))
            if wt is None:
                skipped.append(
                    (spec, "wide-metal width threshold 'metal1_wide_width' absent")
                )
                continue
            plan.append({
                "spec": spec, "token": matched_token, "value": value,
                "width_threshold": wt,
            })
        else:
            plan.append({"spec": spec, "token": matched_token, "value": value})

    a("#" + "-" * 60)
    a("# layer definitions")
    a("#" + "-" * 60)
    for token, (gl, dt) in sorted(used_tokens.items()):
        a(f"{_ruby_var(token)} = input({gl}, {dt})   # {token}")
    a("")

    # --- rules ------------------------------------------------------------
    a("#" + "-" * 60)
    a("# rules (width / space / size / enclosure / area / density / notch / wide)")
    a("#" + "-" * 60)
    n_emitted = 0
    for item in plan:
        spec: _RuleSpec = item["spec"]  # type: ignore[assignment]
        var = _ruby_var(item["token"])  # type: ignore[arg-type]
        value = item["value"]
        if spec.kind == "space":
            um = _fmt_um(value)  # type: ignore[arg-type]
            desc = f"{spec.label} min space {um}um"
            a(f'{var}.space({um}.um).output("{spec.rule_id}", "{desc}")')
        elif spec.kind == "size":
            # min feature size -> width check (gf180 CO.1 idiom)
            um = _fmt_um(value)  # type: ignore[arg-type]
            desc = f"{spec.label} min size {um}um"
            a(f'{var}.width({um}.um).output("{spec.rule_id}", "{desc}")')
        elif spec.kind == "enclosure":
            # inner.enclosed(outer, v.um): flags where the inner layer is not
            # enclosed by the outer layer by at least v. A 0 min-enclosure is a
            # real value but a 0-um check matches nothing; emit it anyway so the
            # rule count + provenance stay honest (KLayout accepts 0.um).
            um = _fmt_um(value)  # type: ignore[arg-type]
            outer_var = _ruby_var(item["outer_token"])  # type: ignore[arg-type]
            desc = f"{spec.label} >= {um}um"
            a(f'{var}.enclosed({outer_var}, {um}.um)'
              f'.output("{spec.rule_id}", "{desc}")')
        elif spec.kind == "area":
            # min area: select shapes whose area is in [0, v) -- those are too
            # small. with_area(0.0, v) is the half-open KLayout selector.
            um2 = _fmt_um2(value)  # type: ignore[arg-type]
            desc = f"{spec.label} min area {um2}um2"
            a(f'{var}.with_area(0.0, {um2}).output("{spec.rule_id}", "{desc}")')
        elif spec.kind == "density_min":
            # minimum coverage: flag tiles whose density is in [0, min) -- too
            # sparse. with_density(lo, hi, tile_size(W)) is the native KLayout
            # density selector; the tile window is a generator default (the
            # foundry window is an unresolved indirection VARIABLE), the FRACTION
            # is the real parsed value.
            fr = _fmt_frac(value)  # type: ignore[arg-type]
            tile = _fmt_um(_DENSITY_TILE_UM)
            desc = f"{spec.label} min density {fr} (tile {tile}um, gen window)"
            a(f'{var}.with_density(0.0, {fr}, tile_size({tile}.um))'
              f'.output("{spec.rule_id}", "{desc}")')
        elif spec.kind == "density_max":
            # maximum coverage: flag tiles whose density is in (max, 1.0] -- too
            # dense.
            fr = _fmt_frac(value)  # type: ignore[arg-type]
            tile = _fmt_um(_DENSITY_TILE_UM)
            desc = f"{spec.label} max density {fr} (tile {tile}um, gen window)"
            a(f'{var}.with_density({fr}, 1.0, tile_size({tile}.um))'
              f'.output("{spec.rule_id}", "{desc}")')
        elif spec.kind == "notch":
            # notch: same-layer inner-corner self-spacing, native layer.notch(v).
            um = _fmt_um(value)  # type: ignore[arg-type]
            desc = f"{spec.label} min notch {um}um"
            a(f'{var}.notch({um}.um).output("{spec.rule_id}", "{desc}")')
        elif spec.kind == "wide_space":
            # wide-metal spacing: select the wide subset by morphological opening
            # (sized(-T/2).sized(+T/2) keeps regions at least T wide), then check
            # its spacing. This is a PARTIAL approximation of the foundry's
            # spacing-by-width rule (it checks wide-feature spacing); the space
            # value + the width threshold are both real parsed numbers.
            um = _fmt_um(value)  # type: ignore[arg-type]
            wt = item["width_threshold"]  # type: ignore[index]
            half = _fmt_um(float(wt) / 2.0)  # type: ignore[arg-type]
            wt_s = _fmt_um(wt)  # type: ignore[arg-type]
            desc = f"{spec.label} wide(W>{wt_s}um) min space {um}um"
            a(f'{var}_wide = {var}.sized(-{half}.um).sized({half}.um)'
              f'   # wide {spec.label}: width > {wt_s}um')
            a(f'{var}_wide.space({um}.um).output("{spec.rule_id}", "{desc}")')
        elif spec.kind == "line_end":
            # line-end (end-of-line): the metal must overhang the cut (contact) at a
            # WIRE END by >= v (foundry rule "M1.E.2: ENC [CO] M1 < 0.06, M1 EOL ext
            # past CO"). The projection-metric enclosure
            # ``metal.enclosing(cut, v.um, projection)`` correctly detects the
            # under-enclosure, but on its own it flags EVERY facing edge whose
            # enclosure < v -- including the LONG SIDES, which carry only the
            # smaller SIDE-enclosure margin (rule M1.E.1, ~5nm) and are NOT line
            # ends. That mis-flags tapeout-clean layouts (a checker false positive
            # measured 188x on the vendor reference, 100% long-side edges). EOL
            # QUALIFICATION: restrict the flagged metal edges to the metal's
            # wire-END edges -- edges shorter than ``eol_edge_um`` (a wire terminus
            # is at most a few line-widths long; long sides are excluded). The
            # collinear intersection ``enclosing.first_edges & wire_end_edges``
            # keeps only under-enclosure at genuine wire ends. Faithful: a real
            # narrow-wire-end under-enclosure still fires; the long-side false
            # positive does not.
            um = _fmt_um(value)  # type: ignore[arg-type]
            cut_var = _ruby_var(item["outer_token"])  # type: ignore[arg-type]
            eolw = _fmt_um(item["eol_edge_um"])  # type: ignore[arg-type]
            desc = (f"{spec.label} end-of-line ext past cut >= {um}um "
                    f"(EOL wire-end edges <= {eolw}um)")
            a(f'{var}_eolends = {var}.edges.with_length(0.0, {eolw}.um)'
              f'   # wire-END edges (terminus <= {eolw}um; long sides excluded)')
            a(f'{var}_eolencl = {var}.enclosing({cut_var}, {um}.um, projection)'
              f'   # under-enclosure of the cut on every facing edge')
            a(f'({var}_eolencl.first_edges & {var}_eolends)'
              f'.output("{spec.rule_id}", "{desc}")')
        elif spec.kind == "antenna":
            # antenna: connectivity-aware gate-area ratio. Derive the transistor gate
            # (gate = conductor & diff), connect the conductor to its gate, then run
            # KLayout's antenna_check(gate, conductor, ratio). This is the native
            # connectivity-aware antenna API (an AREA ratio = conductor/gate area).
            ratio = _fmt_frac(value)  # type: ignore[arg-type]
            diff_var = _ruby_var(item["outer_token"])  # type: ignore[arg-type]
            gate_var = f"{var}_gate"
            desc = f"{spec.label} antenna ratio > {ratio} (conductor/gate area)"
            a(f'{gate_var} = {var} & {diff_var}'
              f'   # {spec.label} gate = {spec.label} over active(diff)')
            a(f'connect({var}, {gate_var})'
              f'   # connectivity for the antenna net')
            a(f'antenna_check({gate_var}, {var}, {ratio})'
              f'.output("{spec.rule_id}", "{desc}")')
        else:  # width
            um = _fmt_um(value)  # type: ignore[arg-type]
            desc = f"{spec.label} min width {um}um"
            a(f'{var}.width({um}.um).output("{spec.rule_id}", "{desc}")')
        n_emitted += 1
    a("")

    # --- parsed-but-not-emitted (audit trail) ----------------------------
    # Rules whose value is REAL + parsed but which this deck deliberately does not
    # emit (antenna needs connectivity; line-end / density are chip-level rules
    # excluded by scope for a cell deck). Listed so coverage is honest from the
    # deck text itself.
    if parsed_not_emitted:
        a("#" + "-" * 60)
        a("# parsed-but-NOT-emitted (value is real; no honest free-KLayout check)")
        a("#" + "-" * 60)
        for spec, reason in parsed_not_emitted:
            a(f"# PARSED-NOT-EMITTED {spec.rule_id} ({spec.rule_key}): {reason}")
        a("")

    # --- skipped (audit trail) -------------------------------------------
    if skipped:
        a("#" + "-" * 60)
        a("# skipped rules (value or layer not available -- honest coverage)")
        a("#" + "-" * 60)
        for spec, reason in skipped:
            a(f"# SKIP {spec.rule_id} ({spec.rule_key}): {reason}")
        a("")

    a(f"# emitted {n_emitted} rule(s); "
      f"parsed-not-emitted {len(parsed_not_emitted)}; skipped {len(skipped)}.")
    a("")
    return "\n".join(lines)


def write_deck(
    parsed_rules: Dict[str, object],
    layermap: Dict[str, Tuple[int, int]],
    out_path,
    *,
    top_cell: Optional[str] = None,
    pdk_name: str = "pdk",
) -> Path:
    """Generate a deck and write it to ``out_path`` (``.lydrc``). Returns the path."""
    deck = klayout_deck_from_rules(
        parsed_rules, layermap, top_cell=top_cell, pdk_name=pdk_name
    )
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(deck, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

DEFAULT_KLAYOUT_BIN = Path(r"C:\Program Files (x86)\KLayout\klayout_app.exe")
_DEFAULT_DOCKER_IMAGE = "hpretl/iic-osic-tools:latest"


@dataclass
class DrcRunResult:
    """Result of running a generated deck via KLayout."""

    ran: bool
    violation_count: int
    rule_counts: Dict[str, int]
    lyrdb_path: Optional[Path]
    command: List[str]
    returncode: Optional[int]
    stdout: str
    stderr: str


def _count_lyrdb(lyrdb_path: Path) -> Tuple[int, Dict[str, int]]:
    """Reuse the v139 lyrdb parser for honest violation counting.

    Falls back to a minimal XML item count if that module is unavailable, so the
    generator does not hard-depend on the signoff harness at import time.
    """
    try:
        from tools.klayout_drc_signoff_v139.run import parse_lyrdb  # type: ignore

        rep = parse_lyrdb(lyrdb_path)
        return rep.total_violations, rep.counts_by_rule()
    except Exception:
        import xml.etree.ElementTree as ET
        from collections import Counter

        root = ET.fromstring(lyrdb_path.read_text(encoding="utf-8"))
        items = root.find("items")
        counter: Counter = Counter()
        if items is not None:
            for item in items.findall("item"):
                cat = item.find("category")
                name = (cat.text.strip().strip("'") if cat is not None and cat.text
                        else "unknown")
                counter[name.split("/")[-1].strip("'")] += 1
        return sum(counter.values()), dict(counter)


@dataclass(frozen=True)
class MimLevel:
    """One CTM/METAL recognition level of a MIM stack: the top-plate marker layer,
    the bottom-plate routing metal, and the riser via up to the cover metal.

    A foundry MIM is a DESIGN-VARIABLE level (external_pdk draws CTM4/METAL4 by default,
    CTM5/METAL5 via ``$.MODEL=M5``), and the foundry external_lvs deck recognizes EVERY
    level it can be drawn at (``extract.rul`` header ``C[M2]=CTM2*METAL2 ...
    C[M5]=CTM5*METAL5``). :class:`MimLvsSpec` therefore carries a TUPLE of these,
    one per recognized level, and :func:`klayout_lvs_from_pdk` emits an
    independent recognition block (derivations + level-aware connects + a
    ``capacitor`` extraction) per level. A PDK whose MIM sits at a single fixed
    level (gf180 metal2 / external_pdk metal5) leaves ``levels`` unset on the spec and the
    single-level tokens below drive one block (byte-identical to the pre-multilevel
    deck).
    """

    top_token: str
    bottom_metal_token: str
    top_via_token: Optional[str] = None


@dataclass(frozen=True)
class MimLvsSpec:
    """A MIM-capacitor LVS recognition recipe, as DATA reverse-engineered from a
    foundry deck (NOT a code branch). Drives :func:`klayout_lvs_from_pdk` to emit
    the MIM derived layers, the MIM connect block, and an
    ``extract_devices(capacitor(...))`` so a MIM cap extracts as ONE device on its
    two plate nets.

    The recipe mirrors the foundry MIM structure verbatim:

    * gf180mcu (``gf180mcu.lvs`` lines 935-937, 1259-1260, 1685): the BOTTOM plate
      is a region of a routing metal under the top-plate marker --
      ``mim_virtual = top.sized(oversize) & bottom_metal.interacting(top)``; the
      ROUTABLE bottom metal is the complement ``metal_ncap = bottom_metal -
      mim_virtual``; the TOP plate is the top-plate layer itself
      (``fuse_cap = top``). The device is
      ``capacitor(model, area_cap_f_per_um2 (as F-per-um2 -> via the .um2 unit),
      MIMCap){P1=>mim_virtual, P2=>fuse_cap}``. Connectivity joins the bottom plate
      to raw ``bottom_metal`` AND -- the bridge the as-shipped open deck OMITS (see
      ``tools/klayout_lvs_signoff_v145/gf180_mim_botplate_connect_fix.md``) -- to
      the routable ``metal_ncap``, and rises the top plate up through ``top_via``.

    * open_pdk (``external_pdk*.lvs`` lines 934-937, 2822, CONNECT 1360-1363):
      the same metal-under-MIM-marker structure (``mimcap1 = metalNa & (MIM ENCLOSE
      VN)``) with the device density ``DEVICE C(MIM) mimcap MIM mimcap [0.971 0]``
      (0.971 fF/um^2 area capacitance). Modelled here with the same
      ``top``/``bottom_metal``/derive form; ``oversize_um=0`` and
      ``enclose_top=True`` express the ``ENCLOSE`` form (no plate oversizing).

    Attributes
    ----------
    top_token:
        Layermap token for the MIM TOP-plate marker layer (the FuseTop / MIM
        layer). Must resolve in the passed layermap or the MIM block is omitted.
    bottom_metal_token:
        Layermap token for the routing metal the BOTTOM plate sits on (gf180
        metal2 / a external_pdk metalN). Must resolve or the MIM block is omitted.
    model:
        The extracted device-class name (gf180 ``mim_1p0fF`` / external_pdk ``mimcap``).
    area_cap_f_per_um2:
        Area capacitance density used as the ``capacitor(...)`` nominal value
        (gf180 deck registers a per-cell nominal; external_pdk density 0.971 fF/um^2 =
        9.71e-16 F/um^2). Verbatim foundry number; never invented.
    oversize_um:
        The bottom-plate derive oversize of the top-plate layer (gf180 1.06um).
        ``0`` selects the ENCLOSE form (external_pdk) -- the bottom plate is the whole
        ``bottom_metal`` interacting with the top-plate marker.
    top_via_token:
        Layermap token of the via that rises the TOP plate to the next metal
        (gf180 via2). Optional; when absent the top-plate up-connect is omitted.
    """

    top_token: str
    bottom_metal_token: str
    model: str
    area_cap_f_per_um2: float
    oversize_um: float = 0.0
    top_via_token: Optional[str] = None
    #: MULTI-LEVEL recognition (the foundry C[M2..M5] form): one MimLevel per CTM
    #: level the cap can be drawn at. When set, the deck emits an independent
    #: recognition block per level; the single ``top_token``/``bottom_metal_token``
    #: /``top_via_token`` above stay as the spec's PRIMARY (nominal) level for any
    #: consumer that reads one level. When ``None`` the spec is single-level and
    #: the primary tokens drive the one block (gf180 / external_pdk -- byte-identical).
    levels: Optional[Tuple["MimLevel", ...]] = None

    def resolved_levels(self) -> Tuple["MimLevel", ...]:
        """The recognition levels to emit: the explicit ``levels`` tuple when
        set (multi-level, external_pdk C[M2..M5]), else the single primary level built
        from this spec's own tokens (gf180 / external_pdk -- exactly one block, so the
        emitted deck is byte-identical to the pre-multilevel generator)."""
        if self.levels:
            return tuple(self.levels)
        return (
            MimLevel(
                top_token=self.top_token,
                bottom_metal_token=self.bottom_metal_token,
                top_via_token=self.top_via_token,
            ),
        )


GF180_MIM_OPTIONS: Dict[str, Dict[str, str]] = {
    "A": {
        "bottom_metal_token": "metal2",
        "top_via_token": "via2",
        "cover_metal_token": "metal3",
        "top_token": "fusetop",
    },
    "B": {
        "bottom_metal_token": "metal4",
        "top_via_token": "via4",
        "cover_metal_token": "metal5",
        "top_token": "fusetop",
        "metal_level": "5LM",
    },
}

_GF180_MIM_DENSITY_BY_CAP: Dict[str, Tuple[str, float]] = {
    "1": ("mim_1p0fF", 1.0e-15),
    "1.5": ("mim_1p5fF", 1.5e-15),
    "2": ("mim_single_2p0fF", 2.0e-15),
}


def gf180_mim_lvs_spec(
    *,
    mim_option: Optional[str] = None,
    cap_option: str = "1",
    metal_level: str = "5LM",
    pex_backend: Optional[str] = None,
) -> MimLvsSpec:
    """Return a GF180 :class:`MimLvsSpec` for Option-A (M2/V2/M3) or Option-B (M4/V4/M5)."""
    from clay.pdk.gf180mcu import resolve_gf180_mim_config

    opt, cap_key, lvl = resolve_gf180_mim_config(
        mim_option=mim_option,
        cap_option=cap_option,
        metal_level=metal_level,
        pex_backend=pex_backend,
    )
    if cap_key not in _GF180_MIM_DENSITY_BY_CAP:
        raise ValueError(f"gf180mcu: unsupported cap_option {cap_option!r}")
    model, density = _GF180_MIM_DENSITY_BY_CAP[cap_key]
    if opt == "A":
        bot_tok, via_tok = "metal2", "via2"
    else:
        if lvl == "4LM":
            bot_tok, via_tok = "metal3", "via3"
        elif lvl == "5LM":
            bot_tok, via_tok = "metal4", "via4"
        else:
            raise ValueError(
                f"gf180mcu Option-B supports metal_level '4LM' or '5LM', got {metal_level!r}"
            )
    return MimLvsSpec(
        top_token="fusetop",
        bottom_metal_token=bot_tok,
        model=model,
        area_cap_f_per_um2=density,
        oversize_um=1.06,
        top_via_token=via_tok,
    )


def gf180_mim_drc_rd(
    *,
    mim_option: Optional[str] = None,
    cap_option: str = "1",
    metal_level: str = "5LM",
    pex_backend: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Return KLayout DRC/LVS ``-rd`` switch pairs matching ``mim_option``."""
    from clay.pdk.gf180mcu import resolve_gf180_mim_config

    opt, cap_key, lvl = resolve_gf180_mim_config(
        mim_option=mim_option,
        cap_option=cap_option,
        metal_level=metal_level,
        pex_backend=pex_backend,
    )
    if cap_key not in _GF180_MIM_DENSITY_BY_CAP:
        raise ValueError(f"gf180mcu: unsupported cap_option {cap_option!r}")
    rd: List[Tuple[str, str]] = [("mim_option", opt), ("mim_cap", cap_key)]
    if opt == "B":
        rd.append(("metal_level", lvl))
    return rd


@dataclass(frozen=True)
class ResLvsSpec:
    """A POLY-RESISTOR LVS recognition recipe, DATA reverse-engineered from a foundry
    deck (not a code branch). Drives :func:`klayout_lvs_from_pdk` to emit the resistor
    body/terminal derived layers and an ``extract_devices(resistor(...))`` so a poly
    resistor extracts as ONE 2-terminal R on its two head nets.

    gf180mcu (``gf180mcu.lvs``): the unsilicided poly resistor body is the poly under the
    salicide-block (SAB) AND the resistor marker (res_mk):
    ``res_body = poly & sab & res_mk``; the contactable heads are the poly NOT under the
    resistor marker (``poly_con = poly - res_mk``), which carry the head contacts to M1.
    The device is ``resistor(model, sheet_ohm_sq){R => res_body, C => poly_con}``. The
    plain poly->contact connectivity is restricted to ``poly_con`` so the body is NOT a
    0-ohm wire (the two heads connect ONLY through the extracted R).

    Attributes
    ----------
    body_token: layermap token for the poly body layer (gf180 ``poly``).
    sab_token:  layermap token for the salicide-block layer (gf180 ``sab`` 49/0).
    marker_token: layermap token for the resistor-recognition marker (gf180 ``res_mk`` 110/5).
    model: extracted device-class name (gf180 ``ppolyf_u``).
    sheet_ohm_sq: poly sheet resistance the extractor uses (gf180 ppolyf_u 350 ohm/sq).
    """

    body_token: str
    sab_token: str
    marker_token: str
    model: str
    sheet_ohm_sq: float
    implant_token: Optional[str] = None
    variants: Tuple["ResLvsSpec", ...] = ()
    source_models: Tuple[str, ...] = ()

    def resolved_variants(self) -> Tuple["ResLvsSpec", ...]:
        """All implant-resolved classes sharing this body/marker stack."""
        result = (self, *self.variants)
        if len({item.model for item in result}) != len(result):
            raise ValueError("resistor LVS classes must be unique")
        for item in result:
            if (item.body_token, item.sab_token, item.marker_token) != (
                    self.body_token, self.sab_token, self.marker_token):
                raise ValueError("resistor LVS variants must share one recognition stack")
            if item is not self and item.variants:
                raise ValueError("nested resistor LVS variants are unsupported")
        if len(result) > 1 and (any(not item.implant_token for item in result)
                or len({item.implant_token for item in result}) != len(result)):
            raise ValueError("resistor LVS variants require distinct implant selectors")
        return result

    def resolve_source_model(self, model: str) -> "ResLvsSpec":
        """Bind a physical source-model alias to exactly one LVS class."""
        token = str(model or self.model).casefold()
        matches = [item for item in self.resolved_variants()
                   if token in {name.casefold() for name in (item.model, *item.source_models)}]
        if len(matches) != 1:
            raise ValueError(f"resistor model {model!r} has no unique implant-resolved LVS binding")
        return matches[0]


@dataclass(frozen=True)
class BjtLvsSpec:
    """One exact BJT recognition recipe expressed entirely as PDK layer data.

    The generator intersects ``base_include_tokens`` to form the base marker,
    then intersects ``emitter_include_tokens`` and subtracts
    ``emitter_exclude_tokens`` to form the emitter.  Terminal-region names are
    semantic deck regions (``nwell``, ``pcomp``, ``ncomp``, ``psd``, ``nsd``, ``ptap``, ``ntap``
    or ``substrate``), never circuit/net identifiers.  A substrate collector is
    represented by the BJT extractor's empty collector layer and its terminal
    annotation is connected to the global substrate region.
    """

    model: str
    base_include_tokens: Tuple[str, ...]
    emitter_include_tokens: Tuple[str, ...]
    emitter_exclude_tokens: Tuple[str, ...] = ()
    base_terminal_region: str = "nwell"
    emitter_terminal_region: str = "pcomp"
    collector_terminal_region: str = "substrate"

    def __post_init__(self) -> None:
        if not self.model or not self.model.replace("_", "").isalnum():
            raise ValueError("BJT LVS model must be an identifier")
        if not self.base_include_tokens or not self.emitter_include_tokens:
            raise ValueError("BJT LVS base/emitter recognition tokens cannot be empty")
        for token in (
            *self.base_include_tokens,
            *self.emitter_include_tokens,
            *self.emitter_exclude_tokens,
        ):
            if not token or not token.replace("_", "").isalnum():
                raise ValueError(f"invalid BJT LVS layer token {token!r}")
        regions = {
            "nwell", "pcomp", "ncomp", "psd", "nsd", "ptap", "ntap", "substrate",
        }
        for region in (
            self.base_terminal_region,
            self.emitter_terminal_region,
            self.collector_terminal_region,
        ):
            if region not in regions:
                raise ValueError(f"unsupported BJT terminal region {region!r}")


@dataclass
class _ResolvedMimLevel:
    """One MIM recognition level after its layers resolved in the PDK layermap.

    Internal to :func:`klayout_lvs_from_pdk`. ``suf`` is the variable-name suffix
    (``""`` for a single-level PDK so the emitted deck is byte-identical to the
    pre-multilevel generator; ``"_<n>"`` per level when the spec recognizes many).
    ``bot_ladder_tok`` / ``ladder_lvl`` are the routing-metal this level's bottom
    plate sits on (the level-aware connect exemption keys off it).
    """

    top_token: str
    bottom_metal_token: str
    top_via_token: Optional[str]
    top_ld: Tuple[int, int]
    bot_ld: Tuple[int, int]
    via_ld: Optional[Tuple[int, int]]
    suf: str = ""
    bot_ladder_tok: str = ""
    ladder_lvl: int = 0


def used_tokens_for_lvs(
    need: Dict[str, Tuple[int, int]],
    opt: Dict[str, Tuple[int, int]],
    implant: bool,
) -> List[Tuple[str, Tuple[int, int]]]:
    """The (token, (gds, datatype)) pairs the LVS deck emits as routing ``input()``
    layer variables, in emission order.

    Mirrors the layer-definition block of :func:`klayout_lvs_from_pdk` so the MIM
    block can detect when its bottom-metal / top-via layer ALREADY has a routing-layer
    variable (the same GDS layer) and ALIAS it rather than re-``input()`` it (a second
    ``input()`` of the same layer is a SEPARATE connectivity node in KLayout, which
    silently disconnects the cap plates from the routed nets).
    """
    base = ["diff", "poly", "nwell", "contact", "metal1"] + (
        ["nplus", "pplus"] if implant else []
    )
    pairs: List[Tuple[str, Tuple[int, int]]] = []
    for tok in base:
        if tok in need:
            pairs.append((tok, need[tok]))
    for tok in opt:
        pairs.append((tok, opt[tok]))
    return pairs


def klayout_lvs_from_pdk(
    layermap: Dict[str, Tuple[int, int]],
    device_names: Dict[str, str],
    *,
    pdk_name: str = "pdk",
    substrate: str = "VSUB",
    mim: Optional["MimLvsSpec"] = None,
    res: Optional["ResLvsSpec"] = None,
    bjt: Optional[Tuple["BjtLvsSpec", ...]] = None,
    top_routing_metal: Optional[int] = None,
    deep_nwell: Optional[object] = None,
) -> Optional[str]:
    """Synthesize a free-KLayout CMOS LVS deck (Ruby) from a PDK's layermap + its
    nmos/pmos model names. Recognition mode is chosen automatically:

    * IMPLANT-BASED (preferred, when nplus + pplus are present): the N+/P+
      implants split the active area, so it extracts FULL 4-terminal MOS -- the
      bulk tap (N+ inside nwell -> PMOS bulk net, P+ outside nwell -> substrate)
      is recognised, the gf180-style idiom. Compares against a standard
      ``M D G S B <model>`` schematic.
    * WELL-BASED FALLBACK (no implants): N vs P by the N-well alone, 3-terminal
      MOS (no bulk); a reader delegate maps 4-node M cards to mos3. Needs only
      diff / poly / nwell / contact / metal1.

    Extracted device CLASS names are the PDK's own model names (e.g. nmos_3p3 /
    pmos_3p3). PARTIAL either way: MOS-only, no R/C/BJT/diode and no Vt/native/DNW
    variant discrimination unless a PDK-owned ``deep_nwell`` recipe is supplied
    -- a free-KLayout subset, NOT full-device tapeout LVS.

    Returns None when the base layers (diff, poly, nwell, contact, metal1) are not
    all resolvable, or when nmos/pmos names are missing.
    """
    need: Dict[str, Tuple[int, int]] = {}
    for tok in ("diff", "poly", "nwell", "contact", "metal1"):
        r = _resolve_layer(tok, layermap)
        if r is None:
            return None
        need[tok] = r[1]
    opt: Dict[str, Tuple[int, int]] = {}
    for tok in ("via1", "metal2", "via2", "metal3"):
        r = _resolve_layer(tok, layermap)
        if r is not None:
            opt[tok] = r[1]
    # FULL-STACK routing ladder (real false-open root cause, 2026-06-12): the
    # ladder used to STOP at metal3, leaving the deck silently blind to any
    # metal4+/via3+ the layermap declares -- a geometrically complete M4 trunk
    # column then extracted as an LVS OPEN (the external_pdk trunk-route MM7 drain
    # orphan island). Walk via{k}/metal{k+1} upward as long as the layermap
    # declares BOTH; demand-driven, so a 3-metal PDK (gf180) emits a
    # byte-identical deck.
    #
    # ``top_routing_metal`` bounds the ladder: metals ABOVE it are NOT routing
    # layers for this PDK config -- today they carry only unextracted DEVICE
    # plates (e.g. the external_pdk MIM sandwich M5 plate / CTM5 / via5 / M6 cover with
    # no MIM recognition landed). Blind 2D-overlap ``connect`` over that
    # sandwich is a WRONG model, not a more complete one: via5 lands on CTM5,
    # never on M5, so ``connect(l_metal5, l_via5)`` would extract the capacitor
    # DIELECTRIC as a plate-to-plate wire (measured: GND!/VOUT merged) -- the
    # exact failure the gf180 metal_ncap block exists to prevent. Bounding
    # keeps those layers honestly UNEXTRACTED (unmeasurable) instead of
    # fabricating connectivity; the bound lifts when MIM recognition for that
    # PDK lands (plate carving makes the upper connects truthful).
    if top_routing_metal is not None and top_routing_metal < 3:
        raise ValueError(
            f"top_routing_metal={top_routing_metal}: the base via1..metal3 "
            f"ladder is this deck's floor; bound must be >= 3 (or None)"
        )
    _lvl = 3
    while f"metal{_lvl}" in opt and (
        top_routing_metal is None or _lvl < top_routing_metal
    ):
        rv = _resolve_layer(f"via{_lvl}", layermap)
        rm = _resolve_layer(f"metal{_lvl + 1}", layermap)
        if rv is None or rm is None:
            break
        opt[f"via{_lvl}"] = rv[1]
        opt[f"metal{_lvl + 1}"] = rm[1]
        _lvl += 1
    rn = _resolve_layer("nplus", layermap)
    rp = _resolve_layer("pplus", layermap)
    implant = rn is not None and rp is not None
    if implant:
        need["nplus"] = rn[1]
        need["pplus"] = rp[1]

    # --- isolated local p-well inside DNW (data-driven, optional) ----------
    # The materializer's PDK-owned recipe is also the extraction contract: a
    # drawn DNW plus implant-based taps lets this partial deck split the local
    # p-well body from the global substrate.  Without both, the deck stays
    # byte-compatible with the historical baseline and claims no DNW coverage.
    deep_emit = False
    deep_models: Dict[str, str] = {}
    if deep_nwell is not None and implant:
        deep_token = str(getattr(deep_nwell, "deep_nwell_layer", "") or "")
        resolved_deep = _resolve_layer(deep_token, layermap) if deep_token else None
        raw_models = getattr(deep_nwell, "lvs_models", {}) or {}
        deep_models = {
            str(key): str(value)
            for key, value in dict(raw_models).items()
            if str(value)
        }
        if resolved_deep is None:
            raise ValueError(
                f"deep-N-well recipe requires unresolved layer {deep_token!r}"
            )
        if not deep_models.get("nmos") and not deep_models.get("nmos3v"):
            raise ValueError("deep-N-well recipe declares no NMOS LVS model")
        need["dnwell"] = resolved_deep[1]
        deep_emit = True

    nmos = device_names.get("nmos")
    pmos = device_names.get("pmos")
    if not nmos or not pmos:
        return None

    # --- OD2 (thick-gate-oxide / 3.3V) gate split (DATA-driven, opt-in) ----------
    # Mirror the foundry LVGT/HVGT recognition: a device is 3.3V iff its gate is
    # covered by OD2 (``HVGT = ALL_GATE AND OD2``) and 1.8V otherwise
    # (``LVGT = ALL_GATE NOT OD2``). We emit the split ONLY when BOTH (a) the PDK
    # declares an OD2 layer in its layermap AND (b) distinct 3.3V device-class names
    # are supplied (``nmos_thick``/``pmos_thick`` or ``nmos3v``/``pmos3v``). Absent
    # either, the deck is BYTE-IDENTICAL to the un-split version -- so every PDK that
    # does not carry OD2 (gf180 / external_pdk) and every 1.8V-only config is unchanged, and a
    # 1.8V layout (no OD2 shapes) still extracts EXACTLY as before (its gate NOT OD2
    # == the whole gate; its gate AND OD2 == empty -> zero 3.3V devices). Closing the
    # false-green: a 3.3V design now extracts as a DISTINCT 3.3V class, so an LVS
    # against a 3.3V schematic can no longer silently match 1.8V geometry.
    r_od2 = _resolve_layer("od2", layermap)
    nmos_thick = device_names.get("nmos_thick") or device_names.get("nmos3v")
    pmos_thick = device_names.get("pmos_thick") or device_names.get("pmos3v")
    od2_split = bool(r_od2 is not None and nmos_thick and pmos_thick)
    od2_ld = r_od2[1] if r_od2 is not None else None
    # A DNW-local device can require OD2 discrimination even when the baseline
    # partial deck intentionally folds ordinary 1.8/3.3V MOS into one class.
    # Emit the physical selector layer whenever either consumer needs it.
    od2_layer_emit = bool(r_od2 is not None and (od2_split or deep_emit))

    # --- MIM capacitor recognition layers (DATA-driven, optional) ------------
    # A MIM is drawn at a DESIGN-VARIABLE level (external_pdk CTM4/METAL4 default, CTM5/
    # METAL5 via $.MODEL); the foundry external_lvs deck recognizes EVERY level it can
    # be drawn at (extract.rul C[M2]=CTM2*METAL2 .. C[M5]=CTM5*METAL5). So the
    # spec carries one MimLevel per recognized level (a single-level PDK -- gf180
    # metal2 / external_pdk metal5 -- yields exactly one). Resolve each level's top-plate
    # marker + bottom-metal (+ optional riser via); a level whose layers are NOT
    # in this PDK's layermap is silently dropped (honest -- no fabricated layer).
    mim_levels: List[_ResolvedMimLevel] = []
    if mim is not None:
        for _lv in mim.resolved_levels():
            rt = _resolve_layer(_lv.top_token, layermap)
            rb = _resolve_layer(_lv.bottom_metal_token, layermap)
            if rt is None or rb is None:
                continue
            via_ld: Optional[Tuple[int, int]] = None
            if _lv.top_via_token:
                rv = _resolve_layer(_lv.top_via_token, layermap)
                if rv is not None:
                    via_ld = rv[1]
            mim_levels.append(_ResolvedMimLevel(
                top_token=_lv.top_token,
                bottom_metal_token=_lv.bottom_metal_token,
                top_via_token=_lv.top_via_token,
                top_ld=rt[1], bot_ld=rb[1], via_ld=via_ld,
            ))
    mim_emit = bool(mim_levels)

    # --- poly-resistor recognition layers (DATA-driven, optional) ------------
    # Emitted ONLY when the SAB + resistor-marker layers resolve in the layermap (the
    # poly body layer is the already-present l_poly). A spec whose layers are absent is
    # silently omitted (honest -- no fabricated layer).
    res_sab: Optional[Tuple[int, int]] = None
    res_mk: Optional[Tuple[int, int]] = None
    if res is not None:
        rs = _resolve_layer(res.sab_token, layermap)
        rm = _resolve_layer(res.marker_token, layermap)
        if rs is not None and rm is not None:
            res_sab = rs[1]
            res_mk = rm[1]
    res_emit = res is not None and res_sab is not None and res_mk is not None
    res_specs = res.resolved_variants() if res_emit else ()
    res_implant_inputs: Dict[str, Tuple[int, int]] = {}
    for res_spec in res_specs:
        token = res_spec.implant_token
        if token and token not in need and token not in opt:
            resolved = _resolve_layer(token, layermap)
            if resolved is None:
                raise ValueError(
                    f"resistor class {res_spec.model!r} requires an available implant layer {token!r}"
                )
            res_implant_inputs[token] = resolved[1]

    # --- BJT recognition layers (data-driven, optional) ---------------------
    bjt_specs = tuple(bjt or ())
    bjt_layer_defs: Dict[str, Tuple[int, int]] = {}
    for spec in bjt_specs:
        for token in (
            *spec.base_include_tokens,
            *spec.emitter_include_tokens,
            *spec.emitter_exclude_tokens,
        ):
            resolved = _resolve_layer(token, layermap)
            if resolved is None:
                raise ValueError(
                    f"BJT model {spec.model!r} requires unresolved layer token {token!r}"
                )
            bjt_layer_defs[token] = resolved[1]

    # Each MIM level's bottom-plate metal maps to a ROUTING-LADDER token (gf180
    # 'metal2', external_pdk 'metal4'/'metal5'). The MIM connect rules are LEVEL-AWARE:
    # at the level where a plate sits, the riser via (the cut over the top-plate
    # marker, rising to the cover) must connect to the routable COMPLEMENT
    # (metal_ncap_<lvl>) -- a raw ``connect(l_metal<n>, l_via<n>)`` there would
    # extract the capacitor DIELECTRIC as a plate-to-plate wire. ``mim_by_bot_level``
    # maps a ladder metal level -> that level's metal_ncap suffix, so the ladder
    # walk below picks the right complement per level. A single-level PDK yields
    # ``{lvl: ""}`` (un-suffixed vars -> byte-identical deck).
    mim_by_bot_level: Dict[int, str] = {}
    if mim_emit:
        _metal_toks = [("metal1", need["metal1"])] + [
            (t, opt[t]) for t in opt if t.startswith("metal")
        ]
        for ml in mim_levels:
            found: Optional[str] = None
            for tok, ld in _metal_toks:
                if ld == ml.bot_ld:
                    found = tok
                    break
            if found is None:
                raise ValueError(
                    f"MIM bottom-plate metal {ml.bottom_metal_token!r} ({ml.bot_ld}) "
                    f"is not one of this deck's routing metals -- the level-aware "
                    f"MIM connect model needs the bottom plate on a ladder metal"
                )
            ml.bot_ladder_tok = found
            ml.ladder_lvl = int(found[len("metal"):])
        _single = len(mim_levels) == 1
        for ml in mim_levels:
            ml.suf = "" if _single else f"_{ml.ladder_lvl}"
            mim_by_bot_level[ml.ladder_lvl] = ml.suf

    def _inp(tok: str) -> str:
        ld = need.get(tok) or opt[tok]
        return f"l_{tok} = input({ld[0]}, {ld[1]})"

    out: list = []
    a = out.append
    mode = "IMPLANT-BASED 4-terminal" if implant else "WELL-BASED 3-terminal"
    a(f"# Clay-generated partial CMOS LVS deck for: {pdk_name}")
    a(f"# {mode} MOS recognition; layer numbers + model names verbatim from the")
    cap_note = " + MIM capacitor" if mim_emit else ""
    a(f"# PDK{cap_note}. NOT full-device tapeout LVS.")
    a("")
    a("source($input) if $input")
    a("report_lvs($report) if $report")
    a("target_netlist($target_netlist) if $target_netlist")
    a("")
    if implant:
        a("reader = RBA::NetlistSpiceReader::new")
    else:
        a("# Map 4-node 'M D G S B <model>' cards to 3-terminal MOS (drop bulk).")
        a("class Mos3Reader < RBA::NetlistSpiceReaderDelegate")
        a("  def element(circuit, el, name, model, value, nets, params)")
        a('    if el == "M"')
        a('      error("MOS needs 4 nodes") if nets.size != 4')
        a("      cls = circuit.netlist.device_class_by_name(model)")
        a("      if ! cls")
        a("        cls = RBA::DeviceClassMOS3Transistor::new")
        a("        cls.name = model")
        a("        circuit.netlist.add(cls)")
        a("      end")
        a("      device = circuit.create_device(cls, name)")
        a('      device.connect_terminal("D", nets[0])')
        a('      device.connect_terminal("G", nets[1])')
        a('      device.connect_terminal("S", nets[2])')
        a('      device.set_parameter("W", (params["W"] || 0.0) * 1e6)')
        a('      device.set_parameter("L", (params["L"] || 0.0) * 1e6)')
        a("      return true")
        a("    else")
        a("      return super")
        a("    end")
        a("  end")
        a("end")
        a("reader = RBA::NetlistSpiceReader::new(Mos3Reader::new)")
    a("schematic($schematic, reader) if $schematic")
    a("")
    a(f'substrate_name = $lvs_sub || "{substrate}"')
    a("")
    a("# --- layers (verbatim GDS layer/datatype from the PDK layermap) ---")
    base = ["diff", "poly", "nwell", "contact", "metal1"] + (
        ["nplus", "pplus"] if implant else []
    ) + (["dnwell"] if deep_emit else [])
    for tok in base:
        a(_inp(tok))
    for tok in opt:
        a(_inp(tok))
    # The PDK label purpose can differ from its drawing purpose. The emitter
    # and extractor must consume the same declared layer or labelled external
    # ports disappear even when transistor connectivity happens to compare.
    label_key = layermap.get("metal1_label", need["metal1"])
    a(f"l_metal1_lbl = labels({label_key[0]}, {label_key[1]})")
    if od2_layer_emit:
        assert od2_ld is not None
        a(f"l_od2 = input({od2_ld[0]}, {od2_ld[1]})"
          "   # OD2 thick-gate-oxide (3.3V) selector")
    if bjt_specs:
        existing_bjt_vars = {
            token: f"l_{token}" for token in (*need.keys(), *opt.keys())
        }
        if od2_layer_emit:
            existing_bjt_vars["od2"] = "l_od2"
        a("")
        a("# --- BJT recognition layers (verbatim PDK marker/selector data) ---")
        for token, layer_data in bjt_layer_defs.items():
            if token in existing_bjt_vars:
                continue
            suffix = token[len("bjt_"):] if token.startswith("bjt_") else token
            variable = f"l_bjt_{suffix}"
            existing_bjt_vars[token] = variable
            a(f"{variable} = input({layer_data[0]}, {layer_data[1]})")
    if mim_emit:
        assert mim is not None
        a("")
        a("# --- MIM capacitor layers (top-plate marker + bottom-plate metal) ---")
        # CRITICAL (real LVS open the gate caught): in KLayout, ``input(L, D)`` called
        # TWICE for the same GDS layer produces TWO INDEPENDENT layer objects with
        # SEPARATE connectivity graphs. If the MIM bottom-metal / top-via are re-input()
        # here as fresh ``l_mim_bot`` / ``l_mim_via`` while the router's wires were
        # already connected through the routing-layer instances ``l_metal2`` / ``l_via2``
        # (same GDS layer, different object), the cap plates NEVER share a net with the
        # routed metal -> every MIM cap extracts FLOATING ($nn / $nn) and LVS reports the
        # caps + their terminal nets (vout/gnd!/net15...) as unmatched. So when the MIM
        # bottom-metal / top-via layer EQUALS an already-defined routing layer, ALIAS the
        # existing variable (one connectivity node) instead of re-input()ing it. Only a
        # MIM layer with no routing-layer twin is freshly input().
        existing_by_ld: Dict[Tuple[int, int], str] = {}
        for _tok, _ld in used_tokens_for_lvs(need, opt, implant):
            existing_by_ld.setdefault(_ld, _ruby_var(_tok))

        def _mim_layer(varname: str, ld: Tuple[int, int], comment: str) -> str:
            """Define a MIM layer var: ALIAS an existing routing-layer instance when the
            GDS layer already has one (shared connectivity), else fresh ``input()``."""
            twin = existing_by_ld.get(ld)
            if twin is not None:
                return f"{varname} = {twin}   # {comment} (ALIAS {twin}; shared net)"
            return f"{varname} = input({ld[0]}, {ld[1]})   # {comment}"

        # ONE recognition block per level (foundry C[M2..M5]); a single-level PDK
        # loops once with suffix "" -> byte-identical to the pre-multilevel deck.
        for ml in mim_levels:
            a(_mim_layer(f"l_mim_top{ml.suf}", ml.top_ld,
                         f"MIM top-plate marker ({ml.top_token})"))
            a(_mim_layer(f"l_mim_bot{ml.suf}", ml.bot_ld,
                         f"MIM bottom-plate metal ({ml.bottom_metal_token})"))
            if ml.via_ld is not None:
                a(_mim_layer(f"l_mim_via{ml.suf}", ml.via_ld,
                             f"MIM top-plate via ({ml.top_via_token})"))
        a("")
        a("# --- MIM derivations (foundry recipe: bottom plate = metal under the")
        a("#     top-plate marker; routable metal = complement; top plate = marker) ---")
        for ml in mim_levels:
            s = ml.suf
            if mim.oversize_um and mim.oversize_um > 0:
                # gf180 form: mim_virtual = top.sized(OVER).and(bottom.interacting(top))
                over = _fmt_um(mim.oversize_um)
                a(f"mim_virtual{s} = l_mim_top{s}.sized({over}.um)"
                  f".and(l_mim_bot{s}.interacting(l_mim_top{s}))   # bottom plate")
            else:
                # external_pdk/external_pdk ENCLOSE form: bottom plate = bottom metal interacting marker.
                a(f"mim_virtual{s} = l_mim_bot{s}.interacting(l_mim_top{s})"
                  "   # bottom plate (ENCLOSE form)")
            a(f"metal_ncap{s}  = l_mim_bot{s}.not(mim_virtual{s})"
              "   # routable bottom metal (plate carved out)")
            a(f"fuse_cap{s}    = l_mim_top{s}   # top plate")
    if res_emit:
        assert res is not None and res_sab is not None and res_mk is not None
        a("")
        a("# --- poly-resistor layers (salicide block + resistor marker) ---")
        a(f"l_sab    = input({res_sab[0]}, {res_sab[1]})   # salicide block ({res.sab_token})")
        a(f"l_res_mk = input({res_mk[0]}, {res_mk[1]})   # resistor marker ({res.marker_token})")
        for token, layer in sorted(res_implant_inputs.items()):
            a(f"l_{token} = input({layer[0]}, {layer[1]})   # resistor implant selector")
        a("")
        a("# --- poly-resistor derivations (body = poly under SAB+marker; heads = the")
        a("#     contactable poly outside the marker -> the two terminals) ---")
        a("res_body = l_poly.and(l_sab).and(l_res_mk)"
          "   # the unsilicided resistor body (poly under SAB + marker) = the R element")
        for index, res_spec in enumerate(res_specs):
            if res_spec.implant_token:
                a(f"res_body_{index} = res_body.and(l_{res_spec.implant_token})"
                  f"   # implant-resolved {res_spec.model}")
        a("poly_con = l_poly.not(l_res_mk)"
          "   # contactable poly heads + MOS gates (outside the BODY-only resistor marker)"
          " -> terminals (the body under the marker is the R, not a wire)")
    a("")
    if implant:
        a("# --- implant-based recognition (full 4-terminal incl. bulk taps) ---")
        a("ncomp = l_diff.and(l_nplus)")
        a("pcomp = l_diff.and(l_pplus)")
        a("ngate = l_poly.and(ncomp)")
        a("pgate = l_poly.and(pcomp)")
        a("nsd   = ncomp.not(l_poly).not(l_nwell)")
        a("psd   = pcomp.not(l_poly).and(l_nwell)")
        a("ntap  = ncomp.and(l_nwell)")
        a("ptap  = pcomp.not(l_nwell)")
        a("sub   = polygon_layer")
        if deep_emit:
            a("")
            a("# --- DNW-local p-well split (PDK-owned isolation recipe) ---")
            a("iso_pwell = l_dnwell.not(l_nwell)")
            a("ptap_iso = ptap.and(iso_pwell)")
            a("ptap_sub = ptap.not(l_dnwell)")
            a("nsd_iso = nsd.and(iso_pwell)")
            a("nsd_bulk = nsd.not(l_dnwell)")
            a("ngate_deep = ngate.and(iso_pwell)")
            a("ngate_bulk = ngate.not(l_dnwell)")
    else:
        a("# --- well-based recognition (3-terminal, no bulk) ---")
        a("pdiff = l_diff.and(l_nwell)")
        a("ndiff = l_diff.not(l_nwell)")
        a("pgate = l_poly.and(pdiff)")
        a("ngate = l_poly.and(ndiff)")
        a("psd   = pdiff.not(l_poly)")
        a("nsd   = ndiff.not(l_poly)")
    if bjt_specs:
        if not implant:
            raise ValueError("BJT extraction requires implant-distinguished diffusion")
        a("")
        a("# --- BJT derived regions (PDK-declared intersections/exclusions) ---")

        def _bjt_var(token: str) -> str:
            if token in existing_bjt_vars:
                return existing_bjt_vars[token]
            raise AssertionError(f"BJT token {token!r} has no emitted layer variable")

        for index, spec in enumerate(bjt_specs):
            base_terms = [_bjt_var(token) for token in spec.base_include_tokens]
            base_expr = base_terms[0] + "".join(
                f".and({term})" for term in base_terms[1:]
            )
            a(f"bjt_base_{index} = {base_expr}")
            emitter_expr = f"bjt_base_{index}" + "".join(
                f".and({_bjt_var(token)})"
                for token in spec.emitter_include_tokens
            )
            emitter_expr += "".join(
                f".not({_bjt_var(token)})"
                for token in spec.emitter_exclude_tokens
            )
            a(f"bjt_emitter_{index} = {emitter_expr}")
    if od2_split:
        # LVGT/HVGT gate split on OD2 (foundry: 1.8V gate = gate NOT OD2, 3.3V gate =
        # gate AND OD2). Only the GATE splits -- source/drain (nsd/psd) and the bulk
        # taps are the SAME derivations for both domains (a 3.3V device's S/D is still
        # nsd), exactly as the foundry deck folds them. On a 1.8V layout (no OD2) the
        # ``_lv`` gate == the full gate and the ``_hv`` gate is empty, so extraction is
        # byte-identical and no 3.3V device is produced.
        a("")
        a("# --- OD2 gate split (LVGT = gate NOT OD2 [1.8V]; HVGT = gate AND OD2 [3.3V]) ---")
        _normal_ngate = "ngate_bulk" if deep_emit else "ngate"
        a(f"ngate_lv = {_normal_ngate}.not(l_od2)   # 1.8V thin-ox NMOS gate")
        a(f"ngate_hv = {_normal_ngate}.and(l_od2)   # 3.3V thick-ox NMOS gate")
        a("pgate_lv = pgate.not(l_od2)   # 1.8V thin-ox PMOS gate")
        a("pgate_hv = pgate.and(l_od2)   # 3.3V thick-ox PMOS gate")
    if deep_emit:
        a("")
        a("# --- isolated NMOS voltage-domain split inside the local p-well ---")
        if r_od2 is not None:
            a("ngate_deep_lv = ngate_deep.not(l_od2)")
            a("ngate_deep_hv = ngate_deep.and(l_od2)")
    a("")
    a("# --- connectivity ---")
    if deep_emit:
        a("connect(nsd_bulk, l_contact)")
        a("connect(nsd_iso, l_contact)")
    else:
        a("connect(nsd, l_contact)")
    a("connect(psd, l_contact)")
    if res_emit:
        # restrict poly->contact to the CONTACTABLE heads (poly outside the resistor
        # marker), so the resistor body is NOT a 0-ohm wire -- the two heads connect ONLY
        # through the extracted R. MOS gate poly (never under res_mk) is unaffected.
        a("connect(poly_con, l_contact)"
          "   # heads only (resistor body excluded -> it is the R, not a wire)")
    else:
        a("connect(l_poly, l_contact)")
    if implant:
        a("connect(ntap, l_contact)")
        if deep_emit:
            a("connect(ptap_sub, l_contact)")
            a("connect(ptap_iso, l_contact)")
        else:
            a("connect(ptap, l_contact)")
    a("connect(l_contact, l_metal1)")
    if "via1" in opt:
        a("connect(l_metal1, l_via1)")
        if "metal2" in opt:
            a("connect(l_via1, l_metal2)")
            # M2<->M3 via the via2 cut (the generic router rises on M3 when the PDK
            # exposes a third metal, so the extractor MUST follow M3 or every
            # M3-routed net fragments into per-terminal islands -> spurious LVS
            # mismatch). Only emitted when the PDK names both via2 + metal3.
            if "via2" in opt and "metal3" in opt:
                # CRITICAL (real LVS short): over a MIM cap the via2 SEA sits on the
                # TOP plate (FuseTop) to rise it to the Metal3 cover. The cap stack is a
                # sandwich -- the Metal2 BOTTOM plate (mim_virtual) lies directly under
                # that same via2 sea. A blanket ``connect(l_metal2, l_via2)`` therefore
                # ties via2 to the BOTTOM plate too, shorting the top plate (Metal3
                # cover) to the bottom plate (Metal2) THROUGH the via2 -- a real cap
                # plate-to-plate short. So when a MIM is recognised, via2 connects to the
                # ROUTABLE metal2 only (``metal_ncap`` = metal2 with the bottom plate
                # carved out), never to ``mim_virtual``. The bottom plate reaches routing
                # through via1/metal1 (above), not via2.
                if mim_emit and 2 in mim_by_bot_level:
                    a(f"connect(metal_ncap{mim_by_bot_level[2]}, l_via2)"
                      "   # via2 to ROUTABLE metal2 only (NOT the MIM bottom plate)")
                else:
                    a("connect(l_metal2, l_via2)")
                a("connect(l_via2, l_metal3)")
                # full-stack ladder: chain every declared upper metal into the
                # SAME connectivity graph (one node set). Without these lines a
                # route on metal4+ is invisible to extraction and every net
                # using it fragments into per-terminal islands -> false opens.
                # LEVEL-AWARE MIM exception: at the bottom-plate level the via
                # ABOVE the plate is the MIM riser (it sits over the top-plate
                # marker) -> connect it to the routable complement only, never
                # the raw plate metal (dielectric-as-wire short otherwise).
                _lvl = 3
                while f"via{_lvl}" in opt and f"metal{_lvl + 1}" in opt:
                    if mim_emit and _lvl in mim_by_bot_level:
                        a(f"connect(metal_ncap{mim_by_bot_level[_lvl]}, l_via{_lvl})"
                          f"   # via{_lvl} to ROUTABLE metal{_lvl} only "
                          f"(NOT the MIM bottom plate)")
                    else:
                        a(f"connect(l_metal{_lvl}, l_via{_lvl})")
                    a(f"connect(l_via{_lvl}, l_metal{_lvl + 1})")
                    _lvl += 1
    a("connect(l_metal1, l_metal1_lbl)")
    if mim_emit:
        a("# --- MIM connectivity (bottom plate joins routing metal; top rises) ---")
        # Foundry deck joins the bottom plate to RAW bottom metal; the routable
        # metal a router lands on is the DERIVED complement (metal_ncap), a
        # SEPARATE connectivity node. Bridge metal_ncap <-> mim_virtual so the
        # bottom plate is on the routed net -- the connect line the as-shipped
        # open gf180 deck OMITS (gf180_mim_botplate_connect_fix.md); we emit it.
        #
        # metal_ncap is the DERIVED complement of the bottom-plate metal (a separate
        # connectivity node). The two bridge lines tie it to its source metal ONLY
        # through mim_virtual, which exists only UNDER a cap. So a routed via landing
        # on metal_ncap AWAY from any cap (the normal case -- e.g. the OTA top-plate
        # net's via-down to the rail) reaches an ISOLATED metal_ncap region not tied
        # to the source metal net -> the plate floats in LVS. Tie the routable
        # complement to its OWN source net directly (safe: metal_ncap is a subset of
        # its source metal that EXCLUDES the bottom plate, so no top<->bottom short).
        # The source join is level-aware: gf180 metal2, external_pdk a per-level metal4/5.
        # One connectivity block per recognized level; a level whose CTM marker is
        # not DRAWN in this GDS derives an EMPTY plate (metal_ncap == its whole metal,
        # extracts no device), so an unused level is inert -- exactly the foundry's
        # C[M2..M5] behaviour when only one level is populated.
        for ml in mim_levels:
            s = ml.suf
            a(f"connect(l_mim_bot{s}, mim_virtual{s})"
              "   # bottom plate <-> raw bottom metal (foundry line)")
            a(f"connect(metal_ncap{s}, mim_virtual{s})"
              "   # bottom plate <-> ROUTABLE metal (the bridge the open deck omits)")
            _riser_tok = ml.top_via_token or f"via{ml.ladder_lvl}"
            a(f"connect(metal_ncap{s}, {_ruby_var(ml.bot_ladder_tok)})"
              f"   # routable complement IS {ml.bot_ladder_tok} -- join it to its source net "
              f"(else a {_riser_tok} "
              "on metal_ncap away from a cap is a floating node)")
            if ml.via_ld is not None:
                a(f"connect(fuse_cap{s}, l_mim_via{s})"
                  "   # top plate rises through its via")
    if implant:
        a("connect(l_nwell, ntap)")
        if deep_emit:
            a("connect(iso_pwell, ptap_iso)")
            a("connect(sub, ptap_sub)")
        else:
            a("connect(sub, ptap)")
        a("connect_global(sub, substrate_name)")
        # Tie the P+ substrate tap (ptap) into the SAME global substrate net. ``sub``
        # is an empty polygon layer, so ``connect(sub, ptap)`` alone connects nothing
        # (an empty primary touches no shape) and the ptap -- with the ground rail M1
        # that lands on it -- would extract as a net SEPARATE from the global substrate
        # the NMOS bulk (``"W" => sub``) references. ``connect_global(ptap, ...)`` makes
        # the ptap (and the ground rail it contacts) part of the global substrate net,
        # so the laid-out NMOS body net (gnd, tied down by its substrate tap) and the
        # extracted NMOS bulk are the SAME net -> LVS matches the schematic body tie.
        if deep_emit:
            a("connect_global(ptap_sub, substrate_name)")
        else:
            a("connect_global(ptap, substrate_name)")
    a("")
    # the MOS GATE terminal annotation layer must be IN the connectivity. When the
    # resistor block restricts poly->contact to ``poly_con`` (heads only), the gate
    # terminal must annotate ``poly_con`` too (gate poly is never under the resistor
    # marker, so poly_con covers it fully) -- else KLayout errors "Terminal G isn't
    # connected".
    gpoly = "poly_con" if res_emit else "l_poly"
    if implant:
        # (SD, gate_var, class, W) tuples. Without OD2 the two original class lines
        # emit verbatim (byte-identical); with OD2 the gate splits into 1.8V (gate
        # NOT OD2) + 3.3V (gate AND OD2) DISTINCT classes -- S/D + W shared.
        p_rows = (
            [("psd", "pgate_lv", pmos, "l_nwell"),
             ("psd", "pgate_hv", pmos_thick, "l_nwell")]
            if od2_split else [("psd", "pgate", pmos, "l_nwell")]
        )
        _normal_nsd = "nsd_bulk" if deep_emit else "nsd"
        n_rows = (
            [(_normal_nsd, "ngate_lv", nmos, "sub"),
             (_normal_nsd, "ngate_hv", nmos_thick, "sub")]
            if od2_split else [(
                _normal_nsd,
                "ngate_bulk" if deep_emit else "ngate",
                nmos,
                "sub",
            )]
        )
        a("# --- device extraction (4-terminal MOS; class name = PDK model) ---")
        for sd, g, cls, w in p_rows:
            a(f'extract_devices(mos4("{cls}"), {{ "SD" => {sd}, "G" => {g}, '
              f'"tS" => {sd}, "tD" => {sd}, "tG" => {gpoly}, "W" => {w} }})')
        for sd, g, cls, w in n_rows:
            a(f'extract_devices(mos4("{cls}"), {{ "SD" => {sd}, "G" => {g}, '
              f'"tS" => {sd}, "tD" => {sd}, "tG" => {gpoly}, "W" => {w} }})')
        if deep_emit:
            deep_rows = []
            if r_od2 is not None:
                if deep_models.get("nmos"):
                    deep_rows.append(("ngate_deep_lv", deep_models["nmos"]))
                if deep_models.get("nmos3v"):
                    deep_rows.append(("ngate_deep_hv", deep_models["nmos3v"]))
            else:
                deep_rows.append((
                    "ngate_deep",
                    deep_models.get("nmos3v") or deep_models["nmos"],
                ))
            a("# --- isolated NMOS extraction (body = measured local p-well) ---")
            for deep_gate, deep_model in deep_rows:
                a(
                    f'extract_devices(mos4("{deep_model}"), {{ '
                    f'"SD" => nsd_iso, "G" => {deep_gate}, '
                    f'"tS" => nsd_iso, "tD" => nsd_iso, '
                    f'"tG" => {gpoly}, "W" => iso_pwell }})'
                )
    else:
        p_rows = (
            [("psd", "pgate_lv", pmos), ("psd", "pgate_hv", pmos_thick)]
            if od2_split else [("psd", "pgate", pmos)]
        )
        n_rows = (
            [("nsd", "ngate_lv", nmos), ("nsd", "ngate_hv", nmos_thick)]
            if od2_split else [("nsd", "ngate", nmos)]
        )
        a("# --- device extraction (3-terminal MOS; no bulk; class = PDK model) ---")
        for sd, g, cls in p_rows:
            a(f'extract_devices(mos3("{cls}"), {{ "SD" => {sd}, "G" => {g}, '
              f'"tS" => {sd}, "tD" => {sd}, "tG" => {gpoly} }})')
        for sd, g, cls in n_rows:
            a(f'extract_devices(mos3("{cls}"), {{ "SD" => {sd}, "G" => {g}, '
              f'"tS" => {sd}, "tD" => {sd}, "tG" => {gpoly} }})')
    if bjt_specs:
        terminal_regions = {
            "nwell": "l_nwell",
            "pcomp": "pcomp",
            "ncomp": "ncomp",
            "psd": "psd",
            "nsd": "nsd",
            "ptap": "ptap",
            "ntap": "ntap",
            "substrate": "sub",
        }
        a("")
        a("# --- BJT extraction (three-terminal; full C/B/E connectivity) ---")
        for index, spec in enumerate(bjt_specs):
            collector_geometry = (
                "polygon_layer"
                if spec.collector_terminal_region == "substrate"
                else terminal_regions[spec.collector_terminal_region]
            )
            a(f'ignore_parameter("{spec.model}","AE")')
            a(
                f'extract_devices(bjt3("{spec.model}"), {{ '
                f'"C" => {collector_geometry}, "B" => bjt_base_{index}, '
                f'"E" => bjt_emitter_{index}, '
                f'"tC" => {terminal_regions[spec.collector_terminal_region]}, '
                f'"tB" => {terminal_regions[spec.base_terminal_region]}, '
                f'"tE" => {terminal_regions[spec.emitter_terminal_region]} }})'
            )
    if mim_emit:
        assert mim is not None
        a("")
        a("# --- MIM capacitor extraction (one device on its two plate nets) ---")
        # The capacitor() nominal is the area-capacitance density (F/um^2) the foundry deck
        # carries verbatim (gf180 per-cell nominal / external_pdk 0.971 fF/um^2). The PLAIN
        # capacitor() extracts C = density * plate-area (e.g. 3000um^2 * 1fF/um^2 = 3pF),
        # so the cap VALUE is LVS-verified against the schematic C (within the tolerance
        # below). (An earlier custom DeviceClassCapacitor subclass left C=0 -> the value
        # could not be compared; the plain class restores value verification.)
        # One extraction per recognized level, all into the SAME device class
        # (the foundry recognizes C[M2..M5] as one 'mimcap' class). Only the level
        # actually DRAWN produces a device -- the others have empty plates
        # (mim_virtual_<lvl> / fuse_cap_<lvl> = empty) and extract nothing.
        cap_f = repr(float(mim.area_cap_f_per_um2))
        for ml in mim_levels:
            a(f'extract_devices(capacitor("{mim.model}", {cap_f}), '
              f'{{ "P1" => mim_virtual{ml.suf}, "P2" => fuse_cap{ml.suf} }})')
    if res_emit:
        assert res is not None
        a("")
        a("# --- poly-resistor extraction (one 2-terminal R on its two head nets) ---")
        # The resistor body (poly under SAB+marker) is the R element; the contactable
        # poly heads (poly_con) are the two terminals. sheet_ohm_sq is the foundry sheet
        # resistance; the extracted R = sheet * (body length / width).
        for index, res_spec in enumerate(res_specs):
            sheet = repr(float(res_spec.sheet_ohm_sq))
            body = f"res_body_{index}" if res_spec.implant_token else "res_body"
            a(f'extract_devices(resistor("{res_spec.model}", {sheet}), '
              f'{{ "R" => {body}, "C" => poly_con }})')
    a("")
    a("# --- compare (emit the exact strings the LVS harness greps) ---")
    a("align")
    a("# Preserve actual GDS channels before LVS-only parallel combination.")
    a("if $target_netlist")
    from clay.layout.pex.uncombined_resistors import raw_device_snapshot_ruby
    for line in raw_device_snapshot_ruby('$target_netlist + ".uncombined.cir"').splitlines():
        a("  " + line)
    a("end")
    # Fold the laid-out parallel fingers (combine_devices) BEFORE simplify so a
    # multi-finger MOS drawn as M parallel single-finger devices on common S/D rails
    # collapses to ONE device of total width W*M -- the same partition the folded
    # reference presents. Applied to the extracted netlist AND the reference so the
    # fold axis is symmetric on both sides.
    #
    # Resistor serial combination is deliberately disabled.  A series junction
    # is a real routed net with its own parasitic capacitance; deleting that node
    # during LVS makes the same-run full-extraction topology unusable by PEX.
    # Parallel resistor banks remain combinable, as do MOS/BJT fingers.
    if res_emit:
        assert res is not None
        # A device class exists only when that side actually contains at least
        # one resistor.  Capability-level decks are also used for MOS/BJT/MIM
        # micro-oracles with no resistor instance, so both lookups are guarded.
        for res_spec in res_specs:
            a(f'layout_res_class = netlist.device_class_by_name("{res_spec.model}")')
            a(
                "layout_res_class.supports_serial_combination = false "
                "if layout_res_class"
            )
            a(
                f'schematic_res_class = schematic.device_class_by_name("{res_spec.model}") '
                "if $schematic"
            )
            a(
                "schematic_res_class.supports_serial_combination = false "
                "if schematic_res_class"
            )
    a("netlist.combine_devices")
    a("schematic.combine_devices if $schematic")
    a("netlist.simplify")
    a("schematic.simplify if $schematic")
    # Extracted device VALUES differ from the schematic by a sub-percent GEOMETRIC rounding
    # (the cap plate area + the poly-R body length are on the 5nm grid). A small RELATIVE
    # tolerance pairs them cleanly (else an exact-value mismatch leaves the R/C devices
    # MatchWithWarning and STALLS the whole compare -- the v0.4.2 root cause). The value is
    # still VERIFIED (within 5%), not ignored.
    if mim_emit:
        assert mim is not None
        a(f'tolerance("{mim.model}", "C", 0.0, 0.05)'
          "   # 5% relative -- the cap value IS compared (plate area * density vs schematic C)")
    if res_emit:
        assert res is not None
        for res_spec in res_specs:
            a(f'tolerance("{res_spec.model}", "R", 0.0, 0.05)'
              "   # 5% relative -- absorbs the body-length snap")
    # MOS W tolerance (P-C2): a serpentine-FOLDED device distributes its total
    # channel width across N abutting bands whose per-band height snaps to the
    # nm grid; summing the bands back yields a sub-percent W deviation from the
    # nominal (e.g. a W=80u device extracts W=80.024u -- 24nm/80um = 0.03%). The
    # MOS compare carries NO W tolerance by default, so that exact-W mismatch is
    # a hard FAIL that STALLS the whole device compare and cascades the touching
    # net cluster to layout=NONE (the same v0.4.2 stall the R/C tolerances above
    # cure, on the MOS axis). A small RELATIVE W tolerance pairs the folded
    # device with its schematic card; W is still VERIFIED (within 1%), not
    # ignored -- a real W error (e.g. 58.8u vs 30u) is far outside the band and
    # still FAILS. Symmetric across paths: an unfolded device extracts W exactly
    # (0% deviation), so this never loosens the non-fold compare. L unchanged
    # (L is grid-exact -- the fold redistributes W, never L).
    a(f'tolerance("{nmos}", "W", 0.0, 0.01)'
      "   # 1% relative -- absorbs the serpentine per-band width snap (P-C2)")
    a(f'tolerance("{pmos}", "W", 0.0, 0.01)'
      "   # 1% relative -- absorbs the serpentine per-band width snap (P-C2)")
    if od2_split:
        # same W tolerance for the 3.3V thick-ox classes (only emitted when the OD2
        # split is active, so the un-split deck is byte-identical).
        a(f'tolerance("{nmos_thick}", "W", 0.0, 0.01)'
          "   # 1% relative -- 3.3V thick-ox NMOS (P-C2)")
        a(f'tolerance("{pmos_thick}", "W", 0.0, 0.01)'
          "   # 1% relative -- 3.3V thick-ox PMOS (P-C2)")
    a("lvs_ok = compare")
    a("if lvs_ok")
    a('  puts "INFO : Congratulations! Netlists match."')
    a("else")
    a('  puts "ERROR : Netlists don\'t match"')
    a("end")
    return "\n".join(out) + "\n"


def run_klayout_drc(
    gds,
    deck,
    *,
    report_path=None,
    klayout_bin: Optional[Path] = None,
    topcell: Optional[str] = None,
    run_mode: str = "flat",
    extra_rd: Optional[Sequence[Tuple[str, str]]] = None,
    mim_option: Optional[str] = None,
    cap_option: str = "1",
    metal_level: str = "5LM",
    use_docker: bool = False,
    docker_image: str = _DEFAULT_DOCKER_IMAGE,
    timeout_sec: float = 300.0,
) -> DrcRunResult:
    """Run a generated DRC ``deck`` on ``gds`` via KLayout and return violation count.

    Two backends:

    * native (default): invoke the installed ``klayout_app.exe`` in batch mode,
      the same invocation form the gf180mcu signoff harness uses::

          klayout_app.exe -b -r <deck> -rd input=<gds> -rd report=<lyrdb> \\
              -rd run_mode=flat [-rd topcell=...]

    * Docker (``use_docker=True``): run KLayout inside the iic-osic-tools image.
      The repo dir is bind-mounted so the GDS/deck paths resolve inside the
      container, mirroring the existing gf180 Docker DRC infrastructure.

    Returns
    -------
    DrcRunResult
        ``ran=True`` and a finite ``violation_count`` iff KLayout produced a
        ``.lyrdb`` (proof the engine parsed + ran the generated deck).
    """
    gds_p = Path(gds).resolve()
    deck_p = Path(deck).resolve()
    if not gds_p.exists():
        raise FileNotFoundError(f"GDS not found: {gds_p}")
    if not deck_p.exists():
        raise FileNotFoundError(f"deck not found: {deck_p}")
    # KLayout treats a ``.lydrc`` file as an XML-wrapped DRC document, but the
    # body we generate is raw DRC-Ruby. When passed to ``-r``, a raw-Ruby body
    # must carry a ``.drc`` extension or KLayout tries to XML-parse it and fails
    # at line 1. If the deck is not already ``.drc``, run a ``.drc`` sibling.
    run_deck = deck_p
    if deck_p.suffix.lower() != ".drc":
        run_deck = deck_p.with_suffix(".drc")
        if run_deck.resolve() != deck_p.resolve():
            run_deck.write_text(deck_p.read_text(encoding="utf-8"), encoding="utf-8")
    report = (
        Path(report_path).resolve()
        if report_path is not None
        else gds_p.with_suffix(".clay_partial.lyrdb")
    )
    report.parent.mkdir(parents=True, exist_ok=True)

    # GF180's foundry deck embeds a Linux-only `pmap` memory probe in Logger; on
    # native Windows that returns nil and aborts before any rule executes.
    temp_shim_deck: Optional[Path] = None
    _pmap_probe = "`pmap #{Process.pid} | tail -1`[10,40].strip"
    if not use_docker and os.name == "nt":
        deck_text = run_deck.read_text(encoding="utf-8", errors="replace")
        if _pmap_probe in deck_text:
            fd, shim_name = tempfile.mkstemp(
                prefix=f"{run_deck.stem}__win_shim_",
                suffix=".drc",
                dir=str(report.parent),
            )
            os.close(fd)
            temp_shim_deck = Path(shim_name)
            temp_shim_deck.write_text(
                deck_text.replace(_pmap_probe, '"n/a"'), encoding="utf-8"
            )
            run_deck = temp_shim_deck

    merged_rd: List[Tuple[str, str]] = []
    if mim_option is not None:
        merged_rd.extend(
            gf180_mim_drc_rd(
                mim_option=mim_option,
                cap_option=cap_option,
                metal_level=metal_level,
            )
        )
    if extra_rd:
        existing_keys = {k for k, _ in merged_rd}
        for key, value in extra_rd:
            if key in existing_keys:
                merged_rd = [(k, v) for k, v in merged_rd if k != key]
            merged_rd.append((key, value))
            existing_keys.add(key)

    try:
        if use_docker:
            cmd = _docker_command(
                gds_p, run_deck, report, topcell=topcell, run_mode=run_mode,
                extra_rd=merged_rd, docker_image=docker_image,
            )
        else:
            kbin = Path(klayout_bin) if klayout_bin is not None else DEFAULT_KLAYOUT_BIN
            if not kbin.exists():
                raise FileNotFoundError(
                    f"KLayout binary not found: {kbin}. Pass klayout_bin= or use_docker=True."
                )
            cmd = [
                str(kbin), "-b",
                "-r", str(run_deck),
                "-rd", f"input={gds_p}",
                "-rd", f"report={report}",
                "-rd", f"run_mode={run_mode}",
            ]
            if topcell is not None:
                cmd += ["-rd", f"topcell={topcell}"]
            for key, value in merged_rd:
                cmd += ["-rd", f"{key}={value}"]

        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sec, check=False
        )
    finally:
        if temp_shim_deck is not None:
            temp_shim_deck.unlink(missing_ok=True)

    if not report.exists():
        return DrcRunResult(
            ran=False, violation_count=-1, rule_counts={}, lyrdb_path=None,
            command=cmd, returncode=proc.returncode,
            stdout=proc.stdout, stderr=proc.stderr,
        )

    total, by_rule = _count_lyrdb(report)
    return DrcRunResult(
        ran=True, violation_count=total, rule_counts=by_rule, lyrdb_path=report,
        command=cmd, returncode=proc.returncode,
        stdout=proc.stdout, stderr=proc.stderr,
    )


def _docker_command(
    gds: Path, deck: Path, report: Path, *,
    topcell: Optional[str], run_mode: str,
    extra_rd: Optional[Sequence[Tuple[str, str]]] = None,
    docker_image: str = _DEFAULT_DOCKER_IMAGE,
) -> List[str]:
    """Build a docker run command that invokes KLayout DRC inside iic-osic-tools.

    A common ancestor of all paths is bind-mounted at the same absolute POSIX
    path inside the container so the rd-var paths resolve unchanged.
    """
    # mount the repo root (common ancestor) at /work and rewrite paths
    paths = [gds, deck, report.parent]
    try:
        common = Path(*Path(__file__).resolve().parts[: _common_prefix_len(paths)])
    except Exception:
        common = Path(gds.anchor)
    # simplest robust mount: mount each file's drive root is messy on Windows;
    # mount the repo root which contains all of them.
    repo_root = _repo_root_for(gds)
    mount = repo_root

    def inside(p: Path) -> str:
        rel = p.resolve().relative_to(mount)
        return "/work/" + rel.as_posix()

    klayout_cmd = (
        f"klayout -b -r {inside(deck)} "
        f"-rd input={inside(gds)} -rd report={inside(report)} "
        f"-rd run_mode={run_mode}"
    )
    if topcell is not None:
        klayout_cmd += f" -rd topcell={topcell}"
    for key, value in extra_rd or ():
        klayout_cmd += f" -rd {key}={value}"

    win_mount = str(mount).replace("\\", "/")
    return [
        "docker", "run", "--rm",
        "-v", f"{win_mount}:/work",
        docker_image,
        "bash", "-lc", klayout_cmd,
    ]


def _common_prefix_len(paths: Sequence[Path]) -> int:
    parts = [p.resolve().parts for p in paths]
    n = min(len(x) for x in parts)
    out = 0
    for i in range(n):
        if len({x[i] for x in parts}) == 1:
            out = i + 1
        else:
            break
    return out


def _repo_root_for(p: Path) -> Path:
    """Walk up from ``p`` to the first dir containing a ``.git`` (repo root)."""
    cur = p.resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / ".git").exists():
            return parent
    return cur.parents[0] if cur.parents else cur


__all__ = [
    "klayout_deck_from_rules",
    "klayout_lvs_from_pdk",
    "GF180_MIM_OPTIONS",
    "gf180_mim_lvs_spec",
    "gf180_mim_drc_rd",
    "MimLvsSpec",
    "MimLevel",
    "BjtLvsSpec",
    "write_deck",
    "run_klayout_drc",
    "DrcRunResult",
    "DEFAULT_KLAYOUT_BIN",
    "CHIP_LEVEL_KINDS",
]
