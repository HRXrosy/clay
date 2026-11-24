"""Machine-readable gf180mcu design rules, extracted from the real DRC deck.

Source of truth (every value below is transcribed verbatim, not approximated):
    AnalogGym/PDK/gf180mcu_pdk/libraries/gf180mcu_fd_pr/latest/rules/klayout/
    drc/gf180mcu.drc

The KLayout DRC deck encodes each rule as a comment line of the form::

    # Rule PL.1_3.3V: Interconnect Width (outside PLFUSE). is 0.18µm

followed by a ``layer.width(0.18.um, ...)`` / ``layer.space(...)`` /
``layer.enclosing(...)`` call carrying the same micron literal.

This table captures the canonical 3.3V (core analog) flavour of the key
analog layers (Poly2/gate, COMP/diffusion, Contact, Metal1/2/3, Nwell).
Where the deck splits a rule into 3.3V and 5V variants we record the 3.3V
number as the default and keep the 5V number alongside for reference.

Units in the deck are micrometres; accessors return nanometres
(microns * 1000) as integers to avoid floating point drift.

Every entry cites its rule-id and the source line in the deck so the value
can be re-verified against the deck at any time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# Path of the deck the values below were extracted from (repo-relative).
DECK_RELPATH = (
    "AnalogGym/PDK/gf180mcu_pdk/libraries/gf180mcu_fd_pr/latest/"
    "rules/klayout/drc/gf180mcu.drc"
)


@dataclass(frozen=True)
class Rule:
    """A single design-rule value extracted from the gf180mcu DRC deck.

    Attributes:
        rule_id: Deck rule identifier (e.g. "PL.1_3.3V", "CO.1", "M1.1").
        value_um: The micron literal exactly as written in the deck.
        deck_line: 1-based line number of the ``# Rule ...`` comment.
        description: Human-readable rule description from the deck.
    """

    rule_id: str
    value_um: float
    deck_line: int
    description: str

    @property
    def value_nm(self) -> int:
        """Value in integer nanometres (microns * 1000)."""
        # Round to avoid 0.18 * 1000 == 179.999... style float drift.
        return int(round(self.value_um * 1000.0))


@dataclass(frozen=True)
class LayerRules:
    """Min-width and min-spacing rules for one logical layer.

    Enclosures are keyed by the *inner* layer they enclose so that e.g. the
    COMP layer can record "COMP enclosure of contact" (CO.4) and "COMP
    extension beyond gate" (DF.6).
    """

    layer_name: str
    min_width: Rule
    min_spacing: Rule
    # Map inner-layer-token -> enclosure Rule. Inner token is upper-cased,
    # e.g. "CONTACT", "GATE".
    enclosures: Dict[str, Rule] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extracted rule table. Every value below is a verbatim transcription of the
# 3.3V (core analog) flavour from the deck; see deck_line for the source.
# ---------------------------------------------------------------------------

# POLY2 / gate -------------------------------------------------------------
# PL.1_3.3V (line 1441): poly2 interconnect width 0.18um.
# PL.3a_3.3V (line 1489): poly2 space on COMP/Field 0.24um.
# PL.4_3.3V (line 1507): poly2 extension beyond COMP to form end cap 0.22um.
# PL.2_3.3V (line 1465): gate width (channel length) 0.28um -> recorded as
#   the "GATE" enclosure-ish gate-length min on poly over the channel.
_POLY = LayerRules(
    layer_name="POLY",
    min_width=Rule("PL.1_3.3V", 0.18, 1441, "Interconnect Width (outside PLFUSE)."),
    min_spacing=Rule("PL.3a_3.3V", 0.24, 1489, "Space on COMP/Field."),
    enclosures={
        # Poly2 extension beyond COMP (poly end cap over diffusion).
        "COMP": Rule("PL.4_3.3V", 0.22, 1507,
                     "Extension beyond COMP to form Poly2 end cap."),
    },
)

# Gate channel-length minimum (PL.2). Kept separate because it is not a
# generic poly width; it is the drawn poly width where it crosses COMP.
GATE_LENGTH = Rule("PL.2_3.3V", 0.28, 1465, "Gate Width (Channel Length).")

# COMP / diffusion ---------------------------------------------------------
# DF.1a_3.3V (line 960): min COMP width 0.22um.
# DF.3a_3.3V (line 1018): min COMP space 0.28um.
# CO.4 (line 2284): COMP overlap (enclosure) of contact 0.07um.
# DF.6_3.3V (line 1179): COMP extend beyond gate (S/D overhang) 0.24um.
_COMP = LayerRules(
    layer_name="COMP",
    min_width=Rule("DF.1a_3.3V", 0.22, 960, "Min. COMP Width."),
    min_spacing=Rule("DF.3a_3.3V", 0.28, 1018, "Min. COMP Space."),
    enclosures={
        "CONTACT": Rule("CO.4", 0.07, 2284, "COMP overlap of contact."),
        # COMP extend beyond gate == source/drain overhang past the gate.
        "GATE": Rule("DF.6_3.3V", 0.24, 1179,
                     "Min. COMP extend beyond gate (source/drain overhang)."),
    },
)

# CONTACT ------------------------------------------------------------------
# CO.1 (line 2247): min/max contact size 0.22um (used here as min width).
# CO.2a (line 2253): min contact spacing 0.25um.
# CO.3 (line 2274): Poly2 overlap of contact 0.07um (poly enclosure of CO).
# CO.4 (line 2284): COMP overlap of contact 0.07um (comp enclosure of CO).
_CONTACT = LayerRules(
    layer_name="CONTACT",
    min_width=Rule("CO.1", 0.22, 2247, "Min/max contact size."),
    min_spacing=Rule("CO.2a", 0.25, 2253, "Min. contact spacing."),
    enclosures={
        # Enclosures of contact *by* an outer layer, recorded here for
        # convenience; also mirrored on the outer layer (POLY/COMP).
        "POLY": Rule("CO.3", 0.07, 2274, "Poly2 overlap of contact."),
        "COMP": Rule("CO.4", 0.07, 2284, "COMP overlap of contact."),
    },
)
# Poly2 enclosure of contact (CO.3) lives on the poly layer too.
_POLY.enclosures["CONTACT"] = Rule(
    "CO.3", 0.07, 2274, "Poly2 overlap of contact."
)

# METAL1 -------------------------------------------------------------------
# M1.1 (line 2394): min metal1 width 0.23um.
# M1.2a (line 2400): min metal1 spacing 0.23um.
_METAL1 = LayerRules(
    layer_name="M1",
    min_width=Rule("M1.1", 0.23, 2394, "Min. metal1 width."),
    min_spacing=Rule("M1.2a", 0.23, 2400, "Min. metal1 spacing."),
)

# METAL2 -------------------------------------------------------------------
# M2.1 (line 2422): min metal2 width 0.28um.
# M2.2a (line 2428): min metal2 spacing 0.28um.
_METAL2 = LayerRules(
    layer_name="M2",
    min_width=Rule("M2.1", 0.28, 2422, "Min. metal2 width."),
    min_spacing=Rule("M2.2a", 0.28, 2428, "Min. metal2 spacing."),
)

# METAL3 -------------------------------------------------------------------
# M3.1 (line 2450): min metal3 width 0.28um.
# M3.2a (line 2456): min metal3 spacing 0.28um.
_METAL3 = LayerRules(
    layer_name="M3",
    min_width=Rule("M3.1", 0.28, 2450, "Min. metal3 width."),
    min_spacing=Rule("M3.2a", 0.28, 2456, "Min. metal3 spacing."),
)

# NWELL --------------------------------------------------------------------
# NW.1a_3.3V (line 827): min Nwell width 0.86um.
# NW.2a_3.3V (line 859): min Nwell space (equi-potential) 0.6um.
#   NB: NW.2b (different potential) is 1.4um; we record the equi-potential
#   value as min_spacing and keep 2b reachable via NWELL_SPACE_DIFF_POT.
_NWELL = LayerRules(
    layer_name="NWELL",
    min_width=Rule("NW.1a_3.3V", 0.86, 827, "Min. Nwell Width."),
    min_spacing=Rule("NW.2a_3.3V", 0.6, 859,
                     "Min. Nwell Space (equi-potential)."),
)

# Different-potential Nwell spacing (NW.2b_3.3V, line 871): 1.4um.
NWELL_SPACE_DIFF_POT = Rule(
    "NW.2b_3.3V", 1.4, 871,
    "Min. Nwell Space (Outside DNWELL) [Different potential]."
)


# 5V (dualgate) variants of the same rules, for reference / cross-check.
# These are NOT the default returned by the accessors below.
FIVE_VOLT_VARIANTS: Dict[str, Rule] = {
    "POLY.width": Rule("PL.1_5V", 0.2, 1447,
                       "Interconnect Width (outside PLFUSE)."),
    "GATE.length": Rule("PL.2_5V", 0.5, 1475, "Gate Width (Channel Length)."),
    "COMP.width": Rule("DF.1a_5V", 0.3, 966, "Min. COMP Width."),
    "COMP.space": Rule("DF.3a_5V", 0.36, 1024, "Min. COMP Space."),
    "COMP.gate_overhang": Rule("DF.6_5V", 0.4, 1185,
                               "Min. COMP extend beyond gate."),
}


# Master registry, keyed by upper-cased logical layer name.
GF180_RULES: Dict[str, LayerRules] = {
    "POLY": _POLY,
    "COMP": _COMP,
    "CONTACT": _CONTACT,
    "M1": _METAL1,
    "M2": _METAL2,
    "M3": _METAL3,
    "NWELL": _NWELL,
}

# Aliases so callers using diffusion/gate/poly2/contact tokens still resolve.
_LAYER_ALIASES: Dict[str, str] = {
    "POLY2": "POLY",
    "GATE": "POLY",
    "DIFF": "COMP",
    "DIFFUSION": "COMP",
    "CO": "CONTACT",
    "CONT": "CONTACT",
    "METAL1": "M1",
    "METAL2": "M2",
    "METAL3": "M3",
    "NW": "NWELL",
}


def _normalize_layer(layer: str) -> str:
    key = layer.upper()
    key = _LAYER_ALIASES.get(key, key)
    if key not in GF180_RULES:
        raise KeyError(
            f"Unknown gf180mcu layer {layer!r}; known layers: "
            f"{sorted(GF180_RULES)} (aliases: {sorted(_LAYER_ALIASES)})"
        )
    return key


def get_layer_rules(layer: str) -> LayerRules:
    """Return the full LayerRules entry for a logical layer (case-insensitive)."""
    return GF180_RULES[_normalize_layer(layer)]


def min_width_nm(layer: str) -> int:
    """Return the minimum drawn width of ``layer`` in nanometres.

    Args:
        layer: Logical layer name or alias (e.g. "poly", "comp", "m1",
            "diff", "nwell"); case-insensitive.

    Raises:
        KeyError: If the layer is unknown.
    """
    return get_layer_rules(layer).min_width.value_nm


def min_spacing_nm(layer: str) -> int:
    """Return the minimum same-layer spacing of ``layer`` in nanometres.

    For NWELL this is the equi-potential value (NW.2a); the different-
    potential value is available via ``NWELL_SPACE_DIFF_POT``.

    Raises:
        KeyError: If the layer is unknown.
    """
    return get_layer_rules(layer).min_spacing.value_nm


def enclosure_nm(outer: str, inner: str) -> int:
    """Return the enclosure of ``inner`` by ``outer`` in nanometres.

    Examples:
        ``enclosure_nm("comp", "contact")`` -> 70  (CO.4, 0.07um)
        ``enclosure_nm("poly", "comp")``    -> 220 (PL.4, poly end cap)
        ``enclosure_nm("comp", "gate")``    -> 240 (DF.6, S/D overhang)

    Args:
        outer: Enclosing layer name or alias.
        inner: Enclosed layer/feature token (e.g. "contact", "comp", "gate").

    Raises:
        KeyError: If the outer layer is unknown or the (outer, inner)
            enclosure pair is not recorded.
    """
    rules = get_layer_rules(outer)
    inner_key = inner.upper()
    inner_key = _LAYER_ALIASES.get(inner_key, inner_key)
    # Allow GATE to map back to its own token rather than POLY for enclosures.
    if inner.upper() == "GATE":
        inner_key = "GATE"
    if inner.upper() in ("CONTACT", "CO", "CONT"):
        inner_key = "CONTACT"
    if inner_key not in rules.enclosures:
        raise KeyError(
            f"No recorded enclosure of {inner!r} by {outer!r}; "
            f"available inner tokens for {rules.layer_name}: "
            f"{sorted(rules.enclosures)}"
        )
    return rules.enclosures[inner_key].value_nm


def get_rule(rule_id: str) -> Optional[Rule]:
    """Return the Rule object for a given deck rule-id, or None if absent.

    Searches min_width, min_spacing, enclosures across all layers plus the
    standalone gate-length / different-potential-nwell rules.
    """
    standalone = (GATE_LENGTH, NWELL_SPACE_DIFF_POT)
    for r in standalone:
        if r.rule_id == rule_id:
            return r
    for lr in GF180_RULES.values():
        for r in (lr.min_width, lr.min_spacing, *lr.enclosures.values()):
            if r.rule_id == rule_id:
                return r
    return None


def all_rules() -> Tuple[Rule, ...]:
    """Return every distinct Rule recorded in this module."""
    seen: Dict[str, Rule] = {}
    for r in (GATE_LENGTH, NWELL_SPACE_DIFF_POT):
        seen[r.rule_id] = r
    for lr in GF180_RULES.values():
        for r in (lr.min_width, lr.min_spacing, *lr.enclosures.values()):
            seen[r.rule_id] = r
    return tuple(seen.values())
