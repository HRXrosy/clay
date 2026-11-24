"""R174 (BL-174A) -- structured PDK-rule container for Clay.

Clay's gf180mcu design-rule numbers (min width / spacing / enclosure, contact /
via sizes) are presently hard-coded as module-level ``GF180_*_NM`` constants
scattered across many layout modules -- ``gf180_device_geom``,
``coclean_5tota_v163``, ``coclean_cascode_v163``, ``magical_comparator_v171``,
``substrate_tap_tie``, ``m2_crossing_route_v163``, and ~20 more. A single
hand-typed rule that is wrong for the process is the root of a whole bug class
(the historic ``fin_pitch`` 48/280 nm error is the canonical example), and the
scatter blocks a principled solver: a MILP placer or A* router cannot consume
rules that live as literals inside geometry builders.

This module is the single queryable source of truth those tools will read. It
follows the TED vPDK pattern (a structured container sourced from the real
foundry deck, every downstream tool queries it, swap PDK by data not code) but
is a fresh reimplementation for gf180mcu:

  * :class:`LayerRule` -- an interned (flyweight) per-(layer, datatype) object
    exposing ``min_width`` / ``min_spacing`` / ``min_area`` /
    ``enclosure(other)`` and an optional width-dependent spacing table.
  * :class:`ViaRule` -- contact / via size + spacing + lower/upper metal
    enclosure, also interned.
  * :class:`PdkRules` -- the container keyed by a PDK id (default
    ``gf180mcu``). It holds the layermap (name -> (layer, datatype)) and the
    LayerRule / ViaRule set and exposes the query API ``rule_for`` /
    ``min_spacing`` / ``min_width`` / ``min_area`` / ``enclosure`` / ``via`` /
    ``layer``.

Data, not code
==============
Every rule carries a :class:`Provenance`:

  * ``PARSED_DECK``  -- the value was read at construction from the REAL
    ``gf180mcu.drc`` rule decks (the ``.output("ID", "... : 0.XXum")``
    descriptions). When the decks are present on disk this is the source.
  * ``CONSOLIDATED`` -- the value was seeded from Clay's existing consolidated
    constants (e.g. ``gf180_device_geom.GF180_CONTACT_SIZE_NM``) and
    cross-checked against the deck / known gf180 numbers. Used when the deck
    expression is not cleanly parseable as a single scalar (enclosures encoded
    inside ``enclosing(...)`` calls rather than the description string).
  * ``DOCUMENTED``   -- the value is a documented PDK number with no single
    clean deck literal (track pitches, derived constants).

The honesty contract is in the data: ``provenance_breakdown()`` reports exactly
how many rules came from each source, and ``rule_for(...).source`` /
``via(...).source`` is queryable per rule. The deck is parsed when available;
nothing is fabricated -- a value that cannot be parsed is seeded from the
consolidated constant and labelled as such.

Second PDK by data
===================
:class:`PdkRules` is constructed from a :class:`PdkData` bundle (layermap +
LayerRule set + ViaRule set). ``PdkRules.gf180mcu()`` builds the fully populated
gf180mcu bundle; ``PdkRules.sky130()`` shows the SAME container accepts a second
PDK as data (a different layermap + a partially populated rule set seeded from
``layer_pitch.SKY130_LAYER_PITCH``) with no code change. Adding or swapping a
PDK is a data operation.

All linear dimensions are stored in nanometres (integers, dbu = 1 nm for both
gf180mcu and sky130) to avoid floating-point drift; areas are stored in nm^2.
"""
from __future__ import annotations

import enum
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple


# --- provenance --------------------------------------------------------------


class Provenance(enum.Enum):
    """Where a rule value came from. Encoded in the data so the container is an
    honest single-source-of-truth, never a silent divergent copy."""

    PARSED_DECK = "parsed-deck"
    CONSOLIDATED = "consolidated"
    DOCUMENTED = "documented"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class RuleValue:
    """One scalar design-rule value (nm or nm^2) with its provenance + rule id.

    Attributes:
        value: the value in nm (lengths) or nm^2 (areas).
        rule_id: the originating foundry rule id (e.g. ``"M1.1"``, ``"CO.4"``).
        source: where the value came from (parsed deck / consolidated / doc).
    """

    value: int
    rule_id: str
    source: Provenance


# --- deck parser -------------------------------------------------------------

# The gf180mcu KLayout DRC decks describe each rule with an ``.output`` call
# whose second argument is a human-readable description that ENDS in the rule's
# numeric value, e.g.::
#
#     m11_l1.output("M1.1", "M1.1 : min. metal1 width : 0.23um")
#     co1_l1.output("CO.1", "CO.1 : Min/max contact size. : 0.22um")
#     v11_l1.output("V1.1", "V1.1 : Min/max Via1 size . : 0.26um")
#     df9_l1.output("DF.9_3.3V", "DF.9_3.3V : Min. COMP area (um2). : 0.2025um2")
#
# (The deck source uses the micro sign before "m"; we match the unit loosely.)
# This regex captures (rule_id, value_um, is_area). Rules whose description has
# no trailing scalar (pure-topology checks like CO.9 / DF.12) are skipped.
_OUTPUT_RE = re.compile(
    r"""\.output\(\s*"(?P<id>[^"]+)"\s*,\s*"
        [^"]*?:\s*
        (?:nil\s*,\s*)?                # DF.2a style "nil,0.22um" lower-open band
        (?P<val>\d+(?:\.\d+)?)\s*
        (?:\xb5|u|\xc2\xb5)?m          # micron unit (ascii u, micro sign, utf-8)
        (?P<area>2|\xb2|\^2)?          # optional area marker (um2 / um^2)
        \s*"\s*\)""",
    re.VERBOSE,
)


def _um_to_nm(um: float) -> int:
    return int(round(um * 1000.0))


def _um2_to_nm2(um2: float) -> int:
    return int(round(um2 * 1_000_000.0))


def parse_deck_rules(deck_dir: str) -> Dict[str, Tuple[int, bool]]:
    """Parse the gf180mcu ``rule_decks/*.drc`` files into ``{rule_id: (value, is_area)}``.

    ``value`` is nm for length rules and nm^2 for area rules; ``is_area`` flags
    the latter. Only rules whose ``.output`` description ends in a scalar are
    captured (pure-topology rules carry no number and are skipped). When the
    same rule id appears more than once (e.g. ``M1.2a`` is not, but some are
    duplicated across 3.3V/5V banks) the FIRST occurrence wins.

    Returns an empty dict if ``deck_dir`` does not exist (the container then
    falls back entirely to the consolidated constants -- see
    :meth:`PdkRules.gf180mcu`).
    """
    out: Dict[str, Tuple[int, bool]] = {}
    if not deck_dir or not os.path.isdir(deck_dir):
        return out
    for name in sorted(os.listdir(deck_dir)):
        if not name.endswith(".drc"):
            continue
        path = os.path.join(deck_dir, name)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:  # pragma: no cover - defensive
            continue
        for m in _OUTPUT_RE.finditer(text):
            rid = m.group("id")
            if rid in out:
                continue
            val = float(m.group("val"))
            is_area = m.group("area") is not None
            out[rid] = (_um2_to_nm2(val) if is_area else _um_to_nm(val), is_area)
    return out


# repo root: clay/layout/pdk_rules_v174.py -> parents[2]
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
# The gf180mcu rule_decks ship in two trees: the in-tree open-PDK install under
# pdks/ (present in this environment) and the historic AnalogGym vendored copy.
# Resolve to the first that exists so rule provenance parses from the real deck
# regardless of host layout; AnalogGym kept as fallback. Same foundry runset.
_GF180_RULE_DECKS_REL = os.path.join(
    "libraries", "gf180mcu_fd_pr", "latest", "rules", "klayout", "drc", "rule_decks",
)
_GF180_DECK_DIR_CANDIDATES = (
    os.path.join(_REPO_ROOT, "pdks", "gf180mcu", _GF180_RULE_DECKS_REL),
    os.path.join(_REPO_ROOT, "AnalogGym", "PDK", "gf180mcu_pdk", _GF180_RULE_DECKS_REL),
)
DEFAULT_GF180_DECK_DIR = next(
    (d for d in _GF180_DECK_DIR_CANDIDATES if os.path.isdir(d)),
    _GF180_DECK_DIR_CANDIDATES[0],
)


# --- interned layer / via rules ----------------------------------------------


@dataclass(frozen=True)
class WidthSpacingEntry:
    """One row of a width-dependent spacing table: shapes at least ``wide_ge``
    nm wide (and/or run-length >= ``run_length_ge``) require ``spacing`` nm."""

    wide_ge: int
    spacing: RuleValue
    run_length_ge: int = 0
    # Optional exact merged-geometry contract. A scalar width table alone does
    # not define how narrow features attached to a wide polygon are classified.
    # These dimensions describe the foundry opening and attached-metal halo;
    # consumers must not infer a component width from its bounding box.
    opening_nm: Optional[int] = None
    attached_halo_nm: Optional[int] = None


class LayerRule:
    """Per-layer design rules, interned (flyweight) by ``(pdk_id, layer, datatype)``.

    Construct via :meth:`intern` so two requests for the same physical layer of
    the same PDK return the SAME object (identity equality holds), exactly like
    TED's per-layer ``Layer`` objects. Direct construction is private.

    Exposes ``min_width`` / ``min_spacing`` / ``min_area`` as :class:`RuleValue`
    (value + rule id + provenance), ``enclosure(other_layer)`` for a named
    enclosure (e.g. ``"contact"``, ``"comp"``, ``"ngate"``), and
    ``spacing_for_width(width_nm, run_length_nm)`` for the width-dependent
    spacing lookup (returns ``min_spacing`` when no wider band applies).
    """

    _CACHE: Dict[Tuple[str, int, int], "LayerRule"] = {}

    def __init__(self, _token: object, pdk_id: str, name: str, layer: int, datatype: int) -> None:
        if _token is not _INTERN_TOKEN:
            raise TypeError("LayerRule must be created via LayerRule.intern(...)")
        self.pdk_id = pdk_id
        self.name = name
        self.layer = layer
        self.datatype = datatype
        self._min_width: Optional[RuleValue] = None
        self._min_spacing: Optional[RuleValue] = None
        self._min_area: Optional[RuleValue] = None
        self._enclosures: Dict[str, RuleValue] = {}
        self._wide_spacing: List[WidthSpacingEntry] = []

    _5LM_CACHE: Dict[Tuple[str, int, int], "LayerRule"] = {}

    @classmethod
    def intern(cls, pdk_id: str, name: str, layer: int, datatype: int) -> "LayerRule":
        key = (pdk_id, int(layer), int(datatype))
        obj = cls._CACHE.get(key)
        if obj is None:
            obj = cls(_INTERN_TOKEN, pdk_id, name, layer, datatype)
            cls._CACHE[key] = obj
        return obj

    @classmethod
    def _intern_stack_variant(
        cls, pdk_id: str, stack: str, name: str, layer: int, datatype: int
    ) -> "LayerRule":
        """Return an isolated flyweight rule for non-default BEOL stacks (e.g. 5LM MetalTop)."""
        key = (f"{pdk_id}:{stack}", int(layer), int(datatype))
        obj = cls._5LM_CACHE.get(key)
        if obj is None:
            obj = cls(_INTERN_TOKEN, pdk_id, name, layer, datatype)
            cls._5LM_CACHE[key] = obj
        return obj

    @classmethod
    def clear_cache(cls) -> None:
        """Drop all interned LayerRule objects (test isolation helper)."""
        cls._CACHE.clear()
        cls._5LM_CACHE.clear()

    # -- population (used only by the PdkData builders) --
    def set_min_width(self, v: RuleValue) -> "LayerRule":
        self._min_width = v
        return self

    def set_min_spacing(self, v: RuleValue) -> "LayerRule":
        self._min_spacing = v
        return self

    def set_min_area(self, v: RuleValue) -> "LayerRule":
        self._min_area = v
        return self

    def set_min_hole_area(self, v: RuleValue) -> "LayerRule":
        self._min_hole_area = v
        return self

    def set_enclosure(self, other: str, v: RuleValue) -> "LayerRule":
        self._enclosures[other] = v
        return self

    def add_wide_spacing(self, entry: WidthSpacingEntry) -> "LayerRule":
        # LayerRule objects are interned across equivalent PdkRules builders.
        # Reconstructing the same authority must not append duplicate facts or
        # change the canonical routing digest merely due to construction count.
        if entry not in self._wide_spacing:
            self._wide_spacing.append(entry)
        self._wide_spacing.sort(key=lambda e: e.wide_ge)
        return self

    # -- query API --
    @property
    def key(self) -> Tuple[int, int]:
        return (self.layer, self.datatype)

    @property
    def min_width(self) -> Optional[RuleValue]:
        return self._min_width

    @property
    def min_spacing(self) -> Optional[RuleValue]:
        return self._min_spacing

    @property
    def min_area(self) -> Optional[RuleValue]:
        return self._min_area

    @property
    def min_hole_area(self) -> Optional[RuleValue]:
        return getattr(self, "_min_hole_area", None)

    def enclosure(self, other_layer: str) -> Optional[RuleValue]:
        """Return the enclosure of ``other_layer`` by this layer (nm), or None.

        ``other_layer`` is the inner / enclosed layer name, e.g.
        ``metal1.enclosure("contact")`` is CO.6 (metal1 overlap of contact),
        ``comp.enclosure("contact")`` is CO.4, ``nplus.enclosure("ngate")`` is
        NP.5a, ``nwell.enclosure("pcomp")`` is DF.4c.
        """
        return self._enclosures.get(other_layer)

    def enclosure_targets(self) -> FrozenSet[str]:
        return frozenset(self._enclosures)

    def spacing_for_width(self, width_nm: int, run_length_nm: int = 0) -> Optional[RuleValue]:
        """Width/run-length-dependent spacing. Returns the widest matching band's
        spacing, else the base ``min_spacing``."""
        best: Optional[WidthSpacingEntry] = None
        for e in self._wide_spacing:
            if width_nm >= e.wide_ge and run_length_nm >= e.run_length_ge:
                if best is None or e.wide_ge > best.wide_ge:
                    best = e
        if best is not None:
            return best.spacing
        return self._min_spacing

    def width_spacing_entries(self) -> Tuple[WidthSpacingEntry, ...]:
        """Immutable width bands including their declared geometry semantics."""
        return tuple(self._wide_spacing)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"LayerRule({self.pdk_id}:{self.name} {self.layer}/{self.datatype} "
            f"w={self._min_width.value if self._min_width else None} "
            f"s={self._min_spacing.value if self._min_spacing else None})"
        )


class ViaRule:
    """Contact / via cut rules, interned (flyweight) by ``(pdk_id, name)``.

    A gf180 cut layer (Contact 33/0, Via1 35/0, ...) has an EXACT (min == max)
    cut size, a cut-to-cut spacing, and an enclosure by the layer below and the
    layer above. Exposes ``size`` / ``spacing`` / ``enclosure_below`` /
    ``enclosure_above`` as :class:`RuleValue`.
    """

    _CACHE: Dict[Tuple[str, str], "ViaRule"] = {}

    def __init__(
        self,
        _token: object,
        pdk_id: str,
        name: str,
        layer: int,
        datatype: int,
        below: str,
        above: str,
    ) -> None:
        if _token is not _INTERN_TOKEN:
            raise TypeError("ViaRule must be created via ViaRule.intern(...)")
        self.pdk_id = pdk_id
        self.name = name
        self.layer = layer
        self.datatype = datatype
        self.below = below
        self.above = above
        self._size: Optional[RuleValue] = None
        self._spacing: Optional[RuleValue] = None
        self._enc_below: Optional[RuleValue] = None
        self._enc_above: Optional[RuleValue] = None

    @classmethod
    def intern(
        cls,
        pdk_id: str,
        name: str,
        layer: int,
        datatype: int,
        below: str,
        above: str,
    ) -> "ViaRule":
        key = (pdk_id, name)
        obj = cls._CACHE.get(key)
        if obj is None:
            obj = cls(_INTERN_TOKEN, pdk_id, name, layer, datatype, below, above)
            cls._CACHE[key] = obj
        return obj

    @classmethod
    def clear_cache(cls) -> None:
        cls._CACHE.clear()

    def set_size(self, v: RuleValue) -> "ViaRule":
        self._size = v
        return self

    def set_spacing(self, v: RuleValue) -> "ViaRule":
        self._spacing = v
        return self

    def set_enclosure_below(self, v: RuleValue) -> "ViaRule":
        self._enc_below = v
        return self

    def set_enclosure_above(self, v: RuleValue) -> "ViaRule":
        self._enc_above = v
        return self

    @property
    def key(self) -> Tuple[int, int]:
        return (self.layer, self.datatype)

    @property
    def size(self) -> Optional[RuleValue]:
        return self._size

    @property
    def spacing(self) -> Optional[RuleValue]:
        return self._spacing

    @property
    def enclosure_below(self) -> Optional[RuleValue]:
        return self._enc_below

    @property
    def enclosure_above(self) -> Optional[RuleValue]:
        return self._enc_above

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"ViaRule({self.pdk_id}:{self.name} {self.layer}/{self.datatype} "
            f"size={self._size.value if self._size else None})"
        )


# Token guarding the interned constructors -- callers cannot pass it, so the
# only construction path is the .intern(...) classmethod.
_INTERN_TOKEN = object()


# --- the PDK bundle + container ----------------------------------------------


@dataclass
class PdkData:
    """A data-only bundle describing one PDK: a layermap + the interned rule set.

    Attributes:
        pdk_id: the PDK identifier (e.g. ``"gf180mcu"``, ``"sky130"``).
        dbu_nm: database unit in nm (1 for both gf180mcu and sky130).
        layermap: name -> (layer, datatype). The canonical drawing layers.
        layer_rules: layer name -> LayerRule (interned).
        via_rules: via name -> ViaRule (interned).
        deck_dir: the rule-deck directory parsed (informational; "" if none).
    """

    pdk_id: str
    dbu_nm: int
    layermap: Dict[str, Tuple[int, int]]
    layer_rules: Dict[str, LayerRule]
    via_rules: Dict[str, ViaRule]
    deck_dir: str = ""
    layout_authority_validator: Optional[Callable[[], None]] = None


class PdkRules:
    """The single queryable PDK-rule container a placer / router consumes.

    Built from a :class:`PdkData` bundle. The downstream query API:

      * ``layer(name)`` -> (layer, datatype) GDS key.
      * ``rule_for(layer_name)`` -> :class:`LayerRule`.
      * ``min_width(layer)`` / ``min_spacing(layer)`` / ``min_area(layer)``
        -> nm (raises ``KeyError`` if the layer is unknown, ``None`` if the
        layer exists but the rule is not populated).
      * ``enclosure(inner, outer)`` -> nm enclosure of ``inner`` by ``outer``
        (e.g. ``enclosure("contact", "metal1")`` = CO.6).
      * ``via(name)`` -> :class:`ViaRule`.
      * ``provenance_breakdown()`` -> {Provenance: count} over every captured
        rule value (the honesty report).
    """

    def __init__(self, data: PdkData) -> None:
        self._data = data

    # -- factories --
    @classmethod
    def gf180mcu(
        cls,
        deck_dir: Optional[str] = DEFAULT_GF180_DECK_DIR,
        *,
        metal_level: Optional[str] = None,
    ) -> "PdkRules":
        return cls(build_gf180mcu_data(deck_dir, metal_level=metal_level))

    @classmethod
    def sky130(cls) -> "PdkRules":
        return cls(build_sky130_data())

    @classmethod
    def for_pdk(cls, pdk_id: str) -> "PdkRules":
        """Build the container for a PDK id by data. Adding a PDK is a data op.

        gf180mcu / sky130 use their dedicated, fully-populated builders (their
        output is BYTE-IDENTICAL to before -- they are the parity oracle). Any
        OTHER PDK is built generically from its registered adapter's parsed
        ``drc_rules()`` / ``layout_min()`` / ``via_rules()`` / ``layermap()``,
        with a node-scaled provenance-tagged fallback (``CONSOLIDATED``) where the
        deck is silent. This is the P0 generalization: a placer/router can consume
        a rule container for any adapter, not just the two hand-wired PDKs.
        """
        key = pdk_id.lower()
        if key == "gf180mcu":
            return cls.gf180mcu()
        if key == "sky130":
            return cls.sky130()
        return cls(build_adapter_data(pdk_id))

    # -- identity / introspection --
    @property
    def pdk_id(self) -> str:
        return self._data.pdk_id

    @property
    def dbu_nm(self) -> int:
        return self._data.dbu_nm

    @property
    def deck_dir(self) -> str:
        return self._data.deck_dir

    def _validate_layout_authority(self) -> None:
        validator = self._data.layout_authority_validator
        if validator is not None:
            validator()

    def layer_names(self) -> List[str]:
        self._validate_layout_authority()
        return sorted(self._data.layermap)

    def via_names(self) -> List[str]:
        return sorted(self._data.via_rules)

    # -- layer queries --
    def layer(self, name: str) -> Tuple[int, int]:
        """Return the GDS (layer, datatype) for a layer name."""
        self._validate_layout_authority()
        try:
            return self._data.layermap[name]
        except KeyError:
            raise KeyError(
                f"layer {name!r} not in {self.pdk_id} layermap; "
                f"known: {self.layer_names()}"
            ) from None

    def rule_for(self, layer_name: str) -> LayerRule:
        """Return the interned LayerRule for a layer name."""
        self._validate_layout_authority()
        try:
            return self._data.layer_rules[layer_name]
        except KeyError:
            raise KeyError(
                f"no rule for layer {layer_name!r} in {self.pdk_id}; "
                f"known: {sorted(self._data.layer_rules)}"
            ) from None

    def min_width(self, layer_name: str) -> Optional[int]:
        rv = self.rule_for(layer_name).min_width
        return rv.value if rv is not None else None

    def min_spacing(self, layer_name: str) -> Optional[int]:
        rv = self.rule_for(layer_name).min_spacing
        return rv.value if rv is not None else None

    def min_area(self, layer_name: str) -> Optional[int]:
        rv = self.rule_for(layer_name).min_area
        return rv.value if rv is not None else None

    def min_hole_area(self, layer_name: str) -> Optional[int]:
        rv = self.rule_for(layer_name).min_hole_area
        return rv.value if rv is not None else None

    def enclosure(self, inner: str, outer: str) -> Optional[int]:
        """Return the enclosure of ``inner`` by ``outer`` (nm), or None.

        Example: ``enclosure("contact", "metal1")`` is CO.6 (metal1 overlap of
        contact); ``enclosure("contact", "comp")`` is CO.4; ``enclosure("via1",
        "metal1")`` is the via1 lower-metal enclosure.
        """
        rv = self.rule_for(outer).enclosure(inner)
        return rv.value if rv is not None else None

    def spacing_for_width(
        self, layer_name: str, width_nm: int, run_length_nm: int = 0
    ) -> Optional[int]:
        rv = self.rule_for(layer_name).spacing_for_width(width_nm, run_length_nm)
        return rv.value if rv is not None else None

    def width_spacing_entries(self, layer_name: str) -> Tuple[WidthSpacingEntry, ...]:
        return self.rule_for(layer_name).width_spacing_entries()

    # -- via queries --
    def via(self, name: str) -> ViaRule:
        self._validate_layout_authority()
        try:
            return self._data.via_rules[name]
        except KeyError:
            raise KeyError(
                f"no via {name!r} in {self.pdk_id}; known: {self.via_names()}"
            ) from None

    def via_size(self, name: str) -> Optional[int]:
        rv = self.via(name).size
        return rv.value if rv is not None else None

    def via_spacing(self, name: str) -> Optional[int]:
        rv = self.via(name).spacing
        return rv.value if rv is not None else None

    # -- honesty report --
    def all_rule_values(self) -> List[RuleValue]:
        """Every captured RuleValue across all layers + vias (for auditing)."""
        out: List[RuleValue] = []
        for lr in self._data.layer_rules.values():
            for rv in (lr.min_width, lr.min_spacing, lr.min_area, lr.min_hole_area):
                if rv is not None:
                    out.append(rv)
            for tgt in sorted(lr.enclosure_targets()):
                out.append(lr.enclosure(tgt))  # type: ignore[arg-type]
            for e in lr._wide_spacing:
                out.append(e.spacing)
        for vr in self._data.via_rules.values():
            for rv in (vr.size, vr.spacing, vr.enclosure_below, vr.enclosure_above):
                if rv is not None:
                    out.append(rv)
        return out

    def rule_count(self) -> int:
        return len(self.all_rule_values())

    def provenance_breakdown(self) -> Dict[Provenance, int]:
        counts: Dict[Provenance, int] = {p: 0 for p in Provenance}
        for rv in self.all_rule_values():
            counts[rv.source] += 1
        return counts


# --- gf180mcu data builder ---------------------------------------------------

# gf180mcu canonical drawing layers (layer, datatype). These mirror the verified
# values in clay.layout.gf180_layermap.GF180_VERIFIED_SOURCES (sourced from the
# real gf180mcu.lyp); duplicated here so the container is self-contained data.
_GF180_LAYERMAP: Dict[str, Tuple[int, int]] = {
    "comp": (22, 0),      # COMP diffusion / active
    "dnwell": (12, 0),
    "nwell": (21, 0),
    "lvpwell": (204, 0),
    "poly2": (30, 0),     # gate poly
    "nplus": (32, 0),     # n+ implant
    "pplus": (31, 0),     # p+ implant
    # Passive-device marker layers (gf180mcu.lyp / KLayout decks ground truth):
    # SAB = salicide block 49/0 (gf180mcu.drc sab.drc), RES_MK = resistor marker
    # 110/5 (gf180mcu.lvs res_mk). Both are required to draw a DRC-legal poly
    # resistor whose body the LVS deck extracts as ``ppolyf_u``/``npolyf_u``
    # (poly2 ∩ implant ∩ sab ∩ res_mk).
    "sab": (49, 0),
    "res_mk": (110, 5),
    # MIM-capacitor marker / plate layers (gf180mcu.lyp / gf180mcu.lvs ground
    # truth). FUSETOP = the MIM top plate 75/0 (gf180mcu.lvs ``fusetop``);
    # CAP_MK = the cap marker 117/5 (must enclose the FuseTop top plate, MIM.7,
    # and is one of the two markers ``fuse_cap`` ANDs); MIM_L_MK = the MIM-L
    # marker 117/10 (the second marker ``fuse_cap = fusetop.interacting(cap_mk)
    # .interacting(mim_l_mk)`` requires). The MIM bottom plate is plain Metal2
    # (36/0) and the plate-to-routing cut is Via2 (38/0), both already present.
    "fusetop": (75, 0),
    "cap_mk": (117, 5),
    "mim_l_mk": (117, 10),
    "contact": (33, 0),
    "metal1": (34, 0),
    "via1": (35, 0),
    "metal2": (36, 0),
    "via2": (38, 0),
    "metal3": (42, 0),
    "via3": (40, 0),
    "metal4": (46, 0),
    "via4": (41, 0),
    "metal5": (81, 0),
}


def _deck(parsed: Dict[str, Tuple[int, bool]], rule_id: str, fallback: int) -> RuleValue:
    """Pick a value from the parsed deck if present (PARSED_DECK), else use the
    consolidated fallback (CONSOLIDATED)."""
    hit = parsed.get(rule_id)
    if hit is not None:
        return RuleValue(value=hit[0], rule_id=rule_id, source=Provenance.PARSED_DECK)
    return RuleValue(value=fallback, rule_id=rule_id, source=Provenance.CONSOLIDATED)


def _consolidated(value: int, rule_id: str) -> RuleValue:
    return RuleValue(value=value, rule_id=rule_id, source=Provenance.CONSOLIDATED)


def _documented(value: int, rule_id: str) -> RuleValue:
    return RuleValue(value=value, rule_id=rule_id, source=Provenance.DOCUMENTED)


def build_gf180mcu_data(
    deck_dir: Optional[str] = DEFAULT_GF180_DECK_DIR,
    *,
    metal_level: Optional[str] = None,
) -> PdkData:
    """Build the fully populated gf180mcu rule bundle.

    Width / spacing / size / area rules that the deck states as a single trailing
    scalar (M1.1, M2.1, V1.1, CO.1, CO.2a, DF.1a, DF.3a, PL.1/2/3a, NP/PP, NW.1a,
    M1.3/M2.3/DF.9) are taken PARSED_DECK from the real ``gf180mcu.drc`` decks
    when present. Enclosures (CO.3/CO.4/CO.6, V1.3/V1.4, PL.4, NP.5a/PP.5a,
    DF.4c) live inside ``enclosing(..., 0.XX.um, ...)`` expressions rather than
    the description string, so they are seeded CONSOLIDATED from Clay's existing
    constants and cross-checked against the deck literals in the audit.
    """
    pdk_id = "gf180mcu"
    deck = deck_dir or ""
    parsed = parse_deck_rules(deck)

    # interned layer rules
    def L(name: str) -> LayerRule:
        layer, dt = _GF180_LAYERMAP[name]
        return LayerRule.intern(pdk_id, name, layer, dt)

    comp = L("comp")
    poly2 = L("poly2")
    nplus = L("nplus")
    pplus = L("pplus")
    nwell = L("nwell")
    metal1 = L("metal1")
    metal2 = L("metal2")
    metal3 = L("metal3")

    # COMP (DF.*) -- width/space/area parsed; contact enclosure (CO.4) is the
    # consolidated 70 nm (= gf180_device_geom.GF180_COMP_CONTACT_ENCLOSURE_NM).
    comp.set_min_width(_deck(parsed, "DF.1a_3.3V", 220))
    comp.set_min_spacing(_deck(parsed, "DF.3a_3.3V", 280))
    comp.set_min_area(_deck(parsed, "DF.9_3.3V", 202500))  # 0.2025 um^2
    comp.set_enclosure("contact", _consolidated(70, "CO.4"))

    # Poly2 (PL.*) -- width/space parsed; contact enclosure (CO.3) + endcap
    # (PL.4) consolidated.
    poly2.set_min_width(_deck(parsed, "PL.1_3.3V", 180))
    poly2.set_min_spacing(_deck(parsed, "PL.3a_3.3V", 240))
    poly2.set_enclosure("contact", _consolidated(70, "CO.3"))
    poly2.set_enclosure("comp", _consolidated(220, "PL.4_3.3V"))  # poly endcap over COMP
    # gate (channel length) min as a named documented value (PL.2_3.3V parsed).
    poly2.set_enclosure("gate_length", _deck(parsed, "PL.2_3.3V", 280))

    # Implants -- width/space parsed; gate overlap enclosure (NP.5a/PP.5a)
    # consolidated; COMP extension (NP.5b/PP.5b) consolidated.
    nplus.set_min_width(_deck(parsed, "NP.1", 400))
    nplus.set_min_spacing(_deck(parsed, "NP.2", 400))
    nplus.set_min_area(_deck(parsed, "NP.8a", 350000))  # 0.35 um^2
    nplus.set_enclosure("ngate", _consolidated(230, "NP.5a"))
    nplus.set_enclosure("comp", _consolidated(160, "NP.5b"))

    pplus.set_min_width(_deck(parsed, "PP.1", 400))
    pplus.set_min_spacing(_deck(parsed, "PP.2", 400))
    pplus.set_enclosure("pgate", _consolidated(230, "PP.5a"))
    pplus.set_enclosure("comp", _consolidated(160, "PP.5b"))

    # Nwell -- width parsed (NW.1a); pcomp enclosure (DF.4c) consolidated 430 nm
    # (= gf180_pmos_geom.GF180_NWELL_PCOMP_ENCLOSURE_NM); ncomp enclosure (DF.4d).
    nwell.set_min_width(_deck(parsed, "NW.1a_3.3V", 860))
    nwell.set_min_spacing(_deck(parsed, "NW.2b_3.3V", 1400))  # different-potential
    nwell.set_enclosure("pcomp", _consolidated(430, "DF.4c_3.3V"))
    nwell.set_enclosure("ncomp", _consolidated(120, "DF.4d_3.3V"))

    # Metal1 -- width/space/area parsed; contact enclosure (CO.6) consolidated
    # 60 nm (= coclean_5tota GF180_METAL1_CONTACT_ENCLOSURE_NM); + wide-metal
    # spacing band (M1.2b: 0.3 um to wide metal).
    metal1.set_min_width(_deck(parsed, "M1.1", 230))
    metal1.set_min_spacing(_deck(parsed, "M1.2a", 230))
    metal1.set_min_area(_deck(parsed, "M1.3", 144400))  # 0.1444 um^2
    metal1.set_enclosure("contact", _consolidated(60, "CO.6"))
    metal1.set_enclosure("via1", _consolidated(60, "V1.3c"))
    metal1.add_wide_spacing(
        WidthSpacingEntry(wide_ge=10000, spacing=_deck(parsed, "M1.2b", 300), run_length_ge=10000)
    )

    # Metal2 -- width/space/area parsed; via enclosures consolidated.
    metal2.set_min_width(_deck(parsed, "M2.1", 280))
    metal2.set_min_spacing(_deck(parsed, "M2.2a", 280))
    metal2.set_min_area(_deck(parsed, "M2.3", 144400))
    metal2.set_enclosure("via1", _consolidated(60, "V1.4b"))
    metal2.set_enclosure("via2", _consolidated(60, "V2.3c"))
    metal2.add_wide_spacing(
        WidthSpacingEntry(wide_ge=10000, spacing=_deck(parsed, "M2.2b", 300), run_length_ge=10000)
    )

    # Metal3 -- width/space parsed; via enclosures consolidated.
    metal3.set_min_width(_deck(parsed, "M3.1", 280))
    metal3.set_min_spacing(_deck(parsed, "M3.2a", 280))
    metal3.set_enclosure("via2", _consolidated(60, "V2.4b"))
    metal3.set_enclosure("via3", _deck(parsed, "V3.3c", 60))

    # The installed foundry deck defines M4/M5 and V3/V4, not merely their
    # layermap keys. Keep every existing lower-layer fact unchanged while
    # exposing the real upper routing prefix to placement, routing and LVS.
    upper_metals = {}
    is_5lm = str(metal_level or "").strip().upper() == "5LM"
    for level in (4, 5):
        name = f"metal{level}"
        if level == 5 and is_5lm:
            # In GF180MCU 5LM (e.g. gf180mcuD Magic/Option-B), Metal5 IS MetalTop
            # (9K/11K thick top metal): MT.1=440, MT.2a=460, MT.2b=600 (gf180mcuD.tech),
            # MT.4=562500 nm^2 (0.5625 um^2 KLayout metaltop.drc >= 526500 nm^2 Magic).
            layer, dt = _GF180_LAYERMAP[name]
            metal = LayerRule._intern_stack_variant(pdk_id, "5LM", name, layer, dt)
            metal.set_min_width(_documented(440, "MT.1"))
            metal.set_min_spacing(_documented(460, "MT.2a"))
            metal.set_min_area(_documented(562500, "MT.4"))
            metal.set_enclosure("via4", _deck(parsed, "V4.4b", 60))
            metal.add_wide_spacing(WidthSpacingEntry(
                wide_ge=10000,
                spacing=_documented(600, "MT.2b"),
                run_length_ge=10000,
            ))
            upper_metals[name] = metal
            continue
        metal = L(name)
        metal.set_min_width(_deck(parsed, f"M{level}.1", 280))
        metal.set_min_spacing(_deck(parsed, f"M{level}.2a", 280))
        metal.set_min_area(_deck(parsed, f"M{level}.3", 144400))
        metal.set_enclosure(f"via{level - 1}", _deck(parsed, f"V{level - 1}.4b", 60))
        if level == 4:
            metal.set_enclosure("via4", _deck(parsed, "V4.3c", 60))
        metal.add_wide_spacing(WidthSpacingEntry(
            wide_ge=10000, spacing=_deck(parsed, f"M{level}.2b", 300),
            run_length_ge=10000,
        ))
        upper_metals[name] = metal

    layer_rules = {
        "comp": comp,
        "poly2": poly2,
        "nplus": nplus,
        "pplus": pplus,
        "nwell": nwell,
        "metal1": metal1,
        "metal2": metal2,
        "metal3": metal3,
        **upper_metals,
    }

    # interned via / contact rules
    def V(name: str, below: str, above: str) -> ViaRule:
        layer, dt = _GF180_LAYERMAP[name]
        return ViaRule.intern(pdk_id, name, layer, dt, below, above)

    # Contact (CO.*): size CO.1 + spacing CO.2a parsed; COMP/poly enclosure
    # below = CO.4/CO.3 (70 nm consolidated); metal1 enclosure above = CO.6 (60).
    contact = V("contact", "comp", "metal1")
    contact.set_size(_deck(parsed, "CO.1", 220))
    contact.set_spacing(_deck(parsed, "CO.2a", 250))
    contact.set_enclosure_below(_consolidated(70, "CO.4"))   # COMP overlap of contact
    contact.set_enclosure_above(_consolidated(60, "CO.6"))   # metal1 overlap of contact

    # Via1 (V1.*): size V1.1 + spacing V1.2a parsed; metal1/metal2 enclosure
    # consolidated 60 nm (EOL overlap V1.3c/V1.4b, = coclean_5tota
    # GF180_M1_VIA1_ENCLOSURE_NM).
    via1 = V("via1", "metal1", "metal2")
    via1.set_size(_deck(parsed, "V1.1", 260))
    via1.set_spacing(_deck(parsed, "V1.2a", 260))
    via1.set_enclosure_below(_consolidated(60, "V1.3c"))
    via1.set_enclosure_above(_consolidated(60, "V1.4b"))

    # Via2 (V2.*): size + spacing parsed; metal2/metal3 enclosure consolidated.
    via2 = V("via2", "metal2", "metal3")
    via2.set_size(_deck(parsed, "V2.1", 260))
    via2.set_spacing(_deck(parsed, "V2.2a", 260))
    via2.set_enclosure_below(_consolidated(60, "V2.3c"))
    via2.set_enclosure_above(_consolidated(60, "V2.4b"))

    via_rules = {"contact": contact, "via1": via1, "via2": via2}
    for level in (3, 4):
        name = f"via{level}"
        via = V(name, f"metal{level}", f"metal{level + 1}")
        via.set_size(_deck(parsed, f"V{level}.1", 260))
        via.set_spacing(_deck(parsed, f"V{level}.2a", 260))
        via.set_enclosure_below(_deck(parsed, f"V{level}.3c", 60))
        via.set_enclosure_above(_deck(parsed, f"V{level}.4b", 60))
        via_rules[name] = via

    return PdkData(
        pdk_id=pdk_id,
        dbu_nm=1,
        layermap=dict(_GF180_LAYERMAP),
        layer_rules=layer_rules,
        via_rules=via_rules,
        deck_dir=deck if parsed else "",
    )


# --- sky130 data builder (proves a 2nd PDK is data, not code) ----------------

# sky130 canonical drawing layers (subset). Values from the public skywater PDK.
_SKY130_LAYERMAP: Dict[str, Tuple[int, int]] = {
    "diff": (65, 20),
    "poly": (66, 20),
    "licon1": (66, 44),   # local-interconnect contact
    "li1": (67, 20),
    "mcon": (67, 44),
    "met1": (68, 20),
    "via": (68, 44),
    "met2": (69, 20),
    "via2": (69, 44),
    "met3": (70, 20),
    "capm": (89, 44),  # native SKY130 MIM selector, not routing metal
    "via3": (70, 44),
    "met4": (71, 20),
    "via4": (71, 44),
    "met5": (72, 20),
    "nwell": (64, 20),
}


def build_sky130_data() -> PdkData:
    """SKY130 routing rules, in nm, from sky130A_mr.drc / sky130A.tech.

    Via enclosures use the larger directional minimum on all four edges;
    this is the existing isotropic ViaRule currency. No source device is
    resized by these routing construction rules.
    """
    pdk_id = "sky130"

    def L(name: str) -> LayerRule:
        layer, dt = _SKY130_LAYERMAP[name]
        return LayerRule.intern(pdk_id, name, layer, dt)

    # sky130 metal1/metal2: width=140 space=140 (skywater met1.1/met1.2,
    # met2.1/met2.2). poly width=150 space=210. diff width=290 space=270.
    poly = L("poly").set_min_width(_documented(150, "poly.1a")).set_min_spacing(
        _documented(210, "poly.2")
    )
    diff = L("diff").set_min_width(_documented(290, "difftap.1")).set_min_spacing(
        _documented(270, "difftap.3")
    )
    # Ordinary-area local interconnect rules from the bound SKY130 deck.
    li1 = L("li1").set_min_width(_documented(170, "li.1")).set_min_spacing(
        _documented(170, "li.3")
    ).set_min_area(_documented(56100, "li.6"))
    met1 = (
        L("met1")
        .set_min_width(_documented(140, "met1.1"))
        .set_min_spacing(_documented(140, "met1.2"))
        .set_enclosure("mcon", _documented(30, "met1.4"))
        .set_enclosure("via", _documented(85, "via.5a"))
        .set_enclosure("via1", _documented(85, "via.5a"))
        .set_min_area(_documented(83000, "met1.6"))
        .set_min_hole_area(_documented(140000, "met1.7"))
    )
    met2 = (
        L("met2")
        .set_min_width(_documented(140, "met2.1"))
        .set_min_spacing(_documented(140, "met2.2"))
        .set_enclosure("via", _documented(85, "met2.5"))
        .set_enclosure("via1", _documented(85, "met2.5"))
        .set_enclosure("via2", _documented(85, "via2.5"))
        .set_min_area(_documented(67600, "met2.6"))
        .set_min_hole_area(_documented(140000, "met2.7"))
    )
    nwell = L("nwell").set_min_width(_documented(840, "nwell.1")).set_min_spacing(
        _documented(1270, "nwell.2a")
    )

    layer_rules = {"poly": poly, "diff": diff, "li1": li1, "met1": met1, "met2": met2, "nwell": nwell}
    for level, width, spacing, area in ((3, 300, 300, 240000),
                                       (4, 300, 300, 240000),
                                       (5, 1600, 1600, 4000000)):
        name = f"met{level}"
        layer_rules[name] = (L(name).set_min_width(_documented(width, name + ".1"))
            .set_min_spacing(_documented(spacing, name + ".2"))
            .set_min_area(_documented(area, name + (".6" if level == 3 else ".4a" if level == 4 else ".4")))
            .set_min_hole_area(_documented(200000 if level in (3, 4) else 140000, f"{name}.7")))
    layer_rules["met3"].set_enclosure("via2", _documented(65, "met3.4"))
    layer_rules["met3"].set_enclosure("via3", _documented(90, "via3.5"))
    layer_rules["met4"].set_enclosure("via3", _documented(65, "met4.3"))
    layer_rules["met4"].set_enclosure("via4", _documented(190, "via4.4"))
    layer_rules["met5"].set_enclosure("via4", _documented(310, "met5.3"))
    for level in range(1, 5):
        layer_rules[f"met{level}"].add_wide_spacing(WidthSpacingEntry(
            wide_ge=3005, spacing=_documented(280 if level < 3 else 400,
                f"met{level}.3ab" if level < 3 else f"met{level}.wide"),
            # sky130A_mr.drc: shrink/grow 1.5um and exposed-edge spacing.
            # sky130A.tech: m*_large_halo=(grow(space, huge)-huge)&allm*,
            # then cifspacing on that actual attached metal (touching_ok).
            opening_nm=1500, attached_halo_nm=280 if level < 3 else 400))
    for level in range(1, 6):
        layer_rules[f"metal{level}"] = layer_rules[f"met{level}"]

    def V(name: str, below: str, above: str) -> ViaRule:
        layer, dt = _SKY130_LAYERMAP[name]
        return ViaRule.intern(pdk_id, name, layer, dt, below, above)

    mcon = (
        V("mcon", "li1", "met1")
        .set_size(_documented(170, "mcon.1"))
        .set_spacing(_documented(190, "mcon.2"))
        .set_enclosure_above(_documented(30, "met1.4"))
    )
    via = (
        V("via", "met1", "met2")
        .set_size(_documented(150, "via.1a"))
        .set_spacing(_documented(170, "via.2"))
        .set_enclosure_below(_documented(85, "via.5a"))
        .set_enclosure_above(_documented(85, "met2.5"))
    )

    via_rules = {"mcon": mcon, "via": via, "via1": via}
    for level, size, spacing, below, above, below_rule, above_rule in (
            (2, 200, 200, 85, 65, "via2.5", "met3.4"),
            (3, 200, 200, 90, 65, "via3.5", "met4.3"),
            (4, 800, 800, 190, 310, "via4.4", "met5.3")):
        name = f"via{level}"
        via_rules[name] = (V(name, f"met{level}", f"met{level+1}")
            .set_size(_documented(size, name + ".1"))
            .set_spacing(_documented(spacing, name + ".2"))
            .set_enclosure_below(_documented(below, below_rule))
            .set_enclosure_above(_documented(above, above_rule)))

    return PdkData(
        pdk_id=pdk_id,
        dbu_nm=1,
        layermap=dict(_SKY130_LAYERMAP),
        layer_rules=layer_rules,
        via_rules=via_rules,
        deck_dir="",
    )


# --- generic adapter-driven builder (any PDK by data) ------------------------


def build_adapter_data(pdk_id: str) -> PdkData:
    """Build a :class:`PdkData` bundle for ANY registered PDK adapter.

    Reads the adapter's parsed surface -- ``layermap()`` (GDS keys),
    ``drc_rules()`` / ``layout_min()`` (width/space/size), ``via_rules()`` (cut
    rules) -- and packs them into the interned LayerRule / ViaRule container the
    placer/router consumes. Every value carries provenance: ``PARSED_DECK`` when
    it traced to the foundry deck, ``CONSOLIDATED`` when it came from a
    node-scaled fallback (the adapter's ``via_rules()`` already flags
    ``calibrated=False`` for those, and ``layout_min()`` substitutes a node scale
    where the deck is silent).

    This never mutates the gf180mcu / sky130 builders; it is the catch-all path
    :meth:`PdkRules.for_pdk` uses for every other PDK.
    """
    from clay.pdk import get_pdk

    LayerRule.clear_cache()
    ViaRule.clear_cache()
    adapter = get_pdk(pdk_id)
    lm = adapter.layermap()                 # canonical token -> (layer, datatype)
    parsed = adapter.drc_rules()            # rule_key -> {value_um, value_nm, line}
    lmin = adapter.layout_min()             # {m1_w, m1_s, poly_w, poly_s, co_w, co_s} (um)

    def _deck_nm(key: str) -> Optional[int]:
        e = parsed.get(key)
        if not e:
            return None
        if e.get("value_nm") is not None:
            return int(e["value_nm"])
        if e.get("value_um") is not None:
            return int(round(float(e["value_um"]) * 1000.0))
        return None

    def _exact_diff_nm(key: str) -> Optional[int]:
        entry = parsed.get(key)
        if entry is None:
            return None
        value = entry.get("value_nm")
        if type(value) is not int:
            raise TypeError(f"{key} must be an exact integer number of nanometres")
        if not 20 <= value <= 2000:
            raise ValueError(f"{key} is outside the accepted foundry-rule range")
        return value

    def _deck_area_nm2(key: str) -> Optional[int]:
        """Min-area value in nm^2 from a parsed area entry.

        Area entries carry ``value_um2`` (and a ``value_nm`` that is already nm^2,
        e.g. 0.2 um^2 -> 200000). Prefer ``value_um2`` (um^2 -> *1e6 -> nm^2); fall
        back to the area entry's ``value_nm`` (already nm^2). Distinct from
        ``_deck_nm`` because a LENGTH entry's ``value_um`` scales by 1e3, an AREA's
        by 1e6 -- mixing them would mis-scale the min-area by 1000x."""
        e = parsed.get(key)
        if not e:
            return None
        if e.get("value_um2") is not None:
            return int(round(float(e["value_um2"]) * 1.0e6))
        if e.get("value_nm") is not None:
            return int(e["value_nm"])  # area entries store value_nm as nm^2
        return None

    def _layer(name: str) -> Optional[LayerRule]:
        key = lm.get(name)
        if key is None:
            return None
        return LayerRule.intern(pdk_id, name, int(key[0]), int(key[1]))

    def _rv(deck_key: str, fallback_um: float, rule_id: str) -> RuleValue:
        nm = _deck_nm(deck_key)
        if nm is not None:
            return RuleValue(value=nm, rule_id=rule_id, source=Provenance.PARSED_DECK)
        return RuleValue(
            value=int(round(fallback_um * 1000.0)),
            rule_id=rule_id, source=Provenance.CONSOLIDATED,
        )

    layer_rules: Dict[str, LayerRule] = {}

    def _area_rv(deck_key: str, fallback_width_nm: int, rule_id: str) -> RuleValue:
        """A min-area RuleValue (nm^2): the parsed deck value when present, else a
        node-scaled CONSOLIDATED fallback of ``min_width^2`` (the smallest legal
        square feature on the layer). The fallback is silent provenance, consistent
        with the other node-scaled rules."""
        a = _deck_area_nm2(deck_key)
        if a is not None:
            return RuleValue(value=a, rule_id=rule_id, source=Provenance.PARSED_DECK)
        return RuleValue(
            value=int(fallback_width_nm) * int(fallback_width_nm),
            rule_id=rule_id, source=Provenance.CONSOLIDATED,
        )

    metal1 = _layer("metal1")
    if metal1 is not None:
        metal1.set_min_width(_rv("metal1_width", lmin["m1_w"], "metal1_width"))
        metal1.set_min_spacing(_rv("metal1_space", lmin["m1_s"], "metal1_space"))
        metal1.set_min_area(
            _area_rv("metal1_min_area", metal1.min_width.value, "metal1_min_area")
        )
        m1_enc = _deck_nm("contact_enclosure_metal1")
        if m1_enc is not None:
            metal1.set_enclosure("contact", RuleValue(m1_enc, "contact_enclosure_metal1", Provenance.PARSED_DECK))
        layer_rules["metal1"] = metal1

    poly = _layer("poly")
    if poly is not None:
        poly.set_min_width(_rv("poly_width", lmin["poly_w"], "poly_width"))
        poly.set_min_spacing(_rv("poly_space", lmin["poly_s"], "poly_space"))
        p_enc = _deck_nm("contact_enclosure_poly")
        if p_enc is not None:
            poly.set_enclosure("contact", RuleValue(p_enc, "contact_enclosure_poly", Provenance.PARSED_DECK))
        layer_rules["poly"] = poly

    diff = _layer("diff")
    if diff is not None:
        d_width = _exact_diff_nm("diff_width")
        if d_width is not None:
            diff.set_min_width(
                RuleValue(d_width, "diff_width", Provenance.PARSED_DECK)
            )
        d_space = _exact_diff_nm("diff_space")
        if d_space is not None:
            diff.set_min_spacing(
                RuleValue(d_space, "diff_space", Provenance.PARSED_DECK)
            )
        d_enc = _deck_nm("contact_enclosure_diff")
        if d_enc is not None:
            diff.set_enclosure("contact", RuleValue(d_enc, "contact_enclosure_diff", Provenance.PARSED_DECK))
        layer_rules["diff"] = diff

    # Every upper metal the layermap exposes (metal2, metal3, ... metalN) gets a rule:
    # the parsed deck width/space when present (external_pdk M2..M5 carry their own Mn.W.1/Mn.S.1),
    # else the metal1 scale (node-scaled CONSOLIDATED). Generalised from a hard-coded
    # (metal2, metal3) pair so a 6-metal PDK routes its full stack -- a metal with no
    # rule is invisible to the router.
    import re as _re_metals
    _upper = sorted(
        (nm for nm in lm if _re_metals.fullmatch(r"metal\d+", nm) and nm != "metal1"),
        key=lambda s: int(s[5:]),
    )
    for upper in _upper:
        lr = _layer(upper)
        if lr is not None:
            lr.set_min_width(_rv(f"{upper}_width", lmin["m1_w"], f"{upper}_width"))
            lr.set_min_spacing(_rv(f"{upper}_space", lmin["m1_s"], f"{upper}_space"))
            lr.set_min_area(
                _area_rv(f"{upper}_min_area", lr.min_width.value, f"{upper}_min_area")
            )
            layer_rules[upper] = lr

    for implant in ("nplus", "pplus", "nwell"):
        lr = _layer(implant)
        if lr is not None:
            layer_rules[implant] = lr

    # via / contact rules from the adapter's via_rules() (already calibrated-tagged)
    via_rules: Dict[str, ViaRule] = {}
    for name, vr in adapter.via_rules().items():
        key = lm.get(name)
        if key is None:
            # contact has no dedicated layermap token in some PDKs; use 'contact'
            key = lm.get("contact") if name == "contact" else None
        if key is None:
            continue
        prov = Provenance.PARSED_DECK if vr.calibrated else Provenance.CONSOLIDATED
        obj = ViaRule.intern(pdk_id, name, int(key[0]), int(key[1]), vr.below, vr.above)
        obj.set_size(RuleValue(vr.cut_size_nm, f"{name}_size", prov))
        obj.set_spacing(RuleValue(vr.spacing_nm, f"{name}_space", prov))
        obj.set_enclosure_below(RuleValue(vr.enclosure_below_nm, f"{name}_enc_below", prov))
        obj.set_enclosure_above(RuleValue(vr.enclosure_above_nm, f"{name}_enc_above", prov))
        via_rules[name] = obj

    return PdkData(
        pdk_id=pdk_id,
        dbu_nm=1,
        layermap=dict(lm),
        layer_rules=layer_rules,
        via_rules=via_rules,
        deck_dir="",
        layout_authority_validator=adapter.layout_authority_validator,
    )


__all__ = [
    "Provenance",
    "RuleValue",
    "WidthSpacingEntry",
    "LayerRule",
    "ViaRule",
    "PdkData",
    "PdkRules",
    "parse_deck_rules",
    "build_gf180mcu_data",
    "build_sky130_data",
    "build_adapter_data",
    "DEFAULT_GF180_DECK_DIR",
]
