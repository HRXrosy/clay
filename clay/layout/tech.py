"""LayoutTech -- the single PDK-technology object a placer / router / device
builder consumes (Clay intent-driven P&R, Phase P0).

Motivation
==========
Before P0 the layout subsystem reached into a dozen sources for PDK facts: GDS
layer keys in ``backend.LogicalLayer`` tables, design-rule numbers in
``pdk_rules_v174.PdkRules``, device bindings + KLayout decks on the
``clay.pdk.PdkAdapter``, and a scatter of ``GF180_*_NM`` module constants. A
placer or router had to know which source held which fact, and adding a PDK meant
touching all of them.

``LayoutTech`` is the flyweight that aggregates these behind ONE object. It is
the contract P1/P3/P4/P5 build on:

  * layer keys -- ``metal_key`` / ``via_key`` / ``label_key`` /
    ``build_layer`` (diff/poly/contact/implant/well/tap), all as GDS
    ``(layer, datatype)`` tuples;
  * spacing / width / enclosure -- ``min_space`` / ``min_width`` /
    ``enclosure`` / ``via_rule`` / ``implant_well`` from the parsed rule
    container;
  * provenance surfaces -- ``device_bindings`` / ``klayout_decks`` /
    ``lvs_reference_models`` / ``substrate_net`` / ``bulk_topology`` straight
    from the adapter.

Parity contract
===============
For gf180mcu the layer keys + spacing constants this object returns are
BYTE-IDENTICAL to the legacy hard-coded gf180 values (``GF180_FIN_PITCH_NM`` /
``GF180_MIN_POLY_SPACE_NM`` in ``backend.py``; the metal/via keys + spacing
minima in ``generic_router`` / ``m2_crossing_route_v163``). The proven L5
co-clean cells depend on that parity; ``tests/test_layout_tech_seam.py`` is the
oracle.

Flyweight
=========
``LayoutTech.resolve(pdk_id)`` caches one instance per PDK id. The aggregated
rule container (``PdkRules``) is itself interned, so two resolves of the same PDK
share the same underlying LayerRule / ViaRule objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .backend import (
    GdsLayer,
    LogicalLayer,
    Gf180mcuLayerMap,
    Sky130LayerMap,
)
from .pdk_rules import PdkRules

LayerKey = Tuple[int, int]


@dataclass(frozen=True)
class ResolvedMimMetalGeometryRule:
    rule_id: str
    metal_layer: LayerKey
    marker_layer: LayerKey
    predicate: str
    grow_nm: int
    spacing_nm: int
    metric: str


# Per-PDK GDS datatype used for METAL text labels (the label layer is the metal
# layer's drawing GDS layer with a label datatype). gf180mcu uses datatype 10
# (= coclean_5tota_v163.GF180_METAL1_LABEL_KEY (34, 10)). SKY130's Magic
# sky130A.tech cifinput maps METnTXT to datatype 5 (``labels METnTXT text``)
# and METnPIN to datatype 16 (``labels METnPIN port``). A drawing-purpose
# text names a node but does not declare a subcircuit port.
# Unknown PDKs retain the drawing-datatype fallback.
_LABEL_DATATYPE: Dict[str, int] = {
    "gf180mcu": 10,
    "sky130": 5,
}
_PIN_LABEL_DATATYPE: Dict[str, int] = {"sky130": 16}


class LayoutTech:
    """Aggregated PDK technology surface for the layout stack (a flyweight).

    Construct via :meth:`resolve`. The object is immutable after construction;
    all accessors are pure reads.
    """

    _CACHE: Dict[str, "LayoutTech"] = {}
    _VARIANT_CACHE: Dict[Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]], "LayoutTech"] = {}

    # ---- construction ----
    def __init__(
        self,
        _token: object,
        pdk_id: str,
        *,
        mim_option: Optional[str] = None,
        cap_option: Optional[str] = None,
        metal_level: Optional[str] = None,
        pex_backend: Optional[str] = None,
    ) -> None:
        if _token is not _TECH_TOKEN:
            raise TypeError("LayoutTech must be created via LayoutTech.resolve(...)")
        self.pdk_id = pdk_id
        self._mim_option = mim_option
        self._cap_option = cap_option
        self._metal_level = metal_level
        self._pex_backend = pex_backend
        # the parsed rule container (gf180/sky130 byte-identical; 5LM gf180mcu uses MetalTop MT.* rules)
        self._rules: PdkRules = (
            PdkRules.gf180mcu(metal_level=metal_level)
            if pdk_id.lower() == "gf180mcu" and str(metal_level or "").strip().upper() == "5LM"
            else PdkRules.for_pdk(pdk_id)
        )
        # the registered adapter (device bindings / decks / lvs / substrate)
        from clay.pdk import get_pdk
        self._adapter = get_pdk(pdk_id)
        # the backend logical-layer map for build-time layer-key resolution.
        # gf180/sky130 have a hand-verified table; other PDKs resolve build layers
        # through the rule container's layermap by canonical token instead.
        self._logical: Optional[object] = _logical_map_for(pdk_id)

    @property
    def mim_option(self) -> Optional[str]:
        """Explicitly bound MIM stack option (e.g. 'B' on gf180mcu), or None for default."""
        return self._mim_option

    @property
    def cap_option(self) -> Optional[str]:
        """Explicitly bound MIM capacitance density option, or None for default."""
        return self._cap_option

    @property
    def metal_level(self) -> Optional[str]:
        """Explicitly bound metal stack level (e.g. '5LM'), or None for default."""
        return self._metal_level

    @property
    def pex_backend(self) -> Optional[str]:
        """Explicitly bound PEX backend hint (e.g. 'magic'), or None for default."""
        return self._pex_backend

    def with_mim_config(
        self,
        *,
        mim_option: Optional[str] = None,
        cap_option: Optional[str] = None,
        metal_level: Optional[str] = None,
        pex_backend: Optional[str] = None,
    ) -> "LayoutTech":
        """Return a ``LayoutTech`` bound to the requested MIM configuration without mutating flyweights."""
        eff_mim = self._mim_option if mim_option is None else mim_option
        eff_cap = self._cap_option if cap_option is None else cap_option
        eff_lvl = self._metal_level if metal_level is None else metal_level
        eff_pex = self._pex_backend if pex_backend is None else pex_backend
        return type(self).resolve(
            self.pdk_id,
            mim_option=eff_mim,
            cap_option=eff_cap,
            metal_level=eff_lvl,
            pex_backend=eff_pex,
        )

    @classmethod
    def resolve(
        cls,
        pdk_id: str,
        *,
        mim_option: Optional[str] = None,
        cap_option: Optional[str] = None,
        metal_level: Optional[str] = None,
        pex_backend: Optional[str] = None,
    ) -> "LayoutTech":
        """Return the (cached) LayoutTech for ``pdk_id``.

        When MIM options (or ``pex_backend='magic'`` on ``gf180mcu``) are
        provided, returns a cached option-bound variant without mutating the
        default PDK flyweight in ``_CACHE``.

        Raises ``KeyError`` if no PDK adapter is registered for ``pdk_id``.
        """
        key = pdk_id.lower()
        eff_pex = str(pex_backend).strip().lower() if pex_backend is not None and str(pex_backend).strip() else None
        if key == "gf180mcu" and (
            mim_option is not None
            or cap_option is not None
            or metal_level is not None
            or eff_pex == "magic"
        ):
            from clay.pdk.gf180mcu import resolve_gf180_mim_config

            norm_opt, norm_cap, norm_lvl = resolve_gf180_mim_config(
                mim_option=mim_option,
                cap_option=cap_option,
                metal_level=metal_level,
                pex_backend=eff_pex,
            )
            if norm_opt != "A" or cap_option is not None or metal_level is not None or eff_pex == "magic":
                vkey = (key, norm_opt, norm_cap, norm_lvl, eff_pex)
                vobj = cls._VARIANT_CACHE.get(vkey)
                if vobj is None:
                    vobj = cls(
                        _TECH_TOKEN,
                        key,
                        mim_option=norm_opt,
                        cap_option=norm_cap,
                        metal_level=norm_lvl if norm_opt == "B" or metal_level is not None else None,
                        pex_backend=eff_pex,
                    )
                    cls._VARIANT_CACHE[vkey] = vobj
                return vobj
        obj = cls._CACHE.get(key)
        if obj is None:
            obj = cls(_TECH_TOKEN, key)
            cls._CACHE[key] = obj
        return obj

    @classmethod
    def clear_cache(cls) -> None:
        """Drop the flyweight cache (test isolation helper)."""
        cls._CACHE.clear()
        cls._VARIANT_CACHE.clear()

    # ---- identity / underlying objects ----
    @property
    def rules(self) -> PdkRules:
        """The underlying :class:`PdkRules` rule container (for advanced queries
        the convenience accessors below do not cover)."""
        self._validate_layout_authority()
        return self._rules

    @property
    def adapter(self):
        """The underlying :class:`clay.pdk.PdkAdapter` (sim / deck surface)."""
        return self._adapter

    def _validate_layout_authority(self) -> None:
        validator = getattr(self._adapter, "layout_authority_validator", None)
        if validator is not None:
            validator()

    # ---- layer-key accessors (GDS (layer, datatype) tuples) ----------------
    def metal_key(self, n: int) -> LayerKey:
        """GDS key for metal layer ``n`` (1-based: ``metal_key(1)`` is M1).

        Resolves through the rule container layermap (``metal1`` / ``metal2`` /
        ``met1`` / ``met2`` ...). Raises ``KeyError`` if the PDK has no such metal.
        """
        self._validate_layout_authority()
        for name in (f"metal{n}", f"met{n}", f"m{n}"):
            key = self._rules._data.layermap.get(name)
            if key is not None:
                return key
        raise KeyError(f"{self.pdk_id} has no metal{n} layer")

    # ---- per-metal preferred routing DIRECTION (the HVH layer discipline) ---
    def metal_preferred_axis_declared(self, n: int) -> Optional[str]:
        """PDK-DECLARED preferred routing axis for metal ``n`` (1-based), or ``None``.

        Reads the adapter's ``preferred_routing_axes`` (process-convention DATA:
        which physical metal runs H vs V in this stack's standard detailed-routing
        discipline -- NEVER a reference-layout measurement). Returns ``'H'`` / ``'V'``
        when the adapter declares metal ``n``, else ``None`` so the caller can keep
        its historical default. An adapter that declares no axes returns ``None`` for
        every metal -> byte-identical direction behaviour."""
        axes = getattr(self._adapter, "preferred_routing_axes", None) or {}
        a = axes.get(n) or axes.get(int(n))
        s = str(a).strip().upper() if a is not None else ""
        return s if s in ("H", "V") else None

    @property
    def manufacturing_grid_nm(self) -> Optional[int]:
        """Explicit foundry grid; ``None`` asks the shared reader to parse a deck."""
        self._validate_layout_authority()
        return self._adapter.manufacturing_grid_nm

    def metal_preferred_axis(self, n: int) -> str:
        """Preferred routing axis for metal ``n`` (1-based) -- ``'H'`` or ``'V'``.

        The PDK-declared axis wins; absent a declaration this falls back to the
        generic HVH alternation ODD metals horizontal / EVEN metals vertical (M1=H,
        M2=V, M3=H, ...), the historical ``CONVENTIONAL_AXES`` /
        ``_Tech.preferred_horizontal`` default. So an undeclared PDK is unchanged."""
        declared = self.metal_preferred_axis_declared(n)
        if declared is not None:
            return declared
        return "H" if (n % 2 == 1) else "V"

    def metal_prefers_horizontal(self, n: int) -> bool:
        """True iff metal ``n`` (1-based) prefers HORIZONTAL routing."""
        return self.metal_preferred_axis(n) == "H"

    def calibrated_metal_sheet_resistance_ohm_sq(
        self, n: int
    ) -> Optional[float]:
        """Foundry-derived sheet resistance for metal ``n``, or ``None``.

        Electrical-budget materialization must not consume the explicit 0.080
        ohm/sq fallback carried by incomplete commercial profiles.  Only a
        profile whose complete electrical source is marked calibrated reaches
        the router.
        """

        from clay.pdk.pex_profile import get_profile

        profile = get_profile(self.pdk_id)
        if profile is None or profile.calibrated is not True:
            return None
        aliases = {f"metal{n}", f"met{n}", f"m{n}", f"mt{n}"}
        for conductor in profile.conductors:
            if str(conductor.name).casefold() not in aliases:
                continue
            value = conductor.sheet_rho_ohm_sq
            if type(value) is float and value > 0.0:
                return value
        return None

    def via_key(self, n: int) -> LayerKey:
        """GDS key for the cut layer between metal ``n`` and metal ``n+1``
        (``via_key(1)`` is Via1). ``via_key(0)`` is the diffusion/poly contact."""
        self._validate_layout_authority()
        if n == 0:
            return self.contact_key()
        for name in (f"via{n}", f"vi{n}", f"v{n}"):
            key = self._rules._data.layermap.get(name)
            if key is not None:
                return key
        # sky130's M1<->M2 cut is named "via"
        key = self._rules._data.layermap.get("via")
        if n == 1 and key is not None:
            return key
        raise KeyError(f"{self.pdk_id} has no via{n} layer")

    def contact_key(self) -> LayerKey:
        """GDS key for the diffusion/poly -> metal1 contact cut layer."""
        self._validate_layout_authority()
        for name in ("contact", "licon1", "mcon", "co"):
            key = self._rules._data.layermap.get(name)
            if key is not None:
                return key
        return self._build_key(LogicalLayer.CONTACT)

    def label_key(self, metal_n: int = 1) -> LayerKey:
        """GDS key for METAL text labels on metal ``metal_n``.

        The drawing layer's GDS layer number with the per-PDK label datatype
        (gf180mcu: datatype 10; sky130: datatype 5). Other PDKs retain the
        drawing-datatype fallback. Source ports may have a separate purpose,
        exposed by :meth:`pin_label_key`.
        """
        layer, _dt = self.metal_key(metal_n)
        dt = _LABEL_DATATYPE.get(self.pdk_id)
        if dt is None:
            return (layer, _dt)
        return (layer, dt)

    def pin_label_key(self, metal_n: int = 1) -> Optional[LayerKey]:
        """Distinct text purpose declaring a source port, when the PDK has one.

        SKY130 Magic reads METnPIN text as ports, separately from METnTXT node
        names. ``None`` preserves the existing label contract for other PDKs;
        it does not invent a port purpose for an undeclared technology.
        """
        layer, _dt = self.metal_key(metal_n)
        dt = _PIN_LABEL_DATATYPE.get(self.pdk_id)
        return None if dt is None else (layer, dt)

    # ---- build-time layer keys (device builder) ----------------------------
    def assembly_boundary_key(self) -> Optional[LayerKey]:
        """PDK-declared nonconducting assembly outline, when one is required."""
        self._validate_layout_authority()
        key = self._adapter.assembly_boundary_layer
        if key is None:
            return None
        if (type(key) is not tuple or len(key) != 2
                or any(type(value) is not int or not 0 <= value <= 65535
                       for value in key)):
            raise ValueError("assembly boundary layer must be an exact GDS layer key")
        return key

    def build_layer(self, kind: str) -> Optional[LayerKey]:
        """GDS key for a build-time layer a device builder draws.

        ``kind`` is one of:
          ``"ndiff"`` / ``"pdiff"`` / ``"diff"``  -- active / diffusion
          ``"poly"``                              -- gate poly
          ``"contact"``                           -- diff/poly contact cut
          ``"metal1"``                            -- first metal
          ``"nplus"`` / ``"pplus"``               -- s/d implant markers
          ``"nwell"``                             -- n-well
          ``"pwell"``                             -- drawn p-well (None when the
                                                     baseline flow uses the global
                                                     substrate)
          ``"dnwell"``                            -- deep n-well
          ``"tap"``                               -- the substrate-tap implant
                                                     (p+ on a p-substrate process)

        Returns ``None`` when the PDK does not draw that layer (e.g. ``pwell`` on
        a non-deep-nwell baseline). Raises ``KeyError`` for an unknown ``kind``.
        """
        k = kind.lower()
        logical_map = {
            "ndiff": LogicalLayer.NDIFF,
            "pdiff": LogicalLayer.PDIFF,
            "diff": LogicalLayer.NDIFF,
            "poly": LogicalLayer.POLY,
            "contact": LogicalLayer.CONTACT,
            "metal1": LogicalLayer.METAL1,
            "nplus": LogicalLayer.NPLUS,
            "pplus": LogicalLayer.PPLUS,
            "nwell": LogicalLayer.NWELL,
            "pwell": LogicalLayer.PWELL,
            "dnwell": LogicalLayer.DNWELL,
        }
        if k == "tap":
            # the substrate tap implant: p+ ties the p-substrate (NMOS bulk).
            return self._build_key(LogicalLayer.PPLUS)
        if k not in logical_map:
            raise KeyError(
                f"unknown build-layer kind {kind!r}; known: "
                f"{sorted(logical_map) + ['tap']}"
            )
        return self._build_key(logical_map[k])

    def marker_key(self, name: str) -> Optional[LayerKey]:
        """GDS key for a non-routing MARKER layer resolved by rule-container token
        (e.g. ``"sab"`` salicide-block, ``"res_mk"`` resistor marker, ``"resistor"``
        high-Rs poly marker). Returns ``None`` when the PDK does not declare the
        marker. Used by the passive-device builder, which needs the gf180
        salicide-block + resistor-marker layers a MOS never draws."""
        self._validate_layout_authority()
        for cand in (name, name.replace("_", ""), name + "_mk"):
            key = self._rules._data.layermap.get(cand)
            if key is not None:
                return key
        return None

    def _build_key(self, logical: LogicalLayer) -> Optional[LayerKey]:
        """Resolve a LogicalLayer to a GDS key via the backend map (gf180/sky130)
        or the rule-container layermap by canonical token (generic PDKs)."""
        self._validate_layout_authority()
        if self._logical is not None:
            g: Optional[GdsLayer] = self._logical.gds_layer(logical)
            if g is not None:
                return (g.layer, g.datatype)
            # backend map returned None (e.g. PWELL_MARKER): for the base members
            # this means "not drawn"; fall through to the token map for the new
            # implant members in case a generic token exists.
        # generic: map the logical name to a canonical layermap token
        token = _LOGICAL_TOKEN.get(logical)
        if token is None:
            return None
        return self._rules._data.layermap.get(token)

    # ---- spacing / width / enclosure ---------------------------------------
    def min_width(self, layer_name: str) -> Optional[int]:
        """Min width (nm) for a rule-container layer name (e.g. ``"metal1"``)."""
        return self._rules.min_width(layer_name)

    def min_space(self, layer_or_pair) -> Optional[int]:
        """Min edge-to-edge spacing (nm).

        Accepts either a single rule-container layer name (e.g. ``"metal1"`` ->
        the same-layer min spacing) or a 2-tuple ``(layer_a, layer_b)`` of layer
        names (returns the conservative max of the two same-layer minima, which is
        the clearance a router lane must hold so it satisfies whichever layer the
        path uses). Returns ``None`` if neither layer carries a spacing rule.
        """
        if isinstance(layer_or_pair, (tuple, list)):
            a, b = layer_or_pair
            sa = self._rules.min_spacing(a)
            sb = self._rules.min_spacing(b)
            vals = [v for v in (sa, sb) if v is not None]
            return max(vals) if vals else None
        return self._rules.min_spacing(layer_or_pair)

    def min_area(self, layer_name: str) -> Optional[int]:
        """Min polygon area (nm^2) for a rule-container layer name (e.g.
        ``"metal1"``), or ``None`` when the PDK declares no min-area for it.

        Sourced from the deck's ``with_area`` rule (the foundry parsed
        ``<layer>_min_area``); for an adapter that is silent the rule container
        seeds a node-scaled CONSOLIDATED fallback (``min_width^2`` -- the smallest
        legal square feature), the same provenance convention the other
        node-scaled rules use. A router / device builder uses this to grow an
        under-area metal pad to a min-area-satisfying rectangle. gf180mcu /
        sky130 return their hand-verified deck values (parity)."""
        return self._rules.min_area(layer_name)

    def min_hole_area(self, layer_name: str) -> Optional[int]:
        """Minimum enclosed hole area on *layer_name* in nm^2, or ``None``."""
        return self._rules.min_hole_area(layer_name)

    def width_spacing_entries(self, layer_name: str):
        """Canonical immutable width bands and optional merged-geometry rules."""
        return self._rules.width_spacing_entries(layer_name)

    def enclosure(self, inner: str, outer: str) -> Optional[int]:
        """Enclosure (nm) of ``inner`` by ``outer`` (e.g.
        ``enclosure("contact", "metal1")``). Delegates to the rule container."""
        return self._rules.enclosure(inner, outer)

    def line_end_ext(self, layer_name: str) -> Optional[int]:
        """Metal end-of-line (line-end) extension (nm) past a cut for a metal layer
        name (e.g. ``"metal1"``), or ``None`` when the PDK declares no line-end rule.

        Sourced from the deck's per-edge end-of-line rule -- the
        ``metal.enclosing(cut, v.um, projection)`` idiom the synthesized DRC deck
        emits as ``M1.LE.1`` (parsed key ``<layer>_line_end``). The value is how far
        a metal wire END must overhang the contact/via cut it lands on (the
        line-end-side projection). A device/router builder uses it to extend an
        M1-over-contact pad past the cut so the foundry line-end check passes.

        gf180mcu has NO parsed metal line-end rule -> returns ``None`` (the caller
        treats that as a 0-nm extension, so gf180 geometry is byte-unchanged). open_pdk
        returns 60 nm (M1 EOL ext past CT = 0.06um)."""
        try:
            parsed = self._adapter.drc_rules()
        except Exception:
            return None
        entry = parsed.get(f"{layer_name}_line_end") if parsed else None
        if not entry:
            return None
        v_nm = entry.get("value_nm")
        if v_nm is not None:
            return int(v_nm)
        v_um = entry.get("value_um")
        return int(round(float(v_um) * 1000.0)) if v_um is not None else None

    def via_rule(self, name: str):
        """The interned :class:`~clay.layout.pdk_rules.ViaRule` for a cut
        layer (``"contact"`` / ``"via1"`` / ``"via2"``): size / spacing /
        enclosure_below / enclosure_above."""
        return self._rules.via(name)

    def via_size(self, name: str) -> Optional[int]:
        """Exact cut size (nm) for a via/contact name."""
        return self._rules.via_size(name)

    def via_spacing(self, name: str) -> Optional[int]:
        """Cut-to-cut spacing (nm) for a via/contact name."""
        return self._rules.via_spacing(name)

    def mim_forbidden_cut_touch_rules(
        self,
    ) -> Tuple[Tuple[LayerKey, LayerKey, LayerKey], ...]:
        """Resolved ``(cut, bottom_metal, marker)`` MIM touch exclusions.

        The adapter declares canonical layer tokens from its foundry deck.  The
        router receives only resolved GDS keys and applies the same component
        selection for every PDK; an undeclared PDK returns an empty tuple.
        """

        self._validate_layout_authority()
        if self.pdk_id == "gf180mcu" and self._mim_option == "B":
            from clay.pdk.gf180mcu import gf180_mim_stack_spec

            stack = gf180_mim_stack_spec(
                "B", metal_level=self._metal_level or "5LM"
            )
            declared = (
                (
                    str(stack["forbidden_cut"]),
                    str(stack["bottom_metal"]),
                    str(stack["top_plate"]),
                ),
            )
        else:
            declared = tuple(
                getattr(self._adapter, "mim_forbidden_cut_touch", ()) or ()
            )
        resolved = []
        layermap = self._rules._data.layermap
        for rule in declared:
            if len(rule) != 3:
                raise ValueError(
                    "MIM forbidden-cut rule must be "
                    "(cut, bottom_metal, top_marker)"
                )
            keys = []
            for token in rule:
                name = str(token).strip().casefold()
                key = layermap.get(name)
                if key is None:
                    raise KeyError(
                        f"{self.pdk_id} MIM cut-touch rule references "
                        f"unknown layer token {token!r}"
                    )
                keys.append((int(key[0]), int(key[1])))
            resolved.append(tuple(keys))
        return tuple(resolved)

    def mim_bottom_routing_clearance_rules(
        self,
    ) -> Tuple[Tuple[LayerKey, LayerKey, int], ...]:
        """Legacy strict bands only, preserving wire/via/fill and placement use.

        A complete-union rule has no fixed band projection: a local attachment
        can make otherwise forbidden metal legal.  Returning such a rule here
        would silently change the established GF180 contract for every caller.
        """
        return tuple((rule.metal_layer, rule.marker_layer, rule.spacing_nm)
                     for rule in self.mim_routing_clearance_rules()
                     if rule.predicate == "strict_bottom_band")

    def mim_metal_geometry_rules(self) -> Tuple[ResolvedMimMetalGeometryRule, ...]:
        """Complete-union view of the same canonical MIM declaration."""
        return tuple(rule for rule in self.mim_routing_clearance_rules()
                     if rule.predicate != "strict_bottom_band")

    def mim_routing_clearance_rules(self) -> Tuple[ResolvedMimMetalGeometryRule, ...]:
        """Resolve legacy strict and typed complete-union rules in one place."""
        from clay.pdk import MimMetalGeometryRule

        self._validate_layout_authority()
        resolved = []
        identifiers = set()
        layermap = self._rules._data.layermap
        if self.pdk_id == "gf180mcu" and self._mim_option == "B":
            from clay.pdk.gf180mcu import gf180_mim_stack_spec

            stack = gf180_mim_stack_spec(
                "B", metal_level=self._metal_level or "5LM"
            )
            clearance_decls = (
                (
                    str(stack["bottom_metal"]),
                    str(stack["top_plate"]),
                    int(stack["bottom_routing_clearance_nm"]),
                ),
            )
        else:
            clearance_decls = getattr(self._adapter, "mim_bottom_routing_clearance_nm", ()) or ()
        for declaration in clearance_decls:
            if isinstance(declaration, MimMetalGeometryRule):
                rule = declaration
            else:
                if len(declaration) != 3:
                    raise ValueError("MIM bottom clearance rule must be (bottom_metal, top_marker, clearance_nm)")
                metal, marker, clearance = declaration
                clearance = int(clearance)
                if clearance < 0:
                    raise ValueError("MIM bottom routing clearance must be non-negative")
                rule = MimMetalGeometryRule(
                    f"strict_bottom_clearance:{metal}:{marker}", metal, marker,
                    "strict_bottom_band", 0, clearance, "square",
                )
            if not isinstance(rule.rule_id, str) or not rule.rule_id or rule.rule_id in identifiers:
                raise ValueError("MIM geometry rule requires a unique nonempty rule id")
            if rule.predicate not in {"strict_bottom_band", "derived_unrelated", "local_surround"}:
                raise ValueError("unsupported MIM metal geometry predicate")
            strict = rule.predicate == "strict_bottom_band"
            if rule.metric != ("square" if strict else "euclidian"):
                raise ValueError("unsupported MIM metal geometry metric")
            if (type(rule.grow_nm) is not int or rule.grow_nm < 0
                    or type(rule.spacing_nm) is not int or rule.spacing_nm < (0 if strict else 1)
                    or (rule.predicate != "derived_unrelated" and rule.grow_nm != 0)):
                raise ValueError("invalid MIM metal geometry distances")
            layers = []
            for token in (rule.metal_layer, rule.marker_layer):
                key = layermap.get(str(token).strip().casefold())
                if key is None:
                    raise KeyError(f"{self.pdk_id} MIM geometry rule references unknown layer {token!r}")
                layers.append((int(key[0]), int(key[1])))
            identifiers.add(rule.rule_id)
            resolved.append(ResolvedMimMetalGeometryRule(
                rule.rule_id, layers[0], layers[1], rule.predicate,
                rule.grow_nm, rule.spacing_nm, rule.metric,
            ))
        return tuple(resolved)

    def mim_top_via_marker_clearance_rules(
        self,
    ) -> Tuple[Tuple[LayerKey, LayerKey, int], ...]:
        """Resolved ``(top_via, marker, clearance_nm)`` MIM rules."""

        self._validate_layout_authority()
        declared = tuple(
            getattr(
                self._adapter, "mim_top_via_marker_clearance_nm", ()
            ) or ()
        )
        resolved = []
        layermap = self._rules._data.layermap
        for rule in declared:
            if len(rule) != 3:
                raise ValueError(
                    "MIM top-via clearance rule must be "
                    "(top_via, top_marker, clearance_nm)"
                )
            via_token, marker_token, clearance_nm = rule
            layers = []
            for token in (via_token, marker_token):
                name = str(token).strip().casefold()
                key = layermap.get(name)
                if key is None:
                    raise KeyError(
                        f"{self.pdk_id} MIM top-via-clearance rule references "
                        f"unknown layer token {token!r}"
                    )
                layers.append((int(key[0]), int(key[1])))
            clearance = int(clearance_nm)
            if clearance < 0:
                raise ValueError(
                    "MIM top-via marker clearance must be non-negative"
                )
            resolved.append((layers[0], layers[1], clearance))
        return tuple(resolved)

    def implant_well(self):
        """The :class:`~clay.pdk.ImplantWellRules` (nplus/pplus min width+space,
        gate-overlap enclosure, nwell-to-pcomp enclosure) for this PDK."""
        return self._adapter.implant_well_rules()

    # ---- provenance / sim / signoff surfaces (delegate to the adapter) ------
    def device_bindings(self) -> Dict[str, str]:
        """Clay device-class -> concrete model/subckt name (from the adapter)."""
        return dict(self._adapter.device_bindings)

    def klayout_decks(self) -> Dict[str, Optional[object]]:
        """Executable KLayout DRC/LVS deck paths (``{"drc": .., "lvs": ..}``)."""
        return self._adapter.klayout_decks()

    def drc_rules(self) -> Dict[str, object]:
        """The PARSED foundry DRC rule values (the same ``{"rule_key": {value_nm,
        value_um, ...}}`` map the synthesized KLayout deck consumes), delegated to
        the adapter. ``{}`` when the PDK exposes no parsed deck rules (its rules
        come through the static builder + a real foundry deck instead)."""
        try:
            return self._adapter.drc_rules()
        except Exception:
            return {}

    def lvs_reference_models(self) -> Dict[str, str]:
        """device_class -> model name the LVS extractor compares against."""
        return self._adapter.lvs_reference_models()

    def substrate_net(self) -> str:
        """The global substrate net name (NMOS bulk tie)."""
        return self._adapter.substrate_net()

    def bulk_topology(self) -> Dict[str, object]:
        """How this PDK distinguishes N/P bulk + taps (see
        :meth:`clay.pdk.PdkAdapter.bulk_topology`)."""
        return self._adapter.bulk_topology()

    def deep_nwell(self):
        """The adapter-owned triple-well recipe, or ``None`` if unproven."""

        return self._adapter.deep_nwell()

    # ---- introspection ----
    def layer_names(self) -> list:
        """Rule-container layer names available for ``min_width`` / ``min_space``."""
        return self._rules.layer_names()

    def via_names(self) -> list:
        """Cut-layer names available for ``via_rule`` / ``via_size``."""
        return self._rules.via_names()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"LayoutTech({self.pdk_id})"


# canonical layermap token for each LogicalLayer (generic-PDK build-key fallback)
_LOGICAL_TOKEN: Dict[LogicalLayer, str] = {
    LogicalLayer.NDIFF: "diff",
    LogicalLayer.PDIFF: "diff",
    LogicalLayer.POLY: "poly",
    LogicalLayer.CONTACT: "contact",
    LogicalLayer.METAL1: "metal1",
    LogicalLayer.NWELL: "nwell",
    LogicalLayer.NPLUS: "nplus",
    LogicalLayer.PPLUS: "pplus",
    LogicalLayer.PWELL: "pwell",
    LogicalLayer.DNWELL: "dnwell",
}


def _logical_map_for(pdk_id: str):
    """Return the backend LogicalLayer map for gf180/sky130, else None (generic
    PDKs resolve build keys through the rule-container layermap token table)."""
    key = pdk_id.lower()
    if key == "gf180mcu":
        return Gf180mcuLayerMap()
    if key == "sky130":
        return Sky130LayerMap()
    return None


# Token guarding the constructor -- only resolve(...) can build a LayoutTech.
_TECH_TOKEN = object()


__all__ = ["LayoutTech", "LayerKey"]
