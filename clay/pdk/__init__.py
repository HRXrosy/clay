"""
Clay PDK adapters.

Each PDK lives in its own submodule (e.g. `clay.pdk.sky130`,
`clay.pdk.gf180mcu`). A PDK adapter publishes:

  * `device_bindings`: mapping from Clay device-class strings (the
    process-independent kind, e.g. 'nmos', 'pmos_pair') to the
    simulator's concrete subckt / model name.
  * `mu_cox`: μ·C_ox lookup for the closed-form sizing solver
    (per device kind). Replaced by a real BSIM-extracted LUT in a
    later version.
  * `corner_file(corner_name)`: filesystem path to the simulator's
    corner include file.
  * Process constants (e.g. min_L, vdd_nom).

The simulator drivers (`clay.drivers.ngspice` etc.) consume this
information through the `get_pdk(...)` registry — they no longer
hard-code any PDK-specific knowledge.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Mapping, Optional, Tuple

from .bjt import BjtBinding
from .native_spice import NativeSpiceModelContract
from .geometry import NativeGeometryProvider

if TYPE_CHECKING:
    from clay.pdk._source_snapshot import PdkSourceSnapshot

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PDKS_DIR = _REPO_ROOT / "pdks"


@dataclass(frozen=True)
class Device:
    """A core MOS instantiation handle: a plain `.model` (M-card) or a `.subckt`
    wrapper (X-card), with the node order at the card."""
    kind: str          # "model" or "subckt"
    name: str
    nodes: str = "d g s b"


@dataclass(frozen=True)
class ViaRule:
    """One cut layer's geometry rules in nm, with a ``calibrated`` honesty flag.

    A cut layer (contact / via1 / via2) has an EXACT cut size, a cut-to-cut
    spacing, and the metal/diff enclosure required below and above the cut. Every
    value is nm (integer). ``calibrated`` is True only when the value came from
    the foundry deck; a node-scaled provenance fallback sets it False so a
    consumer can tell a real rule from an estimate.

    This is the LIGHT, adapter-facing via record (distinct from the interned
    ``clay.layout.pdk_rules.ViaRule`` flyweight, which the placer/router
    container uses). ``provenance`` carries a short tag (e.g. ``"parsed-deck"``,
    ``"node-scaled"``, ``"documented"``).
    """

    name: str               # "contact" | "via1" | "via2"
    cut_size_nm: int
    spacing_nm: int
    enclosure_below_nm: int
    enclosure_above_nm: int
    below: str = ""         # the layer below (e.g. "metal1")
    above: str = ""         # the layer above (e.g. "metal2")
    calibrated: bool = False
    provenance: str = "node-scaled"


@dataclass(frozen=True)
class MimMetalGeometryRule:
    """A foundry MIM rule evaluated on complete drawn metal and marker unions.

    Layer names are canonical layermap tokens.  ``derived_unrelated`` grows
    the marker/metal intersection before checking unrelated components;
    ``local_surround`` checks exposed metal edges against marker edges.  These
    distinct predicates must not be collapsed into one clearance radius.
    """

    rule_id: str
    metal_layer: str
    marker_layer: str
    predicate: str
    grow_nm: int
    spacing_nm: int
    metric: str = "euclidian"


@dataclass(frozen=True)
class ImplantWellRules:
    """Implant + well design-rule values (nm) needed to draw + check implant
    geometry, with a ``calibrated`` honesty flag.

    Fields map to the gf180 rule ids cited in the docstrings:

      * ``nplus_min_width`` / ``nplus_min_space`` -- NP.1 / NP.2
      * ``pplus_min_width`` / ``pplus_min_space`` -- PP.1 / PP.2
      * ``nplus_gate_enclosure`` / ``pplus_gate_enclosure`` -- NP.5a / PP.5a
        (the implant must overlap the matching gate by this much)
      * ``nwell_pcomp_enclosure`` -- DF.4c (nwell overlap of the enclosed PCOMP)
      * ``nwell_ncomp_enclosure`` -- DF.4d (nwell overlap / spacing to NCOMP)

    ``calibrated`` is True when the values trace to the foundry deck. A value the
    deck is silent about uses a node-scaled fallback and ``calibrated=False``.
    """

    nplus_min_width: int
    nplus_min_space: int
    pplus_min_width: int
    pplus_min_space: int
    nplus_gate_enclosure: int
    pplus_gate_enclosure: int
    nwell_pcomp_enclosure: int
    nwell_ncomp_enclosure: int
    calibrated: bool = False
    provenance: str = "node-scaled"


@dataclass(frozen=True)
class DeepNWellRules:
    """PDK-owned recipe for one isolated NMOS p-well inside deep N-well.

    All distances are foundry-rule nanometres.  The generic materializer owns
    only the topology (body p-tap, deep-well n-tap, optional outer substrate
    guard); layer tokens, enclosures and extracted model names stay here.
    """

    # Some foundries draw a dedicated low-voltage p-well marker (GF180), while
    # others derive the isolated p-well as the hole of an N-well ring inside
    # DNW (external_pdk/external_pdk).  ``None`` means the p-well is derived and must not be
    # fabricated as a marker polygon.
    pwell_layer: Optional[str]
    deep_nwell_layer: str
    pwell_enclose_nactive_nm: int
    pwell_enclose_ptap_nm: int
    deep_nwell_enclose_pwell_nm: int
    deep_nwell_enclose_ntap_nm: int
    ntap_to_pwell_nm: int
    outer_ptap_to_deep_nwell_nm: int
    outer_substrate_guard_required: bool
    lvs_models: Mapping[str, str]
    # Simulator models for the full-extraction classes above.  The keys are
    # semantic device classes (``nmos`` / ``nmos3v``), not circuit or instance
    # names.  Keeping this beside ``lvs_models`` makes the isolated-device PEX
    # binding an explicit PDK fact instead of a composer-side model guess.
    sim_models: Mapping[str, str] = field(default_factory=dict)
    # How the DNW is contacted. ``direct_dnw`` puts the N+ tap directly in a
    # drawable DNW (GF180). ``nwell_ring`` retains/builds the foundry-required
    # N-well support ring which overlaps DNW on its inner edge and extends past
    # it on its outer edge.
    deep_tap_well_mode: str = "direct_dnw"
    nwell_inner_overlap_nm: int = 0
    nwell_outer_extension_nm: int = 0
    # Optional foundry device-recognition markers drawn over the isolated
    # device region (for example external_pdk MSE DNWTR/MOSCKT/NFDMK for its 4T class).
    selector_layers: Tuple[str, ...] = ()
    calibrated: bool = False
    provenance: str = "unsupported"


@dataclass
class PdkAdapter:
    pdk_id: str
    # The legacy sizing surface (gf180/sky130/open_pdk use it). All optional now so a
    # commercial HSPICE-lib PDK adapter need only declare what it actually uses.
    device_bindings: Mapping[str, str] = field(default_factory=dict)
    mu_cox: Mapping[str, float] = field(default_factory=dict)
    corner_file_resolver: Optional[Callable[[str], Optional[Path]]] = None
    min_L: float = 0.0
    vdd_nom: float = 0.0
    extra: Mapping[str, object] = field(default_factory=dict)
    # Semantic PNP/NPN -> foundry model and fixed/PCell geometry evidence.
    bjt_bindings: Mapping[str, BjtBinding] = field(default_factory=dict)
    # v0.0.91 (LC-VCO accumulation-mode varactor):
    # Some PDK device models depend on parameter sets that live OUTSIDE the
    # per-MOSFET corner file. For sky130 in particular, the accumulation-mode
    # N-well varactor (`sky130_fd_pr__cap_var_*`) subckt references resistor
    # sheet-rho params (`rnw`, `rp1`, `rcn`, `rcp1`) that are defined in
    # `r+c/res_typical__cap_typical.spice` — not in `corners/tt.spice`. The
    # driver consults `extra_includes(corner)` to emit additional `.include`
    # directives after the primary corner file. Default returns an empty
    # list — adapters that don't need it ignore this hook.
    extra_includes_resolver: Optional[Callable[[str], list[Path]]] = None

    # --- layout SIGNOFF deck surface (optional) ---
    # An EXECUTABLE KLayout DRC/LVS deck for this PDK -- distinct from the numeric
    # rules drc_rules() parses: a deck is what KLayout actually RUNS. The flow's
    # DRC/LVS steps resolve their deck through get_pdk(pdk).klayout_decks() so
    # signoff is PDK-data-driven instead of hardcoding one PDK. A PDK that ships
    # (or generates) decks sets this resolver; others leave it None and the
    # signoff step skips gracefully.
    klayout_deck_resolver: Optional[Callable[[], Mapping[str, Optional[Path]]]] = None
    # When True (and no explicit resolver), klayout_decks() SYNTHESIZES a free
    # KLayout DRC deck from THIS PDK's own parsed rules + layermap -- the deck a
    # closed foundry never ships for free tools. Commercial adapters (external_pdk/external_pdk)
    # set this; their only in-repo decks are proprietary external_drc/external_lvs.
    klayout_drc_from_rules: bool = False
    # When True (and no explicit resolver), klayout_decks() SYNTHESIZES a free
    # KLayout 3-terminal CMOS LVS deck from THIS PDK's layermap + nmos/pmos model
    # names (well-based recognition; bulk not extracted without implant layers).
    klayout_lvs_from_rules: bool = False
    # Optional PDK-owned, geometry-only failure knowledge.  The resolver receives
    # this adapter so each record can bind the current deck and shared
    # implementation digests; stale serialized evidence can never remain active.
    failure_rule_resolver: Optional[Callable[["PdkAdapter"], Tuple[object, ...]]] = None

    # --- commercial HSPICE-lib sim binding (external_pdk/external_pdk; optional) ---
    # When `model_lib_rel` is set, the PDK is bound by `.lib "<abs>" <corner>` (+ extra
    # sections + param overrides) rather than a single corner include. This is the
    # surface the portable harness + the layout/L5 pipeline use; it lives on the ONE
    # PdkAdapter so there is no parallel per-PDK recipe table.
    # Optional authoritative install resolver.  Commercial adapters whose PDK
    # tree can live outside this checkout bind their module-level environment
    # resolver here.  When present it is strict: ``None``/invalid never falls
    # through to the legacy worktree-adjacent ``dir_prefix`` discovery path.
    install_root_resolver: Optional[Callable[[], Optional[Path]]] = None
    install_root_declaration_resolver: Optional[Callable[[], bool]] = None
    dir_prefix: Optional[str] = None        # legacy: resolves pdks/<dir> by prefix
    node_nm: Optional[int] = None
    lmin_um: Optional[float] = None         # process min channel length (um)
    model_lib_rel: Optional[str] = None
    lib_corner: Optional[str] = None        # the .lib typical section name (e.g. "TT")
    extra_lib_sections: Tuple[str, ...] = ()
    extra_lib_sections_after: Tuple[str, ...] = ()
    sim_corner_aliases: Mapping[str, str] = field(default_factory=dict)
    param_overrides: Mapping[str, float] = field(default_factory=dict)
    nmos: Optional[Device] = None
    pmos: Optional[Device] = None
    runnable: bool = False                  # proven to simulate on ngspice
    drc_dialect: Optional[str] = None       # passed to the shared deck parser; None = auto

    # --- P0 bulk / substrate surface (optional, per-adapter override) ---
    # The global substrate net name (the NMOS bulk tie on a p-substrate process).
    # Defaults to "VSS"; an adapter on a different convention overrides it.
    substrate_net_name: str = "VSS"
    # A small per-adapter override for bulk_topology() fields that cannot be
    # derived from the layermap alone (e.g. drawn_p_tap). Keys mirror the
    # bulk_topology() dict; present keys win over the layermap-derived defaults.
    bulk_topology_override: Mapping[str, object] = field(default_factory=dict)
    # Optional triple-well recipe.  ``None`` means this adapter has not proven a
    # materializable deep-N-well path and an isolation declaration fails closed.
    deep_nwell_override: Optional[DeepNWellRules] = None
    # Optional LVS reference-model override: device_class -> model name + an
    # optional poly-R model / sheet-rho. When unset, lvs_reference_models()
    # derives nmos/pmos from device_bindings.
    lvs_models_override: Mapping[str, str] = field(default_factory=dict)
    poly_res_model: Optional[str] = None    # poly resistor model name, if known
    poly_sheet_rho_ohm_sq: Optional[float] = None  # poly sheet resistance, if known
    # Optional explicit via / implant rule overrides. An adapter whose root() does
    # not resolve a parseable deck in this environment (gf180mcu ships its rules
    # via the static pdk_rules_v174 builder, not via dir_prefix) wires its REAL,
    # calibrated values here so via_rules()/implant_well_rules() return them
    # directly. When unset, the methods derive from the parsed deck + node-scaled
    # fallback. Keys: via_rules_override -> {name: ViaRule}.
    via_rules_override: Optional[Mapping[str, "ViaRule"]] = None
    implant_well_override: Optional["ImplantWellRules"] = None
    # Same-layer placement clearances not covered by the compact router rule
    # surface. Keys are canonical layermap tokens and values are exact nm from
    # the foundry deck. The same-net table is a signed, narrower override used
    # only when topology proves both owners share that layer's electrical domain.
    placement_layer_spacing_nm: Mapping[str, int] = field(default_factory=dict)
    placement_same_net_layer_spacing_nm: Mapping[str, int] = field(
        default_factory=dict
    )
    # Non-conducting enclosure/selector layers that may geometrically merge
    # when placement proves both device owners belong to one bulk domain.
    # Spacing still applies when the shapes remain disjoint; active, contacts,
    # and routing metals are intentionally never listed here.
    placement_same_domain_mergeable_layers: Tuple[str, ...] = ()
    # Foundry-declared MIM cut-touch exclusions, expressed as canonical
    # ``(cut, bottom_metal, top_marker)`` layer tokens.  A rule means the cut
    # may not touch any connected bottom-metal component selected by the marker.
    # This is PDK data (for example GF180 MIM.10), not a router/device-name
    # special case.  Adapters whose deck has no such rule leave it empty.
    mim_forbidden_cut_touch: Tuple[Tuple[str, str, str], ...] = ()
    # Foundry MIM routing clearances.  Legacy triples
    # ``(bottom_metal, top_marker, clearance_nm)`` retain their strict net-blind
    # band semantics.  Typed rules extend this SAME declaration for predicates
    # that require complete drawn unions: local attachment can change their
    # result, so partial search and placement cannot lower them to fixed bands.
    mim_bottom_routing_clearance_nm: Tuple[
        Tuple[str, str, int] | MimMetalGeometryRule, ...
    ] = ()
    # Foundry MIM separation between an ordinary top-via cut and the MIM top
    # marker, ``(cut, top_marker, clearance_nm)``.  This is distinct from an
    # enclosure rule for the intentional via that forms the top-plate terminal.
    mim_top_via_marker_clearance_nm: Tuple[Tuple[str, str, int], ...] = ()
    # Optional extra layermap entries merged INTO the parsed layermap() result.
    # The default layermap parser reads a PDK's external_drc DRC deck, which for some
    # foundries only declares the lower routing metals (e.g. external_pdk's deck stops at
    # M3). An adapter that knows additional drawn layers from the foundry stream
    # map (e.g. upper metals M4-M6 / vias V3-V5) wires them here, sourced from a
    # cited PDK file. Keys are canonical tokens (metal4/via3/...) -> (gds, dt).
    # Present keys win over the parsed layermap (explicit override). Default empty
    # -> layermap() is byte-unchanged for every adapter that does not set it.
    layermap_extra: Mapping[str, Tuple[int, int]] = field(default_factory=dict)
    # Optional fail-closed installed-source validator for adapters that retain
    # cited layout facts when their proprietary PDK is absent. LayoutTech calls
    # this before reading its cached layer snapshot, so a later root switch or
    # same-length authority-file rewrite cannot bypass the adapter's byte check.
    layout_authority_validator: Optional[Callable[[], None]] = None
    # Per-metal PREFERRED routing direction (the HVH layer discipline), keyed by
    # 1-based PDK metal index -> 'H' | 'V'. This is process-convention DATA (which
    # physical metal runs horizontal vs vertical in the standard detailed-routing
    # discipline of THIS stack), the analogue of a LEF routing-layer DIRECTION
    # attribute -- NOT a value measured from any reference layout. A router uses it
    # as a soft per-layer step-cost preference. Default empty -> the layout tech
    # falls back to the generic alternation (odd metals horizontal, even vertical),
    # so every adapter that does not declare axes is byte-identical. An adapter whose
    # analog stack runs its local device metal vertically (M1 following vertical
    # poly fingers) declares the opposite parity here (M1=V, M2=H, M3=V, ...).
    preferred_routing_axes: Mapping[int, str] = field(default_factory=dict)
    # Manufacturing grid from the installed foundry rule authority, in integer
    # nanometres. An explicit fact is needed when signoff runs in a container and
    # its deck is unavailable on the host. None retains local-deck discovery.
    manufacturing_grid_nm: Optional[int] = None
    # Optional nonconducting cell/assembly boundary purpose. The canonical
    # emitter encloses actual artwork and original source-owned abutment boxes
    # on this layer; it must never infer a larger outline from density targets.
    assembly_boundary_layer: Optional[Tuple[int, int]] = None
    # Explicit native X-subcircuit contracts. This does not imply a physical
    # builder; imported native layouts use independently bound signoff decks.
    native_spice_models: Mapping[str, "NativeSpiceModelContract"] = field(default_factory=dict)
    # Installed implementation, separate from simulation-only model contracts.
    # The validator is source-only; the materializer may lazily import its tools.
    native_geometry_provider: Optional[NativeGeometryProvider] = None

    # Parsed foundry facts are cached by the digest of the exact source deck
    # contents, never by adapter identity alone.  Source paths are retained only
    # to make the steady-state freshness check a direct stat/read rather than a
    # recursive PDK discovery walk.  Public accessors still return copies because
    # several legacy consumers treat these dictionaries as mutable scratch data.
    _layermap_cache: Dict[str, Dict[str, Tuple[int, int]]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _drc_rules_cache: Dict[str, Dict[str, dict]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _layermap_source_paths: Tuple[Path, ...] = field(
        default=(), init=False, repr=False, compare=False
    )
    _drc_rule_source_paths: Tuple[Path, ...] = field(
        default=(), init=False, repr=False, compare=False
    )
    _deck_digest_cache: Dict[
        Tuple[str, ...],
        Tuple[Tuple[Tuple[int, int, int, int], ...], str],
    ] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _source_snapshot_cache_root: Optional[Path] = field(
        default=None, init=False, repr=False, compare=False
    )
    _source_snapshot_cache: Optional["PdkSourceSnapshot"] = field(
        default=None, init=False, repr=False, compare=False
    )

    # back-compat aliases for the commercial fields (older callers / the recipe shim).
    @property
    def corner(self) -> Optional[str]:
        return self.lib_corner

    @property
    def extra_sections(self) -> Tuple[str, ...]:
        return self.extra_lib_sections

    @property
    def vdd(self) -> float:
        return self.vdd_nom

    @property
    def lmin(self) -> Optional[float]:
        return self.lmin_um

    def corner_file(self, corner: str = "tt") -> Optional[Path]:
        if self.corner_file_resolver is None:
            return None
        return self.corner_file_resolver(corner)

    # ---- commercial sim binding + physical surface (delegate to shared libraries) ----
    def root(self) -> Optional[Path]:
        if self.install_root_resolver is not None:
            raw = self.install_root_resolver()
            if raw is None:
                return None
            try:
                resolved = Path(raw).resolve(strict=True)
            except (OSError, TypeError, ValueError):
                return None
            return resolved if resolved.is_dir() else None
        if not self.dir_prefix or not _PDKS_DIR.is_dir():
            return None
        exact = _PDKS_DIR / self.dir_prefix
        if exact.is_dir():
            return exact.resolve()
        for d in sorted(_PDKS_DIR.iterdir()):
            if d.is_dir() and d.name.lower().startswith(self.dir_prefix.lower()):
                return d.resolve()
        return None

    @staticmethod
    def _confined_existing_file(root: Path, relative: str | Path) -> Optional[Path]:
        """Resolve one regular file without allowing it to escape ``root``."""

        raw = Path(relative)
        if raw.is_absolute():
            return None
        try:
            resolved_root = root.resolve(strict=True)
            candidate = (resolved_root / raw).resolve(strict=True)
            candidate.relative_to(resolved_root)
        except (OSError, ValueError):
            return None
        return candidate if candidate.is_file() else None

    @staticmethod
    def _confined_source_paths(
        root: Path, paths: Tuple[Path, ...]
    ) -> Tuple[Path, ...]:
        """Resolve parser authority files and require every one under ``root``."""

        resolved_root = root.resolve(strict=True)
        confined = []
        for raw in dict.fromkeys(Path(path) for path in paths):
            candidate = raw.resolve(strict=True)
            try:
                candidate.relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError(
                    f"{candidate} escapes PDK install root {resolved_root}"
                ) from exc
            confined.append(candidate)
        return tuple(confined)

    @staticmethod
    def _cached_sources_belong_to_root(
        root: Path, paths: Tuple[Path, ...]
    ) -> bool:
        if not paths:
            return False
        try:
            PdkAdapter._confined_source_paths(root, paths)
        except (OSError, ValueError):
            return False
        return True

    def model_lib_path(self) -> Optional[Path]:
        r = self.root()
        if r is None or not self.model_lib_rel:
            return None
        p = self._confined_existing_file(r, self.model_lib_rel)
        if p is not None:
            return p
        # Some foundry OA distributions nest the real tree one level down in a
        # container dir (e.g. pdks/open_pdk/external_pdk/models/...), so root()
        # returns the container and the model lib lives one level deeper. Descend
        # into immediate subdirs to find it. Additive: flat installs (open_pdk/65)
        # already matched above, so this only fires for the nested layout.
        for sub in sorted(r.iterdir()):
            if sub.is_dir():
                try:
                    nested = sub.resolve(strict=True).relative_to(
                        r.resolve(strict=True)
                    )
                except (OSError, ValueError):
                    continue
                q = self._confined_existing_file(
                    r, nested / Path(self.model_lib_rel)
                )
                if q is not None:
                    return q
        return None

    def lib_available(self) -> bool:
        return self.model_lib_path() is not None

    def sim_preamble(self, corner: Optional[str] = None) -> str:
        """`.lib` lines (abs C:/ path) + param overrides for ngspice."""
        lib = self.model_lib_path()
        if lib is None:
            from circuits._harness import PdkNotFound
            raise PdkNotFound(
                f"{self.pdk_id} model lib not found under pdks/ "
                f"(expected <{self.dir_prefix}*>/{self.model_lib_rel})."
            )
        sec = self.resolve_sim_corner(corner)
        libp = lib.as_posix()
        lines = [f'.lib "{libp}" {s}' for s in self.extra_lib_sections]
        lines.append(f'.lib "{libp}" {sec}')
        lines.extend(
            f'.lib "{libp}" {s}' for s in self.extra_lib_sections_after
        )
        lines += [f".param {k}={v}" for k, v in self.param_overrides.items()]
        return "\n".join(lines)

    def resolve_sim_corner(self, corner: Optional[str] = None) -> str:
        """Map canonical Clay corners to this model library's section names."""
        requested = str(corner or self.lib_corner or "tt")
        return str(
            self.sim_corner_aliases.get(
                requested.casefold(),
                self.sim_corner_aliases.get(requested, requested),
            )
        )

    def device(self, which: str) -> Optional[Device]:
        return self.nmos if which.lower().startswith("n") else self.pmos

    def device_card(self, which: str, inst: str, nodes: str, *, w: float, l: float,
                    w_unit: str = "u", l_unit: str = "u") -> str:
        dev = self.device(which)
        prefix = "M" if dev.kind == "model" else "X"
        return f"{prefix}{inst} {nodes} {dev.name} W={w}{w_unit} L={l}{l_unit}"

    def _deck_content_digest(self, paths: Tuple[Path, ...]) -> Optional[str]:
        """Return a stable digest for exact deck bytes, with cheap steady state.

        The fact caches are keyed by the returned content digest.  File metadata
        only memoizes recomputation of that digest; it is never itself a fact-cache
        key.  A read/stat race or transient filesystem failure returns ``None`` so
        the caller cannot reuse or publish facts under an unproved source identity.
        """

        normalized = tuple(dict.fromkeys(Path(path).resolve() for path in paths))
        if not normalized:
            return None

        def signatures() -> Tuple[Tuple[int, int, int, int], ...]:
            values = []
            for path in normalized:
                stat = path.stat()
                values.append((
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                    int(stat.st_ctime_ns),
                    int(getattr(stat, "st_ino", 0)),
                ))
            return tuple(values)

        cache_key = tuple(str(path) for path in normalized)
        try:
            before = signatures()
        except OSError:
            return None
        cached = self._deck_digest_cache.get(cache_key)
        if cached is not None and cached[0] == before:
            return cached[1]

        digest = hashlib.sha256()
        try:
            for ordinal, path in enumerate(normalized):
                digest.update(ordinal.to_bytes(4, "big"))
                with path.open("rb") as stream:
                    while True:
                        chunk = stream.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(len(chunk).to_bytes(8, "big"))
                        digest.update(chunk)
            after = signatures()
        except OSError:
            return None
        if after != before:
            return None
        value = digest.hexdigest()
        self._deck_digest_cache[cache_key] = (after, value)
        return value

    @staticmethod
    def _report_source_paths(
        report: Mapping[str, object], root: Path
    ) -> Tuple[Path, ...]:
        paths = []
        for key in ("deck", "antenna_deck"):
            raw = report.get(key)
            if raw is None or not str(raw):
                continue
            path = Path(str(raw))
            paths.append(path if path.is_absolute() else root / path)
        return tuple(dict.fromkeys(paths))

    def source_snapshot(
        self, *, refresh: bool = False
    ) -> Optional["PdkSourceSnapshot"]:
        """Return the process-local source authority for the selected install.

        A physical flow treats its declared PDK install as immutable input.  The
        first access captures and parses that authority; geometry getters reuse
        the same snapshot instead of rediscovering and reparsing the PDK for
        every device.  ``refresh=True`` starts a new session snapshot explicitly.
        """

        root = self.root()
        if root is None:
            self._source_snapshot_cache_root = None
            self._source_snapshot_cache = None
            return None
        if (
            not refresh
            and self._source_snapshot_cache is not None
            and self._source_snapshot_cache_root == root
        ):
            return self._source_snapshot_cache
        from clay.pdk._source_snapshot import capture_pdk_source_snapshot

        captured = capture_pdk_source_snapshot(self, root)
        self._source_snapshot_cache_root = root
        self._source_snapshot_cache = captured
        return captured

    def layermap(self) -> Dict[str, Tuple[int, int]]:
        if self.install_root_resolver is not None:
            snapshot = self.source_snapshot()
            if snapshot is None:
                explicitly_declared = (
                    self.install_root_declaration_resolver is not None
                    and self.install_root_declaration_resolver()
                )
                # A clean checkout may compile/layout from the tracked cited
                # layer table, but an explicitly selected invalid install is a
                # failed authority and must expose no layout facts.
                return {} if explicitly_declared else dict(self.layermap_extra)
            return snapshot.layermap()
        r = self.root()
        if r is None:
            return dict(self.layermap_extra)
        if self._cached_sources_belong_to_root(r, self._layermap_source_paths):
            digest = self._deck_content_digest(self._layermap_source_paths)
            cached = None if digest is None else self._layermap_cache.get(digest)
            if cached is not None:
                lm = dict(cached)
                lm.update(self.layermap_extra)
                return lm
        from clay.frontend.pdk_layermap import _detect_source, parse_layermap
        # Exceptions deliberately propagate and no fact cache is written.  The
        # next call retries the same source instead of freezing an empty map.
        parsed = dict(parse_layermap(str(r)))
        _kind, source_path = _detect_source(Path(r))
        sources = self._confined_source_paths(
            r, () if source_path is None else (Path(source_path),)
        )
        digest = self._deck_content_digest(sources)
        self._layermap_source_paths = sources
        if digest is not None:
            self._layermap_cache = {digest: dict(parsed)}
        else:
            self._layermap_cache.clear()
        parsed.update(self.layermap_extra)
        return parsed

    def drc_rules(self) -> Dict[str, dict]:
        if self.install_root_resolver is not None:
            snapshot = self.source_snapshot()
            return {} if snapshot is None else snapshot.rules()
        r = self.root()
        if r is None:
            return {}
        if self._cached_sources_belong_to_root(r, self._drc_rule_source_paths):
            digest = self._deck_content_digest(self._drc_rule_source_paths)
            cached = None if digest is None else self._drc_rules_cache.get(digest)
            if cached is not None:
                return deepcopy(cached)
        from clay.frontend.pdk_register import pdk_rules_from_deck
        try:
            _, report = pdk_rules_from_deck(str(r))
            if not isinstance(report, Mapping):
                raise TypeError("PDK deck parse report must be a mapping")
            raw_parsed = report.get("parsed", {})
            if not isinstance(raw_parsed, Mapping):
                raise TypeError("PDK parsed rule facts must be a mapping")
            parsed = deepcopy(dict(raw_parsed))
        except Exception:
            # Preserve the historical fail-soft public result, but never cache a
            # failure as an authoritative empty deck.  A later call retries.
            return {}
        sources = self._confined_source_paths(
            r, self._report_source_paths(report, Path(r))
        )
        digest = self._deck_content_digest(sources)
        self._drc_rule_source_paths = sources
        if digest is not None:
            self._drc_rules_cache = {digest: deepcopy(parsed)}
        else:
            self._drc_rules_cache.clear()
        return deepcopy(parsed)

    def layout_min(self) -> Dict[str, float]:
        """Min width/space (um) the placer/router pitch derives from -- real parsed
        values, with node-scaled fallbacks only where the deck is silent."""
        p = self.drc_rules()

        def val(key, default):
            e = p.get(key)
            return float(e["value_um"]) if e and e.get("value_um") else default

        f = self.lmin_um or ((self.node_nm / 1000.0) if self.node_nm else 0.1)
        return {
            "m1_w": val("metal1_width", f), "m1_s": val("metal1_space", f),
            "poly_w": val("poly_width", f), "poly_s": val("poly_space", 2 * f),
            "co_w": val("contact_size", f), "co_s": val("contact_space", f),
        }

    # ---- P0: via / implant / LVS-model / bulk capabilities ----
    def _node_scale_nm(self) -> float:
        """The per-PDK length scale (nm) the node-scaled fallbacks key off:
        process min channel length when known, else node_nm, else 100 nm."""
        if self.lmin_um:
            return float(self.lmin_um) * 1000.0
        if self.node_nm:
            return float(self.node_nm)
        return 100.0

    def via_rules(self) -> Dict[str, ViaRule]:
        """Return ``{via_name: ViaRule}`` (contact / via1 / via2) for this PDK.

        Cut size + spacing are taken from the SAME parsed deck ``drc_rules()``
        reads (``contact_size`` / ``contact_space`` / ``via1_*`` / ``via2_*``);
        enclosures from ``*_enclosure_*`` rule keys. Where the deck is silent the
        value is a node-scaled provenance-tagged fallback with ``calibrated=False``
        so a consumer can tell a real cut rule from an estimate. Returns ``{}``
        only when neither a deck nor a node scale is available (never the case for
        a registered adapter).
        """
        if self.via_rules_override is not None:
            return dict(self.via_rules_override)
        p = self.drc_rules()
        s = self._node_scale_nm()

        def nm(key: str) -> Optional[int]:
            e = p.get(key)
            if e and e.get("value_nm") is not None:
                return int(e["value_nm"])
            if e and e.get("value_um") is not None:
                return int(round(float(e["value_um"]) * 1000.0))
            return None

        def via(name: str, below: str, above: str,
                size_keys, space_keys, enc_below_keys, enc_above_keys,
                size_fb: float, space_fb: float, enc_fb: float) -> ViaRule:
            calibrated = True
            prov = "parsed-deck"

            def pick(keys, fb_nm: float):
                nonlocal calibrated, prov
                for k in keys:
                    v = nm(k)
                    if v is not None:
                        return v
                calibrated = False
                prov = "node-scaled"
                return int(round(fb_nm))

            return ViaRule(
                name=name,
                cut_size_nm=pick(size_keys, size_fb),
                spacing_nm=pick(space_keys, space_fb),
                enclosure_below_nm=pick(enc_below_keys, enc_fb),
                enclosure_above_nm=pick(enc_above_keys, enc_fb),
                below=below, above=above,
                calibrated=calibrated, provenance=prov,
            )

        out: Dict[str, ViaRule] = {}
        out["contact"] = via(
            "contact", "metal1", "poly_or_diff",
            ("contact_size",), ("contact_space",),
            ("contact_enclosure_diff", "contact_enclosure_poly"),
            ("contact_enclosure_metal1",),
            size_fb=1.0 * s, space_fb=1.2 * s, enc_fb=0.4 * s,
        )
        # One ViaRule per cut layer the layermap exposes (via1..viaN), connecting
        # metal{n}<->metal{n+1}. Generalised from a hard-coded via1/via2 pair so a
        # 6-metal PDK's full via stack is routable -- a via with no rule is a layer the
        # router cannot rise through. Parsed size/space/enclosure keys per cut, node-
        # scaled fallback otherwise (calibrated=False).
        lm = self.layermap()
        n = 1
        while f"via{n}" in lm and f"metal{n}" in lm and f"metal{n + 1}" in lm:
            out[f"via{n}"] = via(
                f"via{n}", f"metal{n}", f"metal{n + 1}",
                (f"via{n}_size",), (f"via{n}_space",),
                (f"via{n}_enclosure_metal{n}",), (f"via{n}_enclosure_metal{n + 1}",),
                size_fb=1.2 * s, space_fb=1.2 * s, enc_fb=0.3 * s,
            )
            n += 1
        return out

    def implant_well_rules(self) -> ImplantWellRules:
        """Return the implant + well rules (nm) for nplus/pplus + nwell.

        nplus/pplus min width + space come from the parsed deck when present;
        the gate-overlap enclosures (gf180 NP.5a / PP.5a) and the nwell-to-pcomp
        enclosure (gf180 DF.4c) are sourced from the deck where exposed, else a
        node-scaled provenance-tagged fallback (``calibrated=False``). The implant
        TOKENS already surface in ``layermap()`` (nplus/pplus/nwell); this fills in
        the rule VALUES the layermap does not carry.
        """
        if self.implant_well_override is not None:
            return self.implant_well_override
        p = self.drc_rules()
        s = self._node_scale_nm()
        calibrated = True

        def nm(key: str, fb: float) -> int:
            nonlocal calibrated
            e = p.get(key)
            if e and e.get("value_nm") is not None:
                return int(e["value_nm"])
            if e and e.get("value_um") is not None:
                return int(round(float(e["value_um"]) * 1000.0))
            calibrated = False
            return int(round(fb))

        rules = ImplantWellRules(
            nplus_min_width=nm("nplus_width", 2.0 * s),
            nplus_min_space=nm("nplus_space", 2.0 * s),
            pplus_min_width=nm("pplus_width", 2.0 * s),
            pplus_min_space=nm("pplus_space", 2.0 * s),
            nplus_gate_enclosure=nm("nplus_gate_enclosure", 1.2 * s),
            pplus_gate_enclosure=nm("pplus_gate_enclosure", 1.2 * s),
            nwell_pcomp_enclosure=nm("nwell_pcomp_enclosure", 2.2 * s),
            nwell_ncomp_enclosure=nm("nwell_ncomp_enclosure", 0.6 * s),
        )
        return ImplantWellRules(
            **{**rules.__dict__,
               "calibrated": calibrated,
               "provenance": "parsed-deck" if calibrated else "node-scaled"},
        )

    def lvs_reference_models(self) -> Dict[str, str]:
        """Return ``{device_class: model_name}`` the LVS extractor compares against.

        Defaults to ``{'nmos': <nmos>, 'pmos': <pmos>}`` from ``device_bindings``;
        an explicit ``lvs_models_override`` wins. The poly-resistor model (when
        the adapter declares ``poly_res_model``) is included under ``'poly_res'``.
        Empty dict when neither bindings nor an override are present.
        """
        out: Dict[str, str] = {}
        for cls in ("nmos", "pmos", "nmos3v", "pmos3v"):
            name = self.device_bindings.get(cls)
            if name:
                out[cls] = name
        out.update({k: v for k, v in self.lvs_models_override.items() if v})
        if self.poly_res_model:
            out.setdefault("poly_res", self.poly_res_model)
        return out

    def substrate_net(self) -> str:
        """The global substrate net name (NMOS bulk tie on a p-substrate
        process). Defaults to ``"VSS"``; overridable via ``substrate_net_name``."""
        return self.substrate_net_name

    def deep_nwell(self) -> Optional[DeepNWellRules]:
        """Return the proven deep-N-well recipe, or ``None`` when unsupported."""

        return self.deep_nwell_override

    def bulk_topology(self) -> Dict[str, object]:
        """Describe how this PDK distinguishes N/P bulk + taps. Derived from the
        layermap presence + a small per-adapter override.

        Returns a dict with::

          implant_distinguished : bool  -- N/P told apart by nplus/pplus implant
                                           (gf180/external_pdk) vs a separate ndiff/pdiff
                                           GDS layer.
          diffusion_distinguished: bool -- the inverse (separate diff layers).
          drawn_p_tap            : bool  -- a P+ substrate tap must be DRAWN
                                           (implant-based bulk) vs a global
                                           substrate net only.
          needs_pwell_marker     : bool  -- a drawn p-well marker layer is
                                           required (deep-nwell / triple-well).
          substrate_net          : str   -- the global substrate net name.

        Layermap-derived defaults: a PDK whose layermap carries ``nplus`` AND
        ``pplus`` is implant-distinguished + needs a drawn P-tap; otherwise it is
        diffusion-distinguished (separate ndiff/pdiff layers). The
        ``bulk_topology_override`` map wins per key.
        """
        lm = {}
        try:
            lm = self.layermap()
        except Exception:
            lm = {}
        has_nplus = "nplus" in lm and lm.get("nplus") is not None
        has_pplus = "pplus" in lm and lm.get("pplus") is not None
        implant_dist = bool(has_nplus and has_pplus)
        out: Dict[str, object] = {
            "implant_distinguished": implant_dist,
            "diffusion_distinguished": not implant_dist,
            "drawn_p_tap": implant_dist,
            "needs_pwell_marker": False,
            "substrate_net": self.substrate_net(),
        }
        out.update(dict(self.bulk_topology_override))
        return out

    def klayout_decks(self) -> Dict[str, Optional[Path]]:
        """Executable KLayout DRC/LVS deck paths for this PDK's layout SIGNOFF,
        or ``{"drc": None, "lvs": None}`` when the PDK has no deck wired.

        Distinct from :meth:`drc_rules` (parsed numeric rules): a deck is the
        file KLayout actually RUNS. The flow's DRC/LVS steps resolve their deck
        through this, so signoff is PDK-data-driven (no hardcoded PDK).
        """
        if self.klayout_deck_resolver is not None:
            out = dict(self.klayout_deck_resolver() or {})
            return {"drc": out.get("drc"), "lvs": out.get("lvs")}
        if self.klayout_drc_from_rules or self.klayout_lvs_from_rules:
            # commercial PDK: synthesize the decks from the PDK's own data. The
            # proprietary external_drc/external_lvs decks cannot run under free KLayout.
            snapshot = (
                self.source_snapshot()
                if self.install_root_resolver is not None
                else None
            )
            return {
                "drc": self.generated_klayout_drc_deck(snapshot=snapshot)
                if self.klayout_drc_from_rules
                else None,
                "lvs": self.generated_klayout_lvs_deck(snapshot=snapshot)
                if self.klayout_lvs_from_rules
                else None,
            }
        return {"drc": None, "lvs": None}

    def pdk_failure_rules(self) -> Tuple[object, ...]:
        """Return typed PDK-scoped failure rules, or no rules when undeclared."""

        if self.failure_rule_resolver is None:
            return ()
        return tuple(self.failure_rule_resolver(self))

    def generated_klayout_drc_deck(
        self, *, snapshot: Optional["PdkSourceSnapshot"] = None
    ) -> Optional[Path]:
        """Synthesize (and cache) a FREE-KLayout partial DRC deck from THIS PDK's
        own parsed rule values + GDS layermap (``klayout_deck_from_rules``), or
        None when root / rules / layermap are not resolvable.

        Partial core-rule DRC (width/space/size/enclosure/...), NOT tapeout
        signoff -- a closed foundry never ships a free deck. This lets the flow
        run adapter-driven DRC on a commercial PDK whose only in-repo decks are
        proprietary external_drc/external_lvs that KLayout cannot execute.
        """
        if self.install_root_resolver is not None:
            captured = snapshot if snapshot is not None else self.source_snapshot()
            if captured is None:
                return None
            if captured.pdk_id != self.pdk_id:
                raise ValueError("PDK source snapshot belongs to another adapter")
            rules = captured.rules()
            lm = captured.layermap()
            if not rules or not lm:
                return None
            from clay.layout.klayout_deck_gen import klayout_deck_from_rules
            from clay.pdk._source_snapshot import publish_content_addressed_deck

            deck = klayout_deck_from_rules(
                rules, lm, pdk_name=captured.pdk_id
            )
            return publish_content_addressed_deck(
                pdk_id=self.pdk_id,
                snapshot_sha256=captured.snapshot_sha256,
                suffix="drc",
                text=deck,
            )

        root = self.root()
        if root is None:
            return None
        try:
            import tempfile
            # .drc (raw-Ruby) NOT .lydrc: KLayout's `-r` treats a .lydrc file as
            # XML; the deck body is raw DRC-Ruby, so a .lydrc extension makes
            # KLayout fail with an XML parse error at line 1 (the flow's
            # run_gf180_drc passes the path verbatim).
            cache = Path(tempfile.gettempdir()) / "clay_pdk_decks" / f"{self.pdk_id}.drc"
            rules = self.drc_rules()
            lm = self.layermap()
            if not rules or not lm:
                return None
            from clay.layout.klayout_deck_gen import klayout_deck_from_rules
            # ALWAYS regenerate (a cheap pure-string transform of parsed rules)
            # and rewrite the cache only when the content differs: an
            # `if cache.is_file(): return cache` early-return served STALE decks
            # after every generator change (real incident: a pre-b318bad cached
            # deck over-flagged 1748 bogus M1.LE.1 hits on a foundry-clean
            # layout for two weeks). The path stays stable for callers; the
            # content is always the CURRENT generator's output.
            deck = klayout_deck_from_rules(rules, lm, pdk_name=root.name)
            if not (cache.is_file() and cache.read_text(encoding="utf-8") == deck):
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(deck, encoding="utf-8")
            return cache
        except Exception:
            return None

    def generated_klayout_lvs_deck(
        self, *, snapshot: Optional["PdkSourceSnapshot"] = None
    ) -> Optional[Path]:
        """Cache the existing full-extraction synthesized deck for this PDK.

        The route LVS oracle owns the foundry-derived MOS/body/MIM/poly-R
        recipes.  The adapter merely exposes that same deck to the canonical
        flow; maintaining a second MOS-only recipe here previously made the
        flow silently miss every commercial-PDK passive device.
        """
        if self.install_root_resolver is not None:
            captured = snapshot if snapshot is not None else self.source_snapshot()
            if captured is None or captured.installed_authority_sha256 is None:
                return None
            if captured.pdk_id != self.pdk_id:
                raise ValueError("PDK source snapshot belongs to another adapter")
            nmos = self.device_bindings.get("nmos")
            pmos = self.device_bindings.get("pmos")
            if not nmos or not pmos:
                return None
            from clay.layout.route.lvs_gate import synthesized_lvs_deck_text
            from clay.pdk._source_snapshot import publish_content_addressed_deck

            deck = synthesized_lvs_deck_text(
                self.pdk_id,
                full_extraction=True,
                deep_nwell_override=self.deep_nwell(),
            )
            return publish_content_addressed_deck(
                pdk_id=self.pdk_id,
                snapshot_sha256=captured.snapshot_sha256,
                suffix="lvs",
                text=deck,
            )

        root = self.root()
        if root is None:
            return None
        nmos = self.device_bindings.get("nmos")
        pmos = self.device_bindings.get("pmos")
        if not nmos or not pmos:
            return None
        try:
            import tempfile
            cache = Path(tempfile.gettempdir()) / "clay_pdk_decks" / f"{self.pdk_id}.lvs"
            from clay.layout.route.lvs_gate import synthesized_lvs_deck_text
            # Same freshness contract as generated_klayout_drc_deck: regenerate
            # every call, rewrite only on content change (never serve stale).
            deck = synthesized_lvs_deck_text(
                self.pdk_id, full_extraction=True
            )
            if not (cache.is_file() and cache.read_text(encoding="utf-8") == deck):
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(deck, encoding="utf-8")
            return cache
        except Exception:
            return None

    def extra_includes(self, corner: str = "tt") -> list[Path]:
        """List of additional include files the simulator needs for this
        corner. Empty list when none required. The driver emits one
        `.include` line per path, after the primary corner_file include.
        """
        if self.extra_includes_resolver is None:
            return []
        return list(self.extra_includes_resolver(corner))

    def gm_id_lut(self):
        """Return the per-PDK gm/Id LUT handle, or None when the Rust
        extension is unavailable / the LUT JSON has not been generated.

        v0.0.99 R76 B replaces the flat `mu_cox: Mapping[str, float]`
        single-point sizing approximation with a 4-D (class, Vov, Vds,
        L) table. The LUT is sourced from real ngspice DC sweeps on
        the per-PDK BSIM models (see ``tools/pdk_lut_generator/
        run_ngspice_sweep.py``). Sizing algorithms that consume the
        LUT (via ``GmIdLutHandle.lut_size_w``) get PDK-invariant
        Vov-vs-gm/Id curves — the LUT abstracts away the per-PDK
        physics so sizing logic stays PDK-agnostic.

        The legacy ``mu_cox`` Mapping is preserved for backwards
        compatibility; callers that don't yet consume the LUT continue
        to use the closed-form square-law solver in
        ``clay/sizing.py::closed_form_W``. The lookup order is:

          1. If the Rust extension is unavailable, return None
             (caller falls back to ``self.mu_cox``).
          2. If the LUT JSON is not on disk (operator has not run
             the generator script), return a handle that carries the
             legacy `for_pdk` table only (``has_grid_data() == False``).
          3. Otherwise return a fully-loaded LUT with grid data.

        The handle is cached on the adapter instance after the first
        successful load so subsequent calls are cheap.
        """
        # Cached handle on first successful load
        cached = getattr(self, "_gm_id_lut_cache", None)
        if cached is not None:
            return cached
        try:
            import clay_ir  # type: ignore[import-not-found]
        except ImportError:
            return None
        # Locate the LUT JSON — search project-relative paths.
        from pathlib import Path as _Path
        # Importing __file__ from clay/pdk/__init__.py: parent is clay/pdk,
        # parent.parent is the clay package root, parent.parent.parent is
        # the repo root.
        here = _Path(__file__).resolve()
        candidates = [
            here.parent.parent.parent / "data" / "pdk_luts" / f"{self.pdk_id}_gm_id.json",
            _Path("data") / "pdk_luts" / f"{self.pdk_id}_gm_id.json",
        ]
        for cand in candidates:
            if cand.exists():
                try:
                    handle = clay_ir.GmIdLutHandle.from_json_path(str(cand))
                    self._gm_id_lut_cache = handle  # type: ignore[attr-defined]
                    return handle
                except Exception:
                    continue
        # No JSON found — return the legacy seed handle so callers get
        # a uniform "GmIdLutHandle | None" type instead of mixing the
        # legacy mu_cox dict with the new LUT API.
        try:
            handle = clay_ir.GmIdLutHandle.for_pdk(self.pdk_id)
            self._gm_id_lut_cache = handle  # type: ignore[attr-defined]
            return handle
        except Exception:
            return None


_REGISTRY: dict[str, PdkAdapter] = {}


def register_pdk(adapter: PdkAdapter) -> None:
    """Register a PDK adapter so drivers can look it up by `pdk_id`."""
    if adapter.pdk_id in _REGISTRY:
        # Idempotent re-registration is OK (e.g. importlib.reload).
        existing = _REGISTRY[adapter.pdk_id]
        if existing is adapter:
            return
    _REGISTRY[adapter.pdk_id] = adapter


def get_pdk(pdk_id: str) -> PdkAdapter:
    """Lazy-load + return the PDK adapter for pdk_id.

    Lazy import is intentional: importing `clay.pdk.sky130` auto-registers
    via top-level side effects. If a user wants a PDK Clay has not yet
    seen, they can install a third-party `clay.pdk.foo` package and pre-
    import it; this function will then find it in the registry.
    """
    if pdk_id in _REGISTRY:
        return _REGISTRY[pdk_id]
    # Try to import the conventional adapter module
    try:
        import importlib
        importlib.import_module(f"clay.pdk.{pdk_id}")
    except ImportError:
        pass
    if pdk_id not in _REGISTRY:
        raise KeyError(
            f"No PDK adapter registered for {pdk_id!r}. "
            f"Built-in PDKs: {list(_REGISTRY)}. "
            f"Third-party adapters should register via clay.pdk.register_pdk()."
        )
    return _REGISTRY[pdk_id]


def exact_source_identity(pdk_id: str) -> dict[str, Optional[str]]:
    """Return the PDK id and its current exact installed-source snapshot.

    A missing snapshot remains explicit.  Layout may still consume cited facts,
    but calibrated electrical signoff cannot promote them as installed authority.
    """

    if type(pdk_id) is not str or not pdk_id:
        raise TypeError("PDK identity requires a non-empty exact string")
    adapter = get_pdk(pdk_id)
    snapshot = adapter.source_snapshot()
    return {
        "schema": "clay.pdk-exact-source-identity/v1",
        "pdk_id": str(adapter.pdk_id),
        "source_snapshot_sha256": (
            None if snapshot is None else snapshot.snapshot_sha256
        ),
    }


def list_pdks() -> list[str]:
    return sorted(_REGISTRY)


# Commercial HSPICE-lib adapter modules (each registers its PdkAdapter on import).
_OPEN_PDKS = ("gf180mcu", "sky130")


def load_all() -> None:
    """Import the per-PDK adapter modules so they register. Idempotent."""
    import importlib
    for mod in _OPEN_PDKS:
        try:
            importlib.import_module(f"clay.pdk.{mod}")
        except Exception:
            pass


def all_adapters() -> list[PdkAdapter]:
    load_all()
    return list(_REGISTRY.values())


def runnable_ids() -> list[str]:
    """PDK ids whose HSPICE lib is proven to simulate on ngspice."""
    load_all()
    return [a.pdk_id for a in _REGISTRY.values() if getattr(a, "runnable", False)]
