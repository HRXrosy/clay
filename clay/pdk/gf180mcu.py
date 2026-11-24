"""
GlobalFoundries gf180mcu PDK adapter for Clay.

Auto-registers with `clay.pdk.get_pdk('gf180mcu')` on import.

Maps Clay device-class strings to gf180mcuC subckt names, provides
the mu*C_ox lookup used by the closed-form sizing solver, and locates
ngspice corner include files via a dedicated env var (see below).

Status
======
This adapter supports two operating modes:

  * SKELETON mode -- when no PDK install is found on disk. Drivers can
    still resolve the adapter, query device bindings, mu*C_ox, and
    corner names. The corner_file() call returns a resolved path that
    may or may not exist; the driver surfaces a separate "corner file
    missing" error.

  * REAL mode -- when CLAY_GF180MCU_PATH points at an installed
    google/gf180mcu-pdk tree with the `gf180mcu_fd_pr` PR library
    populated. In this mode corner_file() lazily generates a small
    Clay-owned wrapper file under <root>/clay_wrappers/ that includes
    design.ngspice and pulls in the right `.lib sm141064.ngspice
    <corner>` section.

Config plumbing
===============
The standard Clay config has a single pdk_root field. The gf180mcu
adapter reads its root in this resolution order:

  1. CLAY_GF180MCU_PATH (dedicated env var)
  2. CLAY_PDK_PATH (global, only used when CLAY_PDK_ID=gf180mcu)
  3. raise RuntimeError when neither is set

PDK install layout
==================
Expected layout under CLAY_GF180MCU_PATH (matches upstream
google/gf180mcu-pdk after recursive submodule init):

  <root>/libraries/gf180mcu_fd_pr/latest/models/ngspice/
    design.ngspice       # global params, sw_stat_global, sw_stat_mismatch
    sm141064.ngspice     # main model library with .LIB <corner> sections

The wrapper files Clay generates live at:

  <root>/libraries/gf180mcu_fd_pr/latest/models/ngspice/clay_wrappers/
    tt.spice  ss.spice  ff.spice  sf.spice  fs.spice

Each wrapper is a 3-line file:
  .option scale=1.0u
  .include "../design.ngspice"
  .lib "../sm141064.ngspice" <gf-corner-name>

Sources for mu*C_ox values
==========================
Approximate strong-inversion values for the 3.3 V devices in the
gf180mcuC open-PDK release, derived from the public PCM data and the
BSIM3v3 model card distributed with sm141064. These are coarse starting
points; the closed-form sizing solver uses them as a first cut.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ._resolve import resolve_root_via_config, root_override_declared
from . import DeepNWellRules, ImplantWellRules, PdkAdapter, ViaRule, register_pdk
from .bjt import BjtBinding, BjtTapBinding
from .failure_evidence import validated_failure_rules


# Env var for the gf180mcu install root
ENV_GF180MCU_PATH = "CLAY_GF180MCU_PATH"


# Device-class binding map: Clay's process-independent device-class names
# -> gf180mcuC subckt names as defined in sm141064.ngspice. The 3p3
# devices are the default 3.3 V core flavour; 6p0 covers the 6 V HV
# option. Pin order in the subckt definitions is `d g s b`.
_BINDINGS: dict[str, str] = {
    "nmos": "nmos_3p3",
    "pmos": "pmos_3p3",
    "nmos3v": "nmos_3p3",
    "pmos3v": "pmos_3p3",
    "nmos_06v": "nmos_6p0",
    "pmos_06v": "pmos_6p0",
    "nmos_nat": "nmos_6p0_nat",  # native (depletion) NMOS variant
}


# Closed-form sizing mu*C_ox table (used by clay.sizing.closed_form_W).
# Coarse strong-inversion approximations at L ~ 0.5 um.
_MU_COX: dict[str, float] = {
    "nmos": 220e-6,  # 3.3 V core NMOS, mu_n*C_ox ~ 220 uA/V^2
    "pmos": 70e-6,  # 3.3 V core PMOS, mu_p*C_ox ~  70 uA/V^2
    # Semantic 3V classes bind to the same physical 3.3 V core family above.
    "nmos3v": 220e-6,
    "pmos3v": 70e-6,
    "nmos_06v": 150e-6,  # 6 V HV NMOS, thicker oxide -> lower C_ox
    "pmos_06v": 45e-6,  # 6 V HV PMOS
    "nmos_nat": 200e-6,  # native (depletion) NMOS variant
}


# Standard process corners. Clay uses the canonical {tt, ss, ff, sf, fs}
# corner-name vocabulary across all PDKs; this adapter maps the Clay
# name to the upstream gf180mcu .LIB section name (which calls the
# typical corner "typical" instead of "tt").
_VALID_CORNERS: frozenset[str] = frozenset({"tt", "ss", "ff", "sf", "fs"})
_CORNER_NAME_MAP: dict[str, str] = {
    "tt": "typical",
    "ss": "ss",
    "ff": "ff",
    "sf": "sf",
    "fs": "fs",
}


# Subpaths under the PDK root.
_PR_NGSPICE_REL = Path("libraries") / "gf180mcu_fd_pr" / "latest" / "models" / "ngspice"
_DESIGN_FILE = "design.ngspice"
_MODEL_FILE = "sm141064.ngspice"
_WRAPPERS_SUBDIR = "clay_wrappers"


def _resolve_pdk_root() -> Optional[Path]:
    """Resolution: CLAY_GF180MCU_PATH env -> ClayConfig.pdk_root (CLAY_PDK_ID=gf180mcu) -> None."""
    return resolve_root_via_config(ENV_GF180MCU_PATH, "gf180mcu")


def _pdk_root_is_declared() -> bool:
    return root_override_declared(ENV_GF180MCU_PATH, "gf180mcu")


def _generate_wrapper(
    wrappers_dir: Path,
    clay_corner: str,
    design_file: Path,
    model_file: Path,
) -> Path:
    """Write a small spice wrapper file for the given corner.

    The wrapper does three things ngspice needs in order to pick up a
    specific corner from the gf180mcu PR library:

      1. .option scale=1.0u - tells ngspice that W/L numbers in the
         netlist are in microns. The sm141064 model card defines
         binning ranges in SI metres (lmin=2.8e-7 etc.); without this
         option a netlist emitting `W=10 L=0.5` would be interpreted
         as 10 metres / 0.5 metres and fail to match any bin. Clay's
         driver emits unitless W/L numbers in microns by convention.
         The upstream gf180mcu PDK leaves the choice to the user, so
         Clay sets it here in the corner wrapper.
      2. .include "<design.ngspice>" - sets global switches like
         sw_stat_global, sw_stat_mismatch, fnoicor and mc_skew
         (defaults from the PDK).
      3. .lib "<sm141064.ngspice>" <gf-corner-name> - pulls in the
         per-device process-corner sections (typical/ff/ss/sf/fs).

    Driver-side, Clay does `.include <wrapper>` so the rest of the
    netlist behaves identically to any other PDK's corner-file path.

    The wrapper is regenerated only when its contents would change
    (mostly a path-portability robustness, since the PDK root rarely
    moves between runs).
    """
    wrappers_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = wrappers_dir / f"{clay_corner}.spice"
    gf_corner = _CORNER_NAME_MAP[clay_corner]
    # Use absolute paths so the wrapper works regardless of how
    # ngspice is invoked (the driver runs from work_dir, not from
    # wrappers_dir).
    body = (
        f"* Clay-generated wrapper for gf180mcu corner {clay_corner!r} "
        f"(gf180mcu section: {gf_corner!r}).\n"
        f"* Auto-regenerated by clay/pdk/gf180mcu.py on every "
        f"corner_file() call.\n"
        f".option scale=1.0u\n"
        f'.include "{design_file.as_posix()}"\n'
        f'.lib "{model_file.as_posix()}" {gf_corner}\n'
    )
    # Only rewrite if changed - keeps mtime stable for downstream
    # caching / file-watcher heuristics.
    if not wrapper_path.exists() or wrapper_path.read_text(encoding="utf-8") != body:
        wrapper_path.write_text(body, encoding="utf-8")
    return wrapper_path


def _corner_file(corner: str) -> Optional[Path]:
    """Locate (or generate) the gf180mcu ngspice corner include file.

    Validates `corner` against the standard corner set
    {tt, ss, ff, sf, fs} and raises ValueError for unknown corners.

    Raises RuntimeError when no PDK root is configured.

    Returns the resolved wrapper path even when the underlying PDK
    files do not exist on disk; the driver surfaces a separate
    "corner file missing" error with the resolved path, which is the
    right place to complain about an incomplete install.

    The wrapper is generated lazily on first request and cached in
    <root>/libraries/gf180mcu_fd_pr/latest/models/ngspice/clay_wrappers/.
    """
    if corner not in _VALID_CORNERS:
        raise ValueError(
            f"Unknown gf180mcu corner {corner!r}. "
            f"Valid corners: {sorted(_VALID_CORNERS)}."
        )
    root = _resolve_pdk_root()
    if root is None:
        raise RuntimeError(
            f"gf180mcu PDK root is not configured. "
            f"Set {ENV_GF180MCU_PATH} to the gf180mcuC install root "
            f"(or set CLAY_PDK_ID=gf180mcu + CLAY_PDK_PATH for the "
            f"global config path)."
        )

    ngspice_root = root / _PR_NGSPICE_REL
    design_file = ngspice_root / _DESIGN_FILE
    model_file = ngspice_root / _MODEL_FILE

    # If the canonical model files are not present, we cannot generate
    # a meaningful wrapper. Return the *expected* wrapper path anyway
    # so the driver's "corner file missing" message points at the
    # right location, but skip the write.
    wrappers_dir = ngspice_root / _WRAPPERS_SUBDIR
    expected_wrapper = wrappers_dir / f"{corner}.spice"
    if not design_file.exists() or not model_file.exists():
        return expected_wrapper

    return _generate_wrapper(wrappers_dir, corner, design_file, model_file)


# Executable KLayout DRC/LVS decks within a gf180mcu installation.  An explicit
# install selection is authoritative; it must never be replaced by a deck from
# another checkout.  The repo-local location remains only a clean-checkout
# fallback when no install root was declared and the actual files exist.
_KLAYOUT_RULES_REL = (
    Path("libraries")
    / "gf180mcu_fd_pr"
    / "latest"
    / "rules"
    / "klayout"
)
_KLAYOUT_RULES_DIR = (
    Path(__file__).resolve().parents[2]
    / "pdks"
    / "gf180mcu"
    / _KLAYOUT_RULES_REL
)


def _existing_klayout_deck(root: Path, relative: Path) -> Optional[Path]:
    try:
        resolved_root = root.resolve(strict=True)
        candidate = (resolved_root / relative).resolve(strict=True)
        candidate.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    return candidate if candidate.is_file() else None


def _klayout_decks() -> dict:
    root = _resolve_pdk_root()
    if root is not None:
        return {
            "drc": _existing_klayout_deck(
                root, _KLAYOUT_RULES_REL / "drc" / "gf180mcu.drc"
            ),
            "lvs": _existing_klayout_deck(
                root, _KLAYOUT_RULES_REL / "lvs" / "gf180mcu.lvs"
            ),
        }
    if _pdk_root_is_declared():
        return {"drc": None, "lvs": None}
    return {
        "drc": _existing_klayout_deck(
            _KLAYOUT_RULES_DIR, Path("drc") / "gf180mcu.drc"
        ),
        "lvs": _existing_klayout_deck(
            _KLAYOUT_RULES_DIR, Path("lvs") / "gf180mcu.lvs"
        ),
    }


# gf180mcu real via / implant rules (nm). The adapter's root() is not configured
# via dir_prefix (gf180 ships its parsed rules through the static
# pdk_rules_v174 builder, sourced from the real gf180mcu.drc rule_decks), so the
# adapter wires the calibrated values directly here. size/spacing are PARSED_DECK
# literals (CO.1/CO.2a, V1.1/V1.2a, NP.1/NP.2, PP.1/PP.2); the enclosures
# (CO.4/CO.6, V1.3c/V1.4b, NP.5a/PP.5a, DF.4c/DF.4d) are the consolidated gf180
# constants cross-checked against the deck -- all real, hence calibrated=True.
_VIA_RULES = {
    "contact": ViaRule(
        name="contact",
        cut_size_nm=220,
        spacing_nm=250,
        enclosure_below_nm=70,
        enclosure_above_nm=60,
        below="comp",
        above="metal1",
        calibrated=True,
        provenance="parsed-deck",
    ),
    "via1": ViaRule(
        name="via1",
        cut_size_nm=260,
        spacing_nm=260,
        enclosure_below_nm=60,
        enclosure_above_nm=60,
        below="metal1",
        above="metal2",
        calibrated=True,
        provenance="parsed-deck",
    ),
    "via2": ViaRule(
        name="via2",
        cut_size_nm=260,
        spacing_nm=260,
        enclosure_below_nm=60,
        enclosure_above_nm=60,
        below="metal2",
        above="metal3",
        calibrated=True,
        provenance="parsed-deck",
    ),
    # Actual installed gf180mcu.drc: V3/V4.1, .2a, .3c and .4b. These
    # complete the existing M1-M5 layermap; no unbound top-metal is invented.
    "via3": ViaRule(
        name="via3", cut_size_nm=260, spacing_nm=260,
        enclosure_below_nm=60, enclosure_above_nm=60,
        below="metal3", above="metal4", calibrated=True,
        provenance="gf180mcu.drc V3.1/V3.2a/V3.3c/V3.4b",
    ),
    "via4": ViaRule(
        name="via4", cut_size_nm=260, spacing_nm=260,
        enclosure_below_nm=60, enclosure_above_nm=60,
        below="metal4", above="metal5", calibrated=True,
        provenance="gf180mcu.drc V4.1/V4.2a/V4.3c/V4.4b",
    ),
}

_IMPLANT_WELL = ImplantWellRules(
    nplus_min_width=400,
    nplus_min_space=400,  # NP.1 / NP.2
    pplus_min_width=400,
    pplus_min_space=400,  # PP.1 / PP.2
    nplus_gate_enclosure=230,
    pplus_gate_enclosure=230,  # NP.5a / PP.5a
    nwell_pcomp_enclosure=430,  # DF.4c
    nwell_ncomp_enclosure=120,  # DF.4d
    calibrated=True,
    provenance="consolidated",
)

# Foundry MIM stack options from gf180mcu.lvs and mim_capacitor_option_{a,b}.drc.
# Option-A places the capacitor on M2/FuseTop/V2/M3; 5LM Option-B places it on
# M4/FuseTop/V4/M5 to match gf180mcuD Magic extraction.
_GF180_MIM_STACKS: dict[str, dict[str, object]] = {
    "A": {
        "option": "A",
        "metal_level": None,
        "bottom_metal": "metal2",
        "top_via": "via2",
        "cover_metal": "metal3",
        "top_plate": "fusetop",
        "recognition_markers": ("cap_mk", "mim_l_mk"),
        "forbidden_cut": "via1",
        "bottom_routing_clearance_nm": 1200,
        "virtual_oversize_nm": 1060,
        "magic_compatible": False,
    },
    "B": {
        "option": "B",
        "metal_level": "5LM",
        "bottom_metal": "metal4",
        "top_via": "via4",
        "cover_metal": "metal5",
        "top_plate": "fusetop",
        "recognition_markers": ("cap_mk", "mim_l_mk"),
        "forbidden_cut": "via3",
        "bottom_routing_clearance_nm": 1200,
        "virtual_oversize_nm": 1060,
        "magic_compatible": True,
        "by_metal_level": {
            "4LM": {
                "bottom_metal": "metal3",
                "top_via": "via3",
                "cover_metal": "metal4",
                "forbidden_cut": "via2",
            },
            "5LM": {
                "bottom_metal": "metal4",
                "top_via": "via4",
                "cover_metal": "metal5",
                "forbidden_cut": "via3",
            },
        },
    },
}


_GF180_MIM_OPTION_ALIASES: dict[str, str] = {
    "A": "A",
    "OPT_A": "A",
    "OPTION_A": "A",
    "OPTIONA": "A",
    "B": "B",
    "OPT_B": "B",
    "OPTION_B": "B",
    "OPTIONB": "B",
}


def normalize_gf180_mim_option(
    mim_option: object = None,
    *,
    default: str = "A",
) -> str:
    """Normalize GF180 MIM stack option strings (``'A'/'option_a'`` or ``'B'/'option_b'``)."""
    raw = default if mim_option is None else mim_option
    key = str(raw).strip().upper().replace("-", "_").replace(" ", "_")
    if not key:
        key = str(default or "A").strip().upper()
    norm = _GF180_MIM_OPTION_ALIASES.get(key)
    if norm is None:
        raise ValueError(
            f"Unsupported gf180mcu mim_option {mim_option!r}; expected 'A' or 'B'"
        )
    return norm


def resolve_gf180_mim_config(
    tech_or_adapter: object = None,
    *,
    mim_option: object = None,
    cap_option: object = None,
    metal_level: object = None,
    pex_backend: object = None,
) -> tuple[str, str, str]:
    """Resolve ``(mim_option, cap_option, metal_level)`` from explicit args or PDK/backend data."""
    adapter = getattr(tech_or_adapter, "adapter", None) or getattr(
        tech_or_adapter, "_adapter", tech_or_adapter
    )
    extra = getattr(adapter, "extra", {}) if adapter is not None else {}
    if not isinstance(extra, dict):
        extra = {}

    if mim_option is not None and str(mim_option).strip():
        opt = normalize_gf180_mim_option(mim_option)
    else:
        configured = getattr(tech_or_adapter, "mim_option", None) or extra.get(
            "active_mim_option"
        )
        if configured is not None and str(configured).strip():
            opt = normalize_gf180_mim_option(configured)
        elif str(pex_backend or "").strip().casefold() == "magic":
            opt = normalize_gf180_mim_option(
                extra.get("default_magic_mim_option", "B"), default="B"
            )
        else:
            opt = "A"

    cap = str(
        cap_option
        if cap_option is not None and str(cap_option).strip()
        else (getattr(tech_or_adapter, "cap_option", None) or extra.get("active_cap_option") or "1")
    ).strip()
    lvl = str(
        metal_level
        if metal_level is not None and str(metal_level).strip()
        else (
            getattr(tech_or_adapter, "metal_level", None)
            or extra.get("active_metal_level")
            or ("5LM" if opt == "B" else "5LM")
        )
    ).strip().upper()
    return opt, cap, lvl


def gf180_mim_stack_spec(
    mim_option: str = "A",
    *,
    metal_level: str = "5LM",
) -> dict[str, object]:
    """Return resolved GF180 MIM stack layer tokens for Option-A or Option-B."""
    opt = normalize_gf180_mim_option(mim_option)
    if opt not in _GF180_MIM_STACKS:
        raise ValueError(
            f"Unsupported gf180mcu mim_option {mim_option!r}; expected 'A' or 'B'"
        )
    spec = dict(_GF180_MIM_STACKS[opt])
    if opt == "B":
        lvl = str(metal_level or "5LM").strip().upper()
        levels = spec["by_metal_level"]
        if lvl not in levels:
            raise ValueError(
                f"Unsupported gf180mcu Option-B metal_level {metal_level!r}; "
                f"supported: {sorted(levels)}"
            )
        spec.update(levels[lvl])
        spec["metal_level"] = lvl
    spec.pop("by_metal_level", None)
    return spec


GF180MCU = PdkAdapter(
    pdk_id="gf180mcu",
    bjt_bindings={
        "pnp": BjtBinding(
            semantic_kind="pnp",
            simulator_model="vpnp_0p42x10",
            lvs_model="vpnp_0p42x10",
            nominal_voltage_v=3.3,
            emitter_area_um2=4.2,
            geometry_provider="fixed_gds",
            geometry_path="pdks/gf180mcu/libraries/gf180mcu_fd_pr/latest/cells/klayout/pymacros/cells/bjt/vpnp_0p42x10.gds",
            geometry_sha256="b526fde50695a97f18d22d419dcedfcd9b2f561c53fd41122b7473f6a58b5c38",
            provenance="GF180 fixed cell plus real gf180mcu.lvs C/B/E microcell match",
            primitive_evidence_state="pass",
            full_extraction_gds="pdks/gf180mcu/libraries/gf180mcu_fd_pr/latest/rules/klayout/lvs/testing/testcases/vpnp_0p42x10.gds",
            full_extraction_cdl="pdks/gf180mcu/libraries/gf180mcu_fd_pr/latest/rules/klayout/lvs/testing/testcases/vpnp_0p42x10.cdl",
            proof_gds_sha256="b526fde50695a97f18d22d419dcedfcd9b2f561c53fd41122b7473f6a58b5c38",
            proof_cdl_sha256="8a9bd95ef6292110d2bd6b7ac826b6a3bca938b556a6db28bd44deea6b2c26ce",
            terminal_label_layer=(34, 0),
            terminal_labels={"c": "C", "b": "B", "e": "E"},
            # Proven metal1 landing centres inside the fixed cell's three
            # role-specific vertical conductors.  Multi-unit banks use these
            # adapter facts to make every physical C/B/E connection explicit.
            array_landing_offsets_nm={
                "c": (375, 3320),
                "b": (1125, 6640),
                "e": (1850, 9960),
            },
            lvs_parallel_parameter="NE",
            tap_binding=BjtTapBinding(
                mode="external_companion",
                where=("substrate",),
                diffusion_layer="diff",
                diffusion_rule_layer="comp",
                implant_layer="nplus",
                excluded_well_layer="nwell",
                contact_via="contact",
                metal_layer="metal1",
                implant_enclosure_nm=230,
                fixed_cell_clearance_nm=400,
                max_distance_nm=20_000,
            ),
            canonical_materializer=True,
        )
    },
    device_bindings=_BINDINGS,
    mu_cox=_MU_COX,
    corner_file_resolver=_corner_file,
    sim_corner_aliases={
        "tt": "tt",
        "ss": "ss",
        "ff": "ff",
        "sf": "sf",
        "fs": "fs",
    },
    klayout_deck_resolver=_klayout_decks,
    failure_rule_resolver=validated_failure_rules,
    min_L=0.28e-6,  # gf180mcuC 3p3 nfet/pfet drawn min L = 0.28 um
    vdd_nom=3.3,  # 3.3 V supply for the 3p3 device family
    node_nm=180,  # gf180mcu is a 0.18 um (180 nm) node
    lmin_um=0.28,  # 3p3 drawn min L
    via_rules_override=_VIA_RULES,
    implant_well_override=_IMPLANT_WELL,
    # gf180mcu.drc MIM.10:
    #   via1.interacting(metal2.interacting(fusetop))
    # The router consumes this through LayoutTech as a generic layer triple.
    mim_forbidden_cut_touch=(("via1", "metal2", "fusetop"),),
    # gf180mcu.drc MIM.1:
    #   metal2.separation(mim_virtual, transparent, 1.2.um)
    mim_bottom_routing_clearance_nm=(("metal2", "fusetop", 1200),),
    # GlobalFoundries 6LM_1TM_9K tech LEF, header + per-layer DIRECTION:
    #   horizontal Metal1/Metal3/Metal5; vertical Metal2/Metal4/MetalTop.
    # This is PDK process-convention data, not a direction inferred from the
    # external_pdk reference layout.  Clay names MetalTop as metal6 in this stack.
    preferred_routing_axes={
        1: "H",
        2: "V",
        3: "H",
        4: "V",
        5: "H",
        6: "V",
    },
    # gf180mcu is implant-distinguished (one COMP diffusion; Nplus/Pplus tell
    # N/P apart) with a drawn P-tap and a global p-substrate; no triple-well
    # marker on the baseline flow.
    bulk_topology_override={
        "implant_distinguished": True,
        "diffusion_distinguished": False,
        "drawn_p_tap": True,
        "needs_pwell_marker": False,
    },
    # Foundry KLayout DRC: DF.8_3.3V / DF.5_3.3V, LPW.3, DF.4b_3.3V,
    # DF.4a_3.3V, DN.3.  LVS extracts the isolated device as nmos_3p3_dw.
    deep_nwell_override=DeepNWellRules(
        pwell_layer="pwell",
        deep_nwell_layer="dnwell",
        pwell_enclose_nactive_nm=430,
        pwell_enclose_ptap_nm=120,
        deep_nwell_enclose_pwell_nm=2500,
        deep_nwell_enclose_ntap_nm=620,
        ntap_to_pwell_nm=120,
        outer_ptap_to_deep_nwell_nm=2500,
        outer_substrate_guard_required=True,
        lvs_models={"nmos": "nmos_3p3_dw", "nmos3v": "nmos_3p3_dw"},
        # sm141064.ngspice models isolated DNW 3.3V NMOS via the 4-terminal
        # `nmos_3p3` subcircuit whose bulk node binds to the isolated LVPWELL.
        sim_models={
            "nmos": "nmos_3p3",
            "nmos3v": "nmos_3p3",
            "nmos_3p3_dw": "nmos_3p3",
        },
        calibrated=True,
        provenance="gf180mcu KLayout DRC DF.4a/4b/5/8, LPW.3, DN.3 + LVS",
    ),
    substrate_net_name="VSS",
    extra={
        "family": "gf180mcu_fd_pr",
        "release": "sm141064",
        "supply_label": "3p3",
        # The adapter currently exposes only the 3.3 V MOS family.  Plain
        # thin-semantic Clay classes therefore retarget explicitly instead of
        # masquerading as preserved thin devices.
        "device_semantic_domains": {
            "nmos": "thick",
            "pmos": "thick",
            "nmos3v": "thick",
            "pmos3v": "thick",
        },
        "env_var": ENV_GF180MCU_PATH,
        "min_W": 0.22e-6,
        # Raw Magic ext2spice subcircuit -> canonical device class.  Joint PEX
        # resolves each class through lvs_reference_models(); the flow never
        # carries a circuit-name or a duplicate model-name table.
        "magic_device_classes": {
            "nfet_03v3": "nmos",
            "pfet_03v3": "pmos",
        },
        # Isolated DNW NMOS aliases produced by Magic or LVS extraction resolve
        # through deep_nwell().sim_models without altering the default
        # `magic_device_classes` baseline table.
        "magic_deep_nwell_device_classes": {
            "nmos_3p3_dw": "nmos",
            "nfet_03v3_dss": "nmos",
            "nfet_03v3_dw": "nmos",
        },
        "magic_bjt_device_classes": {
            "pnp_10p00x00p42": "pnp",
        },
        "mim_options": _GF180_MIM_STACKS,
        "default_magic_mim_option": "B",
        "thick_oxide_is_default": True,
        # nmos_3p3/pmos_3p3 are subcircuits whose primitive MOS is M0.
        # Joint Magic PEX adds an XCORE alias wrapper around that subcircuit.
        "ngspice_saturation_probe_suffix": ".m0",
        "ngspice_instance_prefixes": {name: "X" for name in _BINDINGS},
        "ngspice_magic_saturation_probe_suffix": ".xcore.m0",
    },
)


# Auto-register on import.
register_pdk(GF180MCU)
