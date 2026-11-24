"""
Sky130 PDK adapter for Clay.

Auto-registers with `clay.pdk.get_pdk('sky130')` on import.

Maps Clay device-class strings to sky130_fd_pr subckt names, provides
the μ·C_ox lookup used by the closed-form sizing solver, and locates
the ngspice corner include files via the Clay config layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import get_config
from . import MimMetalGeometryRule, PdkAdapter, register_pdk
from .native_spice import NativeSpiceFamily, NativeSpiceModelContract
from .sky130_native_geometry import NATIVE_GEOMETRY_PROVIDER


# Device-class binding map: Clay's process-independent device-class names
# → sky130_fd_pr concrete subckt names. The ngspice driver instantiates
# each device using the subckt name returned here.
_BINDINGS: dict[str, str] = {
    "nmos":     "sky130_fd_pr__nfet_01v8",
    "pmos":     "sky130_fd_pr__pfet_01v8",
    "nmos_lvt": "sky130_fd_pr__nfet_01v8_lvt",
    "pmos_lvt": "sky130_fd_pr__pfet_01v8_lvt",
    "pmos_hvt": "sky130_fd_pr__pfet_01v8_hvt",
    # v0.0.91 (LC-VCO accumulation-mode varactor):
    # Accumulation-mode N-well varactor — a real MOSCAP device on the N-well
    # (gate over n-well, with c0 = gate, c1 = c2 = body/n-well terminals).
    # Distinct from the PMOS-MOSCAP varactor used by `varactor` device class:
    # the accumulation-mode device sweeps cap from accumulation (large Cox)
    # all the way to depletion (small parasitic floor) across (Vg - Vbody)
    # = [0, ±2 V], giving a 4-6x C_max/C_min ratio versus the PMOS-MOSCAP's
    # ~1.5-2x. That ratio is what unlocks LC-VCO tuning ranges in the
    # textbook 15-20% band on a 1.8 V supply.
    #
    # The LVT variant has a slightly steeper C-V slope and a lower depletion
    # floor than the HVT variant, so LVT is the default for LC-VCO tuning.
    # The HVT variant is available for cases that want flatter C(Vg) near
    # the swing extremes (e.g. PLL where tuning gain Kvco linearity matters).
    "varactor_accum":     "sky130_fd_pr__cap_var_lvt",
    "varactor_accum_lvt": "sky130_fd_pr__cap_var_lvt",
    "varactor_accum_hvt": "sky130_fd_pr__cap_var_hvt",
}


# Closed-form sizing μ·C_ox table (used by clay.sizing.closed_form_W).
# Values are coarse strong-inversion approximations at L≈0.5µm; future
# BSIM-extracted LUT will replace these.
#
# v0.0.91+: accumulation-mode N-well varactors are MOSCAPs — they don't
# participate in the closed-form sizing as transconductance devices, so
# the "μ·C_ox" entries here are NOT used by the W-solver. They exist so
# the registry-level "every binding has a μ·C_ox" contract test passes;
# the values are set to the LVT-NMOS strong-inversion approximation
# because the accumulation-mode device channel is N-type and behaves
# similarly to an LVT NMOS in deep accumulation. The HVT variant
# inherits the standard NMOS value because its body-effect-shifted
# threshold puts it closer to the typical-Vt NMOS μ·C_ox window.
_MU_COX: dict[str, float] = {
    "nmos":     200e-6,
    "pmos":      70e-6,
    "nmos_lvt": 250e-6,
    "pmos_lvt":  90e-6,
    "pmos_hvt":  50e-6,    # higher Vth → smaller effective mobility window
    "varactor_accum":     250e-6,  # MOSCAP — value unused by W-solver
    "varactor_accum_lvt": 250e-6,
    "varactor_accum_hvt": 200e-6,
}


# Standard sky130 process corners. Used for validation in `corner_file()`
# below — unknown corners raise ValueError rather than silently returning
# None, so the driver does not emit a malformed netlist.
_VALID_CORNERS: frozenset[str] = frozenset({"tt", "ss", "ff", "sf", "fs"})


# Native X-card contracts from the installed SKY130 model subcircuits. These
# preserve the original native parameter tokens; they are separate from the
# older portable MOS/ideal-cap driver and its two-terminal varactor abstraction.
_MOS_NATIVE_PARAMETERS = (
    "w", "l", "nf", "ad", "as", "pd", "ps", "nrd", "nrs", "sa", "sb", "sd", "mult", "m",
)
_NATIVE_SPICE_MODELS = {}
for _native_model in (
    "sky130_fd_pr__nfet_01v8", "sky130_fd_pr__nfet_01v8_lvt", "sky130_fd_pr__pfet_01v8",
):
    _NATIVE_SPICE_MODELS[_native_model] = NativeSpiceModelContract(
        model=_native_model, family=NativeSpiceFamily.MOS,
        terminals=(("d", "d"), ("g", "g"), ("s", "s"), ("b", "b")),
        parameters=_MOS_NATIVE_PARAMETERS, required_parameters=("w", "l"),
        positive_parameters=("w", "l", "nf", "mult", "m"), integer_parameters=("nf",),
        expression_parameters=("ad", "as", "pd", "ps", "nrd", "nrs", "sa", "sb", "sd"),
        length_scales=(("w", 1e-6), ("l", 1e-6)),
        total_width_parameter="w", finger_parameter="nf",
        provenance=("sky130_fd_pr spice model .subckt d g s b; model-local .param w/l/nf; "
                    "native numeric w/l are micrometres under SKY130 scale=1u; "
                    "nf divides total W; m is the native subcircuit multiplier"),
    )
_NATIVE_SPICE_MODELS["sky130_fd_pr__cap_var_lvt"] = NativeSpiceModelContract(
    model="sky130_fd_pr__cap_var_lvt", family=NativeSpiceFamily.VARACTOR,
    terminals=(("c0", "p"), ("c1", "n"), ("b", "b")),
    parameters=("w", "l", "vm", "m", "wc", "lc"), required_parameters=("w", "l"),
    positive_parameters=("w", "l", "vm", "m", "wc", "lc"), integer_parameters=(),
    expression_parameters=("wc", "lc"), length_scales=(("w", 1e-6), ("l", 1e-6)),
    provenance=("sky130_fd_pr__cap_var_lvt.model.spice: .subckt c0 c1 b w=5 l=0.5 vm=1; "
                "wc='w*1' lc='l*1'; native W/L in micrometres; distinct c1 and b"),
)
_NATIVE_SPICE_MODELS["sky130_fd_pr__cap_mim_m3_1"] = NativeSpiceModelContract(
    model="sky130_fd_pr__cap_mim_m3_1", family=NativeSpiceFamily.MIM,
    terminals=(("c0", "p"), ("c1", "n")),
    parameters=("w", "l", "mf", "m"), required_parameters=("w", "l"),
    positive_parameters=("w", "l", "mf", "m"), integer_parameters=(), expression_parameters=(),
    length_scales=(("w", 1e-6), ("l", 1e-6)),
    provenance=("sky130_fd_pr__cap_mim_m3_1.model.spice: .subckt c0 c1 w=1 l=1 mf=1; "
                "native W/L in micrometres; mf and native m remain distinct parameters"),
)


def _corner_file(corner: str) -> Optional[Path]:
    """Locate the sky130 ngspice corner include file via Clay config.

    Validates `corner` against the standard sky130 corner set
    {tt, ss, ff, sf, fs} and raises ValueError for unknown corners.
    Returns None only when the configured pdk_root is missing — in that
    case the driver surfaces a separate "PDK corner file not available"
    error with the resolved path, which is the right place to complain
    about install issues.
    """
    if corner not in _VALID_CORNERS:
        raise ValueError(
            f"Unknown sky130 corner {corner!r}. "
            f"Valid corners: {sorted(_VALID_CORNERS)}."
        )
    cfg = get_config()
    return cfg.pdk_corner_file(corner)


def _extra_includes(corner: str) -> list[Path]:
    """List of additional include files needed for full device-class
    coverage at this corner.

    For sky130 the canonical per-MOSFET `corners/<corner>.spice` files
    do NOT pre-load the resistor / capacitor sheet-rho params (`rnw`,
    `rp1`, `rcn`, `rcp1`, ...). Those params live in
    `r+c/res_typical__cap_typical.spice` and are referenced by the
    accumulation-mode N-well varactor subckts (`sky130_fd_pr__cap_var_*`)
    and the polysilicon resistor models. Without them, the cap_var
    subckts fail to elaborate with `Undefined parameter [rnw]` and
    ngspice aborts.

    We include them unconditionally because:
      * Cost: the file is < 100 lines of `.param` statements — no
        nontrivial netlist content, so the driver pays nothing in
        simulation time.
      * Correctness: any cell that uses ONLY MOSFETs ignores the params
        entirely (no element references them), so legacy decks are
        bit-identical.
      * Idempotence: defining `.param rnw=1700` twice in a deck is fine
        in ngspice (the later definition wins; both files set the same
        value).

    The same r+c file is the right choice for every "tt"-style corner
    Clay supports (ss, ff, sf, fs) because those corners only flip
    MOSFET process; resistor / cap process is typical by default.
    For Monte Carlo (mc) corners that vary r/c, callers should use
    the .lib-based corner loader rather than the .include path.
    """
    cfg = get_config()
    pdk_root = cfg.pdk_root
    if pdk_root is None:
        return []
    # sky130 ngspice tech dir lives at:
    #   <pdk_root>/libs.tech/ngspice/r+c/res_typical__cap_typical.spice
    rc_path = (
        pdk_root
        / "libs.tech"
        / "ngspice"
        / "r+c"
        / "res_typical__cap_typical.spice"
    )
    if not rc_path.exists():
        # Honest failure mode: if the user's sky130 install is missing the
        # r+c file the driver should surface the error explicitly when a
        # varactor_accum device is in the netlist, not silently return an
        # empty list. But we return [] here when the file genuinely doesn't
        # exist so non-varactor-accum decks still work — the driver path
        # for the cap_var subckts will get the same "Undefined parameter"
        # error that it would get without our hook, which makes the
        # diagnostic identical to a fresh install issue.
        return []
    return [rc_path]


SKY130 = PdkAdapter(
    pdk_id="sky130",
    device_bindings=_BINDINGS,
    mu_cox=_MU_COX,
    corner_file_resolver=_corner_file,
    extra_includes_resolver=_extra_includes,
    min_L=0.15e-6,
    vdd_nom=1.8,
    extra={"family": "sky130_fd_pr", "supply_label": "1v8"},
    native_spice_models=_NATIVE_SPICE_MODELS,
    native_geometry_provider=NATIVE_GEOMETRY_PROVIDER,
    # sky130_fd_sc_hd__nom.tlef routing DIRECTION declarations.
    preferred_routing_axes={1: "H", 2: "V", 3: "H", 4: "V", 5: "H"},
    # sky130A_mr.drc OFFGRID checks: every manufacturing layer .ongrid(0.005).
    # Also sky130A.tech cifoutput gridlimit 5 (nm). Signoff's pinned Docker
    # runtime binds these files even when the host has no local DRC deck.
    manufacturing_grid_nm=5,
    # Native authority: sky130A_mr.drc capm.2b_a derives
    # (capm & met3).sized(0.14), then separates unrelated M3 by 1.2um.
    # sky130A.tech capm.11 separately checks 1340nm with surround_ok;
    # native DRC enables Euclidean distance.  Remote metal connectivity does
    # not waive a local facing gap.  The two predicates remain independent.
    mim_bottom_routing_clearance_nm=(
        MimMetalGeometryRule("capm.2b_a", "met3", "capm",
                             "derived_unrelated", 140, 1200),
        MimMetalGeometryRule("capm.11", "met3", "capm",
                             "local_surround", 0, 1340),
    ),
    # sky130A.tech cifoutput BOUND/CELLBOUND (FIXED_BBOX) and the installed
    # met_min_ca_density.lydrc chip_boundary input use this nonconducting purpose.
    assembly_boundary_layer=(235, 4),
)


# Auto-register on import.
register_pdk(SKY130)
