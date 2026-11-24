"""
v0.0.99 R76 B / v0.1.00 R77 B — Per-PDK gm/Id LUT generator (ngspice-driven).

R76 B (sky130) — Runs real DC sweeps in ngspice_con on the sky130 1.8 V
device flavors to populate the 4-D LUT at
``data/pdk_luts/sky130_gm_id.json``.

R77 B (gf180mcu) — Extends the same sweep harness to the
GlobalFoundries gf180mcuC 3.3 V / 6.0 V devices, emitting
``data/pdk_luts/gf180mcu_gm_id.json``. Adding gf180mcu validates that
the LUT-based sizing pipeline is PDK-invariant (per PL philosophy
audit §3.5 — layout/PDK decoupling): same Rust loader, same
``GmIdLutHandle.lut_size_w`` API, same downstream sizing semantics;
only the per-PDK device table, voltage grids, and supply rails change.

This is the silicon-grade companion to ``generate_sky130_gm_id_lut.py``
(analytic mode). Both emit the same JSON schema so the Rust loader
(``crates/clay_ir/src/lut.rs::GmIdLut::from_json_str``) does not need
to distinguish between them. The ``source`` field in the JSON header
records provenance (``"ngspice_dc_sweep_sky130_tt"``,
``"ngspice_dc_sweep_gf180mcu_tt"``, or
``"analytic_bsim_inspired"``) for downstream tooling that wants to
report which model was used.

The sweep deck (per class):

    .include <corner_file>
    M1 D G S B <subckt>  L=<L>  W=1u
    VDS D 0 DC <Vds>
    VGS G 0 DC <Vgs>
    VBS B 0 DC 0
    VSS S 0 DC 0
    .dc VGS <Vgs_min> <Vgs_max> <Vgs_step> VDS <Vds_min> <Vds_max> <Vds_step>
    .save i(VDS) @M1[gm]
    .control
    run
    wrdata <out>.csv v(g) v(d) i(VDS) @M1[gm]
    .endc
    .end

We sweep Vgs over [V_TH - 0.2, V_TH + 0.6] V (sky130) or
[V_TH - 0.2, V_TH + 1.2] V (gf180mcu, with its higher supply) to cover
weak through strong inversion. Vds spans [0.05, 1.6] V (sky130) or
[0.1, 3.0] V (gf180mcu). One ngspice run per (class, L) emits a
(Vgs × Vds) grid. We then post-process to:

  1. Compute gm/Id = gm/Id at every (Vgs, Vds) grid point.
  2. Map Vgs → Vov via the per-(L) V_TH extracted from the sweep itself
     (Vth = max-gm point at Vds=Vdd/2; rough but adequate for the
     LUT's coverage axes).
  3. Re-sample onto the canonical Vov grid the analytic model uses, so
     downstream LUT consumers see the same axes.
  4. Emit the JSON in the analytic file's exact schema.

This script is OPERATOR-RUN, not CI. It takes a few minutes per PDK
(5 classes x 7 L values x ngspice startup overhead ~= 1-3 min on a
modern machine). The output overwrites the per-PDK JSON in place.

Usage:
    # sky130 (R76 B baseline; default)
    /c/Users/Administrator/anaconda3/python \\
        tools/pdk_lut_generator/run_ngspice_sweep.py \\
        --pdk sky130 \\
        --output data/pdk_luts/sky130_gm_id.json \\
        [--corner tt]

    # gf180mcu (R77 B)
    CLAY_GF180MCU_PATH=/path/to/gf180mcu_pdk \\
    /c/Users/Administrator/anaconda3/python \\
        tools/pdk_lut_generator/run_ngspice_sweep.py \\
        --pdk gf180mcu \\
        --output data/pdk_luts/gf180mcu_gm_id.json \\
        [--corner tt]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the analytic-mode generator's grids so the two outputs agree
# on the LUT shape.
from generate_sky130_gm_id_lut import (  # type: ignore[import-not-found]
    L_GRID,
    VOV_GRID,
    VDS_GRID,
    _DEVICE_CLASSES,
)


# Map Clay class names to sky130_fd_pr subckt names. These match the
# Python adapter (clay/pdk/sky130.py::_BINDINGS) so the LUT keys stay
# consistent across the registry, the Rust crate, and the simulator.
_CLASS_TO_SUBCKT = {
    "nmos":     "sky130_fd_pr__nfet_01v8",
    "pmos":     "sky130_fd_pr__pfet_01v8",
    "nmos_lvt": "sky130_fd_pr__nfet_01v8_lvt",
    "pmos_lvt": "sky130_fd_pr__pfet_01v8_lvt",
    "pmos_hvt": "sky130_fd_pr__pfet_01v8_hvt",
}


# v0.1.00 R77 B — gf180mcu device-class table.
#
# Mirrors clay/pdk/gf180mcu.py::_BINDINGS so the LUT keys join cleanly
# across the registry / Rust crate / simulator. The vth0 / mu_cox /
# polarity / n_subvt values seed the resampling fallback (when ngspice
# cannot extract Vth) and the JSON header's per-class metadata. They
# match the strong-inversion constants in the gf180mcu adapter +
# clay_ir::lut::for_pdk("gf180mcu") so the seed values agree across
# all three surfaces.
_GF180MCU_DEVICE_CLASSES: dict[str, dict] = {
    "nmos":     {"mu_cox": 220e-6, "vth0": 0.70, "lambda_l": 0.30e-6, "n_subvt": 1.40, "polarity": +1},
    "pmos":     {"mu_cox":  70e-6, "vth0": 0.75, "lambda_l": 0.30e-6, "n_subvt": 1.45, "polarity": -1},
    "nmos_06v": {"mu_cox": 150e-6, "vth0": 0.90, "lambda_l": 0.45e-6, "n_subvt": 1.45, "polarity": +1},
    "pmos_06v": {"mu_cox":  45e-6, "vth0": 0.95, "lambda_l": 0.45e-6, "n_subvt": 1.50, "polarity": -1},
    # Native (depletion) NMOS sits near Vth ~= 0 V; vth0 is a default
    # for when the constant-current extraction fails (rare, since the
    # device turns on near 0 V). The 6 V oxide stack still applies.
    "nmos_nat": {"mu_cox": 200e-6, "vth0": 0.05, "lambda_l": 0.45e-6, "n_subvt": 1.50, "polarity": +1},
}

_GF180MCU_CLASS_TO_SUBCKT: dict[str, str] = {
    "nmos":     "nmos_3p3",
    "pmos":     "pmos_3p3",
    "nmos_06v": "nmos_6p0",
    "pmos_06v": "pmos_6p0",
    "nmos_nat": "nmos_6p0_nat",
}


# Per-class L lower bound (in metres). The gf180mcu BSIM bins do not
# cover the full canonical L grid for every device:
#   * nmos_3p3 / pmos_3p3: full coverage from 0.28 um
#   * nmos_6p0 / pmos_6p0: 0.5 um (pmos) / 0.6 um (nmos) and up
#   * nmos_6p0_nat       : 1.8 um and up (native is large-device only)
# Sweeps below the per-class min_L would hit "could not find a valid
# modelname" because no .model bin covers the (L, W) point. We skip
# those L values per class so the LUT only carries grid points that
# the BSIM model can actually evaluate. The Rust interpolator handles
# variable per-class point counts.
_GF180MCU_PER_CLASS_LMIN: dict[str, float] = {
    "nmos":     0.28e-6,
    "pmos":     0.28e-6,
    "nmos_06v": 0.60e-6,
    "pmos_06v": 0.50e-6,
    "nmos_nat": 1.80e-6,
}


# gf180mcu Vov / Vds / L grids -- wider voltage ranges than sky130 to
# match the 3.3 V (and 6 V tolerant) supply, but the same axis counts
# so the LUT shapes stay consistent across PDKs. The Rust LUT consumer
# does not care about specific grid values -- it interpolates from
# whatever (Vov, Vds, L) tuples the JSON ships -- so widening the axes
# is purely additive to LUT coverage.
GF180MCU_VOV_GRID: list[float] = [round(-0.20 + 0.10 * i, 3) for i in range(17)]  # -0.2 .. 1.4 V, 0.1 V step
GF180MCU_VDS_GRID: list[float] = sorted({round(v, 4) for v in [
    0.10, 0.20, 0.40, 0.60, 0.90, 1.20, 1.60, 2.00, 2.40, 2.70, 3.00,
]})
GF180MCU_L_GRID: list[float] = [
    0.28e-6, 0.36e-6, 0.50e-6, 0.80e-6, 1.20e-6, 1.80e-6, 3.00e-6,
]


# ─── PDK sweep config ──────────────────────────────────────────────
#
# Per-PDK constants. The sweep harness pulls everything PDK-specific
# from this struct so the deck builder / parser / resampler stay
# generic. sky130 and gf180mcu both flow through the same code path;
# adding a third PDK is "write a new _PDK_CONFIGS entry + corner-file
# resolver" only.

@dataclass(frozen=True)
class PdkSweepConfig:
    pdk_id: str
    vdd_nom: float
    class_to_subckt: dict[str, str]
    device_classes: dict[str, dict]
    vov_grid: list[float]
    vds_grid: list[float]
    l_grid: list[float]
    # Voltage sweep extents (Vgs / Vds magnitudes). For sky130 these
    # default to 1.6 V (its supply); for gf180mcu we widen to 3.0 V.
    vgs_max: float = 1.6
    vds_max: float = 1.6
    vgs_step: float = 0.025
    vds_step: float = 0.05
    # When set, the deck inserts `.option scale=1u`. sky130 needs this
    # (the corner file does not set it); gf180mcu's clay_wrappers/*.spice
    # already sets `.option scale=1.0u`, so we leave the deck-level
    # injection off there to avoid clobbering the wrapper's choice.
    inject_scale_option: bool = True
    # Per-PDK corner-file resolver (returns the file the .include line
    # in the deck points at). sky130 reads from clay.config; gf180mcu
    # uses the PDK adapter's auto-generated wrapper.
    corner_file_resolver: Optional[Callable[[str], Path]] = None
    # When True (sky130), the deck also includes the r+c sheet-rho file
    # for resistor params referenced by subckt instances. gf180mcu's
    # subckts do not depend on this so the field is False.
    needs_rc_include: bool = False
    # Constant-current Vth extraction reference. sky130 uses 100 nA;
    # gf180mcu's higher-Vth devices need a smaller reference (10 nA)
    # because the sub-threshold current at the lowest Vds slice is
    # weaker per unit width.
    vth_iref: float = 1e-7
    # Source/body bias rail for PMOS (NMOS always at 0 V). For sky130
    # this is 1.8 V; for gf180mcu we put PMOS at vdd_nom = 3.3 V so the
    # device is reverse-biased the way a real circuit would be.
    pmos_supply: float = 1.8
    # Optional per-class minimum L (in metres). When a class has an
    # entry here, the sweep skips any L below that value -- needed for
    # gf180mcu's 6 V / native devices which have larger BSIM-bin lmin
    # than the canonical L grid. sky130 has no per-class L floor.
    per_class_lmin: dict[str, float] = field(default_factory=dict)


def _sky130_config() -> PdkSweepConfig:
    return PdkSweepConfig(
        pdk_id="sky130",
        vdd_nom=1.8,
        class_to_subckt=_CLASS_TO_SUBCKT,
        device_classes=_DEVICE_CLASSES,
        vov_grid=VOV_GRID,
        vds_grid=VDS_GRID,
        l_grid=L_GRID,
        vgs_max=1.6,
        vds_max=1.6,
        vgs_step=0.025,
        vds_step=0.05,
        inject_scale_option=True,
        corner_file_resolver=_find_sky130_corner_file,
        needs_rc_include=True,
        vth_iref=1e-7,
        pmos_supply=1.8,
    )


def _gf180mcu_config() -> PdkSweepConfig:
    return PdkSweepConfig(
        pdk_id="gf180mcu",
        vdd_nom=3.3,
        class_to_subckt=_GF180MCU_CLASS_TO_SUBCKT,
        device_classes=_GF180MCU_DEVICE_CLASSES,
        vov_grid=GF180MCU_VOV_GRID,
        vds_grid=GF180MCU_VDS_GRID,
        l_grid=GF180MCU_L_GRID,
        # Widen sweep to cover both 3.3 V devices (Vov up to ~1.5 V)
        # and 6 V devices biased ≤ 3.3 V (Vov up to ~2.5 V at Vgs=3.0).
        vgs_max=3.0,
        vds_max=3.0,
        vgs_step=0.05,    # Coarser than sky130: bigger range, same point count
        vds_step=0.10,
        inject_scale_option=False,  # wrapper already sets scale=1.0u
        corner_file_resolver=_find_gf180mcu_corner_file,
        needs_rc_include=False,
        # gf180mcu's nominal Vth is higher (~0.7 V) and its 6 V variants
        # higher still. At Vds=0.1 V (lowest slice) the constant-current
        # crossing for I_ref = 100 nA × W/L = 100 nA × (1µm / L_um) sits
        # near the wide-inversion region. We loosen to 10 nA so the
        # crossing stays in saturation -- shifts the extracted Vth by
        # at most ~50 mV vs the "true" device Vth, which is well within
        # the LUT's resampling tolerance.
        vth_iref=1e-8,
        pmos_supply=3.3,
        per_class_lmin=_GF180MCU_PER_CLASS_LMIN,
    )


def _find_ngspice_con() -> str:
    """Locate ngspice_con.exe.  Mirrors clay._ir_bridge._discover_ngspice_con."""
    bin_path = (
        shutil.which("ngspice_con")
        or shutil.which("ngspice_con.exe")
        or shutil.which("ngspice")
    )
    if bin_path is not None:
        return bin_path
    for cand in (r"C:\Spice64\bin\ngspice_con.exe",
                 r"C:\Program Files\Spice64\bin\ngspice_con.exe"):
        if Path(cand).exists():
            return cand
    raise RuntimeError(
        "ngspice_con not found. Install ngspice (Windows: use the "
        "Spice64 build that ships ngspice_con.exe)."
    )


def _find_sky130_corner_file(corner: str) -> Path:
    """Locate the sky130 corner file via Clay's config."""
    from clay.config import get_config  # type: ignore[import-not-found]
    cfg = get_config()
    cf = cfg.pdk_corner_file(corner)
    if cf is None or not Path(cf).exists():
        raise RuntimeError(
            f"sky130 corner file for {corner!r} not found. "
            f"Check pdk_root: {cfg.pdk_root!r}"
        )
    return Path(cf)


def _find_gf180mcu_corner_file(corner: str) -> Path:
    """Locate the gf180mcu corner file via the gf180mcu PDK adapter.

    The adapter auto-generates a small wrapper file at
    ``<root>/libraries/gf180mcu_fd_pr/latest/models/ngspice/clay_wrappers/<corner>.spice``
    that issues the right ``.option scale=1.0u`` + ``.include
    design.ngspice`` + ``.lib sm141064.ngspice <gf-corner>`` sequence.
    Calling ``adapter.corner_file(corner)`` ensures the wrapper exists
    on disk before we point ngspice at it.
    """
    from clay.pdk import get_pdk  # type: ignore[import-not-found]
    pdk = get_pdk("gf180mcu")
    cf = pdk.corner_file(corner)
    if cf is None or not Path(cf).exists():
        raise RuntimeError(
            f"gf180mcu corner wrapper for {corner!r} not found. "
            f"Set CLAY_GF180MCU_PATH to the gf180mcu install root "
            f"(e.g. .../AnalogGym/PDK/gf180mcu_pdk). Got: {cf!r}"
        )
    return Path(cf)


# ─── Per-PDK configuration registry ─────────────────────────────────

_PDK_CONFIGS: dict[str, Callable[[], PdkSweepConfig]] = {
    "sky130":   _sky130_config,
    "gf180mcu": _gf180mcu_config,
}


def _build_deck(
    class_name: str,
    subckt: str,
    L: float,
    corner_file: Path,
    polarity: int,
    csv_path: Path,
    cfg: PdkSweepConfig,
) -> str:
    """Build the SPICE deck for one (class, L) DC sweep.

    Sweeps Vgs ∈ [0, cfg.vgs_max] (or its negative for PMOS) and
    Vds ∈ [0.05 / 0.1, cfg.vds_max] in a single nested .dc command.
    The output CSV has columns: Vgs Vds Id gm.
    """
    # ngspice's nested .dc sweep is: .dc <outer> <start> <stop> <step>
    #                                <inner> <start> <stop> <step>
    # The CSV `wrdata` then emits one row per (inner × outer) tuple in
    # outer-major order (Vgs varies fastest with VDS, then VGS).
    # NMOS: positive Vgs / Vds; PMOS: negative (we flip via VDS / VGS
    # source polarity so the same expressions extract |Id| / |gm|).
    vgs_step = cfg.vgs_step  # 25 mV (sky130) / 50 mV (gf180mcu)
    vds_step = cfg.vds_step
    vgs_max = cfg.vgs_max
    vds_max = cfg.vds_max
    sign = polarity  # +1 for NMOS, -1 for PMOS

    # ngspice scale=1 (meters); sky130 conventional scale=1u (microns)
    # We emit W and L in meters explicitly and set .option scale=1 so
    # no surprise unit reinterpretation.

    # For PMOS we set Vgs = -|Vgs| and Vds = -|Vds|.  The corner file
    # references body-effect terminals — we tie B to S in this rig.
    # For nat NMOS (vth ~= 0 V) we extend Vgs slightly negative so the
    # constant-current Vth extraction has both sides of the crossing.
    vds_min = 0.05 if cfg.pdk_id == "sky130" else 0.10
    if sign > 0:
        if class_name == "nmos_nat":
            # native: sweep from -0.4 V so weak-inv samples exist
            vgs_start, vgs_stop, vgs_inc = -0.4, vgs_max, vgs_step
        else:
            vgs_start, vgs_stop, vgs_inc = 0.0, vgs_max, vgs_step
        vds_start, vds_stop, vds_inc = vds_min, vds_max, vds_step
    else:
        # PMOS: source/body to vdd. VGS and VDS are differential
        # voltages (G→S and D→S). For PMOS in active region the gate is
        # below source and drain is below source, so both Vgs and Vds
        # are NEGATIVE.
        vgs_start, vgs_stop, vgs_inc = -vgs_max, 0.0, vgs_step
        vds_start, vds_stop, vds_inc = -vds_max, -vds_min, vds_step

    csv_str = str(csv_path).replace("\\", "/")
    corner_str = str(corner_file).replace("\\", "/")

    # Both PDKs interpret w/l in microns when `.option scale=1u` is
    # active. sky130 needs us to issue it; gf180mcu's wrapper already
    # does. Either way, L/W go into the X-instance as micrometers.
    l_um = L * 1e6   # micrometers
    w_um = 1.0       # 1 µm fixed; gm/Id is W-invariant.

    deck = []
    deck.append(f"* {cfg.pdk_id} gm/Id LUT DC sweep -- class={class_name}, L={L:.3e}")
    # Required mismatch/process Monte Carlo switches (sky130 / gf180mcu
    # subckts check these params at elaboration time; missing →
    # "Cannot compute substitute" elaboration failure).
    deck.append(".param mc_mm_switch=0")
    deck.append(".param mc_pr_switch=0")
    if cfg.inject_scale_option:
        deck.append(".option scale=1u")
    deck.append(f'.include "{corner_str}"')
    # sky130-only: include the resistor/cap sheet-rho file that the
    # subckt instances reference. gf180mcu does not need this.
    if cfg.needs_rc_include:
        pdk_root = corner_file.resolve().parent.parent.parent.parent
        rc_path = pdk_root / "libs.tech" / "ngspice" / "r+c" / "res_typical__cap_typical.spice"
        if rc_path.exists():
            rc_str = str(rc_path).replace("\\", "/")
            deck.append(f'.include "{rc_str}"')
    # X-subckt instance with explicit l/w in microns.
    deck.append(f"X1 D G S B {subckt} l={l_um:.4f} w={w_um:.4f}")
    # Source the same way for both polarities:
    #   * VGS connects G to S so V(VGS) = Vgs directly.
    #   * VDS connects D to S so V(VDS) = Vds directly.
    #   * VS pins source to a fixed rail (gnd for NMOS, VDD for PMOS).
    deck.append("VGS G S DC 0")
    deck.append("VDS D S DC 0")
    if sign > 0:
        deck.append("VS S 0 DC 0")
        deck.append("VB B 0 DC 0")
    else:
        deck.append(f"VS S 0 DC {cfg.pmos_supply}")
        deck.append(f"VB B 0 DC {cfg.pmos_supply}")
    deck.append(
        f".dc VGS {vgs_start} {vgs_stop} {vgs_inc} "
        f"VDS {vds_start} {vds_stop} {vds_inc}"
    )
    deck.append(".control")
    deck.append("run")
    # Output Vgs, Vds, Id via voltage nodes; we post-process to derive
    # gm via finite-difference w.r.t. Vgs at fixed Vds.
    deck.append(f"wrdata {csv_str} v(g) v(s) v(d) i(vds)")
    deck.append("quit")
    deck.append(".endc")
    deck.append(".end")
    return "\n".join(deck) + "\n"


def _run_one_sweep(
    ngspice_bin: str,
    deck_text: str,
    timeout_s: int = 120,
) -> tuple[int, str, str]:
    """Run one ngspice deck via stdin (silent batch mode).  Returns
    (rc, stdout, stderr)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sp", delete=False, newline=""
    ) as f:
        f.write(deck_text)
        deck_path = f.name
    try:
        proc = subprocess.run(
            [ngspice_bin, "-b", deck_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    finally:
        try:
            os.unlink(deck_path)
        except OSError:
            pass


def _parse_wrdata_csv(csv_path: Path, sign: int) -> list[tuple[float, float, float, float]]:
    """Parse ngspice's space-separated `wrdata` output.

    Columns are:  sweepvar  v(g)  sweepvar  v(d)  sweepvar  i(vds)
    (one (sweepvar, value) pair per requested variable — 6 columns
    total for our 3-variable wrdata.)

    Returns a list of (vgs, vds, id, gm) tuples with signs normalized
    so all values are positive (Id flipped for PMOS via ``sign``). The
    ``gm`` field is computed via finite-difference of Id w.r.t. Vgs
    after sorting by (Vds, Vgs).
    """
    raw = []
    text = csv_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        toks = line.split()
        # We requested 4 vars: v(g) v(s) v(d) i(vds) → 8 cols.
        if len(toks) < 8:
            continue
        try:
            vg = float(toks[1])
            vs = float(toks[3])
            vd = float(toks[5])
            id_amps = float(toks[7])
        except ValueError:
            continue
        vgs_raw = vg - vs
        vds_raw = vd - vs
        # i(VDS) is the current FROM + (the D node) THROUGH the source
        # TO - (the S node). For NMOS in active region (V(D) > V(S),
        # current enters D externally), i(VDS) is positive. So the
        # drain current is i(VDS) directly (no flip). Re-check: the
        # convention "current through V source is + to -" means
        # i_through_source = (current that LEAVES the + terminal into
        # the source). With VDS having + on D and - on S, "leaves D
        # into source" = current flowing from D node into the VDS
        # source = drain current (for NMOS, this is into the drain
        # node externally and out through VDS). Empirically: NMOS in
        # saturation with our previous sweep emitted negative i(vds);
        # the active-region drain current MUST be positive. Flip:
        id_amps = -id_amps
        # Magnitudes: for PMOS, Vgs and Vds are negative when biased
        # correctly (V(G) < V(S), V(D) < V(S)). The LUT stores |Vgs|
        # and |Vds|, and Id is positive in active region.
        if sign < 0:
            vgs_raw = -vgs_raw
            vds_raw = -vds_raw
            id_amps = -id_amps
        raw.append((vgs_raw, vds_raw, id_amps))

    # Finite-difference gm w.r.t. Vgs at fixed Vds.
    # Group by Vds (rounded to 4 sig figs), sort each group by Vgs,
    # then compute central differences.
    by_vds: dict[float, list[tuple[float, float, float]]] = {}
    for vgs, vds, idx in raw:
        key = round(vds, 4)
        by_vds.setdefault(key, []).append((vgs, vds, idx))

    out: list[tuple[float, float, float, float]] = []
    for vds_key, rows in by_vds.items():
        rows.sort(key=lambda r: r[0])
        n = len(rows)
        for i, (vgs, vds, idx) in enumerate(rows):
            if i == 0 or i == n - 1:
                # forward/backward diff at endpoints
                if i == 0 and n > 1:
                    vgs2, _, id2 = rows[1]
                    dvgs = vgs2 - vgs
                    gm = (id2 - idx) / dvgs if dvgs != 0 else 0.0
                elif i == n - 1 and n > 1:
                    vgs0, _, id0 = rows[i - 1]
                    dvgs = vgs - vgs0
                    gm = (idx - id0) / dvgs if dvgs != 0 else 0.0
                else:
                    gm = 0.0
            else:
                vgs_prev, _, id_prev = rows[i - 1]
                vgs_next, _, id_next = rows[i + 1]
                dvgs = vgs_next - vgs_prev
                gm = (id_next - id_prev) / dvgs if dvgs != 0 else 0.0
            out.append((vgs, vds, idx, gm))
    return out


def _extract_vth(samples: list[tuple[float, float, float, float]],
                 vth_default: float = 0.45,
                 i_ref: float = 1e-7) -> float:
    """Constant-current Vth extraction: find Vgs at which Id (at the
    lowest Vds slice) crosses I_ref. The reference current is PDK-
    specific (passed in by the caller via cfg.vth_iref): sky130 uses
    100 nA, gf180mcu uses 10 nA because its higher-Vth devices put
    weaker sub-threshold current at the lowest Vds slice.

    Returns ``vth_default`` if no crossing is found (e.g. device never
    turned on in the swept range).
    """
    by_vds: dict[float, list[tuple[float, float, float, float]]] = {}
    for vgs, vds, idx, gm in samples:
        by_vds.setdefault(round(vds, 4), []).append((vgs, vds, idx, gm))
    if not by_vds:
        return vth_default
    target_vds = min(by_vds, key=lambda v: abs(v - 0.05))  # lowest Vds slice
    rows = sorted(by_vds[target_vds], key=lambda r: r[0])

    # Linear interpolation between the two samples that bracket I_ref.
    last_below = None
    for vgs, _, idx, _ in rows:
        if idx >= i_ref:
            if last_below is None:
                return vgs
            v0, _, i0, _ = last_below
            # Linear interpolation between samples
            frac = (i_ref - i0) / max(idx - i0, 1e-30)
            return v0 + frac * (vgs - v0)
        last_below = (vgs, _, idx, 0.0)
    return vth_default


def _resample_to_grid(
    samples: list[tuple[float, float, float, float]],
    vth: float,
    L: float,
    polarity: int,
    vov_grid: list[float],
    vds_grid: list[float],
    vgs_step_norm: float,
    vds_step_norm: float,
) -> list[dict]:
    """Re-sample ngspice's (Vgs, Vds) raster onto the canonical
    (Vov, Vds) grid.  Uses nearest-neighbor; the ngspice raster is
    denser than the LUT grid so this introduces < one ngspice-step of
    slack on each axis.
    """
    out: list[dict] = []
    for vov in vov_grid:
        target_vgs = vth + vov
        for vds_target in vds_grid:
            # Find nearest (Vgs, Vds) sample (no Id sign filter — weak
            # inversion has tiny positive Id; gm/Id is still meaningful).
            best = None
            best_dist = 1e9
            for vgs, vds, idx, gm in samples:
                d = ((vgs - target_vgs) ** 2 / vgs_step_norm ** 2
                     + (vds - vds_target) ** 2 / vds_step_norm ** 2)
                if d < best_dist:
                    best_dist = d
                    best = (vgs, vds, idx, gm)
            if best is None:
                continue
            _, _, idx, gm = best
            # Skip if Id is non-positive (off region — gm/Id is undefined)
            if idx <= 1e-15:
                continue
            gm_id = gm / idx if idx > 0 else 0.0
            # Physical clamp: weak-inversion gm/Id is bounded by
            # 1/(n*Vt) where n=1.0 gives the absolute max (~ 40 V^-1
            # at 300 K). Any value above 50 V^-1 is a finite-difference
            # artifact at the deepest sub-Vth corner where Id is on
            # the order of ngspice's numerical noise floor. Sky130's
            # LUT clamps implicitly because its devices have stronger
            # subthreshold slopes; gf180mcu's native NMOS (vth ~= 0 V)
            # exhibits the artifact more often, so we filter here at
            # the resampler stage where ground truth is still
            # available. Skipping the point (rather than clipping) is
            # the safer choice -- it tells downstream sizing code that
            # this corner is out-of-coverage.
            if gm_id > 50.0:
                continue
            # Width was 1 µm in the deck; Id/W = Id / 1e-6
            id_per_w = idx / 1e-6
            out.append({
                "L": L,
                "Vov": vov,
                "Vds": vds_target,
                "gm_id": round(gm_id, 4),
                "id_per_w": float(f"{id_per_w:.4e}"),
            })
    return out


def sweep_pdk(
    corner: str,
    output: Path,
    pdk: str = "sky130",
    classes: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """Drive a full per-PDK sweep, returning the LUT dict.

    Args:
        corner: process corner ("tt" / "ss" / "ff" / ...).
        output: destination JSON path.
        pdk: PDK id -- "sky130" or "gf180mcu". Default "sky130" keeps
            the R76 B byte-exact entrypoint.
        classes: optional subset of device classes to sweep.
        verbose: print per-step progress to stderr.
    """
    if pdk not in _PDK_CONFIGS:
        raise ValueError(
            f"Unknown PDK {pdk!r}. Known PDKs: {list(_PDK_CONFIGS)}. "
            f"Add a new PdkSweepConfig + register in _PDK_CONFIGS."
        )
    cfg = _PDK_CONFIGS[pdk]()
    ngspice_bin = _find_ngspice_con()
    assert cfg.corner_file_resolver is not None
    corner_file = cfg.corner_file_resolver(corner)
    if verbose:
        print(f"PDK:            {cfg.pdk_id}", file=sys.stderr)
        print(f"Using ngspice:  {ngspice_bin}", file=sys.stderr)
        print(f"Corner file:    {corner_file}", file=sys.stderr)
        print(f"Vdd nominal:    {cfg.vdd_nom} V", file=sys.stderr)

    target_classes = classes or list(cfg.class_to_subckt.keys())
    out_classes: dict[str, dict] = {}

    workdir = Path(tempfile.mkdtemp(prefix=f"{cfg.pdk_id}_lut_sweep_"))

    for cname in target_classes:
        subckt = cfg.class_to_subckt[cname]
        params = cfg.device_classes.get(
            cname, {"polarity": +1, "mu_cox": 200e-6, "vth0": 0.45, "n_subvt": 1.35}
        )
        polarity = params["polarity"]

        rows: list[dict] = []
        if verbose:
            print(f"\n=== Class {cname} ({subckt}, polarity={polarity:+d}) ===", file=sys.stderr)

        # Per-class L floor: skip L values the BSIM bins don't cover.
        # gf180mcu's 6 V / native devices have lmin > the canonical L
        # grid's minimum, so attempting those would produce
        # "could not find a valid modelname" errors. sky130 has no
        # floor (cfg.per_class_lmin is empty), so its loop is unchanged.
        class_lmin = cfg.per_class_lmin.get(cname, 0.0)
        for L in cfg.l_grid:
            if L < class_lmin:
                if verbose:
                    print(
                        f"  Skipping L={L*1e9:.0f}n (below {cname} BSIM lmin = {class_lmin*1e6:.2f}u)",
                        file=sys.stderr,
                    )
                continue
            csv_path = workdir / f"{cname}_L{int(L*1e9)}n.csv"
            deck = _build_deck(cname, subckt, L, corner_file, polarity, csv_path, cfg)
            if verbose:
                print(f"  Sweeping L={L*1e9:.0f}n... ", end="", file=sys.stderr, flush=True)
            rc, out_s, err_s = _run_one_sweep(ngspice_bin, deck)
            if rc != 0 or not csv_path.exists():
                if verbose:
                    print(f"FAIL (rc={rc})", file=sys.stderr)
                    print(f"    stderr (last 200): {err_s[-200:]}", file=sys.stderr)
                continue
            samples = _parse_wrdata_csv(csv_path, polarity)
            if not samples:
                if verbose:
                    print(f"NO DATA", file=sys.stderr)
                continue
            vth = _extract_vth(samples, vth_default=params["vth0"], i_ref=cfg.vth_iref)
            new_rows = _resample_to_grid(
                samples, vth, L, polarity,
                cfg.vov_grid, cfg.vds_grid,
                vgs_step_norm=cfg.vgs_step,
                vds_step_norm=cfg.vds_step,
            )
            rows.extend(new_rows)
            if verbose:
                print(f"OK  Vth~={vth:.3f}V, {len(new_rows)} grid pts", file=sys.stderr)

        out_classes[cname] = {
            "mu_cox_strong": params["mu_cox"],
            "vth0": params["vth0"],
            "polarity": polarity,
            "n_subvt": params.get("n_subvt", 1.35),
            "points": rows,
        }

    lut = {
        "schema_version": 1,
        "pdk_id": cfg.pdk_id,
        "vdd_nom": cfg.vdd_nom,
        "source": f"ngspice_dc_sweep_{cfg.pdk_id}_{corner}",
        "vov_grid": cfg.vov_grid,
        "vds_grid": cfg.vds_grid,
        "l_grid": cfg.l_grid,
        "classes": out_classes,
    }
    if verbose:
        total = sum(len(c["points"]) for c in lut["classes"].values())
        print(f"\nTotal grid points: {total} across {len(lut['classes'])} classes", file=sys.stderr)
    return lut


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", "-o", type=Path,
                   default=None,
                   help="Output JSON path (default: data/pdk_luts/<pdk>_gm_id.json)")
    p.add_argument("--pdk", default="sky130", choices=sorted(_PDK_CONFIGS.keys()),
                   help="PDK id (default: sky130)")
    p.add_argument("--corner", "-c", default="tt", choices=("tt", "ss", "ff", "sf", "fs"))
    p.add_argument("--classes", nargs="+",
                   help="Subset of classes to sweep (default: all)")
    p.add_argument("--quiet", "-q", action="store_true")
    args = p.parse_args(argv)

    out_path = args.output or Path(f"data/pdk_luts/{args.pdk}_gm_id.json")

    lut = sweep_pdk(args.corner, out_path,
                    pdk=args.pdk,
                    classes=args.classes,
                    verbose=not args.quiet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(lut, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    size_kb = out_path.stat().st_size / 1024
    if not args.quiet:
        print(f"\nWrote LUT to {out_path} ({size_kb:.1f} kB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
