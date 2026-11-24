"""
Clay runtime configuration.

Single source of truth for paths and tool locations. Resolution order:

  1. Explicit argument to ClayConfig() constructor
  2. Environment variable
  3. clay.toml in current working directory or repo root
  4. Auto-discovery (platform-specific defaults)
  5. None (caller must handle)

Everything that touches the host filesystem or external tools goes through
this module. No hardcoded paths anywhere else.

Open-source readiness: this config layer allows users to point Clay at their
own PDK install, their own ngspice binary, their own work directory, without
patching any source.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import tomllib                                     # Python 3.11+
except ImportError:                                    # pragma: no cover
    import tomli as tomllib                            # type: ignore[import]


# ─── Environment variable names ─────────────────────────────

ENV_NGSPICE_BIN     = "CLAY_NGSPICE_BIN"
ENV_PDK_PATH        = "CLAY_PDK_PATH"            # e.g. /path/to/sky130_pdk
ENV_PDK_ID          = "CLAY_PDK_ID"              # default 'sky130'
ENV_WORK_DIR        = "CLAY_WORK_DIR"
ENV_CONFIG_FILE     = "CLAY_CONFIG"              # explicit clay.toml path
# SystemC behavioral-tier toolchain. CLAY_SYSTEMC_HOME takes precedence; the
# standard SYSTEMC_HOME is honored as a fallback so existing installs work
# unchanged. CLAY_CXX / CXX point at the C++ compiler used to build the
# behavioral models (must match the SystemC build's ABI).
ENV_SYSTEMC_HOME    = "CLAY_SYSTEMC_HOME"
ENV_SYSTEMC_HOME_STD = "SYSTEMC_HOME"
ENV_CXX             = "CLAY_CXX"
ENV_CXX_STD         = "CXX"


# ─── Auto-discovery candidates ──────────────────────────────

_NGSPICE_CANDIDATES_WINDOWS = [
    # Console-only variant is required on Windows; see project memory:
    # feedback-ngspice-con-for-silent
    r"C:\Spice64\bin\ngspice_con.exe",
    r"C:\Program Files\Spice64\bin\ngspice_con.exe",
]
_NGSPICE_CANDIDATES_UNIX = [
    "/usr/local/bin/ngspice",
    "/usr/bin/ngspice",
    "/opt/homebrew/bin/ngspice",
]

# Default sky130 PDK locations relative to repo root (for dev convenience;
# production users set CLAY_PDK_PATH).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SKY130_DEV_DEFAULT = _REPO_ROOT / "AnalogGym" / "PDK" / "sky130_pdk"


# ─── Config dataclass ───────────────────────────────────────

@dataclass
class ClayConfig:
    ngspice_bin:  Optional[Path] = None
    pdk_id:       str           = "sky130"
    pdk_root:     Optional[Path] = None
    work_dir:     Path           = field(default_factory=lambda: Path("tmp_clay_rundir"))
    systemc_home: Optional[Path] = None   # SystemC install prefix (behavioral tier)
    cxx:          Optional[Path] = None   # C++ compiler for behavioral models

    # ─── Loaders ────────────────────────────────────────

    @classmethod
    def auto(cls, *, explicit: Optional["ClayConfig"] = None) -> "ClayConfig":
        """Resolve config from arg + env + file + auto-discovery."""
        # Start from explicit, then fill gaps
        base = explicit or cls()

        # 1. Env override
        ngspice = _first_existing([os.environ.get(ENV_NGSPICE_BIN)]) or base.ngspice_bin
        pdk_id  = os.environ.get(ENV_PDK_ID) or base.pdk_id
        pdk_root_env = os.environ.get(ENV_PDK_PATH)
        pdk_root = Path(pdk_root_env) if pdk_root_env else base.pdk_root
        work_env = os.environ.get(ENV_WORK_DIR)
        work = Path(work_env) if work_env else base.work_dir

        # 2. TOML file
        toml_data = _load_toml_config()
        if toml_data:
            tool = toml_data.get("tool", {}).get("clay", {})
            if not ngspice and "ngspice_bin" in tool:
                ngspice = Path(tool["ngspice_bin"])
            if pdk_id == "sky130" and "pdk_id" in tool:
                pdk_id = tool["pdk_id"]
            if pdk_root is None and "pdk_root" in tool:
                pdk_root = Path(tool["pdk_root"])
            if work == Path("tmp_clay_rundir") and "work_dir" in tool:
                work = Path(tool["work_dir"])

        # SystemC + compiler: env (CLAY_SYSTEMC_HOME, then std SYSTEMC_HOME) ->
        # toml -> auto-discovery. Same precedence chain as every other tool.
        sc_env = os.environ.get(ENV_SYSTEMC_HOME) or os.environ.get(ENV_SYSTEMC_HOME_STD)
        systemc_home = Path(sc_env) if sc_env else base.systemc_home
        cxx_env = os.environ.get(ENV_CXX) or os.environ.get(ENV_CXX_STD)
        cxx = _first_existing([cxx_env]) or (_first_existing([shutil.which(cxx_env)]) if cxx_env else None) or base.cxx

        # 3. Auto-discovery
        if ngspice is None:
            ngspice = _discover_ngspice()
        if pdk_root is None:
            pdk_root = _discover_pdk(pdk_id)

        if toml_data:
            tool = toml_data.get("tool", {}).get("clay", {})
            if systemc_home is None and "systemc_home" in tool:
                systemc_home = Path(tool["systemc_home"])
            if cxx is None and "cxx" in tool:
                cxx = Path(tool["cxx"])
        if systemc_home is None:
            systemc_home = _discover_systemc()
        if cxx is None:
            cxx = _discover_cxx()

        return cls(
            ngspice_bin=ngspice,
            pdk_id=pdk_id,
            pdk_root=pdk_root,
            work_dir=work,
            systemc_home=systemc_home,
            cxx=cxx,
        )

    # ─── Behavioral-tier (SystemC) build command ────────

    def systemc_build_command(self, src: Path, out: Path,
                              std: str = "c++17") -> Optional[list[str]]:
        """Compiler argv to build a SystemC behavioral model, or None if the
        toolchain is unresolved. All paths come from resolved config -- nothing
        hardcoded -- so an open-source user pointing CLAY_SYSTEMC_HOME / CLAY_CXX
        at their own install gets a correct command with no source edits."""
        if self.cxx is None or self.systemc_home is None:
            return None
        inc = self.systemc_home / "include"
        libdir = _first_existing([str(self.systemc_home / "lib"),
                                  str(self.systemc_home / "lib64")])
        if libdir is None or not (inc / "systemc.h").exists():
            return None
        return [
            str(self.cxx), f"-std={std}", "-O2",
            # SC_HAS_PROCESS + custom ctors are deprecated (not removed) in IEEE
            # 1666-2023 / SystemC 3.0; allow them so the common module idiom
            # builds. Harmless / ignored on pre-3.0 installs.
            "-DSC_ALLOW_DEPRECATED_IEEE_API",
            f"-I{inc}", str(src), "-o", str(out),
            f"-L{libdir}", "-lsystemc",
        ]

    # ─── Derived paths ──────────────────────────────────

    def pdk_corner_file(self, corner: str = "tt") -> Optional[Path]:
        """Path to the ngspice corner include file for the configured PDK."""
        if self.pdk_root is None:
            return None
        if self.pdk_id == "sky130":
            return self.pdk_root / "libs.tech" / "ngspice" / "corners" / f"{corner}.spice"
        # Future PDKs: add their layout here
        return None

    def validate(self) -> list[str]:
        """Return list of error messages; empty list means config is usable."""
        errs: list[str] = []
        if self.ngspice_bin is None or not Path(self.ngspice_bin).exists():
            errs.append(
                f"ngspice binary not found. Set {ENV_NGSPICE_BIN} or "
                f"install ngspice (Windows: use ngspice_con.exe)."
            )
        if self.pdk_root is None or not Path(self.pdk_root).exists():
            errs.append(
                f"PDK root not found. Set {ENV_PDK_PATH} to the sky130 "
                f"(or other) PDK install root."
            )
        corner = self.pdk_corner_file("tt")
        if corner is not None and not corner.exists():
            errs.append(f"PDK corner file missing: {corner}")
        return errs


# ─── Helpers ────────────────────────────────────────────────

def _first_existing(candidates: list[Optional[str]]) -> Optional[Path]:
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return None


def _discover_ngspice() -> Optional[Path]:
    # Prefer console-only on Windows; binary name may include .exe
    is_win = platform.system() == "Windows"
    names = ["ngspice_con", "ngspice"]
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
        # also try with .exe on Windows
        if is_win:
            found = shutil.which(name + ".exe")
            if found:
                return Path(found)
    candidates = (_NGSPICE_CANDIDATES_WINDOWS if is_win else _NGSPICE_CANDIDATES_UNIX)
    return _first_existing(list(candidates))


def _discover_pdk(pdk_id: str) -> Optional[Path]:
    if pdk_id != "sky130":
        return None
    # Dev-convenience: the AnalogGym sky130 bundle at repo root
    if _SKY130_DEV_DEFAULT.exists():
        return _SKY130_DEV_DEFAULT
    # System-wide sky130 conventions (Volare / open_pdks installs)
    candidates = [
        _REPO_ROOT / "pdks" / "sky130A",
        _REPO_ROOT / "pdks" / "sky130_pdk",
        Path("/foss/pdks/sky130A"),
        Path.home() / ".volare" / "sky130A",
        Path("/usr/share/pdk/sky130A"),
        Path("/opt/pdk/sky130A"),
    ]
    return _first_existing([str(c) for c in candidates])


def _discover_systemc() -> Optional[Path]:
    """Locate a SystemC install prefix (one containing include/systemc.h).

    Discovery only -- never a hardcoded source path. Open-source users set
    CLAY_SYSTEMC_HOME (or the standard SYSTEMC_HOME); these candidates are the
    same kind of soft fallback used for ngspice/PDK above.
    """
    candidates = [
        Path.home() / "systemc",
        Path("/usr/local/systemc"),
        Path("/opt/systemc"),
        Path("/usr/local"),                # headers under include/systemc.h
    ]
    for c in candidates:
        if (c / "include" / "systemc.h").exists():
            return c
    return None


def _discover_cxx() -> Optional[Path]:
    for name in ["g++", "clang++", "c++"]:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def _load_toml_config() -> dict:
    """Locate and parse clay.toml (or pyproject.toml [tool.clay] section)."""
    explicit = os.environ.get(ENV_CONFIG_FILE)
    search = []
    if explicit:
        search.append(Path(explicit))
    search.append(Path.cwd() / "clay.toml")
    search.append(_REPO_ROOT / "clay.toml")
    search.append(Path.cwd() / "pyproject.toml")
    search.append(_REPO_ROOT / "pyproject.toml")
    for p in search:
        if p.exists():
            try:
                with p.open("rb") as f:
                    return tomllib.load(f)
            except Exception:
                continue
    return {}


# Convenience singleton accessor (lazy)
_GLOBAL_CONFIG: Optional[ClayConfig] = None


def get_config(*, fresh: bool = False) -> ClayConfig:
    global _GLOBAL_CONFIG
    if fresh or _GLOBAL_CONFIG is None:
        _GLOBAL_CONFIG = ClayConfig.auto()
    return _GLOBAL_CONFIG


# ─── Geometric-fidelity threshold bands (the QUALITY-tier gate's parameters) ──
#
# The geometric-fidelity checkers (clay.layout.spec.geometric_fidelity) compare a
# layout's MEASURED metrics against these bands. Two design rules hold here:
#
#   1. THRESHOLDS ARE CONFIG PARAMETERS, never hardcoded constants. They resolve
#      through the same env -> toml -> default chain as every other tool setting,
#      so a future optimizer (tSS-BO / adjoint / `clay tune`) can auto-calibrate
#      a band by exporting CLAY_FIDELITY_* (or writing [tool.clay.fidelity] in
#      clay.toml) WITHOUT patching any source. The checker reads the resolved
#      value; it never embeds a number.
#
#   2. The DEFAULTS are reference-calibrated SCALARS only. The signed external_pdk
#      reference layout (10704 um^2, aspect ~1.67, fill ~33%, enclosure ~85%,
#      ~9 tight per-compact-group rings) calibrates the band edges below. The
#      reference contributes ONLY these scalars -- it never feeds a coordinate
#      into any metric's input (the leakage line).

ENV_FIDELITY_ENCLOSURE_MIN   = "CLAY_FIDELITY_ENCLOSURE_MIN"
ENV_FIDELITY_ASPECT_LO       = "CLAY_FIDELITY_ASPECT_LO"
ENV_FIDELITY_ASPECT_HI       = "CLAY_FIDELITY_ASPECT_HI"
ENV_FIDELITY_FILL_MIN        = "CLAY_FIDELITY_FILL_MIN"
ENV_FIDELITY_RING_TOL        = "CLAY_FIDELITY_RING_TOL"
ENV_FIDELITY_CAP_EDGE_MARGIN_UM = "CLAY_FIDELITY_CAP_EDGE_MARGIN_UM"
ENV_FIDELITY_CONT_PER_FINGER_MIN = "CLAY_FIDELITY_CONT_PER_FINGER_MIN"
ENV_FIDELITY_BBOX_FILL_MIN   = "CLAY_FIDELITY_BBOX_FILL_MIN"
ENV_FIDELITY_ROW_BALANCE_MAX = "CLAY_FIDELITY_ROW_BALANCE_MAX"
ENV_FIDELITY_DEAD_REGION_MAX = "CLAY_FIDELITY_DEAD_REGION_MAX"
ENV_FIDELITY_DEVREGION_DENSITY_LO = "CLAY_FIDELITY_DEVREGION_DENSITY_LO"
ENV_FIDELITY_DEVREGION_DENSITY_HI = "CLAY_FIDELITY_DEVREGION_DENSITY_HI"
ENV_FIDELITY_NP_INTERLEAVE_LO = "CLAY_FIDELITY_NP_INTERLEAVE_LO"
ENV_FIDELITY_NP_INTERLEAVE_HI = "CLAY_FIDELITY_NP_INTERLEAVE_HI"
ENV_FIDELITY_WIRELENGTH_SPAN_MAX = "CLAY_FIDELITY_WIRELENGTH_SPAN_MAX"
ENV_FIDELITY_SIGNAL_FANOUT_MAX = "CLAY_FIDELITY_SIGNAL_FANOUT_MAX"
ENV_FIDELITY_OVER_CELL_MIN = "CLAY_FIDELITY_OVER_CELL_MIN"


@dataclass(frozen=True)
class FidelityThresholds:
    """Reference-calibrated band edges for the geometric-fidelity checkers.

    Every field is a calibrated SCALAR a future optimizer may override via
    ``CLAY_FIDELITY_*`` / ``[tool.clay.fidelity]``; nothing here is a coordinate.

    Fields
    ------
    enclosure_min : float
        Min acceptable (device-OD area inside rings)/(ring-enclosed area). The
        reference rings tightly (~0.85); the band floor sits below that with
        margin. The WRONG state's ~0.33 is far below.
    aspect_lo, aspect_hi : float
        The acceptable cell W/H band, bracketing the reference ~1.67.
    fill_min : float
        Min acceptable OD-area / cell-area (the reference packs ~0.33).
    ring_tol : int
        How many fewer ring frames than the design's compact-matched-group count
        are tolerated (a too-coarse ring scheme -- 3 huge rings vs ~9 -- is RED).
    cap_edge_margin_um : float
        A cap stamp counts as "hugging a long edge" when its nearest-edge gap is
        within this margin (um). The reference caps occupy a long edge.
    cont_per_finger_min : float
        Min acceptable CONT shapes per MOS finger (the per-S/D-face contact-cut
        density). The reference packs full contact columns (~67 CONT/finger on
        the OTA); a single mid-channel cut leaves ~22. The band floor sits below
        the reference with margin (a dense synthesis path clears it; the sparse
        default does not).
    bbox_fill_min : float
        Min acceptable (sum of per-device-footprint bbox area) / (cell-bbox area)
        -- the PLACEMENT-tier occupancy (distinct from fill_density's
        device-layer OD/cell). The reference packs the cell tightly in 2D
        (~0.66 bbox occupancy); the band-stack ROW paradigm leaves the narrow
        row's tail empty -> the band-stack measures ~0.50 (RED). The band floor
        sits below the reference-calibrated occupancy with margin so the 2D
        floorplanner's denser pack clears it while the row-stack does not.
    row_balance_max : float
        Max acceptable (widest device-row x-span) / (narrowest device-row
        x-span). A balanced 2D pack keeps rows near-equal width (~1.0-1.3); the
        band-stack's pmos-row (~159 um) vs nmos-row (~71 um) is ~2.2 (RED) --
        the cell width = widest row by construction wastes the narrow row's
        tail. The band ceiling sits above the reference's balance with margin.
    devregion_density_lo, devregion_density_hi : float
        The TWO-SIDED band for device-region density = (merged OD area) /
        (OD bounding-box area). The reference clusters devices densely (~0.50);
        the band-stack spreads them into sparse wide rows (~0.24, below the
        floor). The upper edge guards over-compaction (no internal slack left
        as a routing resource is also a defect).
    np_interleave_lo, np_interleave_hi : float
        The TWO-SIDED band for N/P interleaving = (NWELL bbox area) / (OD bbox
        area). When n- and p-devices interleave across the device region the
        n-well bbox spans nearly the whole OD bbox (reference ~0.998); a
        SEGREGATED layout (all n above all p) gives an n-well bbox far smaller
        than the OD bbox (band-stack ~0.33, below the floor). The upper edge
        guards a well that sprawls well beyond the device diffusion.
    """

    enclosure_min: float = 0.80
    aspect_lo: float = 1.40
    aspect_hi: float = 1.90
    fill_min: float = 0.30
    ring_tol: int = 1
    cap_edge_margin_um: float = 2.0
    cont_per_finger_min: float = 40.0
    bbox_fill_min: float = 0.60
    row_balance_max: float = 1.60
    # MOSAIC-COMPLETENESS band (U4, RETIRES the cap-center proxy): the largest
    # CONTIGUOUS dead rectangle (empty of any element footprint) as a fraction of
    # the cell. A dense all-element mosaic (devices wrapping the cap) leaves only
    # small gaps (the unified placement measures ~0.08 on the OTA); a cap dumped as
    # a side block leaves a big contiguous slab (the retired planar bolt-on ~0.42).
    # The ceiling sits between them so a true mosaic passes + a parked-block layout
    # fails. GENERIC (any layout, any circuit), config-tunable; no coordinate.
    dead_region_max: float = 0.30
    # COMPACTNESS band (R1): the device-REGION must pack densely + N/P
    # interleave, not spread into sparse segregated rows. TWO-SIDED bands
    # (over-compaction is also a defect): the device-region density is OD-area
    # / OD-bbox-area (the reference packs ~0.50; the band-stack spreads to
    # ~0.24 -- below the floor); the N/P interleave is NWELL-bbox-area /
    # OD-bbox-area (the reference's n-well spans the whole device region with
    # p-devices interleaved so nwell-bbox ~ od-bbox ~0.998; the band-stack
    # segregates n above p so nwell-bbox ~0.33 of od-bbox -- below the floor).
    # The reference's 0.4994 / 0.9983 only CALIBRATE these edges; never an input.
    devregion_density_lo: float = 0.40
    devregion_density_hi: float = 0.65
    np_interleave_lo: float = 0.70
    np_interleave_hi: float = 1.40
    # WIRELENGTH-LOCALITY band (the routing/floorplan QUALITY obligation): a
    # LOW-FANOUT signal net's half-perimeter (HPWL) must stay LOCAL -- it must not
    # span most of the cell. Measured as max over low-fanout (<= signal_fanout_max
    # terminals) nets of HPWL / (W + H). The defect this gates (measured on the
    # envelope-min cp OTA): the output net VOUT spans 0.77 of the cell perimeter
    # because the output transistors + the Miller cap scatter to three corners --
    # legal (LVS-matched) but sprawled. The envelope-min objective treats
    # wirelength as a tie-break, so locality is never enforced; this band makes it a
    # gated obligation. Power/ground nets (high fanout) intentionally distribute and
    # are EXCLUDED from the max-span gate (reported separately). RELATIVE fraction
    # (no coordinate, no per-circuit constant); a 2-pin signal net needing more than
    # half the cell perimeter is a placement defect. Config-tunable.
    wirelength_span_max: float = 0.55
    signal_fanout_max: int = 5
    # OVER-CELL-ROUTING band (the ROUTER-CONTRACT obligation the OTA density
    # hard-core experiment named): the fraction of ROUTING-TRUNK metal (M2/M3 --
    # NOT device-local M1 pads) whose area sits DIRECTLY OVER the device-active
    # footprint (OD + poly gates). The reference OTA runs its trunk over the cells
    # (measured 0.577), so clusters abut with no dedicated routing channel and the
    # core stays tight; our router keeps the trunk in the apron beside the cells
    # (~0.24, placement-INVARIANT -- FINE-dense 0.259, production-loose 0.243), so
    # metal sprawls (dense-in != dense-out). The floor sits below the reference's
    # 0.577 with margin so a router that routes over the cells clears it while the
    # apron-sprawl default does not. The reference's 0.577 only CALIBRATES this
    # edge; never an input. RELATIVE area/area (translation-invariant, no
    # coordinate). Config-tunable; the optimizer seam.
    over_cell_min: float = 0.50


_FIDELITY_DEFAULTS = FidelityThresholds()


def _env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


_GLOBAL_FIDELITY: Optional[FidelityThresholds] = None


def fidelity_thresholds(*, fresh: bool = False) -> FidelityThresholds:
    """Resolve the geometric-fidelity threshold bands.

    Resolution order (per field, highest precedence first):
      1. environment variable (``CLAY_FIDELITY_*``) -- the optimizer seam;
      2. ``[tool.clay.fidelity]`` in clay.toml / pyproject.toml;
      3. the reference-calibrated default scalar.

    ``fresh=True`` re-reads the environment (used by an optimizer loop / tests
    that flip a band). Returns an immutable :class:`FidelityThresholds`.
    """
    global _GLOBAL_FIDELITY
    if not fresh and _GLOBAL_FIDELITY is not None:
        return _GLOBAL_FIDELITY

    toml_data = _load_toml_config()
    fid = (toml_data.get("tool", {}).get("clay", {}).get("fidelity", {})
           if toml_data else {})

    def pick_float(env_name, toml_key, default):
        v = _env_float(env_name)
        if v is not None:
            return v
        if toml_key in fid:
            return float(fid[toml_key])
        return default

    def pick_int(env_name, toml_key, default):
        v = _env_int(env_name)
        if v is not None:
            return v
        if toml_key in fid:
            return int(fid[toml_key])
        return default

    th = FidelityThresholds(
        enclosure_min=pick_float(ENV_FIDELITY_ENCLOSURE_MIN, "enclosure_min",
                                 _FIDELITY_DEFAULTS.enclosure_min),
        aspect_lo=pick_float(ENV_FIDELITY_ASPECT_LO, "aspect_lo",
                             _FIDELITY_DEFAULTS.aspect_lo),
        aspect_hi=pick_float(ENV_FIDELITY_ASPECT_HI, "aspect_hi",
                             _FIDELITY_DEFAULTS.aspect_hi),
        fill_min=pick_float(ENV_FIDELITY_FILL_MIN, "fill_min",
                            _FIDELITY_DEFAULTS.fill_min),
        ring_tol=pick_int(ENV_FIDELITY_RING_TOL, "ring_tol",
                          _FIDELITY_DEFAULTS.ring_tol),
        cap_edge_margin_um=pick_float(ENV_FIDELITY_CAP_EDGE_MARGIN_UM,
                                      "cap_edge_margin_um",
                                      _FIDELITY_DEFAULTS.cap_edge_margin_um),
        cont_per_finger_min=pick_float(ENV_FIDELITY_CONT_PER_FINGER_MIN,
                                       "cont_per_finger_min",
                                       _FIDELITY_DEFAULTS.cont_per_finger_min),
        bbox_fill_min=pick_float(ENV_FIDELITY_BBOX_FILL_MIN, "bbox_fill_min",
                                 _FIDELITY_DEFAULTS.bbox_fill_min),
        row_balance_max=pick_float(ENV_FIDELITY_ROW_BALANCE_MAX,
                                   "row_balance_max",
                                   _FIDELITY_DEFAULTS.row_balance_max),
        dead_region_max=pick_float(ENV_FIDELITY_DEAD_REGION_MAX,
                                   "dead_region_max",
                                   _FIDELITY_DEFAULTS.dead_region_max),
        devregion_density_lo=pick_float(ENV_FIDELITY_DEVREGION_DENSITY_LO,
                                        "devregion_density_lo",
                                        _FIDELITY_DEFAULTS.devregion_density_lo),
        devregion_density_hi=pick_float(ENV_FIDELITY_DEVREGION_DENSITY_HI,
                                        "devregion_density_hi",
                                        _FIDELITY_DEFAULTS.devregion_density_hi),
        np_interleave_lo=pick_float(ENV_FIDELITY_NP_INTERLEAVE_LO,
                                    "np_interleave_lo",
                                    _FIDELITY_DEFAULTS.np_interleave_lo),
        np_interleave_hi=pick_float(ENV_FIDELITY_NP_INTERLEAVE_HI,
                                    "np_interleave_hi",
                                    _FIDELITY_DEFAULTS.np_interleave_hi),
        wirelength_span_max=pick_float(ENV_FIDELITY_WIRELENGTH_SPAN_MAX,
                                       "wirelength_span_max",
                                       _FIDELITY_DEFAULTS.wirelength_span_max),
        signal_fanout_max=pick_int(ENV_FIDELITY_SIGNAL_FANOUT_MAX,
                                   "signal_fanout_max",
                                   _FIDELITY_DEFAULTS.signal_fanout_max),
        over_cell_min=pick_float(ENV_FIDELITY_OVER_CELL_MIN, "over_cell_min",
                                 _FIDELITY_DEFAULTS.over_cell_min),
    )
    _GLOBAL_FIDELITY = th
    return th
