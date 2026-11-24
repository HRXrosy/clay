"""
PLL closed-loop AC testbench (v0.0.95 Round 72 Agent C).

Sibling of:
  * `clay.templates.pll_testbench`       — TRANSIENT PLL rig (lock-time
                                            jitter / waveform aggregates).
  * `clay.templates.charge_pump_testbench` — CP UP/DN current matching.
  * `clay.templates.lc_tank_vco_testbench` — VCO Kvco / f0 sweep.

This module ships the *small-signal AC* rig the v0.0.94 Round 71 Agent B
deferred-path skip kept asking for: the closed-loop PLL open-loop gain
sweep that produces a directly-measured `loop_bw_<name>` crossover
frequency. With it, `pll_loop_filter` graduates from
`skipped + deferred=True` to a fully-runnable real check (same line of
upgrade as v0.0.92 Round 68 Agent H's promotion of the third batch
placeholders).

Why a dedicated module (not another `clay/templates/*` file):

  The transient PLL rig is structurally a wrapper around `OpAnalysis` +
  `TranAnalysis` and consumes the cell's actual port surface (`ref_in`,
  `clk_out`, `cp_out`, `vbias_lf`, `vctrl`, `fb_clk`). The closed-loop
  AC rig is fundamentally a *behavioral-model-rich* netlist: we need
  voltage-controlled current sources (G-elements) for the PFD+CP and
  VCO Kvco-tuned currents, plus an explicit unit-cap integrator to
  realise the VCO's 1/s phase-domain transfer. Clay's neutral IR
  layer (`Source` / `Device`) only exposes V/I DC, V/I AC, V/I PULSE,
  V/I PWL, MOS, R, L, C, varactor, and VCVS — no VCCS (G-element)
  and no B-element. Rather than grow the IR for a one-off rig, this
  module assembles the netlist directly via string emit and calls
  the ngspice binary on it.

Loop-bandwidth physics (the math the rig encodes):

  For a Type-II 2nd-order PLL with passive lead-lag filter the open-
  loop gain in the phase domain is

      T(s) = Kpd · F(s) · Kvco / s · 1/N

  where
    Kpd     = Icp / (2π)        — phase-detector + CP gain  [A/rad]
    F(s)    = V_vctrl(s) / I_cp(s) — loop-filter impedance   [V/A]
    Kvco    = VCO gain                                       [(rad/s)/V]
    N       = divider ratio                                  [—]

  The (open-loop) unity-gain crossover frequency ω_c is the closed-loop
  bandwidth (to the classical Bode approximation that the high-gain
  loop tracks the open-loop ω_c). The rig synthesises a single node
  `v(loop_gain)` whose magnitude is exactly |T(jω)| and runs an AC
  sweep. `freq when vdb(loop_gain)=0 cross=last` extracts the
  crossover frequency.

Behavioral chain (concretely):

  V_PHASE phase_in 0 dc 0 ac 1                 ← AC source, 1V small-signal
  G_CP    0 cp_out phase_in 0 {Kpd}            ← VCCS: I_cp = Kpd · v(phase_in)
                                                  with v(phase_in)=1V AC,
                                                  I_cp = Kpd amperes AC.
  xDUT    cp_out vctrl vss <filter_cell>       ← actual DUT instantiation
  G_VCO   0 vco_int vctrl 0 {Kvco}             ← VCCS: I_int = Kvco · v(vctrl)
  C_INT   vco_int 0 1.0                        ← 1F unit cap → integrator
                                                  V_vco_int(s) = (Kvco / s) · V_vctrl
  R_INT   vco_int 0 1e12                       ← high R, AC-irrelevant; only
                                                  there to give ngspice a DC
                                                  path so the OP point exists
                                                  (the cap floats otherwise).
  E_DIV   loop_gain 0 vco_int 0 {1/N}          ← VCVS: V_loop = V_int / N

  Resulting loop_gain(s) = Kpd · F(s) · (Kvco / s) · 1/N = T(s).
  When the loop filter is the canonical passive lead-lag,

      F(s) = (1 + s·R1·C1) / (s · (C1 + C2) · (1 + s · R1 · C1 · C2 / (C1 + C2)))

  closed-form ω_c is solved iteratively; the AC sweep handles arbitrary
  F(s) shapes (active filters, higher-order, etc.) just as well.

  Important: the chain is *purely linear* (no MOSFETs, no diodes, no
  bias chains). ngspice converges trivially on the OP — every node
  is at 0 V because no source has a DC component. The AC sweep is
  the entire characterisation.

Default parameters (matched to the v0.0.32 `examples/pll_subcells/
loop_filter.clay` reference for the 100 kHz / 60° phase-margin sizing):

  Icp     = 10 µA          → Kpd = Icp / (2π) ≈ 1.591 µA/rad
  Kvco    = 2π · 100 MHz/V → ≈ 6.283e8 (rad/s)/V
  N       = 1024           → /N attenuation
  f_start = 10 Hz
  f_stop  = 100 MHz
  pts/dec = 20             (good resolution near the crossover)

Caller-overridable kwargs let alternate sizings (Icp=5/20/50 µA, Kvco
profiles for ring/LC VCOs, N=1/2/4/8/16/.../1024 for prescaler chains)
flow through without code edits.

Tolerance band (the invariant check that consumes the result):

  The v0.0.94 Round 71 Agent B `_check_pll_loop_filter` applies
  `DEFAULT_TOLERANCE = ±20%` between the declared `loop_bandwidth_target`
  and the measured crossover. Callers that want the v0.0.95 mission's
  ±10% band pass `tolerance=0.10` to `check_functional_invariants` (or
  set it explicitly on the upstream verifier call).

Honest scope of v0.0.95 R72 C:

  * The CP behavioural model is *linear* (Icp drive is a constant Kpd
    transconductance). Real CPs have UP/DN mismatch + dead-zone +
    current-output-impedance dependence on cp_out — none of which a
    pure AC sweep can capture. The rig assumes a well-balanced CP that
    sources its declared Icp on demand.
  * The VCO is *linear* in the same sense — a perfect Kvco around the
    operating point. Real VCOs have Kvco(vctrl) shape; the rig
    measures the SMALL-SIGNAL bandwidth at the implicit linearisation
    point.
  * The divider is *fractional* — N can be any positive real. Real
    integer-only dividers do not change the AC sweep's small-signal
    behaviour, so the simplification is harmless for this purpose.

These honest-scope notes are documented inline at the build site so a
downstream LLM that reads the testbench source understands what the
crossover frequency is and is not measuring.
"""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ─── Defaults ────────────────────────────────────────────────────

DEFAULT_ICP: float = 10e-6                        # 10 µA charge-pump current
DEFAULT_KVCO_HZ_PER_V: float = 100e6              # 100 MHz/V VCO gain
DEFAULT_DIVIDER_N: float = 1024.0                 # /1024 prescaler typical
DEFAULT_F_START: float = 10.0                     # 10 Hz AC sweep start
DEFAULT_F_STOP: float = 100e6                     # 100 MHz AC sweep stop
DEFAULT_PTS_PER_DECADE: int = 20                  # AC sweep resolution
DEFAULT_PDK_CORNER: str = "tt"                    # nominal process
DEFAULT_TEMP_C: float = 27.0                      # room temperature


# Conventional port-name set the loop-filter cell MUST expose so the
# behavioural chain can wire to it. The canonical loop filter has:
#   cp_out — charge-pump output / filter input
#   vctrl  — VCO control / filter output
#   vss    — ground reference
# We accept alternates via `role.attributes` (`charge_pump_node`,
# `vco_ctrl_node`) to match the v0.0.94 R71 keyword schema.
_REQUIRED_LOOP_FILTER_PORTS = ("cp_out", "vctrl", "vss")


# ─── Cell-shape resolution ─────────────────────────────────────


def _resolve_loop_filter_ports(
    cell, role,
) -> tuple[str, str, str]:
    """Return `(cp_node, vctrl_node, vss_node)` for the DUT instantiation.

    Priority:
      1. role attributes `charge_pump_node` / `vco_ctrl_node` (the
         declared schema in v0.0.94 R70 parse coverage).
      2. Default port names `cp_out` / `vctrl` on the cell.
      3. Fall back to literal "cp_out" / "vctrl" / "vss".

    The vss node is always either the cell's literal `vss` port or the
    string `"vss"` when the cell has no `vss` port (the testbench
    creates the rail externally).
    """
    cp_node = role.attributes.get("charge_pump_node") or "cp_out"
    vctrl_node = role.attributes.get("vco_ctrl_node") or "vctrl"
    if cell is not None and getattr(cell, "ports", None) is not None:
        port_set = set(cell.ports)
        # Require the resolved cp / vctrl names appear on the cell
        # surface — otherwise the DUT instantiation cannot wire them.
        # Defensive fallback only when neither resolves: raise an
        # informative error so the caller sees the misconfiguration.
        if cp_node not in port_set or vctrl_node not in port_set:
            # Try the literal defaults
            if "cp_out" in port_set and "vctrl" in port_set:
                cp_node = "cp_out"
                vctrl_node = "vctrl"
            else:
                raise ValueError(
                    f"pll_loop_filter cell {cell.name!r} must expose "
                    f"ports {{{cp_node!r}, {vctrl_node!r}}} for the "
                    f"closed-loop AC testbench (got {sorted(cell.ports)}). "
                    f"Either rename the cell's ports or set "
                    f"`charge_pump_node` / `vco_ctrl_node` in the role "
                    f"declaration."
                )
    vss_node = "vss"
    return cp_node, vctrl_node, vss_node


# ─── PdkAdapter helper ─────────────────────────────────────────

def _resolve_pdk_corner_file(cell, *, corner: str) -> Optional[Path]:
    """Resolve the PDK corner-include path so MOS-containing cells parse.

    For pure-passive loop filters (R / C only) this is unnecessary; for
    cells that include active models (an LDO-style active loop filter,
    for example), the include is required.

    Returns None when the cell has no devices that need a PDK include
    (zero MOS devices) — keeps the testbench leaner.
    """
    if cell is None:
        return None
    needs_pdk = False
    for dev in getattr(cell, "devices", []) or []:
        if dev.device_class in (
            "nmos", "pmos", "varactor", "varactor_accum",
            "varactor_accum_lvt", "varactor_accum_hvt",
        ):
            needs_pdk = True
            break
    if not needs_pdk:
        return None
    try:
        from .pdk import get_pdk
        adapter = get_pdk(getattr(cell, "pdk_id", "sky130") or "sky130")
        path = adapter.corner_file(corner)
        if path is None or not path.exists():
            return None
        return path
    except Exception:
        return None


# ─── Netlist builder ────────────────────────────────────────────


def _scale_to_microns(meters: float) -> float:
    return meters * 1e6


def _emit_dut_subckt(cell, *, vss_alias: str) -> list[str]:
    """Inline the cell's topology as a self-contained subcircuit.

    Mirrors `clay.drivers.ngspice._emit_device` for the device classes
    a typical loop filter needs (R / C / L / MOS). Inlined here to keep
    the closed-loop AC rig free of the full Clay verifier dependency
    graph (the rig is meant to be standalone).
    """
    out = [f".subckt {cell.name} " + " ".join(cell.ports)]
    # Resolve PDK bindings if active devices exist
    bindings: dict[str, str] = {}
    needs_pdk = any(
        getattr(d, "device_class", "") in (
            "nmos", "pmos", "varactor", "varactor_accum",
            "varactor_accum_lvt", "varactor_accum_hvt",
        )
        for d in getattr(cell, "devices", []) or []
    )
    if needs_pdk:
        try:
            from .pdk import get_pdk
            adapter = get_pdk(getattr(cell, "pdk_id", "sky130") or "sky130")
            bindings = dict(adapter.device_bindings)
        except Exception:
            bindings = {}

    for dev in cell.devices:
        cls = dev.device_class
        if cls in bindings:
            model = dev.pdk_variant or bindings[cls]
            d = dev.pins["d"]; g = dev.pins["g"]
            s = dev.pins["s"]; b = dev.pins["b"]
            W = _scale_to_microns(dev.sizing["W"])
            L = _scale_to_microns(dev.sizing["L"])
            M = int(dev.sizing.get("M", 1))
            out.append(
                f"  x{dev.instance_name} {d} {g} {s} {b} {model} "
                f"W={W:g} L={L:g} m={M}"
            )
        elif cls == "cap":
            out.append(
                f"  C{dev.instance_name} {dev.pins['p']} {dev.pins['n']} "
                f"{dev.sizing['value']:g}"
            )
        elif cls == "res":
            out.append(
                f"  R{dev.instance_name} {dev.pins['p']} {dev.pins['n']} "
                f"{dev.sizing['value']:g}"
            )
        elif cls == "ind":
            out.append(
                f"  L{dev.instance_name} {dev.pins['p']} {dev.pins['n']} "
                f"{dev.sizing['value']:g}"
            )
        elif cls == "cur_src":
            out.append(
                f"  I{dev.instance_name} {dev.pins['p']} {dev.pins['n']} "
                f"{dev.sizing['value']:g}"
            )
        else:
            # Unknown device class — surface the error rather than
            # silently emit a malformed line.
            raise ValueError(
                f"pll_loop_ac: cell {cell.name!r} contains device class "
                f"{cls!r} that the closed-loop AC rig does not know how "
                f"to inline. Supported: cap, res, ind, cur_src, nmos, "
                f"pmos, varactor."
            )
    out.append(".ends")
    return out


@dataclass
class PllLoopAcTestbench:
    """Container for a built closed-loop AC testbench."""
    cell_name: str
    netlist_path: Path
    work_dir: Path
    measure_name: str
    icp: float
    kvco_hz_per_v: float
    divider_n: float
    f_start: float
    f_stop: float
    pdk_corner: str
    temp: float


def build_pll_open_loop_ac_testbench(
    cell,
    role,
    *,
    work_dir: Path,
    icp: float = DEFAULT_ICP,
    kvco_hz_per_v: float = DEFAULT_KVCO_HZ_PER_V,
    divider_n: float = DEFAULT_DIVIDER_N,
    f_start: float = DEFAULT_F_START,
    f_stop: float = DEFAULT_F_STOP,
    points_per_decade: int = DEFAULT_PTS_PER_DECADE,
    pdk_corner: str = DEFAULT_PDK_CORNER,
    temp: float = DEFAULT_TEMP_C,
) -> PllLoopAcTestbench:
    """Build the closed-loop AC testbench netlist on disk.

    Args:
      cell:    the loop-filter Cell IR. MUST expose `cp_out` + `vctrl`
               ports (or the names overridden by the role's
               `charge_pump_node` / `vco_ctrl_node` attributes).
      role:    the `pll_loop_filter` `ParsedFunctionalRole`. Read for
               port-name overrides and for the role's `name` attribute
               (used as the measure name suffix).
      work_dir: directory the netlist gets written into.

    Optional kwargs override the PLL surrounding-block models — see
    module-level defaults.

    Returns:
      a `PllLoopAcTestbench` carrying the netlist path + metadata. The
      caller passes this to `_run_and_extract_loop_bw` (or the
      higher-level `simulate_pll_loop_bandwidth`) to actually run the
      simulation.
    """
    if cell is None:
        raise ValueError(
            "pll_loop_ac.build_pll_open_loop_ac_testbench requires a "
            "non-None cell; got None."
        )
    if role is None or getattr(role, "kind", None) != "pll_loop_filter":
        raise ValueError(
            f"pll_loop_ac.build_pll_open_loop_ac_testbench expects a "
            f"pll_loop_filter role; got "
            f"{getattr(role, 'kind', '<no kind>')!r}."
        )
    if icp <= 0.0:
        raise ValueError(f"icp must be positive, got {icp!r}")
    if kvco_hz_per_v <= 0.0:
        raise ValueError(f"kvco_hz_per_v must be positive, got "
                         f"{kvco_hz_per_v!r}")
    if divider_n <= 0.0:
        raise ValueError(f"divider_n must be positive, got {divider_n!r}")
    if f_start <= 0.0 or f_stop <= f_start:
        raise ValueError(
            f"AC sweep range invalid: f_start={f_start!r}, "
            f"f_stop={f_stop!r}"
        )
    if points_per_decade <= 0:
        raise ValueError(
            f"points_per_decade must be positive, got "
            f"{points_per_decade!r}"
        )

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    # Resolve to absolute now so the run-step's subprocess invocation
    # (which fixes cwd to work_dir) doesn't lose the path when the
    # caller passed a relative directory.
    work_dir = work_dir.resolve()

    cp_node, vctrl_node, vss_node = _resolve_loop_filter_ports(cell, role)

    # Derived constants
    kpd = icp / (2.0 * math.pi)
    kvco_rad_per_v = 2.0 * math.pi * kvco_hz_per_v
    # The divider scales V_int → V_loop. To avoid a 1/N near-zero
    # spread in numerical magnitudes for typical N (1024 → 1/1024 ≈ 1e-3),
    # we keep the integration cap at 1F so the synthesised loop_gain
    # node tracks the actual physics directly.

    measure_name = f"loop_bw_{role.name}"

    netlist_path = (work_dir / f"pll_loop_ac_{cell.name}.cir").resolve()

    pdk_corner_file = _resolve_pdk_corner_file(cell, corner=pdk_corner)

    lines: list[str] = []
    lines.append(f"* Clay closed-loop PLL AC testbench for {cell.name!r}")
    lines.append(
        f"* PLL params: Icp={icp:g} A, Kvco={kvco_hz_per_v:g} Hz/V "
        f"({kvco_rad_per_v:g} rad/s/V), N={divider_n:g}"
    )
    lines.append(
        f"* Derived: Kpd = Icp/(2π) = {kpd:g} A/rad"
    )
    lines.append(
        f"* AC sweep: dec {points_per_decade} {f_start:g} {f_stop:g}"
    )
    lines.append("")
    lines.append(".param mc_mm_switch=0")
    lines.append(".param mc_pr_switch=0")
    lines.append(f".option temp = {temp:g}")
    if pdk_corner_file is not None:
        lines.append(f'.include "{pdk_corner_file.as_posix()}"')
    lines.append("")

    # DUT subcircuit definition
    lines.extend(_emit_dut_subckt(cell, vss_alias=vss_node))
    lines.append("")

    # Testbench: vss rail + behavioural PLL chain
    lines.append("* -- Closed-loop PLL chain (behavioural) --")
    # vss is the global ground (node 0). Some cells declare a separate
    # vss port for symmetric DC referencing — alias it to 0 unless the
    # cell has internal devices that require a distinct rail (e.g. the
    # bandgap testbench does this). The loop filter has only passives,
    # so a 0V supply is sufficient.
    lines.append("V_VSS_RAIL vss 0 dc 0")
    lines.append("")
    # AC source: 1V small-signal on phase_in. The DC operating point
    # is 0V everywhere — all linear elements, no convergence concerns.
    lines.append("V_PHASE phase_in 0 dc 0 ac 1")
    # Voltage-controlled current source emulating PFD+CP:
    #   I_cp = Kpd · v(phase_in)
    # With v(phase_in)=1V AC, this drives Kpd amps of AC current into
    # cp_node. The DUT's V_vctrl response is therefore F(s)·Kpd, and
    # the chain output picks up the remaining Kvco/s·(1/N).
    lines.append(
        f"G_CP 0 {cp_node} phase_in 0 {kpd:g}"
    )
    # DUT instantiation. Pin order matches the cell's declared ports.
    dut_pins = " ".join(cell.ports)
    lines.append(f"xDUT {dut_pins} {cell.name}")
    # Defensive DC path to ground at the loop-filter input. Even when
    # the loop filter has a series C (no DC path through the DUT to
    # ground), the G_CP element is a controlled source with no I-V
    # curve at DC — ngspice reports a singular matrix on cp_out. A
    # very high parallel R (1 GΩ) gives ngspice a DC path while the
    # AC sweep is unaffected for f > 1/(2π·R·C) ≈ 16 Hz (with
    # C1 = 100 pF). The default f_start = 10 Hz can dip into this
    # band; callers running sub-10 Hz sweeps should raise R_GUARD via
    # the `r_guard` kwarg (not yet exposed — open as a follow-up if
    # needed).
    lines.append(f"R_GUARD_CP {cp_node} 0 1e9")
    # Mirror DC path on vctrl. The loop-filter's series-R-shunt-C
    # topology already provides a DC path through C2 → vss → 0, but
    # not all loop filters do (active-LF or higher-order variants
    # could leave vctrl DC-isolated). Adding the guard R is harmless
    # for the AC sweep.
    lines.append(f"R_GUARD_VCTRL {vctrl_node} 0 1e9")
    # VCO behavioural integrator. Drives current Kvco · v(vctrl) into
    # a 1 F cap to ground → V_vco_int(s) = Kvco · V_vctrl / s.
    lines.append(
        f"G_VCO 0 vco_int {vctrl_node} 0 {kvco_rad_per_v:g}"
    )
    lines.append("C_INT vco_int 0 1.0")
    # Very large parallel R prevents OP-point convergence issues — the
    # DC operating point of the integrator cap is otherwise undefined
    # (floating node). At AC frequencies above 1/(2π·R·C) ≈ 1.6e-13 Hz
    # the impedance contribution is negligible relative to the 1F cap.
    lines.append("R_INT vco_int 0 1e12")
    # Divider stage: VCVS with gain 1/N. v(loop_gain) = v(vco_int) / N.
    lines.append(
        f"E_DIV loop_gain 0 vco_int 0 {1.0 / divider_n:g}"
    )
    lines.append("")

    # Control deck. ngspice's `.control` block uses `meas` (no leading
    # dot) — `.meas` is the netlist-level statement that lives outside
    # `.control`. Both forms compute the same thing but the in-control
    # form is what we use here so the analyses + measurements live in
    # one ordered block (Clay convention across all testbench builders).
    lines.append(".control")
    lines.append(f"ac dec {int(points_per_decade)} {f_start:g} {f_stop:g}")
    # Find the open-loop unity-gain crossover. `cross=last` picks the
    # highest-frequency crossing — for well-behaved 2nd-order Type-II
    # loops there's exactly one crossing in the swept band, so the
    # `last` qualifier is harmless but defensive against numerical noise
    # that produces multiple very-close crossings near the true ω_c.
    lines.append(
        f"meas ac {measure_name} when vdb(loop_gain)=0 cross=last"
    )
    # Also expose the magnitude at a low reference frequency so the
    # caller can sanity-check the chain is wired correctly (should be
    # large — typically > 100 dB at the start of the sweep for a high-
    # gain PLL).
    lines.append(
        f"meas ac loop_gain_lf find vdb(loop_gain) at={f_start:g}"
    )
    # Echo so the .log captures all measure values.
    lines.append(f"echo {measure_name}=$&{measure_name}")
    lines.append("echo loop_gain_lf=$&loop_gain_lf")
    lines.append("quit")
    lines.append(".endc")
    lines.append("")
    lines.append(".end")

    netlist_path.write_text("\n".join(lines), encoding="utf-8")

    return PllLoopAcTestbench(
        cell_name=cell.name,
        netlist_path=netlist_path,
        work_dir=work_dir,
        measure_name=measure_name,
        icp=icp,
        kvco_hz_per_v=kvco_hz_per_v,
        divider_n=divider_n,
        f_start=f_start,
        f_stop=f_stop,
        pdk_corner=pdk_corner,
        temp=temp,
    )


# ─── Run + parse ─────────────────────────────────────────────


_MEAS_LINE = re.compile(
    r"^\s*(\w+)\s*=\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)",
    re.MULTILINE,
)


def _run_ngspice_silent(
    tb: PllLoopAcTestbench,
    *,
    ngspice_bin: str | Path,
    timeout: float = 120.0,
) -> Path:
    """Run ngspice in batch/silent mode on the testbench netlist.

    Windows-specific: the caller must point `ngspice_bin` at
    `ngspice_con.exe` rather than `ngspice.exe`. The plain `ngspice.exe`
    opens an interactive GUI window which blocks on a fatal error
    dialog box — see global rule "ngspice_con for silent Windows runs".
    """
    netlist_abs = tb.netlist_path.resolve()
    log_path = tb.work_dir / f"{netlist_abs.stem}.log"
    subprocess.run(
        [str(ngspice_bin), "-b", "-o", str(log_path), str(netlist_abs)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(tb.work_dir.resolve()),
        timeout=timeout,
    )
    return log_path


def _parse_measures(
    log_path: Path,
    *,
    wanted: list[str],
) -> dict[str, float]:
    """Extract measure_name → value from the ngspice log file.

    Mirrors the regex in `clay.drivers.ngspice.NgspiceDriver.parse_measures`
    so the parse logic is consistent across drivers. ngspice's `.meas
    when ... cross=last` emits the result both in the .log via the
    `meas: <name> = <value>` standard format AND via our explicit
    `echo <name>=$&<name>` line. The regex matches either flavour.
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, float] = {}
    for m in _MEAS_LINE.finditer(text):
        name, val_s = m.group(1), m.group(2)
        if name in wanted and name not in out:
            try:
                v = float(val_s)
            except ValueError:
                continue
            if math.isfinite(v):
                out[name] = v
    return out


@dataclass
class PllLoopAcResult:
    """The actual closed-loop AC measurement outcome."""
    loop_bw_hz: Optional[float]   # the crossover frequency, None on failure
    loop_gain_lf_db: Optional[float]   # low-freq gain (sanity)
    netlist_path: Path
    log_path: Path
    measure_name: str
    # If `loop_bw_hz` is None, this carries a short explanation of why
    # (regex parse miss, ngspice convergence error, etc.).
    error: Optional[str] = None


def simulate_pll_loop_bandwidth(
    cell,
    role,
    *,
    ngspice_bin: str | Path,
    work_dir: Optional[Path] = None,
    icp: float = DEFAULT_ICP,
    kvco_hz_per_v: float = DEFAULT_KVCO_HZ_PER_V,
    divider_n: float = DEFAULT_DIVIDER_N,
    f_start: float = DEFAULT_F_START,
    f_stop: float = DEFAULT_F_STOP,
    points_per_decade: int = DEFAULT_PTS_PER_DECADE,
    pdk_corner: str = DEFAULT_PDK_CORNER,
    temp: float = DEFAULT_TEMP_C,
    timeout: float = 120.0,
) -> PllLoopAcResult:
    """End-to-end: build → emit → run → parse the closed-loop AC sweep.

    This is the high-level entry point the invariant runtime hits when
    a `pll_loop_filter` role is checked and the upstream simulator hook
    is attached (see `_check_pll_loop_filter` in `clay.invariants`).

    Args:
      cell:        the loop-filter Cell IR. Required.
      role:        the `pll_loop_filter` `ParsedFunctionalRole`. Required.
      ngspice_bin: absolute path to the ngspice binary. On Windows this
                   MUST be `ngspice_con.exe` (see module-level note).
      work_dir:    directory for the netlist + log. If None, a temp
                   directory under `cell.__dict__["_pll_loop_ac_workdir"]`
                   is used; otherwise the caller's choice.

    Returns a `PllLoopAcResult` with the crossover frequency in Hz. If
    the simulation fails (parse miss, ngspice timeout, file IO error),
    `loop_bw_hz` is None and `error` carries the explanation. The
    caller (typically the invariant runtime) checks for None and falls
    back to the deferred-skip path.
    """
    if work_dir is None:
        # Default: a sibling directory under the current working dir.
        # The invariant runtime's caller should generally pass an
        # explicit work_dir for reproducibility.
        work_dir = Path("tmp_clay_rundir") / f"pll_loop_ac_{cell.name}"
    work_dir = Path(work_dir)
    try:
        tb = build_pll_open_loop_ac_testbench(
            cell, role, work_dir=work_dir,
            icp=icp, kvco_hz_per_v=kvco_hz_per_v, divider_n=divider_n,
            f_start=f_start, f_stop=f_stop,
            points_per_decade=points_per_decade,
            pdk_corner=pdk_corner, temp=temp,
        )
    except Exception as exc:  # noqa: BLE001 — surface as a structured fail
        return PllLoopAcResult(
            loop_bw_hz=None,
            loop_gain_lf_db=None,
            netlist_path=Path(),
            log_path=Path(),
            measure_name=f"loop_bw_{getattr(role, 'name', '?')}",
            error=f"build_failed: {type(exc).__name__}: {exc}",
        )
    try:
        log_path = _run_ngspice_silent(
            tb, ngspice_bin=ngspice_bin, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return PllLoopAcResult(
            loop_bw_hz=None,
            loop_gain_lf_db=None,
            netlist_path=tb.netlist_path,
            log_path=Path(),
            measure_name=tb.measure_name,
            error=f"ngspice_timeout: {exc.timeout:g}s",
        )
    except FileNotFoundError as exc:
        return PllLoopAcResult(
            loop_bw_hz=None,
            loop_gain_lf_db=None,
            netlist_path=tb.netlist_path,
            log_path=Path(),
            measure_name=tb.measure_name,
            error=f"ngspice_not_found: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 — surface as a structured fail
        return PllLoopAcResult(
            loop_bw_hz=None,
            loop_gain_lf_db=None,
            netlist_path=tb.netlist_path,
            log_path=Path(),
            measure_name=tb.measure_name,
            error=f"ngspice_run_failed: {type(exc).__name__}: {exc}",
        )

    measures = _parse_measures(
        log_path,
        wanted=[tb.measure_name, "loop_gain_lf"],
    )
    loop_bw = measures.get(tb.measure_name)
    loop_gain_lf = measures.get("loop_gain_lf")

    if loop_bw is None or not math.isfinite(loop_bw) or loop_bw <= 0.0:
        return PllLoopAcResult(
            loop_bw_hz=None,
            loop_gain_lf_db=loop_gain_lf,
            netlist_path=tb.netlist_path,
            log_path=log_path,
            measure_name=tb.measure_name,
            error=(
                f"measure_miss: {tb.measure_name!r} not parseable from "
                f"log {log_path.name!r} (ngspice may have failed to find "
                f"the crossover — check f_start / f_stop covers the "
                f"loop bandwidth)"
            ),
        )

    return PllLoopAcResult(
        loop_bw_hz=loop_bw,
        loop_gain_lf_db=loop_gain_lf,
        netlist_path=tb.netlist_path,
        log_path=log_path,
        measure_name=tb.measure_name,
    )


# ─── Closed-form prediction helper (for sanity / tests) ───────────


def predict_loop_bandwidth_passive_lead_lag(
    *,
    r1: float, c1: float, c2: float,
    icp: float = DEFAULT_ICP,
    kvco_hz_per_v: float = DEFAULT_KVCO_HZ_PER_V,
    divider_n: float = DEFAULT_DIVIDER_N,
) -> float:
    """Closed-form approximation of the loop bandwidth for the canonical
    passive lead-lag filter (3-element R1 + C1 + C2 topology).

    The loop filter impedance has two characteristic frequencies:

      ω_z  = 1 / (R1 · C1)                          — zero (proportional kick-in)
      ω_p2 = (C1 + C2) / (R1 · C1 · C2)              — high-frequency pole

    Below ω_z (typically the case for moderately-sized loops in the
    sub-100 kHz BW range with C1 in the 100 pF range), F(s) is
    integrator-dominated:

        F(jω) ≈ 1 / (jω · C1)

    so the open-loop crossover satisfies

        Kpd · (1 / (ω_c · C1)) · Kvco / (ω_c · N) = 1
        ω_c² ≈ Kpd · Kvco / (C1 · N)

    yielding f_c = ω_c / (2π) ≈ sqrt(Kpd · Kvco_rad / (C1 · N)) / (2π).

    Above ω_z (large R1 · C1 product OR very high-bandwidth designs),
    F(s) ≈ R1 and the simpler

        ω_c ≈ Kpd · R1 · Kvco / N

    applies. The two regimes meet at ω_c ≈ ω_z. This predictor returns
    the MINIMUM of the two estimates — whichever crossover happens at
    the lower frequency is the actual one (the slope of |F| changes
    from -20 dB/dec to 0 dB/dec at ω_z, so crossing happens once and
    only once, at whichever regime gives the smaller ω_c).

    This helper is used by the v0.0.95 R72 C test suite to sanity-check
    the AC sweep — the simulated crossover should land within ~25% of
    the closed-form prediction for textbook sizings (tolerance band
    chosen to absorb the C2-induced pole, divider non-idealities, and
    the small offset between |F|=1/(ωC1) and the exact lead-lag at
    finite ω).
    """
    if c1 <= 0.0 or c2 < 0.0 or r1 <= 0.0:
        raise ValueError(
            f"loop-filter values must be positive, got "
            f"r1={r1!r}, c1={c1!r}, c2={c2!r}"
        )
    if divider_n <= 0.0 or icp <= 0.0 or kvco_hz_per_v <= 0.0:
        raise ValueError(
            f"PLL params must be positive, got "
            f"icp={icp!r}, kvco_hz_per_v={kvco_hz_per_v!r}, "
            f"divider_n={divider_n!r}"
        )
    kpd = icp / (2.0 * math.pi)
    kvco_rad = 2.0 * math.pi * kvco_hz_per_v
    # Integrator-dominated estimate (ω < ω_z regime)
    omega_c_int_sq = kpd * kvco_rad / (c1 * divider_n)
    omega_c_int = math.sqrt(max(omega_c_int_sq, 0.0))
    # Proportional estimate (ω > ω_z regime)
    omega_c_prop = kpd * r1 * kvco_rad / divider_n
    # Loop zero
    omega_z = 1.0 / (r1 * c1)
    # Pick the regime that crosses first (whichever is smaller and
    # lands on the correct side of ω_z).
    if omega_c_int <= omega_z:
        omega_c = omega_c_int
    else:
        omega_c = omega_c_prop
    return omega_c / (2.0 * math.pi)
