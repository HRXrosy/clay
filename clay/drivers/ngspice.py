"""
Ngspice simulator driver.

Owns ALL ngspice-specific dialect knowledge:
  - .option scale=1u and W/L unit conventions
  - sky130 subckt instantiation via xMx prefix
  - .control / .meas syntax
  - mc_mm_switch / mc_pr_switch boilerplate
  - log parsing regexes

Outside this file, the rest of Clay sees only the simulator-neutral IR.

v0.0.29 (G4 multi-setup verify): no driver-side changes were needed
for ACM / PSRR / multi-DUT drive modes. The verifier rewrites the
Testbench's `sources` and `extra_devices` lists upstream (see
`clay.verifier.rewrite_testbench_for_drive_mode`) — by the time a
testbench reaches this driver it already encodes the right drive
shape (common-mode VCVS gains, voltage_ac on vdd/vss, etc.) through
the same Source / Device IR the driver has always consumed. New
drive modes added later (e.g. transient drives) extend the verifier
rewrite — the driver only needs to add a new Source / Device kind
if the IR vocabulary itself grows.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from ..config import ClayConfig, get_config
from ..ir import (
    AcAnalysis, Analysis, DcAnalysis, Device, Measure, OpAnalysis,
    Source, Testbench, TranAnalysis,
)
from ..measure_expr import emit_ngspice_meas, parse as parse_measure_expr
from ..pdk import PdkAdapter, get_pdk
from .base import SimulatorDriver

# PDK device-class bindings are NOT hardcoded here. The driver fetches
# them at run time from clay.pdk.get_pdk(<pdk_id>). Adding a new PDK
# means writing a `clay/pdk/<new_pdk>.py` adapter; no changes to this
# file needed (closed-open principle).


# ─── Emit helpers ─────────────────────────────────────────────

def _scale_to_microns(meters: float) -> float:
    """sky130 ngspice uses .option scale=1u; W/L in netlist are in um."""
    return meters * 1e6


def _emit_device(dev: Device, bindings: dict[str, str]) -> str:
    """One netlist line for a sized Clay device.

    Args:
        dev: the Clay device IR.
        bindings: PDK-specific device_class → simulator subckt name map.
    """
    cls = dev.device_class
    # v0.0.91 (LC-VCO accumulation-mode varactor): handle BEFORE the
    # generic-bindings branch, because varactor_accum has 3 pins (c0=gate,
    # c1=body, b=guard) and the generic d/g/s/b MOSFET emitter would
    # mis-index.
    if cls in ("varactor_accum", "varactor_accum_lvt", "varactor_accum_hvt"):
        model = dev.pdk_variant or bindings.get(cls)
        if model is None:
            raise NotImplementedError(
                f"ngspice driver: {cls!r} primitive requires a "
                f"{cls!r} binding in the PDK device_bindings, but none "
                f"is registered. The accumulation-mode varactor is a "
                f"sky130-specific primitive bound to "
                f"`sky130_fd_pr__cap_var_lvt` / `_hvt`."
            )
        # Pin map: p = gate (control), n = body / n-well terminal. We
        # tie both c1 and the guard-ring `b` pin to the n-well node so
        # the varactor presents a simple 2-terminal interface (gate vs.
        # n-well) to the rest of the cell. That matches the textbook
        # accumulation-mode N-well varactor symbol and lets the LC-VCO
        # template wire it identically to the PMOS-MOSCAP variant.
        gate = dev.pins["p"]; body = dev.pins["n"]
        W = _scale_to_microns(dev.sizing["W"])
        L = _scale_to_microns(dev.sizing["L"])
        # The sky130 cap_var subckt takes `vm` as a parallel multiplier
        # (it scales cmin / cmax linearly), unlike the MOSFET `m=` flag.
        # We map our integer M sizing to vm because that's the supported
        # multi-finger knob on this subckt. M default = 1.
        vm = int(dev.sizing.get("M", 1))
        return (
            f"x{dev.instance_name} {gate} {body} {body} {model} "
            f"w={W:g} l={L:g} vm={vm}"
        )
    if cls in bindings:
        model = dev.pdk_variant or bindings[cls]
        # Pin order for sky130 fets: d g s b
        d = dev.pins["d"]; g = dev.pins["g"]; s = dev.pins["s"]; b = dev.pins["b"]
        W = _scale_to_microns(dev.sizing["W"])
        L = _scale_to_microns(dev.sizing["L"])
        M = int(dev.sizing.get("M", 1))
        return f"x{dev.instance_name} {d} {g} {s} {b} {model} W={W:g} L={L:g} m={M}"
    if cls == "cap":
        p = dev.pins["p"]; n = dev.pins["n"]
        C = dev.sizing["value"]
        return f"C{dev.instance_name} {p} {n} {C:g}"
    if cls == "res":
        p = dev.pins["p"]; n = dev.pins["n"]
        R = dev.sizing["value"]
        return f"R{dev.instance_name} {p} {n} {R:g}"
    if cls == "ind":
        p = dev.pins["p"]; n = dev.pins["n"]
        L = dev.sizing["value"]
        return f"L{dev.instance_name} {p} {n} {L:g}"
    if cls == "varactor":
        # MOSCAP varactor: a PMOS device with drain and source tied to the
        # same net (the "n" / body pin), gate = "p" / control pin. Voltage
        # across (p - n) modulates Cgs+Cgd ~ Cox*W*L when |Vgs| > Vth_p
        # (PMOS on, high cap) versus near 0 when |Vgs| < Vth_p (off, low
        # cap). Tuning range typically 2-3x across the available Vgs swing.
        # Works on any PDK with a PMOS binding (sky130/gf180mcu/open_pdk/open_pdk).
        p = dev.pins["p"]; n = dev.pins["n"]
        W = _scale_to_microns(dev.sizing["W"])
        L = _scale_to_microns(dev.sizing["L"])
        M = int(dev.sizing.get("M", 1))
        # Look up the PMOS subckt name in the PDK bindings so the same
        # primitive works across PDKs without driver-side hardcoding.
        pmos_model = bindings.get("pmos")
        if pmos_model is None:
            raise NotImplementedError(
                f"ngspice driver: varactor primitive requires a 'pmos' "
                f"binding in the PDK device_bindings, but none registered."
            )
        # Pin order: d=n g=p s=n b=n (drain and source tied to body net `n`,
        # gate is the control net `p`). This is the standard MOSCAP wiring.
        return (
            f"x{dev.instance_name} {n} {p} {n} {n} {pmos_model} "
            f"W={W:g} L={L:g} m={M}"
        )
    if cls == "cur_src":
        p = dev.pins["p"]; n = dev.pins["n"]
        I = dev.sizing["value"]
        return f"I{dev.instance_name} {p} {n} {I:g}"
    if cls == "vcvs":
        # E-element: voltage-controlled voltage source
        op = dev.pins["op"]; on = dev.pins["on"]; ip = dev.pins["ip"]; in_ = dev.pins["in"]
        gain = dev.sizing.get("gain", 1.0)
        return f"E{dev.instance_name} {op} {on} {ip} {in_} {gain:g}"
    raise NotImplementedError(f"ngspice driver: unsupported device class {cls!r}")


def _emit_source(src: Source) -> str:
    p = src.pins["p"]; n = src.pins["n"]
    if src.kind == "voltage_dc":
        return f"V{src.instance_name} {p} {n} {src.params['dc']:g}"
    if src.kind == "voltage_ac":
        dc = src.params.get("dc", 0.0); ac = src.params.get("ac", 1.0)
        return f"V{src.instance_name} {p} {n} dc {dc:g} ac {ac:g}"
    if src.kind == "current_dc":
        return f"I{src.instance_name} {p} {n} {src.params['dc']:g}"
    # v0.0.34 (PULSE / PWL source kinds for transient-step drive modes).
    # PULSE syntax: V/I<name> p n PULSE(v_low v_high td tr tf pw per).
    # Defaults match ngspice's documented defaults so a partial params
    # dict still produces a valid simulation.
    if src.kind in ("voltage_pulse", "current_pulse"):
        prefix = "V" if src.kind.startswith("voltage") else "I"
        return _emit_pulse_source(src, prefix=prefix, p=p, n=n)
    if src.kind in ("voltage_pwl", "current_pwl"):
        prefix = "V" if src.kind.startswith("voltage") else "I"
        return _emit_pwl_source(src, prefix=prefix, p=p, n=n)
    raise NotImplementedError(f"ngspice driver: unsupported source kind {src.kind!r}")


def _emit_pulse_source(src: Source, *, prefix: str, p: str, n: str) -> str:
    """Emit an ngspice PULSE(...) source line for src.

    Required params: `v_low`, `v_high`. Other params receive sensible
    defaults:
      * td (delay):       0
      * t_rise:           10n
      * t_fall:           = t_rise
      * pulse_width:      1ms (so the pulse stays high for the full
                          transient window in typical LDO load-step
                          analyses; user overrides if they need a real
                          pulse train)
      * period:           = 2 * pulse_width (so we don't repeat — one
                          edge is what tran_load_step / tran_line_step
                          want)
    """
    if "v_low" not in src.params or "v_high" not in src.params:
        raise ValueError(
            f"source {src.instance_name!r} (kind={src.kind!r}): "
            f"PULSE requires `v_low` and `v_high` in params. "
            f"Got keys: {sorted(src.params)}."
        )
    v_low = float(src.params["v_low"])
    v_high = float(src.params["v_high"])
    td = float(src.params.get("td", 0.0))
    t_rise = float(src.params.get("t_rise", 10e-9))
    t_fall = float(src.params.get("t_fall", t_rise))
    pulse_width = float(src.params.get("pulse_width", 1e-3))
    period = float(src.params.get("period", 2.0 * pulse_width))
    return (
        f"{prefix}{src.instance_name} {p} {n} "
        f"PULSE({v_low:g} {v_high:g} {td:g} {t_rise:g} {t_fall:g} "
        f"{pulse_width:g} {period:g})"
    )


def _emit_pwl_source(src: Source, *, prefix: str, p: str, n: str) -> str:
    """Emit an ngspice PWL(...) source line for src.

    Requires `pwl_points` in params: a flat sequence of alternating
    time/value floats. Length must be even (each time has a value);
    times must be monotonically non-decreasing. The driver does not
    interpolate or enforce strict monotonicity beyond a basic check.
    """
    pts = src.params.get("pwl_points")
    if pts is None:
        raise ValueError(
            f"source {src.instance_name!r} (kind={src.kind!r}): "
            f"PWL requires `pwl_points` in params (flat list of "
            f"time/value pairs)."
        )
    pts = list(pts)
    if len(pts) == 0 or len(pts) % 2 != 0:
        raise ValueError(
            f"source {src.instance_name!r} (kind={src.kind!r}): "
            f"PWL `pwl_points` must be a non-empty list of even length "
            f"(time, value, time, value, ...). Got length {len(pts)}."
        )
    # Basic monotonic-time check on the time axis. ngspice tolerates
    # equal-time points (instantaneous jump) but rejects backwards.
    times = [float(pts[i]) for i in range(0, len(pts), 2)]
    for a, b in zip(times, times[1:]):
        if b < a:
            raise ValueError(
                f"source {src.instance_name!r} (kind={src.kind!r}): "
                f"PWL times must be monotonically non-decreasing; "
                f"got {a:g} → {b:g}."
            )
    inner = " ".join(f"{float(v):g}" for v in pts)
    return f"{prefix}{src.instance_name} {p} {n} PWL({inner})"


def _emit_analysis(an: Analysis) -> str:
    if isinstance(an, AcAnalysis):
        return f"ac {an.sweep_kind} {an.points_per_decade} {an.f_start:g} {an.f_stop:g}"
    if isinstance(an, DcAnalysis):
        return f"dc {an.sweep_source} {an.start:g} {an.stop:g} {an.step:g}"
    if isinstance(an, TranAnalysis):
        return f"tran {an.step:g} {an.t_end:g}"
    if isinstance(an, OpAnalysis):
        return "op"
    raise NotImplementedError(f"ngspice driver: unsupported analysis {type(an).__name__}")


def _emit_measure(m: Measure) -> str:
    """Translate Clay's neutral measure expr to ngspice .meas syntax.

    v0.0.18: replaced the 3-pattern regex matcher with a proper grammar-
    based parser (`clay.measure_expr`). Supports the same expression
    forms as before plus several new ones (any Signal at any AXIS,
    Signal-when-Signal, plain Signal). Driver-specific emission is now
    in `emit_ngspice_meas`.
    """
    ast = parse_measure_expr(m.expr)
    return emit_ngspice_meas(m.name, m.analysis_ref, ast)


# ─── Driver impl ──────────────────────────────────────────────

class NgspiceDriver(SimulatorDriver):
    name = "ngspice"

    def __init__(
        self,
        config: ClayConfig | None = None,
    ):
        """Construct an ngspice driver.

        Args:
            config: explicit ClayConfig. If None, resolved via
                ClayConfig.auto() (env → clay.toml → auto-discovery).
        """
        self.config = config or get_config()
        errs = self.config.validate()
        if errs:
            raise RuntimeError(
                "ngspice driver cannot start: " + "; ".join(errs)
                + ". See clay/config.py for resolution order."
            )

    @property
    def binary(self) -> str:
        assert self.config.ngspice_bin is not None
        return str(self.config.ngspice_bin)

    def _pdk_corner_file(self, corner: str) -> Path:
        pdk = self._pdk()
        path = pdk.corner_file(corner)
        if path is None or not path.exists():
            raise RuntimeError(
                f"PDK corner file not available: pdk_id={self.config.pdk_id!r}, "
                f"corner={corner!r}, pdk_root={self.config.pdk_root}"
            )
        return path

    def _pdk(self) -> PdkAdapter:
        """Resolve the PDK adapter from the config's pdk_id."""
        try:
            return get_pdk(self.config.pdk_id)
        except KeyError as exc:
            raise RuntimeError(
                f"No clay.pdk adapter registered for {self.config.pdk_id!r}. "
                f"Add an adapter file at clay/pdk/{self.config.pdk_id}.py "
                f"(see clay/pdk/sky130.py as reference) or pre-import a "
                f"third-party adapter module."
            ) from exc

    def _bindings(self) -> dict[str, str]:
        return dict(self._pdk().device_bindings)

    def emit(self, tb: Testbench, out_path: Path) -> Path:
        bindings = self._bindings()
        lines: list[str] = []
        lines.append(f"* Clay-generated ngspice netlist for cell '{tb.cell.name}'")
        lines.append(
            f"* PDK={tb.cell.pdk_id}, corner={tb.pdk_corner}, temp={tb.temp:g}"
        )
        lines.append("")
        # PDK mismatch params (silence the agauss errors in sky130 models)
        lines.append(".param mc_mm_switch=0")
        lines.append(".param mc_pr_switch=0")
        # v0.0.24 (G11): nominal simulation temperature in °C. ngspice's
        # `.option temp = <value>` sets the nominal device temperature
        # used by the model evaluator (separate from `.options tnom`
        # which sets the model-parameter reference temp). Default of
        # `tb.temp` is 27 °C, matching the ngspice historical default,
        # so emitting unconditionally is harmless for older clay files
        # that never touch `tb.temp`.
        lines.append(f".option temp = {tb.temp:g}")
        # PDK include
        corner_file = self._pdk_corner_file(tb.pdk_corner).resolve()
        lines.append(f'.include "{corner_file.as_posix()}"')
        # v0.0.91 (LC-VCO accumulation-mode varactor):
        # Some PDK device models depend on parameter sets that live
        # outside the per-corner MOSFET include (sky130's varactor needs
        # r+c sheet-rho params for its internal series-resistance
        # network). The PDK adapter publishes any such extras via the
        # `extra_includes(corner)` hook — typically just one or two
        # lightweight `.param` files. Emitting them unconditionally is
        # harmless for cells that don't touch the affected devices.
        for extra in self._pdk().extra_includes(tb.pdk_corner):
            lines.append(f'.include "{Path(extra).resolve().as_posix()}"')
        lines.append("")

        # DUT subckt (the cell)
        lines.append(f".subckt {tb.cell.name} {' '.join(tb.cell.ports)}")
        for dev in tb.cell.devices:
            lines.append("  " + _emit_device(dev, bindings))
        lines.append(".ends")
        lines.append("")

        # Testbench: instantiate DUT + sources + extra devices
        lines.append("* -- Testbench --")
        for src in tb.sources:
            lines.append(_emit_source(src))
        for dev in tb.extra_devices:
            lines.append(_emit_device(dev, bindings))

        # DUT instance — connect testbench nets to DUT ports by name (1:1)
        # We require sources/extras to use the same net names as DUT ports.
        dut_pins = " ".join(tb.cell.ports)
        lines.append(f"xDUT {dut_pins} {tb.cell.name}")
        lines.append("")

        # Control deck
        lines.append(".control")
        for an in tb.analyses:
            lines.append(_emit_analysis(an))
        for m in tb.measures:
            lines.append(_emit_measure(m))
        # echo results to log
        for m in tb.measures:
            lines.append(f"echo {m.name}=$&{m.name}")
        lines.append("quit")
        lines.append(".endc")
        lines.append("")
        lines.append(".end")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def run(self, netlist_path: Path, work_dir: Path) -> Path:
        work_abs = work_dir.resolve()
        netlist_abs = netlist_path.resolve()
        log_path = work_abs / f"{netlist_abs.stem}.log"
        # Batch + silent: -b + -o for output redirect.
        # Use DEVNULL for stdout/stderr to avoid pipe buffer hangs (Windows
        # subprocess can deadlock if the child writes more than the pipe
        # buffer before exit). The -o flag captures all useful output to a
        # file already.
        subprocess.run(
            [self.binary, "-b", "-o", str(log_path), str(netlist_abs)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(work_abs),
            timeout=120,
        )
        return log_path

    _MEAS_LINE = re.compile(
        r"^\s*(\w+)\s*=\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)",
        re.MULTILINE,
    )

    def parse_measures(
        self,
        log_path: Path,
        wanted: Sequence[str],
    ) -> Mapping[str, float]:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        results: dict[str, float] = {}
        for m in self._MEAS_LINE.finditer(text):
            name, val_s = m.group(1), m.group(2)
            if name in wanted and name not in results:
                try:
                    results[name] = float(val_s)
                except ValueError:
                    pass
        return results
