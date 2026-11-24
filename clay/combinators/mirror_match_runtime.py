"""
Mirror match closed-loop ngspice check (v0.0.91 + v0.1.12 split).

v0.0.90 landed the metadata + structural helpers (in `mirror_match.py`).
The v0.0.91 closure (this module) wires the check to a REAL ngspice
run:

  1. Build the composed cell + a canonical AC/OP testbench.
  2. Hand-emit the netlist via NgspiceDriver, then post-process the
     file to inject `let + echo` lines for `@m.xdut.x<dev>.m<model>[id]`
     per mirror-pair device (the hierarchical OP-probe shape that
     `i(<dev>)` cannot resolve through subckt-wrapped sky130 PFET /
     NFET models).
  3. Run ngspice (silent batch via the driver's `run`).
  4. Parse per-device branch currents out of the log (single-device
     `id` × M multiplier = branch current).
  5. Call `check_mirror_match_invariant` with the parsed measures.

The helper accepts the same composed-stage handle the v0.0.90 metadata
layer already produces — no parser / IR changes touched. Failures
inside the simulator (no log, no probe extracted) surface as a
`measure_missing` violation pointing at the exact missing devices.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ..ir import Cell, Device, Spec, Testbench
from .mirror_match import (
    MirrorMismatchViolation,
    _mirror_pairs_of,
    check_mirror_match_invariant,
)


# Idempotent mapping from a Clay `device_class` to the ngspice subckt-
# wrapped MOSFET name we need to probe inside the hierarchical path.
# Sourced lazily from the PDK adapter so adding a new PDK (or a new
# device flavour like `nmos_lvt`) needs zero changes here — the helper
# below reads the PDK's `device_bindings` map at runtime.
def _pdk_model_for_device(device: Device, pdk_id: str) -> Optional[str]:
    """Resolve the PDK subckt name a Clay device lowers to.

    For `pmos` on sky130 this returns `sky130_fd_pr__pfet_01v8`; the
    hierarchical probe path inside `xDUT.x<dev>` is then
    `@m.xdut.x<dev>.msky130_fd_pr__pfet_01v8[id]`.

    Returns `None` for device classes the PDK adapter has no binding
    for (e.g. caps / resistors / current sources — none of which carry
    a per-device `id` OP quantity worth probing for mirror matching).
    """
    cls = device.device_class
    if cls not in ("nmos", "pmos"):
        return None
    # Honor an explicit `pdk_variant` if the device set one (e.g.
    # `pmos_lvt` cells used `pdk_variant="sky130_fd_pr__pfet_01v8_lvt"`
    # rather than the default binding).
    if device.pdk_variant is not None:
        return device.pdk_variant
    try:
        from ..pdk import get_pdk
        adapter = get_pdk(pdk_id)
    except Exception:  # noqa: BLE001
        return None
    bindings = dict(adapter.device_bindings)
    return bindings.get(cls)


def _build_mirror_op_probe_lines(
    cell: Cell,
    mirror_pairs: List[Tuple[str, str]],
    *,
    pdk_id: str,
    subckt_instance: str = "xdut",
) -> List[str]:
    """Return the `let + echo` deck lines that probe `[id]` for every
    mirror device in `mirror_pairs`.

    For each device we emit:

        let i_<dev>__id = @m.<subckt>.x<dev>.m<pdk_model>[id]
        echo i_<dev>__id=$&i_<dev>__id

    The `i_<dev>__id` key carries the SINGLE-DEVICE current; the caller
    multiplies by M to get the branch current. The double-underscore
    distinguishes the probe key from the existing `i_<dev>` measure
    spelling that `inject_mirror_match_measures` produces (which the
    ngspice subckt scope can't resolve — see v0.0.90 helper docstring).

    Devices whose `device_class` has no PDK binding (e.g. caps, resistors,
    cur_src) are skipped silently — they have no `[id]` quantity worth
    probing. Devices appearing on both sides of a pair are deduplicated.
    """
    by_name: Dict[str, Device] = {d.instance_name: d for d in cell.devices}
    out: List[str] = []
    seen: set[str] = set()
    for (a, b) in mirror_pairs:
        for dev_name in (a, b):
            if dev_name in seen:
                continue
            seen.add(dev_name)
            dev = by_name.get(dev_name)
            if dev is None:
                continue
            model = _pdk_model_for_device(dev, pdk_id)
            if model is None:
                continue
            probe_key = f"i_{dev_name}__id"
            ref = (
                f"@m.{subckt_instance}.x{dev_name}.m{model}[id]"
            )
            out.append(f"let {probe_key} = {ref}")
            out.append(f"echo {probe_key}=$&{probe_key}")
    return out


def _inject_mirror_probe_lines_into_netlist(
    netlist_path: Path,
    probe_lines: List[str],
) -> Path:
    """Insert `probe_lines` into the existing netlist's .control block.

    The injection point is the line immediately preceding `quit` inside
    the .control / .endc block — keeping the existing measure / echo
    ordering intact. The function mutates the file in place and returns
    the same path so it drops into a fluent chain.

    A no-op when `probe_lines` is empty; the netlist is left untouched.
    """
    if not probe_lines:
        return netlist_path
    text = netlist_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    out: List[str] = []
    injected = False
    for line in lines:
        if not injected and line.strip() == "quit":
            out.extend(probe_lines)
            injected = True
        out.append(line)
    if not injected:
        # No `quit` token found — fall back to appending before `.endc`
        # so we still get the probe even on a malformed deck. If neither
        # marker is present, return unchanged.
        out2: List[str] = []
        injected2 = False
        for line in lines:
            if not injected2 and line.strip() == ".endc":
                out2.extend(probe_lines)
                injected2 = True
            out2.append(line)
        if not injected2:
            return netlist_path
        netlist_path.write_text("\n".join(out2), encoding="utf-8")
        return netlist_path
    netlist_path.write_text("\n".join(out), encoding="utf-8")
    return netlist_path


# Probe-line parser for the `name=value` echo output. Reuses the same
# regex shape NgspiceDriver._MEAS_LINE uses so the probe lines land in
# the same name → value dict shape as measures.
_PROBE_LINE_RE = re.compile(
    r"^\s*(\w+)\s*=\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*$",
    re.MULTILINE,
)


def _parse_probe_log(
    log_path: Path, wanted: List[str],
) -> Dict[str, float]:
    """Pull `name=value` pairs out of an ngspice batch log.

    Mirrors `NgspiceDriver.parse_measures` but standalone so we can
    capture the probe lines without changing the driver's `wanted` list
    (which is set from `tb.measures` upstream).

    Args:
      log_path: path to the ngspice log file produced by the driver.
      wanted: list of probe-line keys to extract. Lines whose name is
        not in `wanted` are ignored.

    Returns:
      `{name: value}` map. Keys absent from the log don't land in the
      result — the caller's downstream check sees them as `None`.
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")
    wanted_set = set(wanted)
    out: Dict[str, float] = {}
    for m in _PROBE_LINE_RE.finditer(text):
        name, val_s = m.group(1), m.group(2)
        if name not in wanted_set or name in out:
            continue
        try:
            out[name] = float(val_s)
        except ValueError:
            pass
    return out


def _branch_currents_from_probes(
    cell: Cell,
    mirror_pairs: List[Tuple[str, str]],
    probes: Mapping[str, float],
) -> Dict[str, float]:
    """Convert per-device `id` probes into branch currents `i_<dev>`.

    The probe value is the current through one finger of the device.
    With Clay's `sizing["M"]` multiplier the full branch carries M
    times the probe value. The output dict keys match the form
    `check_mirror_match_invariant` expects on its first resolution
    attempt (`i_<dev>`).

    Devices for which the probe is absent (or M is non-finite) are
    skipped — the downstream check will surface them as
    `measure_missing`.
    """
    by_name: Dict[str, Device] = {d.instance_name: d for d in cell.devices}
    out: Dict[str, float] = {}
    seen: set[str] = set()
    for (a, b) in mirror_pairs:
        for dev_name in (a, b):
            if dev_name in seen:
                continue
            seen.add(dev_name)
            dev = by_name.get(dev_name)
            if dev is None:
                continue
            probe_key = f"i_{dev_name}__id"
            raw = probes.get(probe_key)
            if raw is None:
                continue
            try:
                m_mult = float(dev.sizing.get("M", 1.0))
            except (TypeError, ValueError):
                m_mult = 1.0
            if not math.isfinite(m_mult) or m_mult <= 0:
                m_mult = 1.0
            try:
                branch = float(raw) * m_mult
            except (TypeError, ValueError):
                continue
            if not math.isfinite(branch):
                continue
            out[f"i_{dev_name}"] = branch
    return out


def _default_mirror_match_testbench(cell: Cell) -> Testbench:
    """Build the canonical AC/OP testbench for a mirror_match composition.

    Routing is template-aware: a `mirror_match(two_stage_miller_pmos,
    ...)` cell needs the diff-input AC drive shape; a
    `mirror_match(common_source_pmos_input, ...)` cell needs the
    single-ended CS shape. Defaults to the two-stage Miller AC testbench
    when ports include both `inp` and `inn` (the diff-input shape every
    first-cohort mirror-match topology produces).
    """
    ports = set(cell.ports)
    if {"inp", "inn", "out", "vdd", "vss"}.issubset(ports):
        from ..templates.two_stage_miller_pmos import build_ac_testbench
        return build_ac_testbench(cell)
    if {"in", "out", "vdd", "vss"}.issubset(ports):
        from ..templates.common_source import build_ac_testbench_cs
        return build_ac_testbench_cs(cell)
    # Fall back to the most permissive shape — diff-input. This will
    # not produce valid AC sweeps for non-diff cells but the OP still
    # converges, which is all the mirror-match check actually needs.
    from ..templates.two_stage_miller_pmos import build_ac_testbench
    return build_ac_testbench(cell)


def run_mirror_match_check(
    composed_stage: Any,
    *,
    cell_name: str = "mm_check",
    spec: Optional[Spec] = None,
    pdk_id: str = "sky130",
    corner: str = "tt",
    temp: float = 27.0,
    tolerance: float = 0.10,
    work_dir: Optional[Path] = None,
    testbench: Optional[Testbench] = None,
    cell: Optional[Cell] = None,
) -> Optional[MirrorMismatchViolation]:
    """Closed-loop mirror_match check: build → ngspice → parse → enforce.

    Drives the full enforcement pipeline end-to-end on a real simulator:

      1. Lower `composed_stage` (or accept an already-built `cell` /
         `testbench`).
      2. Build the canonical AC/OP testbench (or use the supplied one).
      3. Emit the netlist via the configured NgspiceDriver.
      4. Post-process the netlist to inject hierarchical
         `@m.xdut.x<dev>.m<pdk_model>[id]` probe lines so ngspice
         actually reports the per-device current for every mirror pair
         (the existing `i(<dev>)` shape does not resolve through sky130
         subckt-wrapped MOSFETs).
      5. Run ngspice in silent batch mode.
      6. Parse the log, multiply the per-device `id` by the M
         multiplier to get branch current, and feed the result into
         `check_mirror_match_invariant`.

    Args:
      composed_stage: a `ComposedStage` (typically from
        `mirror_match(a, b)`). May be `None` if `cell` is supplied
        directly.
      cell_name: name for the built cell. Defaults to `"mm_check"`.
      spec: optional `Spec` for the cell. Defaults to a permissive
        single-field spec (`gain_dc >= 0 dB`) so the simulator just
        needs to converge — the actual decision is on the mirror
        ratio, not the spec.
      pdk_id: PDK adapter to consult for device → model name mapping.
        Defaults to `"sky130"`.
      corner: PDK corner for the simulation. Defaults to `"tt"`.
      temp: nominal simulation temperature (°C). Defaults to `27`.
      tolerance: relative band around the target ratio (1.0). Default
        `0.10` (±10 %).
      work_dir: optional directory for the netlist + log. Defaults to
        a `tempfile.TemporaryDirectory` (auto-cleaned).
      testbench: pre-built `Testbench` to use directly. When supplied,
        bypasses the `composed_stage` → `cell` → testbench path entirely.
      cell: pre-built `Cell` to wrap in the default testbench. Bypasses
        the `composed_stage` → `cell` step.

    Returns:
      A `MirrorMismatchViolation` describing the FIRST out-of-tolerance
      pair, OR `None` when:
        * No mirror pairs are declared on the composition (degenerate
          `mirror_match` on an unsupported topology pair).
        * Every pair's measured branch current ratio sits within
          `1.0 ± tolerance`.

    Raises:
      ValueError: if `tolerance` is non-positive.
      RuntimeError: if no ngspice driver / PDK is available AND no
        pre-built testbench was supplied (the closed-loop path needs
        a working simulator config).
    """
    if tolerance <= 0:
        raise ValueError(
            f"tolerance must be positive, got {tolerance!r}"
        )

    # ─── Stage 1: resolve cell + testbench ────────────────────────
    if testbench is not None:
        tb = testbench
        active_cell = tb.cell
    else:
        if cell is None:
            if composed_stage is None:
                raise ValueError(
                    "run_mirror_match_check: must supply at least one of "
                    "`composed_stage`, `cell`, or `testbench`."
                )
            if spec is None:
                from ..ir import SpecField
                spec = Spec(fields=[SpecField("gain_dc", "db", ">=", 0.0)])
            from ._lower import lower_to_template
            builder = lower_to_template(composed_stage)
            active_cell = builder(cell_name, spec)
        else:
            active_cell = cell
        tb = _default_mirror_match_testbench(active_cell)
        tb.pdk_corner = corner
        tb.temp = temp

    # ─── Stage 2: resolve mirror pairs ────────────────────────────
    pairs = _mirror_pairs_of(composed_stage) if composed_stage is not None else []
    if not pairs:
        # Cell-side metadata path — happens when only the cell was
        # supplied (no composed stage in scope).
        pairs = _mirror_pairs_of(active_cell)
    if not pairs:
        return None

    # ─── Stage 3: emit + post-process netlist ─────────────────────
    from ..drivers import NgspiceDriver
    from ..config import ClayConfig
    cfg = ClayConfig.auto()
    errs = cfg.validate()
    if errs:
        raise RuntimeError(
            "run_mirror_match_check: ngspice / PDK config invalid — "
            + "; ".join(errs)
        )
    drv = NgspiceDriver(cfg)
    use_tempdir = work_dir is None
    if use_tempdir:
        import tempfile
        td_ctx = tempfile.TemporaryDirectory()
        work_path = Path(td_ctx.name)
    else:
        td_ctx = None
        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)
    try:
        netlist_path = drv.emit(tb, work_path / f"{active_cell.name}.cir")
        probe_lines = _build_mirror_op_probe_lines(
            active_cell, pairs, pdk_id=pdk_id,
        )
        _inject_mirror_probe_lines_into_netlist(netlist_path, probe_lines)

        # ─── Stage 4: run ngspice ─────────────────────────────────
        log_path = drv.run(netlist_path, work_path)

        # ─── Stage 5: parse probes ────────────────────────────────
        wanted = [
            f"i_{dev}__id"
            for pair in pairs for dev in pair
        ]
        probes = _parse_probe_log(log_path, wanted)
        op_result = _branch_currents_from_probes(active_cell, pairs, probes)

        # ─── Stage 6: feed into the v0.0.90 invariant runtime ─────
        return check_mirror_match_invariant(
            composed_stage if composed_stage is not None else active_cell,
            op_result,
            tolerance=tolerance,
        )
    finally:
        if td_ctx is not None:
            td_ctx.cleanup()
