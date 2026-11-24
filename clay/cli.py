"""
Clay command-line interface.

Usage:
    python -m clay verify FILE.clay        # parse + run verify jobs

Exit codes:
    0: all verify jobs PASS
    1: at least one verify job FAIL
    2: file not found / parse error / CLI usage error
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Mapping, Sequence

from .compiler import compile_cell_def
from .core_preview import (
    CoreCertificatePreview,
    preview_from_certificate_previews,
    preview_from_multisetup_result,
    preview_from_verify_results,
)
from .core_preview_payload import attach_core_preview_payload
from .core_verify_bridge import boundary_evidence_manifest_from_verify_result
from .drivers.base import SimulatorDriver
from .ir import ClayFile, VerifyJob
from .module import resolve
from .parser import parse_clay_file
from .proof_trace import (
    boundary_evidence_manifest_is_certified,
    compiler_run_manifest_is_certified,
    observation_basis_manifest_is_certified,
    pdk_adequacy_manifest_is_certified,
    repair_operator_manifest_is_certified,
    spice_oracle_manifest_is_certified,
    verify_result_from_multisetup_result,
)
from .stage_templates import get_stage_testbench
from .signoff_obligation_trace import _co_clean_obligation_row, signoff_obligation_rows
from .templates.bandgap_testbench import (
    build_bandgap_testbench, find_vref_ports,
)
from .templates.charge_pump_testbench import (
    build_charge_pump_testbench, find_pfd_ports,
)
from .templates.ldo_testbench import build_ldo_testbench
from .templates.pll_testbench import (
    build_pll_testbench, find_pll_ports,
)
from .templates.sensor_testbench import (
    build_sensor_testbench, find_sensor_ports,
)
from .templates.two_stage_miller_pmos import build_ac_testbench
from .verifier import (
    MultiSetupResult, verify as verify_testbench, verify_multi_setup,
)

# Sentinel prefix used by parser.py to mark verify_jobs whose target is
# an in-source `cell ... { ... }` definition rather than a `pkg::Cell`
# reference. Keep in sync with parser.py.
_CELLDEF_SENTINEL = "__celldef__::"


# v0.0.87 — Marker tokens framing the JSON-serialized invariant
# violations section emitted at the end of each verify run. Stable
# strings (DO NOT rename — `tools/llm_authoring/score_functional.py`
# greps stdout for these EXACT bytes) form the section that lets
# subprocess-driven consumers (the LLM authoring baseline runner) parse
# the structured functional-role invariant outcomes that `verifier.py`
# already produces in-process. The marker pair is emitted ONCE per
# verify job (aggregated across every corner × temp × setup) so a
# downstream parser sees one section per `=== Verify job N ===` header.
INVARIANT_MARKER_BEGIN = "## invariant_violations: BEGIN"
INVARIANT_MARKER_END = "## invariant_violations: END"
BOUNDARY_EVIDENCE_MARKER_BEGIN = "## boundary_evidence: BEGIN"
BOUNDARY_EVIDENCE_MARKER_END = "## boundary_evidence: END"


class _OfflineMeasuresDriver(SimulatorDriver):
    name = "offline-measures"

    def __init__(self, measures: Mapping[str, float]):
        self._measures = {
            str(name): float(value) for name, value in measures.items()
        }

    def emit(self, tb, out_path: Path) -> Path:
        out_path.write_text(f"* offline measures netlist for {tb.cell.name}\n")
        return out_path

    def run(self, netlist_path: Path, work_dir: Path) -> Path:
        log_path = work_dir / f"{netlist_path.stem}.offline.log"
        log_path.write_text(json.dumps(self._measures, sort_keys=True))
        return log_path

    def parse_measures(
        self,
        log_path: Path,
        wanted: Sequence[str],
    ) -> Mapping[str, float]:
        wanted_names = set(wanted)
        return {
            name: value
            for name, value in self._measures.items()
            if name in wanted_names
        }


def _build_pll_testbench_with_supply_scaled_window(cell):
    """Build a PLL testbench whose transient window scales with the
    cell's PDK supply rail (v0.0.53 gf180mcu closure).

    The base `build_pll_testbench` defaults to a 5 µs transient with
    a 1 µs settle window — tuned for sky130's 1.8 V supply, where the
    PFD/CP/LF/VCO loop saturates to one rail within ~3 µs (v0.0.40
    closure measured vctrl_pp ≈ 0.08 V over [1, 5] µs).

    On gf180mcu the same composed netlist needs ~10× longer to reach
    the same saturated steady-state — two compounding causes:

      1. The 3.3 V supply pushes ~3× more charge-pump current per
         half-period through the same loop-filter cap (100 pF), but
         the per-step swing on vctrl is also wider, so the loop drifts
         further before saturating.
      2. The gf180mcu 3p3 devices have slower edges than sky130's
         core MOS at the same (W, L), which stretches the PFD's
         per-cycle reset windows.

    Empirical observation (default 5 µs window):
      sky130:   vctrl_avg ≈ 1.79 V, vctrl_pp ≈ 0.08 V  (saturated to vdd)
      gf180mcu: vctrl_avg ≈ 0.56 V, vctrl_pp ≈ 1.14 V  (still drifting)

    Extending the window to ~50 µs on gf180mcu lets vctrl reach the
    same saturated state:
      gf180mcu (50 µs window): vctrl_avg ≈ 3.30 V, vctrl_pp ≈ 0.3 mV.

    The wrapper:
      1. Picks `tran_end` / `t_settle` based on the cell's PDK supply
         rail. The 3.0 V cutoff picks up gf180mcu (3.3 V) and any
         future 3.3 V+ PDK adapter; sky130 (1.8 V) keeps the
         pre-v0.0.53 5 µs default unchanged.
      2. Calls `build_pll_testbench(cell, tran_end=..., t_settle=...)`
         with the chosen window. All other defaults (f_ref, tran_step,
         etc.) are inherited from the template.

    Cells on sky130 (or any sub-3 V PDK) continue to use the 5 µs
    window unchanged so the v0.0.40 closure baseline is preserved bit-
    for-bit — only the gf180mcu path picks up the extended window.
    """
    from .templates._supply import resolve_supply
    supply = resolve_supply(cell)
    if supply >= 3.0:
        # 50 µs tran window, 40 µs settle exclusion. Captures the
        # post-saturation steady-state on gf180mcu while leaving a
        # 10 µs aggregation window for stable mean / pp measures.
        return build_pll_testbench(cell, tran_end=50e-6, t_settle=40e-6)
    return build_pll_testbench(cell)


def _build_ldo_testbench_with_port_remap(cell):
    """Build an LDO testbench whose VREF/VFB sources wire by the cell's
    actual `vref` / `vfb` port names (v0.0.46 LDO routing).

    `build_ldo_testbench` hardcodes its VREF source on node `inp` and
    its VFB source between nodes `inn` and `out` — that's the
    AnalogGym sloppy convention where the LDO error amp's inputs are
    named like a differential opamp's. For cells that declare proper
    semantic port names (`vref` for the reference input and `vfb` for
    the feedback rail), those template-default node names land on
    dangling testbench-level nodes — the cell's `vref` / `vfb` ports
    become external nodes that no source drives.

    This wrapper:
      1. Picks `vin_min` based on the cell's PDK supply rail so the
         DC sweep starts ABOVE dropout. The template default 1.6 V is
         tuned for sky130 (1.8 V supply); on gf180mcu (3.3 V) the
         1.6 V lower bound dives well into dropout because the LDO
         can't regulate to a 1.8 V reference when Vin < Vref +
         V_dropout — the textbook-sized ldo_simple's open-loop gain
         on gf180mcu also leaves a wide fall-off region above the
         theoretical dropout point. Bumping `vin_min` to ~Vref+1 V
         on a 3.3 V supply keeps the sweep inside the regulation
         region so `vout_pp` reflects real line regulation rather
         than the dropout brick-wall.
      2. Calls `build_ldo_testbench(cell, vin_min=...)` to get the
         standard rig.
      3. Rewrites the VREF source's `p` pin from `inp` to `vref` and
         the VFB source's `p` pin from `inn` to `vfb`. The remap is
         purely a node-name swap; analyses, measures, and the
         rewriter pipeline (which addresses sources by `instance_name`,
         not pin name) are unaffected.

    Cells with `inn` / `inp` port names continue to use
    `build_ldo_testbench` directly (the Python-level path used by
    `test_benchmark_runs_end_to_end` and `test_gf180mcu_ldo_via_ldo_testbench`).
    """
    from .templates._supply import resolve_supply
    # Move sweep lower bound up when the PDK supply is comfortably
    # above the standard 1.8 V reference. The 3.0 V cutoff picks up
    # gf180mcu (3.3 V) without disturbing sky130's defaults.
    supply = resolve_supply(cell)
    if supply >= 3.0:
        tb = build_ldo_testbench(cell, vin_min=2.8)
    else:
        tb = build_ldo_testbench(cell)
    for src in tb.sources:
        if src.instance_name == "VREF" and src.pins.get("p") == "inp":
            src.pins["p"] = "vref"
        elif src.instance_name == "VFB" and src.pins.get("p") == "inn":
            src.pins["p"] = "vfb"
    return tb


def cmd_verify(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"FAIL: file not found: {path}", file=sys.stderr)
        return 2

    try:
        clay_file = parse_clay_file(path)
    except Exception as exc:
        print(f"FAIL: parse error in {path}: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 2

    print(f"Parsed {path}:")
    print(f"  package = {clay_file.package_name}")
    print(f"  pdk     = {clay_file.pdk_id}")
    print(f"  imports = {clay_file.imports}")
    print(f"  verify  = {len(clay_file.verify_jobs)} job(s)")

    # Resolve all imports up front
    for imp in clay_file.imports:
        try:
            resolve(imp)
        except Exception as exc:
            print(f"FAIL: import {imp!r}: {exc}", file=sys.stderr)
            return 1

    # Run each verify job
    overall_ok = True
    # v0.0.87: aggregate functional-role invariant violations across all
    # verify jobs in this file. One marker-delimited JSON section is
    # emitted at the end of the run so a subprocess-driven consumer
    # (e.g. the LLM authoring baseline runner) can parse the structured
    # violation list without rerunning Python inside the .clay verify
    # pipeline. The empty list is still emitted (just `{"violations": []}`)
    # so a parser can tell "verify ran but cell has no functional roles"
    # apart from "verify never produced this section".
    all_violations: list[dict] = []
    core_previews: list[CoreCertificatePreview] = []
    boundary_evidence_reports: list[dict] = []
    simulator_driver = _offline_measures_driver_from_path(
        args.offline_measures_json
    )
    for i, job in enumerate(clay_file.verify_jobs):
        # v0.0.21: --corner overrides whatever the file's `corners { }`
        # block declared. Mirrors the documented help text: "overrides
        # any `corners { }` block declared in the file".
        if args.corner is not None:
            job.corners = [args.corner]
        # Header. Single corner → compact `(corner=tt)`; multi-corner
        # → `(corners=[tt, ss, ff])` so logs are scannable either way.
        # v0.0.24 (G11): same compact/verbose handling for temps so the
        # job header shows both axes of the sweep.
        if len(job.corners) == 1:
            corner_label = f"corner={job.corners[0]}"
        else:
            corner_label = "corners=[" + ", ".join(job.corners) + "]"
        if len(job.temps) == 1:
            temp_label = f"temp={_fmt_temp(job.temps[0])}"
        else:
            temp_label = "temps=[" + ", ".join(_fmt_temp(t) for t in job.temps) + "]"
        print(
            f"\n=== Verify job {i + 1}: {job.cell_path} "
            f"({corner_label}, {temp_label}) ==="
        )
        job_violations: list[dict] = []
        try:
            job_result = _run_one_job(
                job,
                clay_file,
                core_preview_enabled=args.core_preview,
                boundary_evidence_enabled=args.boundary_evidence_report,
                boundary_evidence_reports=boundary_evidence_reports,
                simulator_driver=simulator_driver,
            )
            ok, job_violations = job_result[:2]
            if len(job_result) >= 3 and job_result[2] is not None:
                core_previews.append(job_result[2])
        except Exception as exc:
            print(f"  FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.verbose:
                traceback.print_exc()
            ok = False
        if not ok:
            overall_ok = False
        all_violations.extend(job_violations)

    if overall_ok:
        print("\nAll verify jobs PASSED.")
    else:
        print("\nAt least one verify job FAILED.")

    core_preview = None
    if args.core_preview:
        expected_preview_count = len(clay_file.verify_jobs)
        if len(core_previews) != expected_preview_count:
            core_preview = _rejected_core_preview(
                "Core preview unavailable: expected a Core preview for every "
                f"verify job, found {len(core_previews)} of {expected_preview_count}"
            )
        elif len(core_previews) == 1:
            core_preview = core_previews[0]
        elif len(core_previews) > 1:
            core_preview = preview_from_certificate_previews(core_previews)
        else:
            core_preview = _rejected_core_preview(
                "Core preview unavailable: expected exactly one simple "
                f"VerifyResult, found {len(core_previews)}"
            )
        _emit_core_preview_section(core_preview)

    if args.boundary_evidence_report:
        _emit_boundary_evidence_section(boundary_evidence_reports)

    # v0.0.87: emit the marker-delimited invariant_violations JSON
    # section at the END of the run (after the PASSED/FAILED banner so
    # downstream parsers can still cheaply grep for the banner without
    # the structured payload getting in the way). The payload is one
    # JSON object containing a `violations` list; each entry is the
    # `to_json_dict()` rendering of an `InvariantViolation`.
    _emit_invariant_violations_section(
        all_violations,
        core_preview=core_preview,
        attach_core_preview=args.core_preview,
    )
    return 0 if overall_ok else 1


def _emit_boundary_evidence_section(reports: list[dict]) -> None:
    print(BOUNDARY_EVIDENCE_MARKER_BEGIN)
    print(json.dumps({"boundary_evidence": reports}, ensure_ascii=False, sort_keys=True))
    print(BOUNDARY_EVIDENCE_MARKER_END)


def _offline_measures_driver_from_path(path: str | None) -> SimulatorDriver | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--offline-measures-json must contain a JSON object")
    return _OfflineMeasuresDriver(payload)


def _emit_invariant_violations_section(
    violations: list[dict],
    *,
    core_preview=None,
    attach_core_preview: bool = False,
) -> None:
    """Emit the marker-delimited invariant_violations JSON section.

    The section bytes are stable across releases — `score_functional`
    in the LLM authoring tools greps stdout for the BEGIN / END marker
    pair and reads the JSON payload between them. The payload shape is
    `{"violations": [...]}` where each entry is the dict produced by
    `InvariantViolation.to_json_dict()` (or a passthrough dict carrying
    the same fields).

    Empty `violations` is still emitted as `{"violations": []}` — the
    marker presence tells the consumer "verify ran the invariant
    runtime; the cell just has no roles" instead of "verify died before
    reaching the invariant layer".
    """
    payload = {"violations": [_violation_to_json_dict(v) for v in violations]}
    if attach_core_preview:
        payload = attach_core_preview_payload(payload, core_preview)
    print(INVARIANT_MARKER_BEGIN)
    # Indent kept tight so the payload stays compact in the log; the
    # parser only cares about `json.loads`, not pretty printing.
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    print(INVARIANT_MARKER_END)


def _emit_core_preview_section(preview: CoreCertificatePreview) -> None:
    print("Clay Core Certificate Preview:")
    print(f"status: {preview.status}")
    print(f"reason: {preview.reason}")
    if preview.source_obligation_ids:
        source_ids = ", ".join(preview.source_obligation_ids)
    else:
        source_ids = "0"
    print(f"source_obligation_ids: {source_ids}")
    print(f"row_count: {preview.row_count}")


def _rejected_core_preview(reason: str) -> CoreCertificatePreview:
    return CoreCertificatePreview(
        status="rejected",
        reason=reason,
        certificate=None,
        source_obligation_ids=(),
        row_count=0,
    )


def _boundary_evidence_report_from_verify_result(
    result,
    *,
    cell,
    pdk_id: str,
    pdk_corner: str,
    temp: float,
    specification_measure_names: list[str],
) -> dict:
    try:
        manifest = boundary_evidence_manifest_from_verify_result(
            result,
            pdk_id=pdk_id,
            pdk_corner=pdk_corner,
            devices=getattr(cell, "devices", []),
            rule_deck=_boundary_rule_deck_name(pdk_id, pdk_corner),
            specification_measure_names=specification_measure_names,
            cell=cell,
            temp=temp,
        )
    except Exception as exc:  # noqa: BLE001 - report rejection, do not crash CLI.
        return _rejected_boundary_evidence_report(
            getattr(result, "cell_name", getattr(cell, "name", "")),
            pdk_corner=pdk_corner,
            temp=temp,
            reason=f"{type(exc).__name__}: {exc}",
        )
    return _boundary_evidence_report_from_manifest(manifest, pdk_corner, temp)


def _boundary_evidence_report_from_multisetup_result(
    result,
    *,
    cell,
    pdk_id: str,
    pdk_corner: str,
    temp: float,
) -> dict:
    derived_errors = dict(getattr(result, "derived_errors", {}) or {})
    if derived_errors:
        details = ", ".join(
            f"{name}: {message}" for name, message in sorted(derived_errors.items())
        )
        return _rejected_boundary_evidence_report(
            getattr(result, "cell_name", getattr(cell, "name", "")),
            pdk_corner=pdk_corner,
            temp=temp,
            reason=f"derived errors present: {details}",
        )
    try:
        manifest = boundary_evidence_manifest_from_verify_result(
            verify_result_from_multisetup_result(result),
            pdk_id=pdk_id,
            pdk_corner=pdk_corner,
            devices=getattr(cell, "devices", []),
            rule_deck=_boundary_rule_deck_name(pdk_id, pdk_corner),
            specification_measure_names=_multisetup_setup_measure_names(result),
            cell=cell,
            multisetup_result=result,
            temp=temp,
        )
    except Exception as exc:  # noqa: BLE001 - report rejection, do not crash CLI.
        return _rejected_boundary_evidence_report(
            getattr(result, "cell_name", getattr(cell, "name", "")),
            pdk_corner=pdk_corner,
            temp=temp,
            reason=f"{type(exc).__name__}: {exc}",
        )
    return _boundary_evidence_report_from_manifest(manifest, pdk_corner, temp)


def _boundary_evidence_report_from_manifest(manifest, pdk_corner: str, temp: float) -> dict:
    return {
        "status": (
            "accepted" if boundary_evidence_manifest_is_certified(manifest)
            else "rejected"
        ),
        "cell": manifest.cell,
        "pdk_corner": pdk_corner,
        "temp": temp,
        "compiler": _component_status(
            compiler_run_manifest_is_certified(manifest.compiler)
        ),
        "spice": _component_status(
            spice_oracle_manifest_is_certified(manifest.spice)
        ),
        "pdk": _component_status(
            pdk_adequacy_manifest_is_certified(manifest.pdk)
        ),
        "observations": _component_status(
            observation_basis_manifest_is_certified(manifest.observations)
        ),
        "repairs": _component_status(
            repair_operator_manifest_is_certified(manifest.repairs)
        ),
        "setup_observation_count": len(manifest.observations.specifications),
        "derived_observation_count": len(manifest.observations.derived_measures),
    }


def _rejected_boundary_evidence_report(
    cell: str,
    *,
    pdk_corner: str,
    temp: float,
    reason: str,
) -> dict:
    return {
        "status": "rejected",
        "cell": cell,
        "pdk_corner": pdk_corner,
        "temp": temp,
        "reason": reason,
        "compiler": "rejected",
        "spice": "rejected",
        "pdk": "rejected",
        "observations": "rejected",
        "repairs": "rejected",
        "setup_observation_count": 0,
        "derived_observation_count": 0,
    }


def _component_status(ok: bool) -> str:
    return "accepted" if ok else "rejected"


def _boundary_rule_deck_name(pdk_id: str, pdk_corner: str) -> str:
    return f"{pdk_id}:{pdk_corner}:runtime-boundary"


def _multisetup_setup_measure_names(result) -> list[str]:
    return [
        str(assertion.name)
        for setup_result in dict(result.per_setup).values()
        for assertion in list(getattr(setup_result, "assertions", []))
    ]


def _violation_to_json_dict(v):
    """Coerce an InvariantViolation (or dict-shaped entry) to a JSON dict.

    Accepts:
      * `InvariantViolation` instances (calls `to_json_dict()` for the
        canonical dataclass `asdict` rendering).
      * Plain dicts (passes through verbatim — used by the soft-failure
        path in `verifier.py` which stashes an empty list when the
        invariant runtime itself raises).

    Returns a `dict` ready for `json.dumps`. None of the floats are
    re-rounded; we trust the caller's numerical formatting.
    """
    if hasattr(v, "to_json_dict"):
        return v.to_json_dict()
    if isinstance(v, dict):
        return dict(v)
    # Defensive fallback: best-effort dict via `vars()` so a debug
    # session with an unexpected shape still produces SOMETHING readable
    # in the marker block rather than dying with TypeError.
    try:
        return dict(vars(v))
    except TypeError:
        return {"repr": repr(v)}


def _run_one_job(
    job: VerifyJob, clay_file: ClayFile,
    *,
    core_preview_enabled: bool = False,
    boundary_evidence_enabled: bool = False,
    boundary_evidence_reports: list[dict] | None = None,
    simulator_driver: SimulatorDriver | None = None,
) -> tuple[bool, list[dict], CoreCertificatePreview | None]:
    """Run a verify job at each corner in `job.corners`.

    The job passes only if ALL corners pass. Per-corner sub-headers
    are printed for multi-corner runs so log readers can spot which
    corner fails. Single-corner runs (the common case before v0.0.21,
    and the case under `--corner X`) suppress the sub-header to keep
    output identical to pre-v0.0.21 single-corner behavior.

    Returns:
        ``(all_ok, invariant_violations)`` where ``invariant_violations``
        is the flat list of `InvariantViolation.to_json_dict()` rendered
        entries the verifier produced across every corner × temp ×
        setup point. The list is JSON-ready for the marker section
        emitted by :func:`cmd_verify`. Empty when no functional roles
        are declared on the cell (the runtime returns `[]` in that
        case) — still surfaced so the marker block tells a downstream
        parser "verify ran the invariant layer; no roles to check".
    """
    # Build cell + testbench builder ONCE; iterate corners over the
    # testbench. This mirrors the design constraint: only `pdk_corner`
    # changes between runs, everything upstream (sizing, devices, .meas)
    # is identical so we don't re-solve the closed-form sizing N times.
    # Path A: in-source `cell ... { ... }` definition (v0.0.19).
    if job.cell_path.startswith(_CELLDEF_SENTINEL):
        local_name = job.cell_path[len(_CELLDEF_SENTINEL):]
        cd = next(
            (c for c in clay_file.cell_defs if c.local_name == local_name),
            None,
        )
        if cd is None:
            raise RuntimeError(
                f"verify references undeclared cell {local_name!r}; "
                f"defs={[c.local_name for c in clay_file.cell_defs]}"
            )
        cell = compile_cell_def(cd)
        # In-source cells: prefer the stage template's registered
        # testbench builder (e.g. single-ended CS amps need a different
        # rig than the differential default). Fall back to the generic
        # AC testbench when no stage-specific testbench was registered,
        # UNLESS the cell's port shape matches a known non-amp pattern
        # (currently: any `vref*` port → bandgap testbench). The
        # port-shape heuristic only fires for cells WITHOUT a registered
        # stage style, so opamp cells (which always have a style) are
        # never affected.
        style = cd.topology.get("style") if isinstance(cd.topology.get("style"), str) else None
        stage_tb = get_stage_testbench(style) if style else None
        if stage_tb is not None:
            tb_builder = stage_tb
        elif find_pll_ports(cell):
            # v0.0.37: PLL-shaped cell (has BOTH `ref_in` + `clk_out`
            # ports). Use the PLL testbench so the e2e measurement
            # drives the reference clock + measures VCO output activity
            # rather than gain_dc / gbw measure_missing.
            #
            # Ordered FIRST among the port-shape heuristics because
            # `ref_in` + `clk_out` is the most specific signature — a
            # cell with that pair is unambiguously PLL-shaped, whereas
            # `vref*` (bandgap) / `up`+`dn` (CP) / `vsense*` (sensor)
            # could in principle co-occur with a PLL surface in a
            # future composite cell. Picking PLL first keeps the ladder
            # deterministic and matches the architectural intent: a
            # cell with `ref_in` + `clk_out` is being tested AS a PLL,
            # not as a sub-block.
            #
            # v0.0.53: route through a supply-scaled wrapper so the
            # transient window stretches to ~50 µs on gf180mcu
            # (3.3 V), where the loop needs ~10× longer to saturate
            # than on sky130 (1.8 V). sky130 keeps the v0.0.40 5 µs
            # default unchanged.
            tb_builder = _build_pll_testbench_with_supply_scaled_window
        elif "vref" in cell.ports and "vfb" in cell.ports:
            # v0.0.46: LDO-shaped cell (has BOTH `vref` + `vfb` ports).
            # Use the LDO testbench so the e2e measurement extracts
            # `line_regulation` (vout_pp over the Vin sweep / Vin span)
            # rather than gain_dc / gbw measure_missing.
            #
            # Ordered BEFORE the bandgap (`vref*`) branch because LDO
            # is more specific: every LDO cell with a `vref` port also
            # has a `vfb` port (the feedback rail closing the regulator
            # loop), whereas a bandgap is self-biased and has no `vfb`.
            # Checking `vfb` first short-circuits the ladder before
            # the bandgap branch fires on the `vref` match alone.
            #
            # The LDO testbench wires VREF / VFB sources by the cell's
            # actual `vref` / `vfb` port names (overriding the template
            # default of `inp` / `inn`, which is the AnalogGym sloppy
            # convention) so the regulator loop closes correctly. Pin
            # remapping happens via a thin wrapper around
            # `build_ldo_testbench` — the template stays untouched.
            tb_builder = _build_ldo_testbench_with_port_remap
        elif find_vref_ports(cell):
            # v0.0.35: bandgap-shaped cell (has `vref*` ports). Use the
            # bandgap testbench instead of the differential AC default
            # so the e2e measurement picks up Vref values rather than
            # gain_dc / gbw measure_missing.
            tb_builder = build_bandgap_testbench
        elif find_pfd_ports(cell):
            # v0.0.35 follow-on: charge-pump-shaped cell (has `up`+`dn`
            # ports, possibly also `upb`/`dnb`, or the alternate
            # `up_signal`/`dn_signal` aliases). Use the CP testbench so
            # the e2e measurement reads UP / DN current matching
            # via the 0 V ammeter on cp_out — the standard CP figure
            # of merit. AnalogGym's CP FOM is the (i_up, i_dn) pair;
            # without this branch the cell falls through to the
            # differential AC testbench, which has no `inp`/`inn`
            # ports to drive and would produce measure_missing on
            # every measure.
            tb_builder = build_charge_pump_testbench
        elif find_sensor_ports(cell):
            # v0.0.35 follow-on: sensor-shaped cell (has `vsense*` /
            # `tsense*` / `sensor_out*` port and no `vref*` — the vref
            # check above already returned False to reach this branch).
            # Use the sensor testbench so the e2e measurement picks up
            # V(vsense) at each requested `temps { ... }` point rather
            # than gain_dc / gbw measure_missing. Same port-shape
            # heuristic style as the bandgap branch — no grammar or
            # IR changes needed, the ladder just grows by one entry.
            tb_builder = build_sensor_testbench
        else:
            tb_builder = build_ac_testbench
    else:
        # Path B: classic `pkg::Cell` reference into Python module.
        pkg, _, cell_name = job.cell_path.partition("::")
        mod = resolve(pkg)
        entry = mod.cell_entry(cell_name)
        cell = entry.builder(spec=job.spec, **job.sizing_overrides)
        # Look up the cell's declared default testbench. If a cell didn't
        # register one (older stdlib cells, third-party packages), fall back
        # to the AC testbench. This decouples the CLI from hardcoded
        # testbench types (code review #3).
        tb_builder = entry.default_testbench or build_ac_testbench

    # v0.0.41: the .clay file's `pdk '<id>';` declaration wins over
    # whatever pdk_id the stage builder hardcoded. Lets a single Clay
    # cell parameterized at sky130 (1.8 V) re-target at gf180mcu
    # (3.3 V) by just changing the `pdk` line — the testbench supply
    # auto-resolver downstream picks up the new vdd_nom automatically.
    if clay_file.pdk_id:
        cell.pdk_id = clay_file.pdk_id

    # Edge case: corners / temps lists should never be empty (parser
    # rejects empty blocks; defaults are ["tt"] / [27.0]). Defensive
    # fall-backs preserve old single-point behavior.
    corners = list(job.corners or ["tt"])
    temps = job.temps or [27.0]
    # v0.0.30 (G11 closeout): UNION job-level corners with every
    # `across_corners(...)` clause on a spec field. The verifier scopes
    # each assertion to its declared corners (or all corners when the
    # clause is absent), but every corner appearing anywhere in the
    # union must actually be SIMULATED for the per-field check to be
    # meaningful. Example: `corners { tt }` + `gain across_corners('ss')`
    # → sim runs at BOTH tt and ss; gain is asserted at ss only;
    # other fields are asserted at tt only.
    field_corners: set[str] = set()
    for sf in job.spec.fields:
        if sf.across_corners:
            field_corners.update(sf.across_corners)
    for s in job.setups:
        if s.spec is None:
            continue
        for sf in s.spec.fields:
            if sf.across_corners:
                field_corners.update(sf.across_corners)
    if field_corners:
        # Preserve the explicit declaration order from `corners { }`
        # for the leading entries, then append any extra corners
        # required by field-level clauses in deterministic order.
        existing = set(corners)
        for c in sorted(field_corners):
            if c not in existing:
                corners.append(c)
                existing.add(c)
    # v0.0.24 (G11): cartesian product over corners × temps. We print
    # a per-point sub-header iff more than one point is run — single
    # (corner=tt, temp=27) jobs keep the pre-v0.0.21 output shape.
    multi = (len(corners) * len(temps)) > 1

    all_ok = True
    is_multi_setup = bool(job.setups) or bool(job.derived_measures)
    # v0.0.53: bandgap TC aggregation. Collect per-(corner, port) Vref
    # samples across the temperature sweep so a single temperature
    # coefficient (TC) figure can be derived AFTER the temp loop —
    # AnalogGym's bandgap FOM is `ppavl/avgval/dT` (per-million / °C),
    # which spans temperature points rather than living inside one
    # ngspice run. Only fires when (a) ≥2 temps requested and (b) the
    # cell routed to `build_bandgap_testbench`. Empty for every other
    # testbench shape — no impact on AC / LDO / PLL / CP / sensor.
    is_bandgap = (tb_builder is build_bandgap_testbench)
    # nested dict: corner -> port_name -> list[(temp, vref)]
    bandgap_vref_by_corner: dict[str, dict[str, list[tuple[float, float]]]] = {}
    # v0.0.87: collect the functional-role InvariantViolation entries
    # the verifier stashes on `result.__dict__["invariant_violations"]`
    # across every corner × temp × setup point. Duplicates are
    # de-duplicated below: if the same role check produces an identical
    # JSON dict at multiple sim points (e.g. measure_missing on the
    # same testbench across all 5 corners), only the first copy is
    # kept. The dedup key is the JSON-serialized form so equivalent
    # entries collapse regardless of dict ordering.
    job_violations: list[dict] = []
    job_uncertified: list[str] = []   # functional-role certification non-passes
    core_preview_results = []
    core_preview_job_previews = []
    for corner in corners:
        for temp in temps:
            if multi:
                print(f"\n--- corner={corner} temp={_fmt_temp(temp)} ---")
            if is_multi_setup:
                # v0.0.29 (G4): multi-setup path. Run each setup
                # (with its drive-mode rewrite), then derive top-level
                # measures and apply top-level assertions.
                multi_setup_kwargs = {
                    "tb_builder": tb_builder,
                    "cell": cell,
                    "setups": job.setups,
                    "derived_measures": job.derived_measures,
                    "top_spec": job.spec,
                    "pdk_corner": corner,
                    "temp": temp,
                }
                if simulator_driver is not None:
                    multi_setup_kwargs["driver"] = simulator_driver
                ms_result = verify_multi_setup(**multi_setup_kwargs)
                _print_multi_setup_result(ms_result)
                if core_preview_enabled:
                    core_preview_job_previews.append(
                        preview_from_multisetup_result(ms_result)
                    )
                if boundary_evidence_enabled and boundary_evidence_reports is not None:
                    boundary_evidence_reports.append(
                        _boundary_evidence_report_from_multisetup_result(
                            ms_result,
                            cell=cell,
                            pdk_id=cell.pdk_id,
                            pdk_corner=corner,
                            temp=temp,
                        )
                    )
                if not ms_result.all_passed:
                    all_ok = False
                # v0.0.87: collect per-setup invariant violations.
                # Each setup's VerifyResult carries its own stash from
                # the in-process Stage 3 check; we union them all.
                job_uncertified.extend(_uncertified_reasons(ms_result))
                for setup_result in ms_result.per_setup.values():
                    job_violations.extend(
                        _violations_of(setup_result),
                    )
                    job_uncertified.extend(_uncertified_reasons(setup_result))
            else:
                tb = tb_builder(cell)
                # Override pdk_corner and temp AFTER the testbench is built so
                # the template's defaults are replaced for this point only.
                tb.pdk_corner = corner
                tb.temp = temp
                if simulator_driver is None:
                    result = verify_testbench(tb)
                else:
                    result = verify_testbench(tb, driver=simulator_driver)
                if boundary_evidence_enabled and boundary_evidence_reports is not None:
                    boundary_evidence_reports.append(
                        _boundary_evidence_report_from_verify_result(
                            result,
                            cell=cell,
                            pdk_id=cell.pdk_id,
                            pdk_corner=corner,
                            temp=temp,
                            specification_measure_names=[
                                field.name for field in cell.spec.fields
                            ],
                        )
                    )
                if core_preview_enabled:
                    core_preview_results.append(result)
                for k, v in result.measures.items():
                    print(f"  measured {k}: {v:g}")
                # v0.0.53: capture per-port Vref samples from the
                # "headline" find-at-vdd_nom measures (the suffix-free
                # entries — vref1, vref2 — emitted by
                # `_vref_measures` in bandgap_testbench.py) into the
                # TC aggregator. The _max / _min / _avg variants on
                # the same port are intentionally skipped because they
                # are aggregates of a trivial 3-point VDD sweep at the
                # SAME temp — not independent samples for the cross-
                # temp TC computation.
                if is_bandgap and len(temps) >= 2:
                    port_bucket = bandgap_vref_by_corner.setdefault(
                        corner, {},
                    )
                    headline_vrefs = {
                        p for p in find_vref_ports(cell)
                    }
                    for k, v in result.measures.items():
                        if k in headline_vrefs:
                            port_bucket.setdefault(k, []).append(
                                (float(temp), float(v))
                            )
                for a in result.assertions:
                    mark = "PASS" if a.passed else "FAIL"
                    print(f"  [{mark}] {a.message}")
                if not result.all_passed:
                    all_ok = False
                # v0.0.87: collect this point's invariant violations.
                job_violations.extend(_violations_of(result))
                job_uncertified.extend(_uncertified_reasons(result))
    # v0.0.53: emit TC summary per (corner, vref port) after every
    # temp ran. ppm/°C is the textbook bandgap figure
    # (`ppavl/avgval/dT * 1e6`) — printed alongside the per-temp
    # measures so the user gets a single-number FOM.
    if is_bandgap and bandgap_vref_by_corner:
        for corner, port_samples in bandgap_vref_by_corner.items():
            for port, samples in port_samples.items():
                if len(samples) < 2:
                    continue
                tc_ppm = _bandgap_tc_ppm_per_c(samples)
                if tc_ppm is None:
                    continue
                label = f"{port}_tc_ppm_per_c"
                if len(corners) > 1:
                    label = f"{corner}_{label}"
                print(f"  measured {label}: {tc_ppm:g}")
    # v0.0.87: dedup the collected violations. The same role check
    # runs at every (corner, temp, setup) point; if all reports are
    # bit-identical we surface only one copy in the marker block so
    # the consumer sees N distinct findings, not N × corners copies.
    deduped = _dedup_violations(job_violations)
    # Functional-role invariant violations + obligation-certification non-passes
    # are KERNEL constraints, not metadata -> they gate the verify verdict and the
    # exit code, not just the legacy spec assertions (which see only the simulator
    # + SpecField pass/fail). A spec pass with an invariant mismatch or an
    # unverifiable invariant is NOT a clean verify.
    if deduped or job_uncertified:
        all_ok = False
    for _reason in dict.fromkeys(job_uncertified):   # stable de-dup, surface once
        print(f"  [UNCERTIFIED] {_reason}")
    if core_preview_enabled and core_preview_job_previews:
        if len(core_preview_job_previews) == 1:
            return all_ok, deduped, core_preview_job_previews[0]
        return all_ok, deduped, preview_from_certificate_previews(
            core_preview_job_previews
        )
    if core_preview_enabled and core_preview_results:
        return all_ok, deduped, preview_from_verify_results(core_preview_results)
    return all_ok, deduped, None


def _uncertified_reasons(result) -> list[str]:
    """Reasons the functional-role / obligation certification did NOT clean-pass.

    Functional-role invariants are KERNEL constraints, not metadata: if the
    obligation certification failed, or the invariant runtime could not even run
    the check, the verify result must NOT be reported as a clean pass (and the
    exit code must reflect it) -- a spec-assertion pass does not certify the
    functional contract. Returns [] when the cell carries no obligations or the
    certification cleanly passed."""
    d = getattr(result, "__dict__", {})
    reasons: list[str] = []
    err = d.get("_invariant_runtime_error")
    if err:
        reasons.append(f"functional invariants could not be verified: {err}")
    if d.get("certification_status") == "fail":
        reasons.append("obligation certification failed (functional-role invariant)")
    return reasons


def _violations_of(result) -> list[dict]:
    """Extract invariant violations from a VerifyResult-shaped object.

    The verifier stashes the list on `result.__dict__["invariant_violations"]`
    (it's a runtime attribute, not a dataclass field). Each entry is an
    `InvariantViolation`; we render it via `to_json_dict()` for the
    marker payload. Returns an empty list when the stash is absent or
    empty (cell carries no functional roles, or the soft-failure path
    in `verifier.py` cleared the list).
    """
    if result is None:
        return []
    raw = getattr(result, "__dict__", {}).get("invariant_violations")
    if raw is None:
        # Also try direct attribute access — some result shapes might
        # add it as a normal field in the future. Either path lands
        # the same list.
        raw = getattr(result, "invariant_violations", None)
    if not raw:
        return []
    out: list[dict] = []
    for v in raw:
        out.append(_violation_to_json_dict(v))
    return out


def _dedup_violations(items: list[dict]) -> list[dict]:
    """Stable-order dedup of violation dicts by JSON-canonical equality.

    Two entries that JSON-serialize to the same string (with sorted
    keys) are considered equivalent — the second copy is dropped. The
    relative order of the first occurrences is preserved so log lines
    stay deterministic across runs.

    Returns a new list; does not mutate `items`.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for it in items:
        try:
            key = json.dumps(it, sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            # Defensive: if the entry can't be JSON-serialized (should
            # never happen for InvariantViolation dataclasses), fall
            # back to repr — the goal here is stable dedup, not perfect
            # equality semantics.
            key = repr(it)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _bandgap_tc_ppm_per_c(
    samples: list[tuple[float, float]],
) -> float | None:
    """Compute the temperature coefficient in ppm/°C from per-temp Vref.

    `samples` is a list of `(temp_celsius, vref_volts)` tuples gathered
    across a `temps { ... }` sweep. Returns `(V_pp / V_avg) / dT * 1e6`
    — the AnalogGym `.measure dc TC param='ppavl/avgval/dT'` convention
    scaled into parts-per-million. Returns `None` when the dataset is
    degenerate (single point, zero average, zero temperature span, or
    non-finite reference value), so the caller can suppress the line
    rather than print a meaningless `inf` / `nan`.
    """
    import math
    if len(samples) < 2:
        return None
    temps_c = [t for t, _ in samples]
    vrefs = [v for _, v in samples]
    if any(not math.isfinite(v) for v in vrefs):
        return None
    v_pp = max(vrefs) - min(vrefs)
    v_avg = sum(vrefs) / len(vrefs)
    dT = max(temps_c) - min(temps_c)
    if dT == 0 or v_avg == 0:
        return None
    return abs(v_pp / v_avg / dT) * 1e6


def _print_multi_setup_result(ms: MultiSetupResult) -> None:
    """Pretty-print a MultiSetupResult to stdout.

    Format:

        [setup adm]
          measured gain_dc: 27.3
          measured gbw: 1.8e6
          [PASS] gain_dc: actual=27.3, expected >= 25
        [setup acm]
          measured gain_dc: -42.1
          [PASS] gain_dc: actual=-42.1, expected <= 0
        derived cmrr: 69.4
          [PASS] cmrr: actual=69.4, expected >= 60
    """
    for name, result in ms.per_setup.items():
        print(f"  [setup {name}]")
        for k, v in result.measures.items():
            print(f"    measured {k}: {v:g}")
        for a in result.assertions:
            mark = "PASS" if a.passed else "FAIL"
            print(f"    [{mark}] {a.message}")
        if result.simulator_error is not None:
            print(f"    SIM ERROR: {result.simulator_error}")
    if ms.derived_measures or ms.derived_assertions or ms.derived_errors:
        print("  [derived]")
        for k, v in ms.derived_measures.items():
            print(f"    measured {k}: {v:g}")
        for k, err in ms.derived_errors.items():
            print(f"    [FAIL] {k}: eval error — {err}")
        for a in ms.derived_assertions:
            mark = "PASS" if a.passed else "FAIL"
            print(f"    [{mark}] {a.message}")


def _fmt_temp(t: float) -> str:
    """Compact format for temperature values in CLI output.

    Integer-valued temps print without a decimal point (`27` not `27.0`).
    Fractional temps keep enough precision to disambiguate.
    """
    if float(t).is_integer():
        return f"{int(t)}"
    return f"{t:g}"


def cmd_info(args: argparse.Namespace) -> int:
    """Print registered capabilities: stages, PDKs, corners, version."""
    from . import __version__
    from .stage_templates import list_styles, get_stage, get_stage_testbench

    print(f"Clay v{__version__}")
    print()

    print("Registered stage templates:")
    for style in list_styles():
        builder = get_stage(style)
        tb = get_stage_testbench(style)
        tb_note = builder.__name__ if not tb else f"{builder.__name__} + {tb.__name__}"
        print(f"  {style:<30} {tb_note}")
    print()

    # Registered PDKs. Force-import all submodules of clay.pdk first so each
    # adapter registers via its module-side register_pdk(...) call.
    print("Registered PDKs:")
    try:
        from . import pdk as _pdk_pkg
        import importlib, pkgutil
        for _, modname, _ in pkgutil.iter_modules(_pdk_pkg.__path__):
            try:
                importlib.import_module(f"clay.pdk.{modname}")
            except Exception:
                pass  # adapter failed to import — skip silently
        from .pdk import list_pdks, get_pdk
        for pdk_id in list_pdks():
            adapter = get_pdk(pdk_id)
            supply = (
                getattr(adapter, "supply_voltage", None)
                or getattr(adapter, "vdd_nom", None)
            )
            # 5 standard corners for both sky130 and gf180mcu; if an adapter
            # surfaces its own set (via extra["valid_corners"], used by open_pdk
            # to expose uppercase TT/SS/...), prefer that.
            extra = getattr(adapter, "extra", {}) or {}
            corners = (
                getattr(adapter, "valid_corners", None)
                or extra.get("valid_corners")
                or ["tt", "ss", "ff", "sf", "fs"]
            )
            corner_list = ", ".join(corners)
            print(f"  {pdk_id:<12} supply={supply} V, corners=[{corner_list}]")
    except Exception as exc:
        print(f"  (could not enumerate PDKs: {exc})")
    print()

    return 0


def cmd_onboard(args: argparse.Namespace) -> int:
    """Map a foreign SPICE .cir onto the Clay paradigm (a design.clay draft).

    Chains infer_roles -> emit_design_clay (+ optional PDK-retarget). Emits
    advisory WARNINGS (never fails) when the intent cannot be inferred. Exit 0 on
    success (even with warnings); non-zero only on a hard read/parse failure."""
    import warnings as _warnings
    from clay.frontend.onboard import onboard

    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")  # we print them ourselves, structured
            result = onboard(
                args.cir,
                package=args.package or f"onboard.{Path(args.cir).stem}",
                pdk=args.pdk,
                cell=args.cell,
                retarget=args.retarget,
                tb=args.tb or [],
                emit_warnings=False,
                auto_craft=getattr(args, "auto_craft", True),
                calibrate_op=getattr(args, "calibrate_op", True),
            )
    except Exception as exc:  # noqa: BLE001
        print(f"onboard: hard failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.case:
        from clay.frontend.onboard import scaffold_case
        case = scaffold_case(result, case_dir=args.case, cir=args.cir,
                             top_cell=result.cell, pdk=args.pdk,
                             tb=args.tb or ())
        print(f"scaffolded draft case at {case} "
              f"(source/design.clay + case.yaml binding {args.cir}); "
              f'next: clay flow check --case "{case}" --until source --json', file=sys.stderr)
    elif args.out:
        Path(args.out).write_text(result.design_clay, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(result.design_clay)

    d = result.decomposition
    print(
        f"\n# onboard: {d.cell} — {len(d.devices)} devices, {len(d.roles)} role(s), "
        f"{len(d.matched_pairs)} matched pair(s), axis={d.mirror_axis}; "
        f"intent_inferred={result.intent_inferred}",
        file=sys.stderr,
    )
    for w in result.warnings:
        print(f"⚠ WARNING [{w.code}]: {w.message}", file=sys.stderr)
    for n in result.notes:
        print(f"  note: {n}", file=sys.stderr)
    if result.retargeted_params:
        print(f"  retargeted {len(result.retargeted_params)} param(s)", file=sys.stderr)
    return 0


def cmd_pdk_characterize(args: argparse.Namespace) -> int:
    """Derive a PDK's porting parameters (Vdd / Lmin / Vth / KP) from its model
    cards, so a newly-added PDK is portable to without a hand-coded preset."""
    from clay.frontend.pdk_characterize import characterize_pdk
    try:
        c = characterize_pdk(args.dir, name=args.name)
    except Exception as exc:  # noqa: BLE001
        print(f"pdk-characterize: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    p = c.params
    print(f"PDK {c.pdk}  (mode={c.mode})")
    print(f"  Vdd   = {p.vdd} V")
    print(f"  Lmin  = {p.lmin} um")
    print(f"  Vth_n = {p.vth_n:.3f} V   Vth_p = {p.vth_p:.3f} V")
    print(f"  KP_n  = {p.kp_n:.1f} uA/V^2   KP_p = {p.kp_p:.1f} uA/V^2")
    for k in ("nmos", "pmos"):
        pr = c.provenance.get(k, {})
        print(f"  {k}: model {pr.get('model')} <- {pr.get('file')}")
    for n in c.notes:
        print(f"  ! {n}", file=sys.stderr)
    return 0


def cmd_pdk_add(args: argparse.Namespace) -> int:
    """Admit a real PDK into Clay's flow: characterize (Vdd/Lmin/Vth/KP) +
    optionally sim-refine Vth/KP by ngspice + discover the harness adapter +
    parse the DRC deck into PdkRules. One call so retarget / tune / layout /
    signoff can target the new PDK. Honest about parsed-vs-defaulted + sim cover."""
    from clay.frontend.pdk_register import register_pdk
    try:
        reg = register_pdk(args.dir)
    except Exception as exc:  # noqa: BLE001
        print(f"pdk add: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    char = reg.characterization
    p = char.params
    sim_note = "static model-card parse"
    if args.sim:
        from clay.frontend.pdk_sim_characterize import refine_characterization
        char = refine_characterization(char, args.dir)
        p = char.params
        # refine returns unchanged (mode stays model_card) when sim is unavailable.
        sim_note = ("sim-refined Vth/KP" if char.mode == "sim_refined"
                    else "sim unavailable -> static parse")

    print(f"PDK {reg.pdk_id}")
    print(f"  params: Vdd={p.vdd}V  Lmin={p.lmin}um  "
          f"Vth_n={p.vth_n:.3f}  Vth_p={p.vth_p:.3f}  "
          f"KP_n={p.kp_n:.1f}  KP_p={p.kp_p:.1f}  [{sim_note}]")
    ad = reg.adapter
    print(f"  adapter: marker={ad.get('marker')!r}  corner={ad.get('corner_tt')!r}")
    print(f"           model_lib={ad.get('model_lib')}")
    rep = reg.report or {}
    parsed = rep.get("parsed", rep.get("parsed_from_deck", []))
    defaulted = rep.get("defaulted", [])
    print(f"  rules:   deck={rep.get('deck') or rep.get('source')}  "
          f"dialect={rep.get('dialect')}")
    print(f"           {len(parsed)} parsed-from-deck, {len(defaulted)} defaulted")
    for n in char.notes:
        print(f"  ! {n}", file=sys.stderr)
    return 0


def cmd_pdk_drc_deck(args: argparse.Namespace) -> int:
    """Auto-generate a FREE KLayout DRC deck from a commercial PDK's OWN parsed
    foundry rule values + its GDS layer map -- the deck a closed foundry PDK never
    ships for free tools. Optionally run it (KLayout) on a GDS. Partial core-rule
    DRC (width/space/size), NOT full tapeout sign-off (that needs external_drc)."""
    from pathlib import Path as _P
    from clay.frontend.pdk_register import register_pdk
    from clay.frontend.pdk_layermap import parse_layermap
    from clay.layout.klayout_deck_gen import klayout_deck_from_rules, run_klayout_drc
    try:
        lm = parse_layermap(args.dir)
        rules = register_pdk(args.dir).report.get("parsed", {})
        if not rules:
            print(f"pdk-drc-deck: no rules parsed from {args.dir} (gf180/sky130 ship their "
                  "own KLayout deck; this path targets commercial external_drc-deck PDKs).",
                  file=sys.stderr)
            return 2
        deck = klayout_deck_from_rules(rules, lm, pdk_name=_P(args.dir).name)
    except Exception as exc:  # noqa: BLE001
        print(f"pdk-drc-deck: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    out = args.out or f"{_P(args.dir).name}.drc"
    _P(out).write_text(deck, encoding="utf-8")
    n = deck.count(".output(")
    print(f"wrote {out}  ({n} core rules from real foundry values; layers {dict(lm)})")
    for k, v in rules.items():
        print(f"  {k} = {v.get('value_um')}um")
    if args.run and args.gds:
        res = run_klayout_drc(args.gds, out, report_path=args.report)
        print(f"KLayout DRC on {args.gds}: ran={getattr(res,'ran',None)} "
              f"violations={getattr(res,'violation_count',None)}")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    """Sim-in-loop sizing tune: real tSS-BO + real ngspice push a circuit's
    sizing toward KPI targets (the closed loop). Needs the tSS-BO clone + a
    simulator; reports a clear message and exits non-zero when unavailable."""
    import re as _re
    from clay.frontend.presim_evaluate import run_presim_tune

    try:
        from tools.clay_optimize.tssbo_solver import tssbo_available
        if not tssbo_available():
            print("tune: tSS-BO unavailable — run python tools/clay_optimize/fetch_tssbo.py "
                  "and pip install botorch gpytorch torch", file=sys.stderr)
            return 2
    except Exception as exc:  # noqa: BLE001
        print(f"tune: BO backend import failed: {exc}", file=sys.stderr)
        return 2

    kpi_map = {}
    for item in (args.kpi or []):
        if "=" in item:
            metric, key = item.split("=", 1)
            kpi_map[metric.strip()] = key.strip()
    spec = {}
    for item in (args.target or []):
        m = _re.match(r"\s*(\w+)\s*(>=|<=|>|<|~)\s*([-+0-9.eE]+)\s*$", item)
        if m:
            spec[m.group(1)] = (m.group(2), float(m.group(3)))
    if not kpi_map or not spec:
        print("tune: need at least one --kpi metric=presim_key and one --target 'metric>=value'",
              file=sys.stderr)
        return 2
    tunable = [t.strip() for t in (args.tunable or "").split(",") if t.strip()]

    try:
        r = run_presim_tune(
            presim_module=args.presim, source_attr=args.source_attr,
            kpi_map=kpi_map, spec_targets=spec, cir=args.cir, tunable=tunable,
            rel_window=args.rel_window, n_evals=args.n_evals,
            batch_size=args.batch_size, n_candidates=args.n_candidates, seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"tune: failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"baseline KPIs: { {k: round(v,4) for k,v in r['baseline_kpis'].items()} }")
    print(f"tuned params : {r['tuned_params']}")
    print(f"tuned KPIs   : { {k: round(v,4) for k,v in r['tuned_kpis'].items()} }")
    print(f"loss {r['seed_loss']:.4f} -> {r['tuned_loss']:.4f}  "
          f"({'TARGETS MET' if r['met'] else 'improved' if r['tuned_loss'] < r['seed_loss'] else 'no improvement'})")
    return 0




def cmd_case_validate(args: argparse.Namespace) -> int:
    from .casepkg import validate_case_package

    result = validate_case_package(args.path)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status}: {result.case_id or args.path} ({result.maturity})")
        for check in result.checks:
            print(f"  [{check.status.upper()}] {check.name}: {check.message}")
    return 0 if result.passed else 1


def cmd_case_run(args: argparse.Namespace) -> int:
    from .casepkg import run_case_package

    payload = run_case_package(args.path, execute=args.execute)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "PASS" if payload["passed"] else "FAIL"
        mode = "execute" if payload["execute"] else "dry-run"
        print(f"{status}: {payload.get('case_id') or args.path} ({mode})")
        for command in payload["commands"]:
            print(f"  [{command['name']}] " + " ".join(command["argv"]))
    return 0 if payload["passed"] else 1


def cmd_case_list(args: argparse.Namespace) -> int:
    from .casepkg import (
        discover_case_packages,
        discover_example_records,
        load_seed_catalog,
        records_to_json,
    )

    records = discover_case_packages(args.root)
    if args.catalog:
        catalog_path = Path(args.catalog_path)
        if catalog_path.exists():
            records.extend(load_seed_catalog(catalog_path))
    if args.examples:
        records.extend(discover_example_records(args.examples_root))

    records = sorted(records, key=lambda record: (record.maturity, record.case_id))
    if args.json:
        print(json.dumps(records_to_json(records), indent=2, sort_keys=True))
    else:
        for record in records:
            source = f" {record.source_clay}" if record.source_clay else ""
            print(f"{record.case_id:<48} {record.maturity:<16} {record.title}{source}")
    return 0


def cmd_case_audit(args: argparse.Namespace) -> int:
    from .casepkg import audit_maturity, build_case_inventory

    inventory = build_case_inventory(
        args.repo_root,
        cases_root=args.cases_root,
        catalog_path=args.catalog_path,
        examples_root=args.examples_root,
    )
    payload = inventory.to_dict()
    # Evidence-driven maturity: computed-from-artifacts vs hand-declared, so the
    # audit surfaces invalid declarations + overclaims instead of trusting case.yaml.
    payload["maturity_audit"] = audit_maturity(
        args.repo_root, cases_root=args.cases_root
    ).to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    if args.json:
        print(text)
    else:
        summary = payload["summary"]
        print(f"total: {summary['total']}")
        print("maturity:")
        for key, value in summary["maturity"].items():
            print(f"  {key}: {value}")
        print("origins:")
        for key, value in summary["origins"].items():
            print(f"  {key}: {value}")
        ma = payload["maturity_audit"]["summary"]
        print("maturity audit (evidence-computed vs declared):")
        print(f"  invalid declared: {ma['invalid_declared']} {ma['invalid_ids'] or ''}")
        print(f"  drift: {ma['drift']}  overclaimed: {ma['overclaimed']}")
        for cid in ma["overclaimed_ids"]:
            print(f"    overclaim: {cid}")
        if args.out:
            print(f"wrote: {args.out}")
    return 0


def _resolve_flow_spec(circuit: str):
    """Resolve a CircuitSpec from the discovered registry.

    Returns ``(spec, error_message)``. ``spec`` is None when the registry is
    not yet implemented (WS-D) or the circuit is unknown; the error message is
    a clear, non-crashing explanation for the caller to print + exit non-zero.
    """
    from .flow.spec import discover_registry

    try:
        registry = discover_registry()
    except NotImplementedError:
        return None, (
            "flow registry pending: clay.flow.spec.discover_registry() is not "
            "implemented yet (WS-D owns the flow/quality language lowering). "
            "Once case packages carry flow {}/quality {} blocks, "
            "`clay flow run <circuit>` resolves them automatically."
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"could not discover flow registry: {type(exc).__name__}: {exc}"
    spec = registry.get(circuit)
    if spec is None:
        known = ", ".join(sorted(registry)) or "(none)"
        return None, f"unknown circuit {circuit!r}; known: {known}"
    return spec, None


def cmd_flow_check(args: argparse.Namespace) -> int:
    from .flow.intent_readiness import check_intent_readiness

    result = check_intent_readiness(args.case, project_root=args.project_root, until=args.until)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"{'READY' if result.ready else 'BLOCKED'}: {args.until} preflight (no physical admission)")
        for row in result.checks:
            print(f"  [{row.status}] {row.code}: {row.message}")
    return 0 if result.ready else 1


def _resolve_explicit_flow_spec(case_dir, project_root=None):
    """Compile one declared case through the existing executable spec compiler."""
    from .casepkg import load_case_manifest
    from .flow.spec import compile_flow_spec
    from .parser import parse_clay_file

    case = Path(case_dir).resolve()
    if not case.is_dir():
        raise ValueError("case directory does not exist")
    if project_root is not None and not case.is_relative_to(Path(project_root).resolve()):
        raise ValueError("case directory must stay inside project-root")
    manifest = load_case_manifest(case)
    if manifest.get("schema") != "clay.case/v1":
        raise ValueError("expected clay.case/v1")
    source = manifest.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("clay"), str):
        raise ValueError("case must declare source.clay")
    source_path = (case / source["clay"]).resolve()
    if not source_path.is_relative_to(case):
        raise ValueError("source.clay must stay inside the case package")
    # Existing executable producers consume this path; compiling a different
    # source here would bind one declaration and execute another.
    if source_path != (case / "source" / "design.clay").resolve():
        raise ValueError("executable case source.clay must be source/design.clay")
    parsed = parse_clay_file(source_path)
    top = source.get("top_cell")
    cells = [cell for cell in parsed.cell_defs
             if (top is None or cell.local_name == top)
             and cell.flow_decl is not None and cell.quality_decl is not None]
    if len(cells) != 1:
        raise ValueError("case must select exactly one cell with flow and quality declarations")
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("case metadata must be a mapping")
    metadata = dict(metadata)
    metadata.setdefault("id", manifest.get("id"))
    metadata.setdefault("top_cell", top)
    return compile_flow_spec(
        cells[0], case_dir=case, verify_jobs=parsed.verify_jobs,
        policies=parsed.policies, discover_layout=False, inspect_artifacts=True,
        metadata=metadata,
    )


def _flow_execution_options(args):
    """Only explicitly supplied controls reach run_flow (legacy mocks stay valid)."""
    from .flow.execution_budget import ExecutionBudget
    from .flow.run_admission import RetryRequest

    options = {}
    wall_seconds = getattr(args, "wall_seconds", None)
    if wall_seconds is not None:
        options["budget"] = ExecutionBudget(wall_seconds=wall_seconds)
    stage_unwind = getattr(args, 'stage_unwind_seconds', None)
    if stage_unwind is not None:
        options['stage_unwind_seconds'] = stage_unwind
    role_trace = getattr(args, 'stage_role_trace_seconds', None)
    if role_trace is not None:
        from .flow.execution_budget import _validate_seconds
        role_trace = _validate_seconds(role_trace, 'stage_role_trace_seconds')
        if getattr(args, 'only', None) != 'primitive-observe' or wall_seconds is None:
            raise ValueError('stage role trace requires --only primitive-observe and --wall-seconds')
        if role_trace > options['budget'].wall_seconds:
            raise ValueError('role trace budget must fit inside the stage wall budget')
        options['stage_role_trace_seconds'] = role_trace
    for name in ("project_root", "task_id"):
        value = getattr(args, name, None)
        if value is not None:
            options[name] = Path(value) if name == "project_root" else value
    receipt = getattr(args, "retry_receipt", None)
    reason = getattr(args, "retry_reason", None)
    evidence = getattr(args, "retry_evidence", None)
    if receipt is not None or reason is not None or evidence is not None:
        if receipt is None or reason is None or not evidence:
            raise ValueError("retry requires --retry-receipt, --retry-reason and --retry-evidence")
        options["retry"] = RetryRequest(
            prior_receipt_sha256=receipt, reason=reason,
            evidence=tuple(Path(path) for path in evidence),
        )
    stage_receipt = getattr(args, 'stage_retry_receipt', None)
    stage_reason = getattr(args, 'stage_retry_reason', None)
    stage_evidence = getattr(args, 'stage_retry_evidence', None)
    if stage_receipt is not None or stage_reason is not None or stage_evidence is not None:
        if stage_receipt is None or stage_reason is None or not stage_evidence:
            raise ValueError('stage retry requires --stage-retry-receipt, --stage-retry-reason and --stage-retry-evidence')
        options['stage_retry'] = RetryRequest(prior_receipt_sha256=stage_receipt,
            reason=stage_reason, evidence=tuple(Path(path) for path in stage_evidence))
    return options


def _flow_command_error(reason, message):
    print(json.dumps({"schema": "clay.flow-command/v1", "status": "blocked",
                      "reason": reason, "message": message,
                      "physical_admission": False}, ensure_ascii=False, indent=2, sort_keys=True))
    return 2


def cmd_flow_recover_precreate(args: argparse.Namespace) -> int:
    from .api import recover_precreate_primitive
    from .flow.execution_budget import ExecutionBudget, ExecutionStopped
    from .flow.run_admission import RunAdmissionError
    try:
        spec = _resolve_explicit_flow_spec(args.case, args.project_root)
        result = recover_precreate_primitive(spec, project_root=args.project_root,
            task_id=args.task_id, preparation_manifest=args.preparation_manifest,
            budget=ExecutionBudget(wall_seconds=args.wall_seconds))
    except (RunAdmissionError, ExecutionStopped, ValueError, OSError) as exc:
        return _flow_command_error('PRIMITIVE_RECOVERY_BLOCKED', str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_flow_run(args: argparse.Namespace) -> int:
    case = getattr(args, "case", None)
    circuit = getattr(args, "circuit", None)
    if sum(bool(value) for value in (case, circuit, getattr(args, "all", False))) != 1:
        return _flow_command_error("FLOW_SELECTION_INVALID", "provide exactly one circuit key, --case or --all")
    try:
        options = _flow_execution_options(args)
    except (TypeError, ValueError) as exc:
        return _flow_command_error("FLOW_OPTIONS_INVALID", str(exc))
    if getattr(args, "all", False):
        if getattr(args, 'only', None) == 'primitive-observe':
            return _flow_command_error('FLOW_OPTIONS_REQUIRE_SINGLE_CASE',
                                       'primitive observation applies to one circuit or --case per invocation')
        if ("task_id" in options or "retry" in options or "stage_retry" in options
                or 'stage_role_trace_seconds' in options):
            return _flow_command_error("FLOW_OPTIONS_REQUIRE_SINGLE_CASE",
                                       "task identity and retry controls apply to one circuit or --case per invocation")
        return _cmd_flow_run_all(args, **options)

    from .flow.engine import FlowSelectionError, run_flow
    from .flow.execution_budget import ExecutionStopped
    from .flow.failure_diagnostics import summarize_failure
    from .flow.run_admission import RunAdmissionError

    if case is not None:
        try:
            spec = _resolve_explicit_flow_spec(case, getattr(args, "project_root", None))
        except Exception as exc:  # source/manifest errors are command diagnostics
            return _flow_command_error("CASE_INPUT_INVALID", str(exc))
    else:
        spec, error = _resolve_flow_spec(circuit)
        if spec is None:
            return _flow_command_error("FLOW_SELECTION_INVALID", error)
    out_dir = Path(args.out) if args.out else (spec.case_dir / "_flow_run")
    try:
        result = run_flow(
            spec,
            out_dir=out_dir,
            only=([args.only] if args.only else None),
            force=args.force,
            **options,
        )
    except RunAdmissionError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 1
    except FlowSelectionError as exc:
        return _flow_command_error("STEP_SELECTION", str(exc))
    except ExecutionStopped as exc:
        if getattr(args, 'only', None) != 'primitive-observe':
            raise
        print(json.dumps({'schema': 'clay.flow-command/v1', 'status': 'incomplete',
                          **exc.flow_error_data(), 'cleanup': 'UNKNOWN', 'physical_admission': False,
                          'notes': [note[:512] for note in getattr(exc, '__notes__', ())
                                    if isinstance(note, str)][:8]}, ensure_ascii=False, indent=2, sort_keys=True))
        return 1
    payload = dict(result.to_dict())
    payload["failure_summary"] = summarize_failure(result)
    if hasattr(result, "execution_control"):
        payload["execution_control"] = result.execution_control
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.passed else 1


def _cmd_flow_run_all(args: argparse.Namespace, **controls) -> int:
    """Run the flow for every registered circuit (`clay flow run --all`)."""
    from .flow.engine import run_portfolio

    out_dir = Path(args.out) if args.out else (Path("demoout") / "flow_portfolio")
    report = run_portfolio(
        out_dir=out_dir,
        only=([args.only] if args.only else None),
        force=args.force,
        **controls,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["summary"]["n_fail"] == 0 else 1

    circuits = report["circuits"]
    summary = report["summary"]
    only_note = (
        f"  (only: {', '.join(summary['only'])})" if summary.get("only") else ""
    )
    print(f"Portfolio flow over {summary['n_circuits']} circuit(s){only_note}")
    # Gather the union of step names actually run, in canonical order.
    from .flow.spec import DEFAULT_STEPS

    seen_steps: list[str] = []
    for entry in circuits.values():
        for step_name in entry.get("per_step_status", {}):
            if step_name not in seen_steps:
                seen_steps.append(step_name)
    ordered = [s for s in DEFAULT_STEPS if s in seen_steps]
    ordered += [s for s in seen_steps if s not in ordered]

    key_w = max([len("circuit")] + [len(k) for k in circuits]) + 2
    verdict_w = 8
    header = f"{'circuit':<{key_w}}{'verdict':<{verdict_w}}" + "  ".join(ordered)
    print(header)
    print("-" * len(header))
    for key in sorted(circuits):
        entry = circuits[key]
        per = entry.get("per_step_status", {})
        cells = "  ".join(
            f"{_status_glyph(per.get(s)):<{max(len(s), 4)}}" for s in ordered
        )
        print(f"{key:<{key_w}}{entry['verdict']:<{verdict_w}}{cells}")
        if entry.get("error"):
            print(f"    error: {entry['error']}")
    print("-" * len(header))
    print(
        f"pass={summary['n_pass']}  fail={summary['n_fail']}  "
        f"circuits={summary['n_circuits']}"
    )
    return 0 if summary["n_fail"] == 0 else 1


def _status_glyph(status: str | None) -> str:
    """Compact per-step status glyph for the portfolio/status tables."""
    return {
        "pass": "ok",
        "warn": "warn",
        "fail": "FAIL",
        "skipped": "skip",
        None: "-",
    }.get(status, status or "-")


def cmd_status(args: argparse.Namespace) -> int:
    """One-screen project status: per-circuit computed maturity + last verdict."""
    from .casepkg import assess_maturity
    from .flow.spec import discover_registry

    try:
        registry = discover_registry()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not discover flow registry: {exc}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for key in sorted(registry):
        spec = registry[key]
        try:
            maturity = assess_maturity(spec.case_dir)
        except Exception as exc:  # noqa: BLE001
            maturity = f"error:{type(exc).__name__}"
        verdict = _last_flow_verdict(spec)
        rows.append(
            {
                "circuit": key,
                "display_name": spec.display_name,
                "pdk": spec.pdk,
                "maturity": maturity,
                "last_verdict": verdict,
            }
        )

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if not rows:
        print("no circuits in the flow registry")
        return 0

    key_w = max(len("circuit"), *(len(r["circuit"]) for r in rows)) + 2
    mat_w = max(len("maturity"), *(len(str(r["maturity"])) for r in rows)) + 2
    pdk_w = max(len("pdk"), *(len(r["pdk"]) for r in rows)) + 2
    header = (
        f"{'circuit':<{key_w}}{'pdk':<{pdk_w}}{'maturity':<{mat_w}}{'last_flow_verdict'}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['circuit']:<{key_w}}{r['pdk']:<{pdk_w}}"
            f"{str(r['maturity']):<{mat_w}}{r['last_verdict']}"
        )
    return 0


def _last_flow_verdict(spec) -> str:
    """Read the last persisted flow verdict for a circuit, if any.

    Looks for a ``flow_state.json`` under the conventional run dir
    (``<case_dir>/_flow_run/work``) or the portfolio dir, and derives a verdict
    from the recorded per-step statuses (fail if any step failed, else pass).
    Returns ``"-"`` when no flow has ever run for the circuit.
    """
    candidates = [
        spec.case_dir / "_flow_run" / "work",
        Path("demoout") / "flow_portfolio" / spec.key / "work",
    ]
    for work_dir in candidates:
        state_file = work_dir / "flow_state.json"
        if not state_file.is_file():
            continue
        try:
            doc = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        steps = doc.get("steps") if isinstance(doc, dict) else None
        if not isinstance(steps, dict) or not steps:
            continue
        statuses = [
            (v or {}).get("status") for v in steps.values() if isinstance(v, dict)
        ]
        if any(s == "fail" for s in statuses):
            return "fail"
        return "pass"
    return "-"


def cmd_flow_list(args: argparse.Namespace) -> int:
    from .flow.spec import discover_registry

    try:
        registry = discover_registry()
    except NotImplementedError:
        print(
            "flow registry pending: no flow/quality blocks compiled yet "
            "(WS-D owns discover_registry).",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not discover flow registry: {exc}", file=sys.stderr)
        return 2
    for key in sorted(registry):
        spec = registry[key]
        print(f"{key:<16} {spec.display_name}  pdk={spec.pdk}  steps={','.join(spec.steps)}")
    return 0


def cmd_flow_status(args: argparse.Namespace) -> int:
    from .flow.state import FlowState

    spec, error = _resolve_flow_spec(args.circuit)
    if spec is None:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    out_dir = Path(args.out) if args.out else (spec.case_dir / "_flow_run")
    work_dir = out_dir / "work"
    state = FlowState(work_dir)
    payload = {
        "circuit": spec.key,
        "work_dir": str(work_dir),
        "steps": {name: state.recorded(name) for name in spec.steps},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_flow_suite(args: argparse.Namespace) -> int:
    """Unified multi-circuit E2E & Docker signoff suite entrypoint (`clay flow suite`)."""
    from scripts.run_four_circuit_e2e_flow import main as _suite_main

    argv: list[str] = []
    if getattr(args, "families", None):
        argv.extend(["--families", *args.families])
    if getattr(args, "cases", None):
        argv.extend(["--cases", *args.cases])
    if getattr(args, "include_audit_only", False):
        argv.append("--include-audit-only")
    if getattr(args, "check_only", False):
        argv.append("--check-only")
    if getattr(args, "run_flow", False):
        argv.append("--run-flow")
    if getattr(args, "force", False):
        argv.append("--force")
    if getattr(args, "out_subdir", None):
        argv.extend(["--out-subdir", str(args.out_subdir)])
    if getattr(args, "docker_signoff", False):
        argv.append("--docker-signoff")
    if getattr(args, "report_json", None):
        argv.extend(["--report-json", str(args.report_json)])
    if getattr(args, "report_md", None):
        argv.extend(["--report-md", str(args.report_md)])
    if getattr(args, "retry_reason", None):
        argv.extend(["--retry-reason", str(args.retry_reason)])
    for ev in getattr(args, "retry_evidence", None) or ():
        argv.extend(["--retry-evidence", str(ev)])
    if getattr(args, "python_bin", None):
        argv.extend(["--python", str(args.python_bin)])
    return int(_suite_main(argv))


def cmd_test(args: argparse.Namespace) -> int:
    """Unified high-speed parallel test entrypoint (`python -m clay test`)."""
    from .testing.runner import run_clay_tests

    return int(
        run_clay_tests(
            tier=str(getattr(args, "tier", "fast")),
            targets=tuple(getattr(args, "targets", ()) or ()),
            workers=getattr(args, "workers", "auto"),
            base=getattr(args, "base", None),
            last_failed=bool(getattr(args, "lf", False)),
            keyword=getattr(args, "keyword", None),
            durations=getattr(args, "durations", None),
            list_only=bool(getattr(args, "list_only", False)),
        )
    )




def cmd_build(args: argparse.Namespace) -> int:
    """Build Clay C++20 native PnR extensions and/or pure C++20 native unit tests."""
    import subprocess

    repo_root = Path(__file__).resolve().parent.parent
    build_script = repo_root / "cpp" / "clay_pnr" / "build.py"
    if not build_script.is_file():
        print(f"FAIL: C++20 build script not found: {build_script}", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(build_script), "--target", str(args.target)]
    if getattr(args, "jobs", None) is not None:
        cmd.extend(["--jobs", str(int(args.jobs))])
    if bool(getattr(args, "force", False)):
        cmd.append("--force")
    if bool(getattr(args, "clean", False)):
        cmd.append("--clean")
    if bool(getattr(args, "werror", False)):
        cmd.append("--werror")
    if bool(getattr(args, "test", False)):
        cmd.append("--test")
    if getattr(args, "test_filter", None):
        cmd.extend(["--filter", str(args.test_filter)])
    return int(subprocess.run(cmd, cwd=str(repo_root)).returncode)


def _signoff_obligation_rows_of_cell(cell) -> list[dict]:
    """Extract production signoff proof rows attached to a compiled cell."""
    raw = getattr(cell, "__dict__", {}).get("_signoff_results")
    if raw is None:
        raw = getattr(cell, "signoff_results", None)
    if raw is None:
        raw = getattr(cell, "__dict__", {}).get("signoff_results")
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    rows: list[dict] = []
    for result in raw:
        rows.extend(signoff_obligation_rows(cell.name, result))
    rows = _append_cross_result_coclean_row(cell.name, rows)
    return rows


def _append_cross_result_coclean_row(cell_name: str, rows: list[dict]) -> list[dict]:
    by_kind = {row.get("obligation_kind"): row for row in rows}
    if "co_clean" in by_kind or "drc" not in by_kind or "lvs" not in by_kind:
        return rows
    drc_row = by_kind["drc"]
    lvs_row = by_kind["lvs"]
    co_clean = drc_row.get("status") == "pass" and lvs_row.get("status") == "pass"
    return rows + [
        _co_clean_obligation_row(
            cell_name,
            status="pass" if co_clean else "fail",
            same_gds_required=False,
            same_gds=None,
            co_clean=co_clean,
            drc_status=str(drc_row.get("status", "error")),
            lvs_status=str(lvs_row.get("status", "error")),
        )
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clay",
        description="Clay: LLM-Native analog circuit description language",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show full tracebacks on errors")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_verify = sub.add_parser(
        "verify",
        help="Parse a .clay file and run its verify jobs",
    )
    p_verify.add_argument("file", help="Path to .clay source file")
    p_verify.add_argument(
        "--corner",
        default=None,
        choices=["tt", "ss", "ff", "sf", "fs"],
        help=(
            "PDK process corner for this verify run (overrides the "
            "testbench's default AND any `corners { }` block declared "
            "in the file — collapses the sweep to this single corner). "
            "One of: tt, ss, ff, sf, fs."
        ),
    )
    p_verify.add_argument(
        "--core-preview",
        action="store_true",
        default=False,
        help="print and attach a read-only Clay Core certificate preview",
    )
    p_verify.add_argument(
        "--boundary-evidence-report",
        action="store_true",
        default=False,
        help="emit a structured runtime boundary evidence report",
    )
    p_verify.add_argument(
        "--offline-measures-json",
        default=None,
        help=argparse.SUPPRESS,
    )
    p_verify.set_defaults(func=cmd_verify)

    p_info = sub.add_parser(
        "info",
        help="Print registered stage templates, PDKs, and version info",
    )
    p_info.set_defaults(func=cmd_info)
    p_onboard = sub.add_parser(
        "onboard",
        help="Map a foreign SPICE .cir onto the Clay paradigm (emit a design.clay draft)",
    )
    p_onboard.add_argument("cir", help="Path to the source SPICE .cir")
    p_onboard.add_argument("--cell", default=None, help="Top cell / subckt name (else inferred)")
    p_onboard.add_argument("--package", default=None, help="Clay package name for the draft")
    p_onboard.add_argument("--pdk", default="gf180mcu", help="Target PDK id")
    p_onboard.add_argument("--retarget", action="store_true",
                           help="Also retarget the .param sizing to the target PDK (analytic)")
    p_onboard.add_argument("--tb", action="append", default=[],
                           help="Testbench .sp (repeatable) -- acceptance KPIs are derived from its Spec/.meas")
    p_onboard.add_argument(
        "--auto-craft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically synthesize layout_intent and annotate directives (default: on)",
    )
    p_onboard.add_argument(
        "--calibrate-op",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Calibrate DC operating-point currents/high-Z/pair metadata (default: on)",
    )
    p_onboard.add_argument("--out", default=None, help="Write the design.clay draft to this path")
    p_onboard.add_argument("--case", default=None,
                           help="Scaffold a runnable case package (source/design.clay + case.yaml binding the .cir) at this dir")
    p_onboard.set_defaults(func=cmd_onboard)

    p_tune = sub.add_parser(
        "tune",
        help="Sim-in-loop sizing tune (real tSS-BO + ngspice) toward KPI targets",
    )
    p_tune.add_argument("cir", help="Path to the source SPICE .cir")
    p_tune.add_argument("--presim", required=True,
                        help="Importable presim module, e.g. circuits.fc_ota.presim")
    p_tune.add_argument("--source-attr", required=True,
                        help="The presim module's source-cir Path constant, e.g. FC_OTA_SOURCE")
    p_tune.add_argument("--kpi", action="append", default=[],
                        help="metric=presim_result_key (repeatable), e.g. gbw_hz=gbw_hz")
    p_tune.add_argument("--target", action="append", default=[],
                        help="'metric>=value' (repeatable), e.g. 'gbw_hz>=13e6'")
    p_tune.add_argument("--tunable", default="",
                        help="Comma-separated params to tune (default: W_*/L_*)")
    p_tune.add_argument("--rel-window", type=float, default=0.5)
    p_tune.add_argument("--n-evals", type=int, default=20)
    p_tune.add_argument("--batch-size", type=int, default=4)
    p_tune.add_argument("--n-candidates", type=int, default=8)
    p_tune.add_argument("--seed", type=int, default=0)
    p_tune.set_defaults(func=cmd_tune)

    p_pdkc = sub.add_parser(
        "pdk-characterize",
        help="Derive a PDK's porting params (Vdd/Lmin/Vth/KP) from its model cards",
    )
    p_pdkc.add_argument("dir", help="Path to a pdks/<name> directory")
    p_pdkc.add_argument("--name", default=None, help="PDK display name (else the dir name)")
    p_pdkc.set_defaults(func=cmd_pdk_characterize)

    p_pdka = sub.add_parser(
        "pdk-add",
        help="Admit a real PDK to Clay: characterize + adapter + DRC-derived PdkRules",
    )
    p_pdka.add_argument("dir", help="Path to a pdks/<name> directory")
    p_pdka.add_argument("--sim", action="store_true",
                        help="Sim-refine Vth/KP by ngspice when the PDK is runnable")
    p_pdka.set_defaults(func=cmd_pdk_add)

    p_pdkd = sub.add_parser(
        "pdk-drc-deck",
        help="Generate a free KLayout DRC deck from a commercial PDK's parsed foundry rules",
    )
    p_pdkd.add_argument("dir", help="Path to a pdks/<name> directory")
    p_pdkd.add_argument("--out", default=None, help="Output .drc path (default <pdk>.drc)")
    p_pdkd.add_argument("--gds", default=None, help="GDS to run the generated deck on")
    p_pdkd.add_argument("--run", action="store_true", help="Run KLayout DRC on --gds")
    p_pdkd.add_argument("--report", default=None, help="lyrdb report path when --run")
    p_pdkd.set_defaults(func=cmd_pdk_drc_deck)

    p_case = sub.add_parser(
        "case",
        help="Manage standardized Clay case packages",
    )
    case_sub = p_case.add_subparsers(dest="case_cmd", required=True)

    p_case_validate = case_sub.add_parser(
        "validate",
        help="Validate a case package manifest and fast local obligations",
    )
    p_case_validate.add_argument("path", help="Path to a case package directory")
    p_case_validate.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable validation report",
    )
    p_case_validate.set_defaults(func=cmd_case_validate)

    p_case_run = case_sub.add_parser(
        "run",
        help="Validate a case package and dry-run its declared commands",
    )
    p_case_run.add_argument("path", help="Path to a case package directory")
    p_case_run.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute declared commands instead of dry-running them",
    )
    p_case_run.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable run report",
    )
    p_case_run.set_defaults(func=cmd_case_run)

    p_case_list = case_sub.add_parser(
        "list",
        help="List case packages, curated catalog entries, and/or examples",
    )
    p_case_list.add_argument(
        "root",
        nargs="?",
        default="cases",
        help="Root directory to scan for case packages (default: cases)",
    )
    p_case_list.add_argument(
        "--catalog",
        action="store_true",
        help="Include curated catalog entries",
    )
    p_case_list.add_argument(
        "--catalog-path",
        default="cases/catalog.yaml",
        help="Curated catalog path (default: cases/catalog.yaml)",
    )
    p_case_list.add_argument(
        "--examples",
        action="store_true",
        help="Include virtual L1 records for examples/**/*.clay",
    )
    p_case_list.add_argument(
        "--examples-root",
        default="examples",
        help="Examples root for --examples (default: examples)",
    )
    p_case_list.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable records",
    )
    p_case_list.set_defaults(func=cmd_case_list)

    p_case_audit = case_sub.add_parser(
        "audit",
        help="Build the de-duplicated inventory of all known Clay cases",
    )
    p_case_audit.add_argument(
        "--repo-root",
        default=".",
        help="Repository root for inventory paths (default: current directory)",
    )
    p_case_audit.add_argument(
        "--cases-root",
        default="cases",
        help="Case package root relative to repo root (default: cases)",
    )
    p_case_audit.add_argument(
        "--catalog-path",
        default="cases/catalog.yaml",
        help="Curated catalog path relative to repo root",
    )
    p_case_audit.add_argument(
        "--examples-root",
        default="examples",
        help="Examples root relative to repo root",
    )
    p_case_audit.add_argument(
        "--out",
        default=None,
        help="Write the JSON inventory report to this path",
    )
    p_case_audit.add_argument(
        "--json",
        action="store_true",
        help="Emit the full JSON inventory report",
    )
    p_case_audit.set_defaults(func=cmd_case_audit)

    p_flow = sub.add_parser(
        "flow",
        help="Run the generic, data-driven signoff flow over a circuit",
    )
    flow_sub = p_flow.add_subparsers(dest="flow_cmd", required=True)

    p_flow_check = flow_sub.add_parser("check", help="Check an explicit circuit package without physical execution")
    p_flow_check.add_argument("--case", required=True, help="Directory containing case.yaml/yml/json")
    p_flow_check.add_argument("--project-root", help="Explicit input root (defaults to the case directory)")
    p_flow_check.add_argument(
        "--until",
        default="comparison",
        choices=("source", "placement", "comparison", "preroute"),
        help="Requested capability scope: source, placement, comparison, preroute",
    )
    p_flow_check.add_argument("--json", action="store_true", help="Emit the versioned readiness result")
    p_flow_check.set_defaults(func=cmd_flow_check)

    p_flow_recovery = flow_sub.add_parser('recover-precreate',
        help='Record a failed pre-create primitive stage after verifying its evidence')
    p_flow_recovery.add_argument('--case', required=True)
    p_flow_recovery.add_argument('--project-root', required=True)
    p_flow_recovery.add_argument('--task-id', required=True)
    p_flow_recovery.add_argument('--preparation-manifest', required=True)
    p_flow_recovery.add_argument('--wall-seconds', required=True, type=float)
    p_flow_recovery.set_defaults(func=cmd_flow_recover_precreate)
    p_flow_run = flow_sub.add_parser("run", help="Run the flow for a circuit (or --all)")
    p_flow_run.add_argument(
        "circuit",
        nargs="?",
        default=None,
        help="Circuit key from the flow registry (omit with --all)",
    )
    p_flow_run.add_argument(
        "--all",
        action="store_true",
        help="Run the flow for every registered circuit (portfolio); print a verdict table",
    )
    p_flow_run.add_argument("--case", help="Explicit case package directory (instead of a registry key)")
    p_flow_run.add_argument("--project-root", help="Persistent execution-control project root containing the case")
    p_flow_run.add_argument("--task-id", help="Stable task identity shared by attempts and output directories")
    p_flow_run.add_argument("--wall-seconds", type=float, help="Finite wall-clock budget for this flow invocation")
    p_flow_run.add_argument('--stage-unwind-seconds', type=float,
                            help='Finite independent cleanup-verification budget for an interrupted primitive stage')
    p_flow_run.add_argument('--stage-role-trace-seconds', type=float,
                            help='Finite same-run PDK role-trace observation budget')
    p_flow_run.add_argument("--retry-receipt", help="SHA256 of the exact prior failure receipt")
    p_flow_run.add_argument("--retry-reason", choices=("input_changed", "producer_changed",
                                                      "environment_repaired", "budget_experiment"))
    p_flow_run.add_argument("--retry-evidence", action="append", help="Evidence file for this retry (repeat for multiple files)")
    p_flow_run.add_argument('--stage-retry-receipt', help='SHA256 of the prior primitive-stage failure receipt')
    p_flow_run.add_argument('--stage-retry-reason', choices=('input_changed', 'producer_changed',
                                                             'environment_repaired', 'budget_experiment'))
    p_flow_run.add_argument('--stage-retry-evidence', action='append', help='Evidence file for primitive-stage retry')
    p_flow_run.add_argument("--only", default=None, help="Run only this single step")
    p_flow_run.add_argument(
        "--force", action="store_true", help="Ignore the incremental cache; re-run all steps"
    )
    p_flow_run.add_argument("--out", default=None, help="Output run directory")
    p_flow_run.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of a table"
    )
    p_flow_run.set_defaults(func=cmd_flow_run)

    p_flow_list = flow_sub.add_parser("list", help="List circuits in the flow registry")
    p_flow_list.set_defaults(func=cmd_flow_list)

    p_flow_status = flow_sub.add_parser(
        "status", help="Show the persisted per-step incremental state for a circuit"
    )
    p_flow_status.add_argument("circuit", help="Circuit key from the flow registry")
    p_flow_status.add_argument("--out", default=None, help="Output run directory")
    p_flow_status.set_defaults(func=cmd_flow_status)

    p_flow_suite = flow_sub.add_parser(
        "suite",
        help="Run the unified multi-circuit E2E & Docker signoff suite (OTA/FC_OTA/LDO/BANDGAP/CMP)",
    )
    p_flow_suite.add_argument("--families", nargs="+", default=None, help="Filter by circuit families (e.g. OTA FC_OTA LDO BANDGAP CMP)")
    p_flow_suite.add_argument("--cases", nargs="+", default=None, help="Filter by case_id(s)")
    p_flow_suite.add_argument("--include-audit-only", action="store_true", help="Include audit_only targets")
    p_flow_suite.add_argument("--check-only", action="store_true", help="Run preroute checks and collect receipts without launching flow runs")
    p_flow_suite.add_argument("--run-flow", action="store_true", help="Execute `clay flow run --case` for each selected non-audit target")
    p_flow_suite.add_argument("--force", action="store_true", help="Forward `--force` to `clay flow run`")
    p_flow_suite.add_argument("--out-subdir", default=None, help="Optional output subdir inside each case")
    p_flow_suite.add_argument("--docker-signoff", action="store_true", help="Execute real Docker Magic/Netgen/ngspice signoff")
    p_flow_suite.add_argument("--report-json", default=None, help="Output path for JSON summary report")
    p_flow_suite.add_argument("--report-md", default=None, help="Output path for Markdown summary report")
    p_flow_suite.add_argument(
        "--retry-reason",
        choices=("input_changed", "producer_changed", "environment_repaired", "budget_experiment"),
        default=None,
        help="Optional `--retry-reason` forwarded when retrying a failed ledger attempt",
    )
    p_flow_suite.add_argument("--retry-evidence", action="append", default=[], help="Optional `--retry-evidence` file path(s)")
    p_flow_suite.add_argument("--python", dest="python_bin", default=None, help="Python executable for sub-invocations")
    p_flow_suite.set_defaults(func=cmd_flow_suite)

    p_status = sub.add_parser(
        "status",
        help="One-screen project status: per-circuit maturity + last flow verdict",
    )
    p_status.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of a table"
    )
    p_status.set_defaults(func=cmd_status)

    p_build = sub.add_parser(
        "build",
        help="Build Clay C++20 native PnR extensions and run pure C++20 native unit tests",
    )
    p_build.add_argument(
        "--target",
        choices=("all", "core", "place", "pathfinder", "gridless", "tests"),
        default="all",
        help="Native C++20 target(s) to build (default: all)",
    )
    p_build.add_argument(
        "-j", "--jobs",
        type=int,
        default=None,
        help="Parallel compiler worker jobs",
    )
    p_build.add_argument("--force", action="store_true", help="Ignore content-hash cache and rebuild")
    p_build.add_argument("--clean", action="store_true", help="Clean modular build directory before building")
    p_build.add_argument("--werror", action="store_true", help="Treat C++20 compiler warnings as errors")
    p_build.add_argument("--test", action="store_true", help="Execute pure C++20 native unit tests after build")
    p_build.add_argument("--filter", dest="test_filter", default=None, help="Substring filter for C++20 tests")
    p_build.set_defaults(func=cmd_build)

    p_test = sub.add_parser(
        "test",
        help="Run parallel, OpenBLAS-safe Clay test suites (tiers: fast, core, cpp, regression, all, perf_timing)",
    )
    p_test.add_argument(
        "--tier",
        choices=("fast", "core", "cpp", "regression", "all", "perf_timing"),
        default="fast",
        help="Test tier to execute (default: fast blast-radius)",
    )
    p_test.add_argument(
        "-j", "--workers",
        default="auto",
        help="Parallel pytest-xdist worker count (default: auto; use 1 for serial)",
    )
    p_test.add_argument("--base", default=None, help="Git diff base for `--tier fast`")
    p_test.add_argument("--lf", action="store_true", help="Re-run only last-failed tests")
    p_test.add_argument("-k", dest="keyword", default=None, help="Pytest -k expression")
    p_test.add_argument("--durations", type=int, default=None, help="Report N slowest durations")
    p_test.add_argument("--list", dest="list_only", action="store_true", help="List selected test targets without executing")
    p_test.add_argument("targets", nargs="*", help="Optional explicit test file(s)/nodeid(s)")
    p_test.set_defaults(func=cmd_test)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
