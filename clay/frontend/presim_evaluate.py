"""Close the sizing loop: a real-sim ``evaluate(params) -> KPIs`` bridge so the
PDK-retarget refine (clay.frontend.pdk_retarget.retarget_with_refine) can drive
the adjoint + tssbo optimiser against measured ngspice KPIs -- the loop the
circuits' hand-derived sizing bypassed.

The bridge is GENERAL (no per-circuit branch): it runs a circuit's committed
``presim`` module against a param-OVERRIDDEN copy of its source ``.cir`` by
monkeypatching the module's source-cir constant to a temp copy (non-mutating to
the committed file; restored in a finally). The caller passes the constant name
(e.g. ``"FC_OTA_SOURCE"``) and the KPI key map -- both data, not code branches.

Composition (reuse, do not reinvent):
  * the circuit's ``presim.run`` (real ngspice via the marker harness) = the sim;
  * ``pdk_retarget.retarget_with_refine`` = the analytic seed + the tssbo /
    distance_to_sat refine loop; this module only supplies the ``evaluate``.
"""
from __future__ import annotations

import importlib
import json
import math
import re
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional
from weakref import WeakKeyDictionary

# Factory-owned registry: a caller-provided attribute is not source evidence.
# This binds the canonical source bridge, not the physical signoff outcome.
_SOURCE_BOUND_EVALUATORS = WeakKeyDictionary()


def require_source_bound_joint_evaluator(evaluator, authority, spec):
    """Accept only the unchanged canonical bridge bound to this source."""
    try:
        binding = _SOURCE_BOUND_EVALUATORS.get(evaluator)
    except TypeError as exc:
        raise TypeError("custom sizing evaluator is not a registered source bridge") from exc
    if binding is None:
        raise ValueError("custom sizing evaluator is not a registered source bridge")
    bound, canonical_flow, code, acceptance, acceptance_code, bound_spec, schematic = binding
    from clay.flow.sizing_source import spec_fingerprint
    if (bound is None or not canonical_flow or bound.path != authority.path or bound.sha256 != authority.sha256
            or getattr(evaluator, "__code__", None) is not code
            or getattr(evaluator, "evaluate_acceptance", None) is not acceptance
            or getattr(acceptance, "__code__", None) is not acceptance_code
            or bound_spec is None or spec_fingerprint(spec) != bound_spec):
        raise ValueError("custom sizing evaluator source or physical runner differs")
    bound.verify()
    authority.verify()
    if schematic.path != Path(spec.source_cir).resolve():
        raise ValueError("custom sizing evaluator schematic source differs")
    schematic.verify()


def override_params(cir_text: str, overrides: Mapping[str, str]) -> str:
    """Return ``cir_text`` with the named ``.param`` values replaced.

    Only ``.param`` lines are touched; a ``key = value`` (value up to whitespace
    or a ``$`` comment) is rewritten for each key in ``overrides``. Pure; the
    netlist is otherwise byte-identical."""
    if not overrides:
        return cir_text
    out = []
    for line in cir_text.splitlines():
        if line.lstrip().lower().startswith(".param"):
            for key, val in overrides.items():
                pat = re.compile(rf"(\b{re.escape(str(key))}\s*=\s*)([^\s$]+)")
                line = pat.sub(lambda m, _v=str(val): m.group(1) + _v, line)
        out.append(line)
    return "\n".join(out)


def make_presim_evaluate(
    presim_module: str,
    *,
    source_attr: str,
    kpi_map: Mapping[str, str],
) -> Callable[[Mapping[str, str]], Dict[str, float]]:
    """Build an ``evaluate(params) -> {metric: value}`` over a circuit's real presim.

    ``presim_module`` is an importable module exposing ``run()`` (the circuit's
    ``run_presim`` alias) and a Path constant named ``source_attr`` pointing at its
    source ``.cir``. Each call writes a param-overridden temp copy of that .cir,
    monkeypatches the module constant to it, runs ``presim.run()`` (real ngspice),
    restores the constant, and returns the KPIs selected by ``kpi_map``
    (``{out_name: presim_result_key}``). General -- any circuit whose presim
    follows this shape, no per-circuit code."""
    mod = importlib.import_module(presim_module)
    if not hasattr(mod, source_attr):
        raise AttributeError(
            f"{presim_module} has no source-cir attribute {source_attr!r}"
        )
    if not hasattr(mod, "run"):
        raise AttributeError(f"{presim_module} exposes no run() entry point")
    original = getattr(mod, source_attr)
    src_path = Path(original)

    def evaluate(params: Mapping[str, str]) -> Dict[str, float]:
        text = src_path.read_text(encoding="utf-8")
        patched = override_params(text, dict(params))
        tmpdir = Path(tempfile.mkdtemp(prefix="clay_tune_"))
        tmp_cir = tmpdir / src_path.name
        tmp_cir.write_text(patched, encoding="utf-8")
        try:
            setattr(mod, source_attr, tmp_cir)
            res = mod.run()
        finally:
            setattr(mod, source_attr, original)
            shutil.rmtree(tmpdir, ignore_errors=True)
        out: Dict[str, float] = {}
        for name, key in kpi_map.items():
            if key in res and res[key] is not None:
                out[name] = float(res[key])
        return out

    return evaluate


def canonicalize_candidate_sizing(
    spec, params: Mapping[str, float], *, source_authority=None
) -> dict[str, float]:
    """Return the one physical sizing map shared by simulation and layout."""

    from clay.flow.optimized_clay import canonicalize_variable_value, _canonical_physical_sizing

    intent = getattr(spec, "optimization", None)
    if intent is None:
        raise ValueError("joint evaluation requires source-declared optimization intent")
    unknown = sorted(set(params) - {variable.name for variable in intent.variables})
    if unknown:
        raise ValueError(f"joint evaluation received unknown sizing variables: {unknown}")
    from clay.flow.sizing_source import sizing_source

    authority = sizing_source(spec, source_authority)
    design = authority.path if authority is not None else Path(spec.case_dir) / "source" / "design.clay"
    if design.is_file():
        from clay.parser import parse_clay_file

        parsed = parse_clay_file(design)
        selected = [cell for cell in parsed.cell_defs if cell.local_name == spec.top_cell]
        if len(selected) == 1 and any(device.device_class == "native" for device in selected[0].devices):
            if parsed.pdk_id != spec.pdk:
                raise ValueError("native sizing PDK differs from its source")
            values = _canonical_physical_sizing(
                design.read_text(encoding="utf-8-sig"), top_cell=spec.top_cell,
                optimization=intent,
                raw_sizing={variable.name: params.get(variable.name, variable.default)
                            for variable in intent.variables},
            )
            authority.verify()
            return values
    values: dict[str, float] = {}
    for variable in intent.variables:
        value = float(params.get(variable.name, variable.default))
        if not math.isfinite(value):
            raise ValueError(f"sizing variable {variable.name} is non-finite")
        if value < variable.low and math.isclose(value, variable.low, rel_tol=1e-12, abs_tol=1e-24):
            value = variable.low
        elif value > variable.high and math.isclose(value, variable.high, rel_tol=1e-12, abs_tol=1e-24):
            value = variable.high
        if not variable.low <= value <= variable.high:
            raise ValueError(
                f"sizing variable {variable.name}={value:g} outside "
                f"[{variable.low:g}, {variable.high:g}]"
            )
        canonical = canonicalize_variable_value(variable, value)
        if canonical < variable.low and math.isclose(canonical, variable.low, rel_tol=1e-12, abs_tol=1e-24):
            canonical = variable.low
        elif canonical > variable.high and math.isclose(canonical, variable.high, rel_tol=1e-12, abs_tol=1e-24):
            canonical = variable.high
        if not variable.low <= canonical <= variable.high:
            raise ValueError(
                f"canonical sizing variable {variable.name}={canonical:g} outside "
                f"[{variable.low:g}, {variable.high:g}]"
            )
        values[variable.name] = canonical
    # The optimized derivative is generated from the same source after sizing.
    # Quantize its width DOFs now -- before either simulation phase -- so the
    # eventual integer-nm uniform-finger plan does not create an adjacent,
    # unmeasured circuit revision at layout time.
    if design.is_file():
        from clay.flow.optimized_clay import declared_compact_candidate
        from clay.flow.optimized_layout_style import (
            DEFAULT_OPTIMIZED_FOLD_HEIGHT_UM,
            canonicalize_fold_compatible_sizing,
        )
        from clay.parser import parse_clay_file

        parsed = parse_clay_file(design)
        matches = [
            cell
            for cell in (getattr(parsed, "cell_defs", ()) or ())
            if getattr(cell, "local_name", getattr(cell, "name", None))
            == spec.top_cell
        ]
        if len(matches) == 1:
            from clay.flow.optimized_clay import _effective_sizing_fold_height_um

            compact = declared_compact_candidate(
                design.read_text(encoding="utf-8-sig")
            )
            fold_height_um = _effective_sizing_fold_height_um(matches[0], compact)
            grid_nm = 1
            try:
                from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
                from clay.layout.tech import LayoutTech

                pdk_id = getattr(parsed, "pdk_id", None) or getattr(spec, "pdk", None)
                if pdk_id:
                    grid_nm = max(1, int(round(pdk_manufacturing_grid_nm(LayoutTech.resolve(str(pdk_id))))))
            except Exception:  # noqa: BLE001
                grid_nm = 1
            values = canonicalize_fold_compatible_sizing(
                matches[0],
                intent,
                values,
                fold_height_um=float(fold_height_um),
                grid_nm=grid_nm,
            )
        elif matches:
            raise ValueError(
                f"joint sizing source contains multiple {spec.top_cell!r} cells"
            )
    if authority is not None:
        authority.verify()
    return values


# Backward-compatible private spelling for older internal callers.
_candidate_values = canonicalize_candidate_sizing


def _format_schematic_value_like(
    source_text: str, parameter: str, value: float
) -> str:
    """Format an SI value in the source parameter's declared SPICE scale.

    Sizing intent is lowered to SI, while SPICE geometry parameters are
    conventionally authored as ``24u`` / ``1.2p`` / ``2k``.  Writing the SI
    float bare changes the downstream LVS geometry interpretation when it
    restores the declared micron convention.  Preserve the source token's
    engineering suffix; a genuinely suffix-free token remains suffix-free.
    """
    pattern = re.compile(
        rf"^\s*\.param\b[^\n]*\b{re.escape(parameter)}\s*=\s*([^\s$]+)",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(source_text)
    if match is None:
        raise ValueError(f"schematic binding names missing .param {parameter!r}")
    token = match.group(1).strip().lower()
    suffix = ""
    scale = 1.0
    if token.endswith("meg"):
        suffix, scale = "meg", 1e6
    elif token and token[-1] in _ENG and token[-1] != "e":
        suffix = token[-1]
        scale = _ENG[suffix]
    return f"{float(value) / scale:.12g}{suffix}"


def materialize_joint_schematic(spec, values: Mapping[str, float], root: Path, *, source_authority=None):
    """Write the canonical candidate schematic consumed by every flow phase."""

    from clay.flow.candidate_bindings import value_for_binding

    from clay.flow.sizing_source import sizing_source

    authority = sizing_source(spec, source_authority)
    values = canonicalize_candidate_sizing(spec, values, source_authority=authority)
    root.mkdir(parents=True, exist_ok=True)
    source = Path(spec.source_cir)
    if not source.is_file():
        raise FileNotFoundError(f"joint schematic source missing: {source}")
    source_text = source.read_text(encoding="utf-8")
    schematic_overrides: dict[str, str] = {}
    testbench_params: dict[str, float] = {}
    for variable in spec.optimization.variables:
        value = float(values[variable.name])
        for binding in variable.bindings:
            parts = binding.split(".")
            bound_value = value_for_binding(variable, binding, value)
            if len(parts) == 2 and parts[0] == "schematic":
                schematic_overrides[parts[1]] = _format_schematic_value_like(
                    source_text, parts[1], bound_value
                )
            elif len(parts) == 2 and parts[0] == "testbench":
                testbench_params[parts[1]] = bound_value
            elif len(parts) == 3 and parts[0] == "device":
                continue  # CompileStep consumes this binding from the same intent.
            else:
                raise ValueError(f"unsupported sizing binding {binding!r}")
    candidate_source = root / source.name
    if candidate_source.resolve() == source.resolve():
        raise ValueError("candidate schematic must not overwrite its source")
    design = authority.path if authority is not None else Path(spec.case_dir) / "source" / "design.clay"
    if design.is_file():
        from clay.parser import parse_clay_file
        from clay.frontend.native_spice_import import emit_native_spice
        from clay.flow.candidate_bindings import apply_candidate_device_bindings

        parsed = parse_clay_file(design)
        selected = [cell for cell in parsed.cell_defs if cell.local_name == spec.top_cell]
        if len(selected) == 1 and any(device.device_class == "native" for device in selected[0].devices):
            if parsed.pdk_id != spec.pdk or schematic_overrides:
                raise ValueError("native schematic requires one PDK-bound device parameter authority")
            cell = selected[0]
            baseline = emit_native_spice(cell, pdk_id=spec.pdk, top_name=spec.top_cell)
            if baseline.strip() != source_text.strip():
                raise ValueError("native schematic source differs from the authoring Clay")
            apply_candidate_device_bindings(cell, spec.optimization, values, pdk_id=spec.pdk)
            candidate_source.write_text(
                emit_native_spice(cell, pdk_id=spec.pdk, top_name=spec.top_cell),
                encoding="utf-8", newline="\n",
            )
            authority.verify()
            return candidate_source, testbench_params
    candidate_source.write_text(
        override_params(source_text, schematic_overrides) + "\n",
        encoding="utf-8",
    )
    if authority is not None:
        authority.verify()
    return candidate_source, testbench_params


# Backward-compatible private spelling for older internal callers.  New flow
# code uses the public name so candidate-source materialization has one owner.
_materialize_joint_schematic = materialize_joint_schematic


def _finite_metric_map(value: Any, *, phase: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{phase} evaluator returned no metric mapping")
    out: dict[str, float] = {}
    for key, raw in value.items():
        if str(key).startswith("__"):
            continue
        try:
            numeric = float(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{phase} metric {key!r} is not numeric") from exc
        if not math.isfinite(numeric):
            raise RuntimeError(f"{phase} metric {key!r} is non-finite")
        out[str(key)] = numeric
    if not out:
        raise RuntimeError(f"{phase} evaluator returned no metrics")
    return out


def make_joint_evaluate(
    spec,
    *,
    work_root: Path,
    pre_evaluate: Callable[[Path, Mapping[str, float], Path], Mapping[str, float]],
    flow_runner: Optional[Callable[..., Any]] = None,
    source_authority=None,
):
    """Extend the existing presim bridge through canonical layout/PEX/post.

    Circuit-specific simulation remains behind the injected driver callable;
    candidate materialization and the physical flow are data-driven by the
    source-compiled ``CircuitSpec``.  A pre-only row is explicitly tagged
    ``pre_proxy`` and can never masquerade as post-layout truth.
    """
    if getattr(spec, "optimization", None) is None:
        raise ValueError("joint evaluation requires source-declared optimization intent")
    from clay.flow.engine import run_flow as canonical_flow_runner
    if flow_runner is None:
        flow_runner = canonical_flow_runner
    canonical_flow = flow_runner is canonical_flow_runner

    from clay.flow.sizing_source import sizing_source, SizingSource
    from clay.flow.execution_budget import ExecutionStopped

    authority = sizing_source(spec, source_authority)
    source = SizingSource.capture(spec.source_cir)
    schematic = SizingSource(source.path, source.sha256, 'sizing schematic source')
    work_root = Path(work_root)
    from clay.flow.spec import CircuitSpec
    from clay.flow.sizing_source import SizingSourceChanged, spec_fingerprint, verify_spec_fingerprint
    bound_spec = spec_fingerprint(spec) if isinstance(spec, CircuitSpec) else None

    def verify_spec():
        if bound_spec is not None:
            verify_spec_fingerprint(spec, bound_spec)
        try:
            current_schematic = Path(spec.source_cir).resolve()
        except (OSError, TypeError, ValueError) as exc:
            raise SizingSourceChanged('sizing schematic source path changed') from exc
        if current_schematic != schematic.path:
            raise SizingSourceChanged('sizing schematic source path changed')
        schematic.verify()

    def verify_inputs():
        verify_spec()
        if authority is not None:
            authority.verify()

    def checked_call(function, *args, **kwargs):
        """Observe source drift even when a candidate callback fails."""
        verify_inputs()
        try:
            value = function(*args, **kwargs)
        except ExecutionStopped:
            raise
        except Exception as original:
            try:
                verify_inputs()
            except Exception as changed:
                raise changed from original
            raise
        verify_inputs()
        return value

    # A journal row is not an artifact authority. Retain the actual producer's
    # immutable identity per evaluation instead of trusting a later row/path.
    retained_candidates: dict[int, str] = {}

    def evaluate(
        params: Mapping[str, float],
        *,
        evaluation_index: int,
        run_post: bool,
    ) -> dict[str, Any]:
        verify_spec()
        if authority is not None:
            authority.verify()
        values = checked_call(canonicalize_candidate_sizing, spec, params, source_authority=authority)
        candidate_root = work_root / f"eval_{int(evaluation_index):04d}"
        candidate_source, testbench_params = checked_call(materialize_joint_schematic,
            spec, values, candidate_root / "candidate", source_authority=authority
        )
        from clay.flow.candidate_artifacts import source_identity, require_identity

        source_before = source_identity(candidate_root, candidate_source)
        try:
            pre_values = checked_call(pre_evaluate, candidate_source, testbench_params, candidate_root / "pre")
        except ExecutionStopped:
            raise
        except Exception as original:
            try:
                require_identity(candidate_root, candidate_source, source_before)
            except SizingSourceChanged as changed:
                raise changed from original
            raise
        require_identity(candidate_root, candidate_source, source_before)
        pre = _finite_metric_map(pre_values, phase="pre")
        verify_spec()
        if authority is not None:
            authority.verify()
        require_identity(candidate_root, candidate_source, source_before)
        result: dict[str, Any] = {
            "sizing": values,
            "pre": pre,
            "post": None,
            "truth": "pre_proxy",
            "candidate_source": str(candidate_source),
        }
        if authority is not None:
            result["authoring_design_clay"] = str(authority.path)
            result["authoring_design_clay_sha256"] = authority.sha256
        if not run_post:
            return result

        candidate_spec = replace(spec, source_cir=candidate_source)
        staged = None
        staged_before = None
        authoring_design = authority.path if authority is not None else Path(spec.case_dir) / "source" / "design.clay"
        if authoring_design.is_file():
            from clay.flow.optimized_clay import materialize_sizing_candidate_clay
            from clay.parser import parse_clay_file

            parsed = checked_call(parse_clay_file, authoring_design)
            matches = [
                cell
                for cell in (getattr(parsed, "cell_defs", ()) or ())
                if getattr(cell, "local_name", getattr(cell, "name", None))
                == spec.top_cell
            ]
            if len(matches) > 1:
                raise ValueError(
                    f"joint sizing source contains multiple {spec.top_cell!r} cells"
                )
            if len(matches) == 1:
                candidate_case = candidate_root / "case"
                staged = checked_call(materialize_sizing_candidate_clay,
                    source_path=authoring_design,
                    output_path=candidate_case / "source" / "design.clay",
                    top_cell=spec.top_cell,
                    optimization=spec.optimization,
                    sizing=values,
                    provenance={
                        "engine": spec.optimization.engine,
                        "seed": spec.optimization.seed,
                    },
                )
                src_case_yaml = Path(spec.case_dir) / "case.yaml"
                if src_case_yaml.is_file():
                    dst_case_yaml = candidate_case / "case.yaml"
                    dst_case_yaml.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_case_yaml, dst_case_yaml)
                candidate_spec = replace(
                    candidate_spec,
                    case_dir=candidate_case,
                )
                result["candidate_design_clay"] = str(staged.path.resolve())
                result["candidate_design_clay_sha256"] = staged.sha256
                staged_before = source_identity(candidate_root, staged.path)
                authority.verify()
        try:
            flow_result = checked_call(flow_runner,
                candidate_spec,
                out_dir=candidate_root / "physical",
                only=("compile", "layout", "drc", "lvs", "pex", "post"),
                force=True,
                initial_extra={"candidate_sizing": values, "pvt_mode": "nominal"},
            )
        except ExecutionStopped:
            raise
        except Exception as original:
            try:
                require_identity(candidate_root, candidate_source, source_before)
                if staged is not None:
                    require_identity(candidate_root, staged.path, staged_before)
            except SizingSourceChanged as changed:
                raise changed from original
            raise
        verify_spec()
        if authority is not None:
            authority.verify()
        require_identity(candidate_root, candidate_source, source_before)
        if staged is not None:
            from clay.flow.sizing_source import SizingSource
            SizingSource(staged.path, staged.sha256, 'sizing staged candidate').verify()
        if getattr(flow_result, "verdict", None) != "pass":
            raise RuntimeError("post-PEX candidate physical flow failed")
        post_step = getattr(flow_result, "by_step", {}).get("post")
        if post_step is None or post_step.status != "pass":
            raise RuntimeError("post-PEX candidate produced no passing post step")
        result_path = Path(post_step.artifacts.get("post_results", ""))
        if not result_path.is_file():
            raise RuntimeError("post-PEX candidate produced no post_results artifact")
        pex_step = getattr(flow_result, "by_step", {}).get("pex")
        pex_path = Path(
            (getattr(pex_step, "artifacts", {}) or {}).get("pex_spice", "")
        )
        if not pex_path.is_file():
            raise RuntimeError("post-PEX candidate produced no extracted netlist")
        native_observation = None
        if getattr(spec, "postsim_module", None) == "circuits.jku_comparator.postsim":
            from clay.flow.native_candidate_crossbinding import verify_native_candidate
            layout_step = getattr(flow_result, "by_step", {}).get("layout")
            if layout_step is None or layout_step.status != "pass":
                raise RuntimeError("native candidate produced no passing layout")
            gds_path = Path((layout_step.artifacts or {}).get("layout_gds", ""))
            native_observation = verify_native_candidate(root=candidate_root,
                source=candidate_source, source_before=source_before,
                gds=gds_path, gds_before=source_identity(candidate_root, gds_path),
                pex=pex_path, pex_before=source_identity(candidate_root, pex_path))
        post = _finite_metric_map(
            json.loads(result_path.read_text(encoding="utf-8")), phase="post"
        )
        from clay.flow.candidate_artifacts import CandidateArtifactChanged, seal_candidate

        encoded = seal_candidate(root=candidate_root, evaluation_index=evaluation_index,
            sizing=values, source=candidate_source, source_before=source_before,
            pex=pex_path, staged=None if staged is None else staged.path, staged_before=staged_before,
            native_observation=native_observation)
        verify_inputs()
        key = int(evaluation_index)
        if key in retained_candidates and retained_candidates[key] != encoded:
            raise CandidateArtifactChanged("candidate evaluation identity was reused with different artifacts")
        retained_candidates[key] = encoded
        result.update(
            {
                "post": post,
                "truth": "post_pex",
                "candidate_pex": str(pex_path.resolve()),
                "candidate_artifact_seal": json.loads(encoded),
                "flow": flow_result,
            }
        )
        return result

    def evaluate_acceptance(record: Mapping[str, Any]) -> dict[str, Any]:
        """Reuse one candidate PEX for the declared full-PVT POST sweep."""
        verify_spec()
        if authority is not None:
            authority.verify()
        from clay.flow.candidate_artifacts import verify_candidate

        index = record.get("evaluation_index")
        if type(index) is not int:
            from clay.flow.candidate_artifacts import CandidateArtifactChanged
            raise CandidateArtifactChanged("candidate journal lacks an evaluation identity")
        sizing = canonicalize_candidate_sizing(spec, dict(record.get("sizing") or {}),
                                               source_authority=authority)
        encoded = retained_candidates.get(index)
        verify_candidate(encoded, record, sizing)
        if not getattr(spec, "postsim_module", None):
            raise ValueError("joint evaluation spec declares no postsim_module")
        if not getattr(spec, "postsim_function", None):
            raise ValueError("joint evaluation spec declares no postsim_function")
        pex_spice = Path(str(record.get("candidate_pex") or ""))
        if not pex_spice.is_file():
            raise FileNotFoundError(
                f"candidate full-extraction PEX missing: {pex_spice}"
            )
        module = importlib.import_module(spec.postsim_module)
        # Reuse the exact canonical Driver invocation contract used by the
        # outer PostSimStep.  The only policy difference is explicit here:
        # search samples are nominal; promotion is acceptance PVT.
        from clay.flow.step import PostSimStep

        try:
            result = PostSimStep._invoke_declared(
                module,
                spec.postsim_function,
                pex_spice,
                pdk=spec.pdk,
                candidate_sizing=dict(record.get("sizing") or {}),
                corners=spec.corners,
                temps_c=spec.temps_c,
                pvt_mode="acceptance",
                kpi_bounds={kpi.key: kpi.bound for kpi in spec.kpis},
                simulation_seed=spec.optimization.seed,
            )
        except ExecutionStopped:
            raise
        except Exception as original:
            try:
                verify_candidate(encoded, record, sizing)
                verify_inputs()
            except Exception as changed:
                raise changed from original
            raise
        verify_candidate(encoded, record, sizing)
        verify_spec()
        if authority is not None:
            authority.verify()
        return result

    # Keep the public evaluator callable compatible with the existing sizing
    # problem while making the stronger promotion oracle an explicit part of
    # the same pre/layout/PEX/post bridge.
    evaluate.evaluate_acceptance = evaluate_acceptance  # type: ignore[attr-defined]
    _SOURCE_BOUND_EVALUATORS[evaluate] = (
        authority, canonical_flow, evaluate.__code__, evaluate_acceptance, evaluate_acceptance.__code__, bound_spec,
        schematic,
    )

    return evaluate


def make_joint_evaluate_from_spec(
    spec,
    *,
    work_root: Path,
    flow_runner: Optional[Callable[..., Any]] = None,
    source_authority=None,
):
    """Resolve the pre-simulation Driver from compiled circuit DATA."""
    if not getattr(spec, "presim_module", None):
        raise ValueError("joint evaluation spec declares no presim_module")
    module = importlib.import_module(spec.presim_module)
    function_name = str(getattr(spec, "presim_function", "run_candidate"))
    function = getattr(module, function_name, None)
    if not callable(function):
        raise AttributeError(
            f"{spec.presim_module} exposes no callable {function_name!r}"
        )

    def pre_evaluate(source_cir, testbench_params, work_dir):
        import inspect

        parameters = inspect.signature(function).parameters
        kwargs: dict[str, Any] = {
            "testbench_params": dict(testbench_params),
            "work_dir": Path(work_dir),
        }
        if "pdk" in parameters:
            kwargs["pdk"] = spec.pdk
        if "vdd" in parameters:
            kwargs["vdd"] = spec.vdd
        if "simulation_seed" in parameters:
            kwargs["simulation_seed"] = spec.optimization.seed
        return function(source_cir, **kwargs)

    return make_joint_evaluate(
        spec,
        work_root=work_root,
        pre_evaluate=pre_evaluate,
        flow_runner=flow_runner,
        source_authority=source_authority,
    )


def tune_sizing(
    seed_params: Mapping[str, str],
    src,
    dst,
    *,
    evaluate: Callable[[Mapping[str, str]], Mapping[str, float]],
    spec_targets: Mapping[str, tuple],
    use_optimizer: bool = False,
    max_iters: int = 8,
) -> Dict[str, str]:
    """Retarget + refine the sizing against measured KPIs (thin reuse of
    :func:`clay.frontend.pdk_retarget.retarget_with_refine`).

    The analytic gm/Id retarget is the warm start; with ``use_optimizer`` and a
    real-sim ``evaluate`` the tssbo + distance_to_sat loop refines it toward
    ``spec_targets`` ({metric: (op, target)}). With ``use_optimizer=False`` (or no
    optimiser backend installed) it returns the analytic seed -- graceful, no sim
    needed."""
    from clay.frontend.pdk_retarget import retarget_with_refine

    return retarget_with_refine(
        seed_params, src, dst,
        evaluate=evaluate,
        spec_targets=spec_targets,
        use_optimizer=use_optimizer,
        max_iters=max_iters,
    )


_ENG = {"t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3,
        "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}


def _eng_to_float(tok: str) -> float:
    s = str(tok).strip()
    sl = s.lower()
    if sl.endswith("meg"):
        return float(sl[:-3]) * 1e6
    if sl and sl[-1] in _ENG and sl[-1] != "e":
        try:
            return float(sl[:-1]) * _ENG[sl[-1]]
        except ValueError:
            pass
    return float(s)


def _float_to_um(v: float) -> str:
    return f"{v / 1e-6:.5g}u"


def _hinge_loss(kpis: Mapping[str, float], spec: Mapping[str, tuple]) -> float:
    """Normalised hinge: 0 when every (op, target) is met, else the summed
    relative shortfall. A missing metric is a large penalty."""
    total = 0.0
    for name, (op, tgt) in spec.items():
        m = kpis.get(name)
        if m is None:
            total += 1e6
            continue
        denom = abs(float(tgt)) or 1.0
        if op in (">=", ">"):
            total += max(0.0, float(tgt) - float(m)) / denom
        elif op in ("<=", "<"):
            total += max(0.0, float(m) - float(tgt)) / denom
        else:
            total += abs(float(m) - float(tgt)) / denom
    return total


def bo_tune_sizing(
    seed_params: Mapping[str, str],
    *,
    evaluate: Callable[[Mapping[str, str]], Mapping[str, float]],
    spec_targets: Mapping[str, tuple],
    tunable: Optional[list] = None,
    rel_window: float = 0.4,
    n_evals: int = 40,
    batch_size: int = 8,
    n_candidates: int = 60,
    seed: int = 0,
) -> Dict[str, str]:
    """Refine sizing with the REAL tSS-BO Bayesian optimiser against measured KPIs.

    Tunes the ``tunable`` params (default: keys starting ``W_``/``L_``) in a
    +/-``rel_window`` box around ``seed_params``; the objective maps each candidate
    to ``evaluate``'s KPIs and a normalised-hinge distance to ``spec_targets``
    ({metric: (op, target)}). Returns the best params found (>= the seed). Composes
    ``tools.clay_optimize.tssbo_solver`` (github.com/PascalInn/tSS-BO on
    torch/gpytorch/botorch). Raises if tSS-BO is not fetched (run
    ``tools/clay_optimize/fetch_tssbo.py``)."""
    import numpy as np
    from tools.clay_optimize.tssbo_solver import minimize, tssbo_available

    if not tssbo_available():
        raise RuntimeError(
            "tSS-BO unavailable: run python tools/clay_optimize/fetch_tssbo.py "
            "and pip install botorch gpytorch torch"
        )
    keys = list(tunable) if tunable else [
        k for k in seed_params if k.upper().startswith(("W_", "L_"))
    ]
    if not keys:
        return dict(seed_params)
    base = {k: _eng_to_float(seed_params[k]) for k in keys}
    bounds = [(max(1e-12, base[k] * (1 - rel_window)), base[k] * (1 + rel_window)) for k in keys]

    def _params_for(vec) -> Dict[str, str]:
        # ONLY the tunable overrides -- evaluate's override_params then rewrites
        # just those .param lines, leaving every other definition byte-identical
        # (a full-seed rewrite is lossy on space-around-= / expression params).
        return {k: _float_to_um(float(vec[i])) for i, k in enumerate(keys)}

    def objective(X):
        rows = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
        rows = rows.reshape(-1, len(keys))
        losses = []
        for row in rows:
            kpis = evaluate(_params_for(row))
            losses.append(_hinge_loss(kpis, spec_targets))
        return np.asarray(losses, dtype="float32")

    best_x, _best_y = minimize(
        objective, bounds, n_max=int(n_evals),
        batch_size=batch_size, n_candidates=n_candidates, seed=seed,
    )
    return _params_for(best_x)


def run_presim_tune(
    *,
    presim_module: str,
    source_attr: str,
    kpi_map: Mapping[str, str],
    spec_targets: Mapping[str, tuple],
    cir: str,
    tunable: list,
    rel_window: float = 0.5,
    n_evals: int = 20,
    batch_size: int = 4,
    n_candidates: int = 8,
    seed: int = 0,
) -> Dict[str, object]:
    """End-to-end sim-in-loop sizing tune (the integration of the closed loop).

    Composes make_presim_evaluate (real ngspice KPIs) + bo_tune_sizing (real tSS-BO)
    to push a circuit's ``tunable`` sizing toward ``spec_targets``. Returns the
    baseline + tuned KPIs, the tuned params, and the before/after hinge loss.
    Validated on fc_ota: W_diff 4.4u->7.92u lifts GBW 11.2->14.6 MHz to meet a
    13 MHz target in ~15 ngspice evals."""
    from clay.frontend.onboard import _extract_params

    evaluate = make_presim_evaluate(presim_module, source_attr=source_attr, kpi_map=kpi_map)
    seed_params = _extract_params(Path(cir).read_text(encoding="utf-8"))
    baseline = dict(evaluate({}))  # committed sizing (no override)
    tuned = bo_tune_sizing(
        seed_params, evaluate=evaluate, spec_targets=spec_targets,
        tunable=tunable, rel_window=rel_window, n_evals=n_evals,
        batch_size=batch_size, n_candidates=n_candidates, seed=seed,
    )
    after = dict(evaluate(tuned))
    return {
        "baseline_kpis": baseline,
        "tuned_params": tuned,
        "tuned_kpis": after,
        "seed_loss": _hinge_loss(baseline, spec_targets),
        "tuned_loss": _hinge_loss(after, spec_targets),
        "met": _hinge_loss(after, spec_targets) == 0.0,
    }


__all__ = [
    "override_params",
    "canonicalize_candidate_sizing",
    "make_presim_evaluate",
    "make_joint_evaluate",
    "make_joint_evaluate_from_spec",
    "tune_sizing",
    "bo_tune_sizing",
    "run_presim_tune",
]
