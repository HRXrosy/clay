"""Generic, incremental, recorded flow engine.

``run_flow`` orders the built-in steps by ``spec.steps`` (intersected with
``DEFAULT_STEPS`` so the order is the canonical dependency order), runs each
one, skips unchanged steps via the persisted :class:`FlowState`, emits a
recorder event per step when a recorder is supplied, stops on a blocking
failure, and returns a :class:`FlowResult` with the per-step results and the
overall verdict.

Nothing here is circuit-specific: every step reads what it needs from the
:class:`CircuitSpec` it is handed.
"""
from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from clay.flow.spec import DEFAULT_STEPS, CircuitSpec, FlowContext, FlowStep, StepResult
from clay.flow.execution_budget import CancellationToken, ExecutionBudget
from clay.flow.run_admission import RunAdmission, RetryRequest, current_admission
from clay.flow.run_workspace import RunRequest, artifact_identity
from clay.flow.state import FlowState
from clay.flow.step import BUILTIN_STEPS, _emit_event, make_step


class FlowSelectionError(ValueError):
    """A requested flow must actually select known executable steps."""

    def __init__(self, detail: str):
        super().__init__("STEP_SELECTION: " + detail)

    def flow_error_data(self):
        return {"reason": "STEP_SELECTION", "detail": str(self)}


@dataclass
class FlowResult:
    spec_key: str
    verdict: str  # "pass" | "fail"
    steps: list[StepResult] = field(default_factory=list)
    work_dir: Optional[Path] = None
    artifact_identities: dict[str, Any] = field(default_factory=dict, repr=False)
    execution_control: dict[str, Any] = field(default_factory=dict, repr=False)
    _stage_stop: BaseException | None = field(default=None, repr=False)
    _stage_failure: BaseException | None = field(default=None, repr=False)

    @property
    def by_step(self) -> dict[str, StepResult]:
        return {step.name: step for step in self.steps}

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "spec_key": self.spec_key,
            "verdict": self.verdict,
            "work_dir": str(self.work_dir) if self.work_dir else None,
            "steps": [
                {
                    "name": s.name,
                    "status": s.status,
                    "summary": s.summary,
                    "blocking": s.blocking,
                    "data": s.data,
                    "artifacts": s.artifacts,
                }
                for s in self.steps
            ],
        }
        if self.artifact_identities:
            payload["artifact_identities"] = self.artifact_identities
        return payload


def ordered_steps(spec: CircuitSpec) -> list[str]:
    """The step identities to run, in canonical dependency order.

    Intersect the spec's requested steps with DEFAULT_STEPS (which defines the
    order and the universe of known steps). Unknown step names on the spec are
    ignored; the canonical order is always preserved regardless of the order
    the spec listed them.
    """
    requested = set(spec.steps or DEFAULT_STEPS)
    return [name for name in DEFAULT_STEPS if name in requested and name in BUILTIN_STEPS]


def run_flow(
    spec: CircuitSpec,
    *,
    out_dir: str | Path,
    only: Optional[Iterable[str]] = None,
    force: bool = False,
    recorder: Any = None,
    initial_extra: Optional[Mapping[str, Any]] = None,
    request: RunRequest | None = None,
    budget: ExecutionBudget | None = None,
    cancellation: CancellationToken | None = None,
    resume: bool = False,
    step_factory: Callable[[str], FlowStep] | None = None,
    project_root: str | Path | None = None,
    task_id: str | None = None,
    retry: RetryRequest | None = None,
    stage_retry: RetryRequest | None = None,
    stage_unwind_seconds: float | None = None,
    stage_role_trace_seconds: float | None = None,
    stage_role_trace_version: int | None = None,
    production_quiescence: Any = None,
) -> FlowResult:
    """Run the flow for ``spec`` under ``out_dir``.

    Parameters
    ----------
    only:
        If given, restrict execution to this subset of step names (still in
        canonical order). The ``sign`` step's verdict only reflects the steps
        that actually ran.
    force:
        Ignore the incremental cache and re-run every selected step.
    recorder:
        Optional ``clay.runtime.recorder.RunRecorder``; one ``flow_<step>``
        event is emitted per executed (or skipped) step.
    initial_extra:
        Explicit per-run materialization data shared by canonical steps.  The
        joint sizing bridge uses this for candidate sizing; no module-global
        state or circuit branch is involved. A bounded, admitted, built-in
        layout invocation may explicitly supply ``native_row_candidates`` as
        a tuple of source-bound NativeRowCandidate choices. Without that input
        the original declared native production path is unchanged.
    """
    if not spec.artifacts_resolved:
        raise ValueError("flow artifact inputs have not been resolved; a preflight projection cannot execute")
    selected = ordered_steps(spec)
    if not selected or set(spec.steps) - set(DEFAULT_STEPS):
        raise FlowSelectionError("empty or unknown declared steps")
    primitive_stage = False
    if only is not None:
        only_set = set(only)
        if only_set == {'primitive-observe'}:
            if 'layout' not in selected:
                raise FlowSelectionError('primitive observation requires a declared layout step')
            primitive_stage = True
            selected = ['primitive-observe']
        elif not only_set or only_set - set(selected):
            raise FlowSelectionError("empty, unknown or undeclared selected steps")
        else:
            selected = [name for name in selected if name in only_set]
    if (stage_retry is not None or stage_unwind_seconds is not None
            or stage_role_trace_seconds is not None or stage_role_trace_version is not None or production_quiescence is not None) and not primitive_stage:
        raise ValueError('stage controls require primitive-observe')
    if stage_role_trace_version is not None and (type(stage_role_trace_version) is not int or stage_role_trace_version != 2 or stage_role_trace_seconds is None):
        raise ValueError("role trace version 2 requires a role trace budget")
    if primitive_stage and initial_extra:
        raise ValueError('primitive observation does not accept unbound initial_extra')
    if primitive_stage and step_factory is not None:
        raise ValueError('primitive observation requires the built-in step provider')
    if initial_extra is not None and initial_extra.get('native_composite_observation') is not None:
        from clay.flow.native_composite_observation import NativeCompositeObservation
        if type(initial_extra['native_composite_observation']) is not NativeCompositeObservation:
            raise TypeError('native composite observation requires an exact request')
        if set(initial_extra) - {'native_composite_observation','native_row_candidates'}:
            raise ValueError('native composite observation does not accept other initial data')
        if selected != ['layout'] or request is not None or step_factory is not None:
            raise ValueError('native composite observation requires the built-in in-process layout caller')
        if not isinstance(budget, ExecutionBudget) or budget.wall_seconds is None:
            raise ValueError('native composite observation requires a finite wall budget')
    if initial_extra is not None and initial_extra.get('native_row_candidates') is not None:
        from clay.layout.native_row_candidate import NativeRowCandidate
        choices=initial_extra['native_row_candidates']
        if (type(choices) is not tuple or not choices or
                any(type(choice) is not NativeRowCandidate for choice in choices)):
            raise TypeError('native row candidate request requires a nonempty exact tuple')
        if (selected != ['layout'] or request is not None or step_factory is not None or
                project_root is None or task_id is None or
                not isinstance(budget, ExecutionBudget) or budget.wall_seconds is None):
            raise ValueError('native row candidates require bounded admitted built-in layout')
    if initial_extra is not None:
        group_endpoint = initial_extra.get('diagnostic_route_observer')
        if (type(group_endpoint).__module__ == 'clay.layout.route.group_endpoint_observation'
                and type(group_endpoint).__name__ == 'GroupEndpointObserver'):
            from clay.layout.route.group_endpoint_observation import GroupEndpointObserver
            if type(group_endpoint) is not GroupEndpointObserver:
                raise TypeError('group endpoint observation has a foreign request type')
            if (selected != ['layout'] or request is not None or step_factory is not None
                    or not isinstance(budget, ExecutionBudget) or budget.wall_seconds is None
                    or any(initial_extra.get(key) is not None for key in
                           ('pre_route_gate', 'diagnostic_candidate_request',
                            'diagnostic_candidate_batch_request', 'native_composite_observation'))):
                raise ValueError('group endpoint observation requires one admitted built-in layout and finite budget')
            if project_root is None or task_id is None:
                raise ValueError('group endpoint observation requires an admitted task')
    if initial_extra is not None and initial_extra.get('pre_route_gate') is not None:
        from clay.layout.route.live_diagnostic import PreRouteGate
        if type(initial_extra['pre_route_gate']) is not PreRouteGate:
            raise TypeError('pre-route gate must be an exact PreRouteGate')
        if 'layout' not in selected or request is not None or step_factory is not None:
            raise ValueError('pre-route gate requires the built-in in-process layout caller')
        if 'sizing' in selected:
            raise ValueError('pre-route gate requires sizing to be completed separately')
        if (initial_extra.get('diagnostic_route_observer') is not None
                or initial_extra.get('diagnostic_candidate_request') is not None
                or initial_extra.get('diagnostic_candidate_batch_request') is not None):
            raise ValueError('pre-route gate cannot share a terminal diagnostic observer')
    if initial_extra is not None and initial_extra.get('candidate_route_request') is not None:
        raise ValueError('experimental candidate route requests are no longer supported')
    if initial_extra is not None and initial_extra.get('local_routing_scene_request') is not None:
        from clay.layout.route.local_routing_scene import LocalRoutingSceneRequest
        if type(initial_extra['local_routing_scene_request']) is not LocalRoutingSceneRequest:
            raise TypeError('local routing scene needs a LocalRoutingSceneRequest')
        if (selected != ['layout'] or request is not None or step_factory is not None
                or not isinstance(budget, ExecutionBudget) or budget.wall_seconds is None
                or project_root is None or task_id is None):
            raise ValueError('local routing scene requires a bounded admitted layout-only flow')
    if (initial_extra is not None and initial_extra.get("diagnostic_candidate_request") is not None
            and initial_extra.get("diagnostic_candidate_batch_request") is not None):
        raise ValueError("single and batch candidate diagnostics are mutually exclusive")
    if initial_extra is not None and initial_extra.get('diagnostic_resource_pair_request') is not None:
        if request is not None:
            raise ValueError('UNKNOWN_MANAGED_HANDOFF_UNSUPPORTED')
        from clay.layout.route.resource_pair_transaction import ResourcePairRequest
        pair_request = initial_extra['diagnostic_resource_pair_request']
        if type(pair_request) is not ResourcePairRequest:
            raise TypeError('resource pair requires a typed request')
        pair_request.__post_init__()
        if initial_extra.get('diagnostic_resource_lane_site_request') is None:
            raise ValueError('resource pair requires resource lane site')
    if initial_extra is not None and initial_extra.get('diagnostic_resource_lane_site_request') is not None:
        if request is not None:
            raise ValueError('UNKNOWN_MANAGED_HANDOFF_UNSUPPORTED')
        from clay.layout.route.resource_lane_site import ResourceLaneSiteRequest
        lane_site_request = initial_extra['diagnostic_resource_lane_site_request']
        if type(lane_site_request) is not ResourceLaneSiteRequest:
            raise TypeError('resource lane site requires a typed request')
        lane_site_request.__post_init__()
        if initial_extra.get('diagnostic_resource_setup_request') is None:
            raise ValueError('resource lane site requires resource setup')
        if (initial_extra.get('diagnostic_resource_access_request') is not None
                or initial_extra.get('diagnostic_resource_lane_survey_request') is not None):
            raise ValueError('resource lane site, lane survey and access are mutually exclusive')
    if initial_extra is not None and initial_extra.get('diagnostic_resource_lane_survey_request') is not None:
        if request is not None:
            raise ValueError('UNKNOWN_MANAGED_HANDOFF_UNSUPPORTED')
        from clay.layout.route.resource_lane_survey import ResourceLaneSurveyRequest
        lane_survey_request = initial_extra['diagnostic_resource_lane_survey_request']
        if type(lane_survey_request) is not ResourceLaneSurveyRequest:
            raise TypeError('resource lane survey requires a typed request')
        lane_survey_request.__post_init__()
        if initial_extra.get('diagnostic_resource_setup_request') is None:
            raise ValueError('resource lane survey requires resource setup')
        if initial_extra.get('diagnostic_resource_access_request') is not None:
            raise ValueError('resource lane survey and access are mutually exclusive')
    if initial_extra is not None and initial_extra.get('diagnostic_resource_access_request') is not None:
        if request is not None:
            raise ValueError('UNKNOWN_MANAGED_HANDOFF_UNSUPPORTED')
        from clay.layout.route.resource_access_diagnostic import ResourceAccessRequest
        access_request = initial_extra['diagnostic_resource_access_request']
        if type(access_request) is not ResourceAccessRequest:
            raise TypeError('resource access requires a typed request')
        access_request.__post_init__()
        if initial_extra.get('diagnostic_resource_setup_request') is None:
            raise ValueError('resource access requires resource setup')
    if initial_extra is not None and initial_extra.get('diagnostic_resource_setup_request') is not None:
        if request is not None:
            raise ValueError('UNKNOWN_MANAGED_HANDOFF_UNSUPPORTED')
        from clay.layout.route.resource_setup_diagnostic import ResourceSetupRequest
        resource_request = initial_extra['diagnostic_resource_setup_request']
        if type(resource_request) is not ResourceSetupRequest:
            raise TypeError('resource setup requires a typed request')
        resource_request.__post_init__()
        if (initial_extra.get('diagnostic_candidate_batch_request') is None
                or initial_extra.get('diagnostic_source_handoff_request') is None):
            raise ValueError('resource setup requires candidate batch and source handoff')
    if initial_extra is not None and initial_extra.get('diagnostic_source_handoff_request') is not None:
        if request is not None:
            raise ValueError('UNKNOWN_MANAGED_HANDOFF_UNSUPPORTED')
        from clay.layout.route.source_handoff import SourceHandoffRequest
        if type(initial_extra['diagnostic_source_handoff_request']) is not SourceHandoffRequest:
            raise TypeError('source handoff requires a typed request')
        if initial_extra.get('diagnostic_candidate_batch_request') is None:
            raise ValueError('source handoff requires candidate batch')
    if (initial_extra is not None and (initial_extra.get("diagnostic_candidate_request") is not None
            or initial_extra.get("diagnostic_candidate_batch_request") is not None)
            and "layout" not in selected):
        raise ValueError("candidate diagnostic requires the layout step")
    if request is None and (resume or step_factory is not None):
        raise ValueError("managed execution options require a RunRequest")
    from clay.flow.execution_session import current_session
    if request is not None and (current_admission() is not None or current_session() is not None):
        raise ValueError("nested owned executions require an independent top-level invocation")
    physical = bool(set(selected) & {"sizing", "layout", "drc", "lvs", "pex", "post", "primitive-observe"})
    parent = current_admission()
    if parent is not None and getattr(parent, 'is_primitive_stage', False):
        raise ValueError('nested flow cannot execute under primitive stage admission')
    if parent is not None and request is None:
        if project_root is not None or task_id is not None or retry is not None or budget is not None or cancellation is not None:
            raise ValueError("nested flows inherit the parent execution controls")
        # Sizing candidates are subordinate work inside the parent's durable
        # attempt. They consume its same timer and cancellation token.
        return _execute_legacy(spec, out_dir, selected, force, recorder, initial_extra,
                               execution=current_session())
    if physical:
        from clay.flow.failure_diagnostics import summarize_failure
        if primitive_stage:
            from clay.flow.execution_budget import ExecutionStopped
            from clay.flow.run_admission import RunAdmissionError
            if not isinstance(budget, ExecutionBudget) or budget.wall_seconds is None:
                raise RunAdmissionError('PRIMITIVE_STAGE_REQUIRES_FINITE_WALL_BUDGET')
            if cancellation is not None and not isinstance(cancellation, CancellationToken):
                raise TypeError('primitive stage cancellation must be a CancellationToken')
            if stage_role_trace_seconds is not None:
                from clay.flow.execution_budget import _validate_seconds
                stage_role_trace_seconds = _validate_seconds(stage_role_trace_seconds, 'stage_role_trace_seconds')
                if stage_role_trace_seconds > budget.wall_seconds:
                    raise ValueError('role trace budget must fit inside the stage wall budget')
            stage_wall_start = time.monotonic()
            def stage_precheck():
                if cancellation is not None and cancellation.reason is not None:
                    raise ExecutionStopped('EXECUTION_CANCELLED', cancellation_reason=cancellation.reason)
                elapsed = time.monotonic()-stage_wall_start
                if stage_role_trace_seconds is not None and elapsed >= stage_role_trace_seconds:
                    raise ExecutionStopped('WALL_BUDGET_EXCEEDED', scope='primitive-role-trace',
                                           limit_seconds=stage_role_trace_seconds, consumed_seconds=elapsed)
                if elapsed >= budget.wall_seconds:
                    raise ExecutionStopped('WALL_BUDGET_EXCEEDED', limit_seconds=budget.wall_seconds,
                                           consumed_seconds=elapsed)
            stage_precheck()
            from clay.flow.run_admission import PrimitiveStageAdmission
            from clay import parser as clay_parser
            from clay.flow import (control_stop, managed_execution, native_owned_container,
                                   native_physical, owned_process, primitive_mim_role_trace,
                                   primitive_observation, primitive_role_trace,
                                   primitive_role_trace_capture, resource_budget,
                                   run_admission, run_workspace, step)
            from clay.layout import intent as layout_intent_module, native_sky130
            dependencies = set(native_sky130._producer_dependencies(checkpoint=stage_precheck))
            dependencies.update(str(Path(module.__file__).resolve()) for module in (
                clay_parser, control_stop, layout_intent_module, managed_execution,
                native_owned_container, native_physical, owned_process,
                primitive_mim_role_trace, primitive_observation, primitive_role_trace,
                primitive_role_trace_capture, resource_budget, run_admission,
                run_workspace, step, native_sky130))
            clay_root = Path(__file__).resolve().parents[1]
            for subtree in (clay_root / 'parser_modules', clay_root / 'frontend',
                            clay_root / 'pdk', clay_root / 'verification',
                            clay_root / 'layout' / 'intent_modules'):
                if subtree.is_dir():
                    for path in subtree.rglob('*.py'):
                        stage_precheck()
                        if path.is_file():
                            dependencies.add(str(path.resolve()))
            for path in clay_root.rglob('*.lark'):
                stage_precheck()
                if path.is_file():
                    dependencies.add(str(path.resolve()))
            stage_precheck()
            source_paths = {'design': Path(spec.case_dir)/'source/design.clay',
                            'spice': Path(spec.source_cir),
                            'runtime': Path(spec.native_runtime_manifest) if spec.native_runtime_manifest else None,
                            'dependencies': tuple(sorted(dependencies))}
            with PrimitiveStageAdmission(spec, project_root=project_root if project_root is not None else
                    (request.project_root if request is not None else None), task_id=task_id,
                    budget=budget, retry=retry, stage_retry=stage_retry, source_paths=source_paths,
                    cancellation=cancellation, managed=request is not None,
                    unwind_budget_seconds=stage_unwind_seconds, initial_wall_start=stage_wall_start,
                    role_trace_seconds=stage_role_trace_seconds, role_trace_version=stage_role_trace_version,
                    production_quiescence=production_quiescence) as admission:
                result = _dispatch_flow(spec, out_dir=out_dir, selected=selected, force=True,
                    recorder=recorder, initial_extra=None, request=request, budget=budget,
                    cancellation=admission.cancellation, resume=resume, step_factory=step_factory,
                    admission=admission)
                managed_terminal = result.work_dir.parent/'terminal.json' if request is not None else None
                primitive_steps = [step for step in result.steps if step.name == 'primitive-observe']
                result_step = primitive_steps[0] if len(primitive_steps) == 1 else None
                evidence = result_step.data.get('container_evidence_path') if result_step else None
                diagnostic = {'reason': 'PRIMITIVE_OBSERVATION_UNKNOWN' if result.verdict == 'unknown'
                              else 'PRIMITIVE_OBSERVATION_FAILED',
                              'unknown': ['INPUT_CONSUMPTION_IDENTITY', 'LOADED_CODE_IDENTITY', 'LOADED_DOCKER_CLI_IDENTITY',
                                          'TOOLKIT_SOURCE_EXPECTATION', 'SOURCE_TERMINAL_BINDING',
                                          'BODY_CUT_COVERAGE', 'PRIMITIVE_STAGE_ADMISSION']}
                if admission.role_trace_request is not None and result.verdict != 'unknown' and result_step:
                    diagnostic['producer_failure'] = result_step.data
                if evidence:
                    admission.finish_stage(passed=result.verdict == 'unknown',
                        diagnostic=diagnostic,
                        evidence_path=Path(evidence), checkpoint=admission.checkpoint, stop=result._stage_stop,
                        managed_terminal_path=managed_terminal, original_error=result._stage_failure)
                elif not admission.has_resource_plan:
                    if admission.role_trace_request is None:
                        diagnostic = {'reason': 'PRIMITIVE_OBSERVATION_UNAVAILABLE' if result.verdict == 'unknown'
                                      else 'PRIMITIVE_OBSERVATION_FAILED'}
                    admission.finish_without_launch_unknown(
                        diagnostic=diagnostic,
                        failed=result.verdict != 'unknown', managed_terminal_path=managed_terminal,
                        stop=result._stage_stop, original_error=result._stage_failure)
                elif admission.role_trace_request is not None:
                    original = result._stage_failure or result._stage_stop
                    if original is not None:
                        original.add_note('Primitive resource cleanup is unverified; active stage retained')
                        raise original
                result.execution_control = admission.control
                return result
        with RunAdmission(spec, project_root=project_root if project_root is not None else
                          (request.project_root if request is not None else None),
                          task_id=task_id, budget=budget, retry=retry) as admission:
            if initial_extra and initial_extra.get('native_row_candidates') is not None:
                from dataclasses import asdict
                admission.native_row_candidates=tuple(asdict(choice) for choice in initial_extra['native_row_candidates'])
            result = _dispatch_flow(spec, out_dir=out_dir, selected=selected, force=force or retry is not None,
                                    recorder=recorder, initial_extra=initial_extra, request=request,
                                    budget=budget, cancellation=cancellation, resume=resume,
                                    step_factory=step_factory, admission=admission)
            admission.finish(passed=result.passed, diagnostic=summarize_failure(result))
            result.execution_control = admission.control
            return result
    if retry is not None:
        raise ValueError("retry applies to physical flow attempts")
    return _dispatch_flow(spec, out_dir=out_dir, selected=selected, force=force,
                          recorder=recorder, initial_extra=initial_extra, request=request,
                          budget=budget, cancellation=cancellation, resume=resume,
                          step_factory=step_factory)


def _dispatch_flow(spec, *, out_dir, selected, force, recorder, initial_extra,
                   request, budget, cancellation, resume, step_factory, admission=None):
    if request is not None:
        from clay.flow.managed_execution import execute_managed
        return execute_managed(
            spec, request=request, out_dir=out_dir, selected=selected, budget=budget,
            cancellation=cancellation, resume=resume, recorder=recorder,
            initial_extra=initial_extra, execute=_execute_flow,
            step_factory=step_factory or make_step,
        )
    from clay.flow.execution_session import current_session
    if current_session() is not None:
        if budget is not None or cancellation is not None:
            raise ValueError("nested flows inherit the active execution session")
        return _execute_legacy(spec, out_dir, selected, force, recorder, initial_extra,
                               execution=current_session())
    if admission is not None or budget is not None or cancellation is not None:
        from uuid import uuid4
        from clay.flow.execution_session import ExecutionSession
        invocation_id = admission.invocation_id if admission else uuid4().hex
        event_dir = admission.attempt_dir if admission else Path(out_dir) / "execution" / invocation_id
        session = ExecutionSession(project_id=admission.key if admission else "standalone",
                                   run_id="flow", invocation_id=invocation_id,
                                   event_path=event_dir / "events.jsonl",
                                   budget=budget or ExecutionBudget(), cancellation=cancellation)
        with session.activate():
            return _execute_legacy(spec, out_dir, selected, force, recorder, initial_extra, execution=session)
    return _execute_legacy(spec, out_dir, selected, force, recorder, initial_extra)


def _execute_legacy(spec, out_dir, selected, force, recorder, initial_extra, execution=None):
    work_dir = Path(out_dir) / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    result = None
    try:
        if execution:
            execution.checkpoint()
        result = _execute_flow(spec, work_dir=work_dir, selected=selected, force=force,
                               recorder=recorder, initial_extra=initial_extra, execution=execution)
        if execution:
            execution.checkpoint()
    except Exception as exc:
        if result is None:
            result = FlowResult(spec_key=spec.key, verdict="fail", work_dir=work_dir)
        result.verdict = "fail"
        if selected == ['primitive-observe']:
            from clay.flow.execution_budget import ExecutionStopped
            if isinstance(exc, ExecutionStopped) and result._stage_stop is None:
                if result._stage_failure is None:
                    result._stage_stop = exc
                else:
                    result._stage_failure.add_note('Post-error execution checkpoint: ' + str(exc.flow_error_data()))
            elif not isinstance(exc, ExecutionStopped):
                result._stage_failure = exc
        result.steps.append(StepResult("execution", "fail", str(exc), data=_exception_data(exc), blocking=True))
        from clay.flow.run_workspace import atomic_json
        atomic_json(work_dir / "run_summary.json", result.to_dict())
    return result


def _execute_flow(spec, *, work_dir, selected, force=False, recorder=None,
                  initial_extra=None, execution=None, request=None, step_factory=None):
    """Single execution loop; managed attempts deliberately never reuse FlowState."""
    # Native physical results require sealed raw-report replay and a live bound
    # toolchain. FlowState restores neither; its input-only cache is insufficient.
    # Execute these explicitly opted-in flows afresh rather than reusing a PASS.
    force = force or getattr(spec, "native_runtime_manifest", None) is not None
    ctx = FlowContext(spec=spec, work_dir=work_dir, recorder=recorder, force=force,
                      execution=execution)
    if initial_extra:
        ctx.extra.update(dict(initial_extra))
    owned = request is not None
    state = None if owned else FlowState(work_dir)

    results: list[StepResult] = []
    artifact_identities = {}
    stage_stop = None
    stage_failure = None
    blocked = False
    for name in selected:
        step = None if owned else make_step(name)
        stage_result = None

        if blocked:
            result = StepResult(
                name,
                "skipped",
                "upstream blocking failure; step not run",
                data={"reason": "blocked"},
            )
            ctx.results[name] = result
            results.append(result)
            _emit_event(ctx, result)
            continue

        # Sizing needs its compile-time source identity and produces promoted
        # context. Legacy FlowState restores neither, so reexecute both stages
        # when selected together instead of accepting an unbound cache hit.
        nonrestorable = (name == "sizing" or (name == "compile" and "sizing" in selected)
                         or (name == "layout" and (ctx.extra.get("local_routing_scene_request") is not None
                                                   or ctx.extra.get("diagnostic_route_observer") is not None
                                                   or ctx.extra.get("native_composite_observation") is not None
                                                   or ctx.extra.get("pre_route_gate") is not None
                                                   or ctx.extra.get("diagnostic_candidate_request") is not None
                                                   or ctx.extra.get("diagnostic_candidate_batch_request") is not None)))
        if not nonrestorable and state is not None and state.should_skip(name, _safe_inputs(step, spec, ctx), force):
            prior = state.recorded(name) or {}
            result = StepResult(
                name,
                "skipped",
                f"inputs unchanged (prior status {prior.get('status')})",
                data={"prior_status": prior.get("status"), "input_sha": prior.get("input_sha")},
            )
            ctx.results[name] = result
            results.append(result)
            _emit_event(ctx, result)
            continue

        try:
            with execution.phase("flow." + name) if execution else nullcontext() as phase:
                if owned:
                    step = step_factory(name)
                    if getattr(step, "owned_execution_protocol", None) != "clay.owned-step/v1":
                        raise ValueError("UNSUPPORTED_STEP_EXECUTION: producer has no owned-step contract")
                    _managed_inputs(step, spec, ctx, request)
                result = step.run(spec, ctx)
                if name == 'primitive-observe':
                    stage_result = result
                if execution and (not isinstance(result, StepResult) or result.name != name
                                  or result.status not in {"pass", "warn", "fail", "skipped"}):
                    raise ValueError("INVALID_STEP_RESULT: wrong identity or status")
                if phase and owned:
                    identities = {f"{name}.{key}": artifact_identity(work_dir, path)
                                  for key, path in result.artifacts.items()}
                    artifact_identities.update(identities)
                    phase.annotate(step_status=result.status, artifacts=identities)
                elif phase:
                    phase.annotate(step_status=result.status)
        except Exception as exc:  # noqa: BLE001
            if name == 'primitive-observe':
                from clay.flow.execution_budget import ExecutionStopped
                if isinstance(exc, ExecutionStopped):
                    stage_stop = exc
                else:
                    stage_failure = exc
            failure_data = _exception_data(exc)
            if name == 'primitive-observe' and stage_result is not None:
                evidence = stage_result.data.get('container_evidence_path')
                if isinstance(evidence, str):
                    failure_data['container_evidence_path'] = evidence
            if name == "layout":
                from clay.layout.route.group_fallback_artifacts import observations
                partials = observations(work_dir)
                if partials:
                    failure_data["group_fallback_observations"] = {
                        key: str(path) for key, path in partials.items()}
                    failure_data["group_fallback_scope"] = "partial routing observations; no physical admission"
            result = StepResult(
                name,
                "fail",
                f"{type(exc).__name__}: {exc}",
                data=failure_data,
                blocking=True,
            )

        ctx.results[name] = result
        results.append(result)
        # Re-hash inputs post-run (e.g. layout writes the GDS that DRC hashes).
        if state is not None:
            state.record(name, _safe_inputs(step, spec, ctx), result.status)
        _emit_event(ctx, result)

        if result.blocking and result.status == "fail":
            blocked = True

    verdict = _overall_verdict(results)
    if selected == ['primitive-observe'] and len(results) == 1 and results[0].status == 'warn':
        verdict = 'unknown'
    elif owned and any(result.status != "pass" for result in results):
        verdict = "fail"
    flow_result = FlowResult(
        spec_key=spec.key,
        verdict=verdict,
        steps=results,
        work_dir=work_dir,
        artifact_identities=artifact_identities,
        _stage_stop=stage_stop,
        _stage_failure=stage_failure,
    )
    # Persist the complete ordered step evidence for audit.  This is written
    # after SignStep so its artifact path is included; an atomic replace keeps
    # an interrupted run from masquerading as a complete summary.
    summary_path = work_dir / "run_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(flow_result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    return flow_result


def _managed_inputs(step, spec, ctx, request):
    bound = {entry.path.resolve() for entry in request.files}
    for raw_path in step.inputs(spec, ctx):
        path = Path(raw_path).resolve()
        if path not in bound and not path.is_relative_to(ctx.work_dir):
            raise ValueError("STEP_INPUT_NOT_BOUND: " + str(path))


def _exception_data(exc: BaseException) -> dict[str, Any]:
    """Preserve optional typed failure evidence without masking the failure."""
    data: dict[str, Any] = {
        "error": str(exc),
        "error_type": type(exc).__name__,
    }
    evidence_path = getattr(exc, 'native_container_evidence_path', None)
    if isinstance(evidence_path, str):
        data['container_evidence_path'] = evidence_path
    notes = getattr(exc, "__notes__", None)
    if type(notes) is list:
        captured = [note[:512] for note in notes[:8] if type(note) is str]
        if captured:
            data["exception_notes"] = captured
    provider = getattr(exc, "flow_error_data", None)
    if not callable(provider):
        return data
    try:
        diagnostic = provider()
        if not isinstance(diagnostic, Mapping):
            raise TypeError("flow_error_data() must return a mapping")
        # Bound traversal before serialization; a stopped route must not start
        # an unbounded second job merely to describe the failure.
        from clay.flow.execution_session import _bounded_snapshot, _DetailLimit
        try:
            data["diagnostic"] = _bounded_snapshot(diagnostic, 16 * 1024)
        except _DetailLimit:
            code = diagnostic.get("reason", diagnostic.get("code"))
            data["diagnostic"] = {"reason": code if isinstance(code, str) and len(code) <= 128 else
                                  "DIAGNOSTIC_LIMIT_EXCEEDED", "details_omitted": True}
    except Exception as diagnostic_error:  # noqa: BLE001
        data["diagnostic_error"] = (
            f"{type(diagnostic_error).__name__}: {diagnostic_error}"
        )
    return data


def run_portfolio(
    *,
    out_dir: str | Path,
    only: Optional[Iterable[str]] = None,
    force: bool = False,
    circuits: Optional[Iterable[str]] = None,
    recorder: Any = None,
    budget: ExecutionBudget | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run :func:`run_flow` for every circuit in the discovered registry.

    The portfolio runner is the project-level automation surface: it walks the
    registry (``clay.flow.spec.discover_registry()``) and runs the flow for each
    circuit under ``out_dir/<circuit_key>``, tolerating per-circuit failures so
    one circuit blowing up never aborts the rest.

    Parameters
    ----------
    out_dir:
        Parent directory; each circuit runs under ``out_dir/<key>``.
    only:
        Restrict every circuit's flow to this subset of step names (e.g.
        ``("compile", "layout")`` for the tool-free / no-Docker subset).
    force:
        Ignore the incremental cache for every circuit.
    circuits:
        Optional subset of registry keys to run. ``None`` runs all discovered
        circuits. Unknown keys are reported under ``errors`` rather than raising.
    recorder:
        Optional ``RunRecorder`` threaded into each circuit's flow.

    Returns
    -------
    dict
        ``{"circuits": {key: {...}}, "summary": {...}, "errors": {...}}`` where
        each circuit entry is
        ``{"verdict", "per_step_status", "work_dir", "spec_key", "display_name"}``.
        A circuit whose discovery/flow raised is recorded with
        ``verdict == "error"`` (honest: never hidden) and the exception text.
    """
    from clay.flow.spec import discover_registry

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    only_tuple = tuple(only) if only is not None else None

    registry: dict[str, CircuitSpec] = {}
    discover_error: Optional[str] = None
    try:
        registry = discover_registry()
    except Exception as exc:  # noqa: BLE001 - discovery failure must not raise
        discover_error = f"{type(exc).__name__}: {exc}"

    if circuits is not None:
        requested = list(dict.fromkeys(circuits))
    else:
        requested = sorted(registry)

    circuits_out: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    if discover_error is not None:
        errors["__discover__"] = discover_error

    n_pass = 0
    n_fail = 0
    for key in requested:
        spec = registry.get(key)
        if spec is None:
            errors[key] = (
                "unknown circuit (not in discovered registry)"
                if discover_error is None
                else f"registry unavailable: {discover_error}"
            )
            circuits_out[key] = {
                "spec_key": key,
                "display_name": key,
                "verdict": "error",
                "per_step_status": {},
                "work_dir": None,
                "error": errors[key],
            }
            n_fail += 1
            continue
        try:
            result = run_flow(
                spec,
                out_dir=out_root / key,
                only=only_tuple,
                force=force,
                recorder=recorder,
                **({"budget": budget} if budget is not None else {}),
                **({"project_root": project_root} if project_root is not None else {}),
            )
        except Exception as exc:  # noqa: BLE001 - one circuit must not abort the rest
            from clay.flow.run_admission import RunAdmissionError
            errors[key] = f"{type(exc).__name__}: {exc}"
            circuits_out[key] = {
                "spec_key": spec.key,
                "display_name": spec.display_name,
                "verdict": "error",
                "per_step_status": {},
                "work_dir": str(out_root / key / "work"),
                "error": errors[key],
                **({"execution_control": exc.to_dict()} if isinstance(exc, RunAdmissionError) else {}),
            }
            n_fail += 1
            continue

        per_step_status = {s.name: s.status for s in result.steps}
        from clay.flow.failure_diagnostics import summarize_failure
        circuits_out[key] = {
            "spec_key": result.spec_key,
            "display_name": spec.display_name,
            "verdict": result.verdict,
            "per_step_status": per_step_status,
            "work_dir": str(result.work_dir) if result.work_dir else None,
            "execution_control": result.execution_control,
            "failure_summary": summarize_failure(result),
        }
        if result.passed:
            n_pass += 1
        else:
            n_fail += 1

    summary = {
        "n_circuits": len(requested),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "only": list(only_tuple) if only_tuple is not None else None,
    }
    return {"circuits": circuits_out, "summary": summary, "errors": errors}


def _safe_inputs(step: Any, spec: CircuitSpec, ctx: FlowContext) -> list[Path]:
    try:
        return [Path(p) for p in step.inputs(spec, ctx)]
    except Exception:  # noqa: BLE001
        return []


def _overall_verdict(results: list[StepResult]) -> str:
    """Pass iff no executed step hard-failed.

    A ``sign`` step, when present, encodes the spec-driven gate/KPI/consistency
    verdict; its failure (blocking) propagates here. Steps that are ``warn`` or
    ``skipped`` do not by themselves fail the flow — only a ``fail`` does.
    """
    for result in results:
        if result.status == "fail":
            return "fail"
    return "pass"


__all__ = ["FlowResult", "FlowSelectionError", "ordered_steps", "run_flow", "run_portfolio"]
