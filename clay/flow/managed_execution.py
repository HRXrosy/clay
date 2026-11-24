"""Adapter between the legacy step protocol and owned execution resources."""
from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

from clay.flow.execution_budget import ExecutionBudget, ExecutionStopped
from clay.flow.execution_session import ExecutionSession
from clay.flow.run_workspace import InputBinding, RunWorkspace, atomic_json, json_snapshot
from clay.flow.spec import StepResult


def execute_managed(spec, *, request, out_dir, selected, budget, cancellation,
                    resume, recorder, initial_extra, execute, step_factory):
    """Resume means verified re-execution until steps have a state restore protocol."""
    budget = budget if budget is not None else ExecutionBudget()
    if spec.source_cir.resolve() not in {e.path.resolve() for e in request.files if e.role == "source"}:
        raise ValueError("INPUT_SOURCE_NOT_BOUND: compiled source must be in source closure")
    # The core adapter is always bound; installed producers additionally declare
    # their own transitive implementation, PDK assets and tool identities.
    core = Path(__file__).parent
    implementation = [InputBinding("implementation", core / name) for name in (
        "managed_execution.py", "run_workspace.py", "execution_budget.py", "execution_session.py",
        "engine.py", "spec.py", "state.py", "resource_budget.py", "step.py",
        "run_admission.py", "failure_diagnostics.py", "primitive_observation.py",
    )]
    from clay.flow.run_admission import current_admission
    admission = current_admission()
    if getattr(admission, 'role_trace_request', None) is not None:
        implementation.extend(InputBinding('implementation', core / name) for name in (
            'primitive_role_trace.py', 'primitive_role_trace_capture.py'))
    provider_file = inspect.getsourcefile(step_factory)
    if provider_file is None:
        raise ValueError("STEP_PROVIDER_IDENTITY_REQUIRED: installed Python source must be available")
    implementation.append(InputBinding("implementation", Path(provider_file)))
    request = replace(request, files=tuple(dict.fromkeys((*request.files, *implementation))))
    initial = json_snapshot(initial_extra or {})
    configuration = {"spec": json_snapshot(spec), "steps": selected,
                     "initial_extra": initial, "budget": budget.to_dict(), "resume_policy": "reexecute",
                     "step_provider": {"path": str(Path(provider_file).resolve()),
                                       "name": step_factory.__qualname__}}
    with RunWorkspace(request, out_dir, configuration=configuration, resume=resume) as workspace:
        session = ExecutionSession(project_id=request.project_id, run_id=request.run_id,
                                   invocation_id=workspace.invocation_id, event_path=workspace.event_path,
                                   budget=budget, cancellation=cancellation)
        with session.activate():
            result = execute(spec, work_dir=workspace.work_dir, selected=selected, force=True,
                             recorder=recorder, initial_extra=json_snapshot(initial),
                             execution=session, request=request, step_factory=step_factory)
            try:
                session.checkpoint()
            except ExecutionStopped as exc:
                if result.passed or result.verdict == 'unknown':
                    if selected == ['primitive-observe']:
                        result._stage_stop = exc
                    result.steps.append(StepResult("execution", "fail", str(exc),
                                                   data={"diagnostic": exc.flow_error_data()}, blocking=True))
                    result.verdict = "fail"
        if json_snapshot(spec) != configuration["spec"]:
            raise ValueError("RUN_IDENTITY_CHANGED: compiled spec mutated during execution")
        status = "complete" if result.passed or (selected == ['primitive-observe']
                                                   and result.verdict == 'unknown') else "failed"
        reasons = {step.data.get("diagnostic", {}).get("reason") for step in result.steps}
        if "EXECUTION_CANCELLED" in reasons:
            status = "cancelled"
        elif reasons & {"WALL_BUDGET_EXCEEDED", "DETERMINISTIC_BUDGET_EXCEEDED"}:
            status = "timeout"
        atomic_json(workspace.work_dir / "run_summary.json", result.to_dict())
        artifacts = {f"{step.name}.{key}": value for step in result.steps
                     for key, value in step.artifacts.items()}
        artifacts["run_summary"] = workspace.work_dir / "run_summary.json"
        try:
            workspace.finish(status=status, verdict=result.verdict, artifacts=artifacts,
                             expected_artifacts=result.artifact_identities,
                             checkpoint=session.checkpoint if status == 'complete' else None)
        except ExecutionStopped as exc:
            result.verdict = "fail"
            if selected == ['primitive-observe']:
                result._stage_stop = exc
            result.steps.append(StepResult("execution", "fail", str(exc),
                                           data={"diagnostic": exc.flow_error_data()}, blocking=True))
            atomic_json(workspace.work_dir / "run_summary.json", result.to_dict())
            workspace.stop(exc.reason)
        return result
