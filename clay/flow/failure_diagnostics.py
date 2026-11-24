"""Bounded failure summaries for CLI/API callers; never an admission verdict.

Only explicit producer codes are classified. Messages, process exit codes,
inventory counts and artifact filenames cannot prove a physical rejection.
The original result is neither modified nor serialized wholesale.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import PurePath, PurePosixPath, PureWindowsPath
from typing import Any


SCHEMA = "clay.flow-failure/v1"
MAX_STEPS = 256
MAX_CHECKS = 128
MAX_EVIDENCE_PATHS = 8
MAX_PATH_CHARS = 1024
MAX_PREVIEW_BYTES = 4096

# Exact codes are the compatibility boundary. Unknown new codes remain unknown
# until their producer contract is reviewed; do not replace this with prefixes.
_CODES = {
    "unsupported": frozenset({
        "UNSUPPORTED_STEP_EXECUTION", "LAYOUT_PROVIDER_UNSUPPORTED",
        "PLACEMENT_PRODUCER_UNAVAILABLE", "DEVICE_MATERIALIZER_UNAVAILABLE",
        "SOURCE_DEVICE_UNSUPPORTED", "SOURCE_PROJECTION_UNSUPPORTED",
        "REFERENCE_GENERATION_PROVIDER_UNAVAILABLE", "COMPARISON_PRODUCER_UNAVAILABLE",
        "PDK_ADAPTER_UNAVAILABLE", "NATIVE_MODEL_UNAVAILABLE",
        "NATIVE_GEOMETRY_PROVIDER_UNAVAILABLE", "NATIVE_GEOMETRY_UNSUPPORTED",
        "NATIVE_MIXED_TOPOLOGY_UNSUPPORTED", "NONMOS_MATCHING_UNSUPPORTED",
        "DIAGNOSTIC_OBSERVER_UNAVAILABLE", "CANDIDATE_DIAGNOSTIC_UNAVAILABLE",
    }),
    "input": frozenset({
        "CASE_INPUT_INVALID", "INPUT_MISSING", "INPUT_PATH_OUTSIDE_PROJECT",
        "UNKNOWN_FIELD", "UNKNOWN_REQUEST_STAGE", "INTENT_BINDING_FAILED",
        "INTENT_LOWERING_FAILED", "SOURCE_DEVICE_INVALID", "NATIVE_GEOMETRY_INPUT_INVALID",
        "INPUT_SOURCE_NOT_BOUND", "STEP_INPUT_NOT_BOUND", "BINDING_ROLE_MISSING",
        "DUPLICATE_INPUT_BINDING", "STEP_SELECTION", "RUN_IDENTITY_CHANGED",
        "RUN_ARTIFACT_CHANGED", "UNRESOLVED_COMPACT_INTENT", "CANDIDATE_BOUNDARY_CHANGED",
    }),
    "environment": frozenset({
        "TOOL_UNAVAILABLE", "TOOL_IDENTITY_REQUIRED", "TOOL_NOT_FOUND",
        "DEPENDENCY_UNAVAILABLE", "RUNTIME_UNAVAILABLE", "RUNTIME_VALIDATION_FAILED",
    }),
    "budget_or_cancel": frozenset({
        "EXECUTION_CANCELLED", "WALL_BUDGET_EXCEEDED", "DETERMINISTIC_BUDGET_EXCEEDED",
        "RESOURCE_BUDGET_EXCEEDED", "DIAGNOSTIC_BUDGET_EXCEEDED",
        "EDGE_EXPANSION_BUDGET", "SOLVE_BUDGET_EXCEEDED",
        "NEGOTIATION_WATCHDOG", "TIMEOUT",
    }),
    "incomplete": frozenset({
        "EXECUTION_INCOMPLETE", "RUN_INTERRUPTED", "MISSING_TERMINAL",
        "EVIDENCE_INCOMPLETE", "DIAGNOSTIC_LIMIT_EXCEEDED",
    }),
    "diagnostic_stop": frozenset({"DIAGNOSTIC_OBSERVATION_COMPLETE"}),
    "internal_error": frozenset({
        "INTERNAL_ERROR", "INVALID_STEP_RESULT", "STEP_PROVIDER_IDENTITY_REQUIRED",
        "PDK_ADAPTER_ERROR", "SOURCE_DEVICE_CHECK_ERROR", "NATIVE_GEOMETRY_PROVIDER_ERROR",
        "CAPABILITY_CHECK_FAILED", "INVALID_RUN_TERMINAL",
    }),
}
_PHYSICAL_CODES = frozenset({
    "PHYSICAL_REJECTION", "DRC_VIOLATIONS", "LVS_MISMATCH",
    "PLACEMENT_INFEASIBLE_PROVEN", "ROUTE_INFEASIBLE_PROVEN",
})
_ACTIONS = {
    "unsupported": ("install_provider", "Install or implement the named supported provider before retrying."),
    "input": ("fix_input", "Repair the named input contract and retain the previous failure evidence."),
    "environment": ("repair_environment", "Repair the named tool or runtime dependency before retrying."),
    "budget_or_cancel": ("replay_minimal", "Preserve the stop; review bounded evidence before proposing an authorized minimal replay."),
    "incomplete": ("replay_minimal", "Inspect the retained partial evidence before proposing an authorized minimal replay."),
    "diagnostic_stop": ("review_diagnostic", "Review the observation and bind its retained receipt, concrete evidence and a finite budget before retrying."),
    "internal_error": ("report_backend", "Report the exact producer code and retained evidence to the backend owner."),
    "physical_rejection": ("replay_minimal", "Inspect the referenced physical counterexample with its original input binding."),
    "unknown": ("report_backend", "Retain the original result and request a typed producer diagnostic; do not guess the blocker."),
}


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _code(value: Any) -> str | None:
    return value if type(value) is str and 0 < len(value) <= 128 else None


def _label(value: Any) -> str | None:
    return value[:256] if type(value) is str else None


def _preview(value: Any) -> dict[str, Any]:
    """Bound traversal before encoding, including cycles and enormous integers."""
    remaining = 128
    seen: set[int] = set()
    omitted = False

    def visit(item: Any, depth: int = 0) -> Any:
        nonlocal remaining, omitted
        remaining -= 1
        if remaining < 0 or depth > 6:
            omitted = True
            return {"omitted": "node_or_depth_limit"}
        if item is None or type(item) is bool:
            return item
        if type(item) is int and item.bit_length() <= 256:
            return item
        if type(item) is float and math.isfinite(item):
            return item
        if type(item) is str:
            if len(item) > 256:
                omitted = True
            return item[:256]
        if type(item) not in (dict, list, tuple):
            omitted = True
            return {"omitted": "unsupported_value"}
        if id(item) in seen:
            omitted = True
            return {"omitted": "repeated_reference"}
        seen.add(id(item))
        if type(item) is dict:
            result = {}
            for index, (key, child) in enumerate(item.items()):
                if index >= 16 or remaining <= 0:
                    omitted = True
                    break
                if type(key) is not str or len(key) > 128:
                    omitted = True
                    continue
                result[key] = visit(child, depth + 1)
            return result
        result = []
        for child in item[:16]:
            if remaining <= 0:
                omitted = True
                break
            result.append(visit(child, depth + 1))
        omitted = omitted or len(item) > len(result)
        return result

    preview = visit(value)
    if len(json.dumps(preview, ensure_ascii=True, allow_nan=False).encode("utf-8")) > MAX_PREVIEW_BYTES:
        preview = {"omitted": "byte_limit"}
        omitted = True
    return {"value": preview, "omitted": omitted, "max_bytes": MAX_PREVIEW_BYTES}


def _safe_path(value: Any) -> str | None:
    if isinstance(value, PurePath):
        value = str(value)
    if type(value) is not str or not value or len(value) > MAX_PATH_CHARS:
        return None
    if "://" in value or any(ord(char) < 32 for char in value):
        return None
    if ".." in PurePosixPath(value).parts or ".." in PureWindowsPath(value).parts:
        return None
    return value


def _evidence_paths(step: Any, diagnostic: Mapping) -> tuple[list[str], bool]:
    paths: list[str] = []
    omitted = False

    def add(value: Any) -> None:
        nonlocal omitted
        path = _safe_path(value)
        if path is None:
            omitted = True
        elif path not in paths:
            if len(paths) == MAX_EVIDENCE_PATHS:
                omitted = True
            else:
                paths.append(path)

    artifacts = _field(step, "artifacts", {})
    if isinstance(artifacts, Mapping):
        for index, value in enumerate(artifacts.values()):
            if index >= 32:
                omitted = True
                break
            add(value)
    for key in ("evidence_refs", "evidence_paths"):
        values = diagnostic.get(key, ())
        if type(values) in (list, tuple):
            for value in values[:32]:
                add(value)
            omitted = omitted or len(values) > 32
    return paths, omitted


def _diagnostic(step: Any) -> tuple[Mapping, Mapping, str | None, str | None]:
    data = _field(step, "data", {})
    data = data if isinstance(data, Mapping) else {}
    diagnostic = data.get("diagnostic", data)
    diagnostic = diagnostic if isinstance(diagnostic, Mapping) else {}
    reason, code = _code(diagnostic.get("reason")), _code(diagnostic.get("code"))
    selected = diagnostic
    if reason == "PHYSICAL_SUPPORT_BLOCKED":
        detail = diagnostic.get("detail", {})
        checks = detail.get("checks", ()) if isinstance(detail, Mapping) else ()
        if type(checks) in (list, tuple):
            for check in checks[:MAX_CHECKS]:
                if isinstance(check, Mapping) and check.get("status") != "pass":
                    selected = check
                    code = _code(check.get("code"))
                    break
    return diagnostic, selected, reason, code


def summarize_failure(result: Any) -> dict[str, Any]:
    """Summarize a FlowResult or its mapping without changing the original.

    ``physical_rejection`` describes an explicit producer verdict with provided
    references. This function does not open those references or certify them.
    The first actual blocking failure wins over later skipped/error steps.
    """
    raw_steps = _field(result, "steps", ())
    steps = raw_steps[:MAX_STEPS] if type(raw_steps) in (list, tuple) else ()
    scan_complete = type(raw_steps) in (list, tuple) and len(raw_steps) <= MAX_STEPS
    failed = [step for step in steps if _field(step, "status") in ("fail", "error", "unsupported")]
    blocking = next((step for step in failed if _field(step, "blocking") is True), None)
    selected_step = blocking if blocking is not None else next(iter(failed), None)
    declared_verdict = _label(_field(result, "verdict"))
    failure_present = bool(failed) or declared_verdict == "fail"
    diagnostic, selected, reason, code = _diagnostic(selected_step)
    paths, paths_omitted = _evidence_paths(selected_step, diagnostic)
    codes = tuple(value for value in (reason, code) if value is not None)
    category = "unknown"
    # A stop code cannot be reinterpreted as physical infeasibility by another
    # field, even when a partial tool report also mentions physical violations.
    if any(value in _CODES["budget_or_cancel"] for value in codes):
        category = "budget_or_cancel"
    else:
        effective_codes = (code,) if selected is not diagnostic else codes
        matches = {name for name, registered in _CODES.items()
                   if any(value in registered for value in effective_codes)}
        if len(matches) == 1:
            category = next(iter(matches))
        if (not matches and any(value in _PHYSICAL_CODES for value in effective_codes)
                and selected.get("status") == "fail" and paths):
            category = "physical_rejection"
    action, message = _ACTIONS[category]
    return {
        "schema": SCHEMA,
        "failure_present": failure_present,
        "declared_verdict": declared_verdict,
        "category": category,
        "first_blocking_stage": _label(_field(blocking, "name")),
        "diagnostic_stage": _label(_field(selected_step, "name")),
        "reason": reason,
        "code": code,
        "diagnostic_preview": _preview(_field(selected_step, "data", {})),
        "evidence_paths": paths,
        "evidence_paths_omitted": paths_omitted,
        "evidence_verified": False,
        "scan_complete": scan_complete,
        "recommended_next_action": {"kind": action, "message": message} if failure_present else None,
        "physical_admission": False,
    }
