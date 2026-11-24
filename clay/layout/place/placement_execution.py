"""Invocation telemetry at the CP-SAT solve boundaries in cp_model_builder.

These phases measure Solve plus its adapter overhead, not model construction.
They cover strip, banded, X-legalizer/diagnostic and floorplan CP-SAT calls only;
full2d, LP and MILP solvers remain outside this adapter. Their native work is
not charged here, so this is not a global deterministic solver limit. All
observations go to the active session's sidecar, never placement certificates.
"""
from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from typing import Any, Iterator

from clay.flow.execution_session import current_session

_BANDED_PROTO_MAX_BYTES = 16 * 1024 * 1024


def _banded_area_primary(proto: Any, parameters: Any, phase: str) -> bool:
    """Select banded area solves except those using fixed-hint parameters.

    Materialization replay can also use this phase without fixed hints, so an
    invocation may retain multiple primary models. Their phase identities and
    invocation records distinguish them; this predicate does not classify them.
    """
    if phase != "placement.banded.solve.primary" or parameters.fix_variables_to_their_hinted_value:
        return False
    names = tuple(variable.name for variable in proto.variables)
    return any(0 <= index < len(names) and names[index] == "area_nm2"
               for index in proto.objective.vars)


def _save_banded_proto(session: Any, handle: Any, proto: Any, kind: str) -> dict[str, Any]:
    """Write bounded, invocation-local diagnostics without altering the solve."""
    try:
        size = proto.ByteSize()
        if size > _BANDED_PROTO_MAX_BYTES:
            return {kind + "_omitted_bytes": size}
        data = proto.SerializeToString(deterministic=True)
        path = session.event_path.parent / f"banded-area-cp-{handle.phase_id}-{kind}.pb"
        with path.open("xb") as stream:
            stream.write(data)
        return {kind + "_sha256": hashlib.sha256(data).hexdigest(),
                kind + "_path": str(path), kind + "_bytes": size}
    except Exception as exc:  # Diagnostic failure cannot change the native result.
        return {kind + "_write_error": type(exc).__name__}


def _banded_attempt_start(session: Any) -> dict[str, Any]:
    """Bind the existing invocation start record when one actually exists."""
    path = session.event_path.parent / "start.json"
    try:
        with path.open("rb") as stream:
            data = stream.read(_BANDED_PROTO_MAX_BYTES + 1)
        if len(data) > _BANDED_PROTO_MAX_BYTES:
            return {"attempt_start_omitted_bytes": len(data)}
        return {"attempt_start_path": str(path),
                "attempt_start_sha256": hashlib.sha256(data).hexdigest()}
    except FileNotFoundError:
        return {"attempt_start_status": "absent"}
    except OSError as exc:
        return {"attempt_start_error": type(exc).__name__}


@contextmanager
def cp_solve_phase(solver: Any, model: Any, *, phase: str) -> Iterator[None]:
    """Surround a caller's native Solve without introducing caller-frame locals.

    CP-SAT deterministic seconds retain their native unit independently of wall
    seconds. A null native/effective limit in events denotes positive infinity.
    Unmanaged callers yield without model/solver probing. Use without an ``as``
    binding to retain the original native caller's code, globals and locals.
    The yielded body must perform exactly one Solve; its response is charged
    only after normal return. Only limit fields changed here are restored.
    """
    session = current_session()
    if session is None:
        yield
        return

    with session.phase(phase) as handle:
        proto = model.Proto()
        capture = _banded_area_primary(proto, solver.parameters, phase)
        model_details = {
            "variables": len(proto.variables),
            "constraints": len(proto.constraints),
            "model_sha256": hashlib.sha256(
                proto.SerializeToString(deterministic=True)
            ).hexdigest(),
        }
        if capture:
            model_details.update(_save_banded_proto(session, handle, proto, "model"))
            model_details.update(_banded_attempt_start(session))
        handle.annotate(**model_details)
        session.checkpoint()
        parameters = solver.parameters
        native_deterministic = parameters.max_deterministic_time
        native_wall = parameters.max_time_in_seconds
        remaining_deterministic = session.remaining_deterministic_seconds()
        remaining_wall = session.remaining_wall_seconds()
        effective_deterministic = (
            native_deterministic if remaining_deterministic is None
            else min(native_deterministic, remaining_deterministic)
        )
        effective_wall = (
            native_wall if remaining_wall is None else min(native_wall, remaining_wall)
        )
        limits = {
            "native_deterministic_seconds": native_deterministic,
            "native_wall_seconds": native_wall,
            "effective_deterministic_seconds": effective_deterministic,
            "effective_wall_seconds": effective_wall,
        }
        handle.annotate(**{
            name: None if value == math.inf else value
            for name, value in limits.items()
        })
        handle.annotate(
            num_search_workers=parameters.num_search_workers,
            stop_after_first_solution=parameters.stop_after_first_solution,
        )
        changed: list[tuple[str, float, bool]] = []
        try:
            for name, native, effective in (
                ("max_deterministic_time", native_deterministic, effective_deterministic),
                ("max_time_in_seconds", native_wall, effective_wall),
            ):
                if effective != native:
                    changed.append((name, native, parameters.HasField(name)))
                    setattr(parameters, name, effective)
            if capture:
                handle.annotate(**_save_banded_proto(session, handle, parameters, "parameters"))
            session.checkpoint()
            yield
            response = solver.ResponseProto()
            deterministic_seconds = float(response.deterministic_time)
            handle.annotate(
                solver_status=int(response.status),
                solver_deterministic_seconds=deterministic_seconds,
                **(_save_banded_proto(session, handle, response, "response") if capture else {}),
            )
            session.charge_deterministic(deterministic_seconds)
        finally:
            # Restoring the whole message would hide unrelated native mutations
            # from the caller's full-parameter replay comparison.
            for name, native, was_present in reversed(changed):
                if was_present:
                    setattr(parameters, name, native)
                else:
                    parameters.ClearField(name)


def solve_cp_sat(solver: Any, model: Any, *, phase: str) -> Any:
    """Convenience entry for callers that do not require a native Solve frame."""
    with cp_solve_phase(solver, model, phase=phase):
        return solver.Solve(model)
