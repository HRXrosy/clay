"""Invocation-owned diagnostic debits for selected resource consumers.

This meter is independent of source route policy and does not measure all
physical, native or serialization work. Its native budget delegates retain
their own original limits and exception semantics.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect
import math
import time
from types import MappingProxyType
from typing import Any, Iterator

from clay.flow.execution_budget import ExecutionStopped
from clay.flow.execution_session import current_session
from clay.flow.run_admission import current_admission
from clay.flow.control_stop import record_unwind_failure
from clay.layout.place.native_stage_observation import _execution_owner


class ResourceMeterScopeError(ValueError):
    """A diagnostic meter is closed, foreign or bound to another world."""


class ResourceMeterStopped(RuntimeError):
    """A diagnostic allowance stopped work; this does not prove infeasibility."""

    def __init__(self, reason: str, *, phase: str, net: str | None, used: int):
        super().__init__(reason)
        self.reason, self.phase, self.net, self.used = reason, phase, net, used

    def flow_error_data(self) -> dict[str, Any]:
        return {"reason": self.reason, "scope": "resource_meter", "phase": self.phase,
                "net": self.net, "used": self.used, "physical_admission": False}


class DiagnosticResourceMeter:
    """A nonserializable meter for covered budget debits in one live session."""

    def __init__(self, *, producer_scope: Any,
                 max_operations: int, per_net_operations: int | None,
                 wall_seconds: float):
        from clay.layout.route.source_handoff import _ACTIVE_SCOPE, _SourceHandoffScope
        from clay.layout.route.resource_world import pdk_provenance_digest

        session = current_session()
        admission = current_admission()
        if (type(producer_scope) is not _SourceHandoffScope
                or _ACTIVE_SCOPE.get() is not producer_scope or session is None
                or admission is None or session.remaining_wall_seconds() is None
                or producer_scope._projected or producer_scope._cap is None
                or not producer_scope._cap._active):
            raise ResourceMeterScopeError("an active admitted source/candidate producer scope is required")
        producer_scope._check()
        world_sha256 = producer_scope._world.sha256()
        pdk_sha256 = pdk_provenance_digest(producer_scope._tech)
        for value in (world_sha256, pdk_sha256):
            if (not isinstance(value, str) or len(value) != 64 or
                    any(ch not in "0123456789abcdef" for ch in value)):
                raise ValueError("canonical world and PDK digests are required")
        if type(max_operations) is not int or max_operations <= 0:
            raise ValueError("max_operations must be a positive exact integer")
        if per_net_operations is not None and (
                type(per_net_operations) is not int or per_net_operations <= 0):
            raise ValueError("per_net_operations must be a positive exact integer or None")
        if (isinstance(wall_seconds, bool) or not isinstance(wall_seconds, (int, float))
                or not math.isfinite(wall_seconds) or wall_seconds <= 0):
            raise ValueError("wall_seconds must be finite and positive")
        self._session = session
        self._admission = admission
        self._owner = _execution_owner()
        self._scope = producer_scope
        self._ticket = producer_scope._ticket
        self._candidate = self._ticket.candidate_sha
        self._world_object = producer_scope._world
        self._tech_object = producer_scope._tech
        self._world = world_sha256
        self._pdk = pdk_sha256
        self._max = max_operations
        self._per_net_max = per_net_operations
        self._deadline = time.monotonic() + float(wall_seconds)
        self._closed = False
        self._phase = "SETUP"
        self._net: str | None = None
        self._total = 0
        self._by_net: Counter[str] = Counter()
        self._by_kind: Counter[str] = Counter()
        self._by_phase: Counter[str] = Counter()
        self._by_reason: Counter[tuple[str, str]] = Counter()
        self._unmetered_entries: Counter[str] = Counter()
        self._resource_ledgers: dict[int, Any] = {}
        self._incidence_indices: dict[int, tuple[Any, Any, Any]] = {}

    def __reduce_ex__(self, protocol: int) -> None:
        raise TypeError("live resource meter cannot be serialized")

    def _check_owner(self) -> None:
        from clay.layout.route.source_handoff import _ACTIVE_SCOPE
        if (self._closed or _execution_owner() != self._owner
                or current_session() is not self._session or current_admission() is not self._admission
                or _ACTIVE_SCOPE.get() is not self._scope or not self._scope._active
                or self._scope._projected or self._scope._cap is None or not self._scope._cap._active
                or self._scope._ticket is not self._ticket
                or self._ticket.candidate_sha != self._candidate
                or self._scope._world is not self._world_object or self._scope._tech is not self._tech_object):
            raise ResourceMeterScopeError("resource meter is closed or has a foreign producer scope")

    def check_boundary(self) -> None:
        self.check()
        self._scope._check()
        self.check()

    def check(self, *, world_sha256: str | None = None, pdk_sha256: str | None = None) -> None:
        self._check_owner()
        if world_sha256 is not None and world_sha256 != self._world:
            raise ResourceMeterScopeError("resource meter belongs to a foreign world")
        if pdk_sha256 is not None and pdk_sha256 != self._pdk:
            raise ResourceMeterScopeError("resource meter belongs to a foreign PDK")
        self._session.checkpoint()
        if time.monotonic() >= self._deadline:
            raise ResourceMeterStopped("METER_WALL", phase=self._phase, net=self._net, used=self._total)

    @contextmanager
    def scope(self, phase: str, *, net: str | None = None) -> Iterator[None]:
        self._check_owner()
        if not isinstance(phase, str) or not phase or (net is not None and (not isinstance(net, str) or not net)):
            raise ValueError("phase and optional net must be nonempty strings")
        previous = self._phase, self._net
        self._phase, self._net = phase, net
        try:
            self.check()
            yield
            self.check()
        except (ExecutionStopped, ResourceMeterStopped):
            raise
        except Exception as original:
            try:
                self.check()
            except Exception as ending:
                if isinstance(ending, (ExecutionStopped, ResourceMeterStopped)) and _local_budget_error(original):
                    raise
                record_unwind_failure(original, ending)
            raise
        finally:
            self._phase, self._net = previous

    def reserve(self, kind: str, reason: str) -> None:
        self.check()
        if self._total >= self._max:
            raise ResourceMeterStopped("METER_SHARED_LIMIT", phase=self._phase, net=self._net, used=self._total)
        if self._net is not None and self._per_net_max is not None and self._by_net[self._net] >= self._per_net_max:
            raise ResourceMeterStopped("METER_NET_LIMIT", phase=self._phase, net=self._net, used=self._total)
        self._total += 1
        self._by_kind[kind] += 1
        self._by_phase[self._phase] += 1
        self._by_reason[(kind, reason)] += 1
        if self._net is not None:
            self._by_net[self._net] += 1

    def snapshot(self) -> dict[str, Any]:
        self.check()
        return {"world_sha256": self._world, "pdk_sha256": self._pdk,
                "candidate_sha256": self._candidate, "total": self._total,
                "by_net": dict(self._by_net), "by_kind": dict(self._by_kind),
                "by_phase": dict(self._by_phase),
                "unmetered_entries": dict(self._unmetered_entries),
                "by_reason": [(kind, reason, count) for (kind, reason), count in sorted(self._by_reason.items())],
                "scope": "per-instance covered resource-budget debits only; not a candidate-wide limit or routing proof"}

    def note_unmetered(self, reason: str) -> None:
        """Record an entered cache-sensitive operation without inventing a debit."""
        self.check()
        self._unmetered_entries[reason] += 1

    def register_ledger(self, ledger: Any, geometry_budget: Any) -> None:
        self.check()
        from clay.layout.route.resource_world import (ExplicitRoutingResourceLedger,
                                                       enumerate_explicit_routing_resources)
        producer = enumerate_explicit_routing_resources.__wrapped__
        caller = inspect.currentframe().f_back
        try:
            if (caller.f_code is not producer.__code__ or caller.f_globals is not producer.__globals__
                    or caller.f_locals.get("resource_meter") is not self
                    or caller.f_locals.get("ledger") is not ledger
                    or caller.f_locals.get("geometry_budget") is not geometry_budget
                    or budget_meter(geometry_budget) is not self
                    or type(ledger) is not ExplicitRoutingResourceLedger
                    or caller.f_locals.get("tech") is not self._tech_object
                    or caller.f_locals.get("world").sha256() != self._world
                    or ledger.routing_world_provenance_digest != self._world
                    or id(ledger) in self._resource_ledgers):
                raise ResourceMeterScopeError("ledger registration requires the actual resource producer")
        finally:
            del caller
        self._resource_ledgers[id(ledger)] = ledger

    def require_ledger(self, ledger: Any) -> None:
        self.check()
        if self._resource_ledgers.get(id(ledger)) is not ledger:
            raise ResourceMeterScopeError("ledger was not produced in this meter scope")

    def register_incidence(self, index: Any, ledger: Any, conductors: Any) -> None:
        self.check()
        self.require_ledger(ledger)
        from clay.layout.route.resource_conductors import OwnedConductorComponentIndex
        from clay.layout.route.pattern_compile import LaneIncidenceIndex, build_lane_incidence_index
        producer = build_lane_incidence_index.__wrapped__
        caller = inspect.currentframe().f_back
        try:
            if (caller.f_code is not producer.__code__ or caller.f_globals is not producer.__globals__
                    or caller.f_locals.get("resource_meter") is not self
                    or caller.f_locals.get("index") is not index
                    or caller.f_locals.get("ledger") is not ledger
                    or caller.f_locals.get("conductor_index") is not conductors
                    or budget_meter(caller.f_locals.get("incidence_budget")) is not self
                    or type(index) is not LaneIncidenceIndex
                    or type(conductors) is not OwnedConductorComponentIndex
                    or conductors.world_digest != self._world
                    or conductors.sha256() != ledger.owned_conductor_index_digest
                    or index.routing_world_provenance_digest != self._world
                    or index.routing_resource_ledger_digest != ledger.sha256()
                    or index.owned_conductor_index_digest != conductors.sha256()
                    or id(index) in self._incidence_indices):
                raise ResourceMeterScopeError("incidence registration requires the actual producer")
        finally:
            del caller
        self._incidence_indices[id(index)] = (index, ledger, conductors)

    def require_incidence(self, index: Any, ledger: Any = None, conductors: Any = None) -> None:
        self.check()
        entry = self._incidence_indices.get(id(index))
        if (entry is None or entry[0] is not index
                or (ledger is not None and entry[1] is not ledger)
                or (conductors is not None and entry[2] is not conductors)):
            raise ResourceMeterScopeError("incidence index was not produced in this meter scope")

    def close(self) -> None:
        self._check_owner()
        self._closed = True


class _MeteredBudget:
    """Keep native mutable budget and exception semantics as the authority."""

    def __init__(self, delegate: Any, meter: DiagnosticResourceMeter, kind: str):
        object.__setattr__(self, "delegate", delegate)
        object.__setattr__(self, "meter", meter)
        object.__setattr__(self, "kind", kind)

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("metered budget supports consume-only consumers")

    def __getattr__(self, name: str) -> Any:
        if name not in ("limit", "used", "reason_counts"):
            raise AttributeError("metered budget supports consume-only consumers")
        value = getattr(self.delegate, name)
        return MappingProxyType(value) if name == "reason_counts" else value

    def consume(self, reason: str) -> None:
        self.meter.check()
        if self.delegate.used >= self.delegate.limit:
            return self.delegate.consume(reason)
        self.meter.reserve(self.kind, reason)
        return self.delegate.consume(reason)


def bind_budget(delegate: Any, meter: DiagnosticResourceMeter | None, kind: str) -> Any:
    if meter is None:
        return delegate
    if type(meter) is not DiagnosticResourceMeter:
        raise TypeError("exact diagnostic resource meter required")
    meter.check()
    if isinstance(delegate, _MeteredBudget):
        if delegate.meter is not meter or delegate.kind != kind:
            raise ResourceMeterScopeError("budget is bound to a different meter or kind")
        return delegate
    from clay.layout.route.resource_world import _GeometryEvaluationBudget
    from clay.layout.route.pattern_compile import _IncidenceBudget
    expected = {"geometry": _GeometryEvaluationBudget, "incidence": _IncidenceBudget}.get(kind)
    if expected is None or type(delegate) is not expected:
        raise TypeError("meter only supports native consume-only geometry/incidence budgets")
    return _MeteredBudget(delegate, meter, kind)


def budget_meter(budget: Any) -> DiagnosticResourceMeter | None:
    return budget.meter if isinstance(budget, _MeteredBudget) else None


def _local_budget_error(error: Exception) -> bool:
    from clay.layout.route.resource_world import RoutingResourceEnumerationLimitError
    from clay.layout.route.pattern_compile import _IncidenceBudgetExceeded
    return isinstance(error, (RoutingResourceEnumerationLimitError, _IncidenceBudgetExceeded))


def checked_call(meter: DiagnosticResourceMeter | None, *, identities: Any,
                 call: Any) -> Any:
    """Check an optional meter at entry, all normal exits and error boundaries."""
    if meter is None:
        return call()
    if type(meter) is not DiagnosticResourceMeter:
        raise TypeError("exact diagnostic resource meter required")
    def check() -> None:
        meter.check_boundary()
        world_sha256, pdk_sha256 = identities()
        meter.check(world_sha256=world_sha256, pdk_sha256=pdk_sha256)

    check()
    try:
        result = call()
    except (ExecutionStopped, ResourceMeterStopped):
        raise
    except Exception as original:
        try:
            check()
        except Exception as ending:
            if isinstance(ending, (ExecutionStopped, ResourceMeterStopped)) and _local_budget_error(original):
                raise
            record_unwind_failure(original, ending)
        raise
    check()
    return result


_LANE_SITE_BOUNDARY = ContextVar('lane_site_boundary_transaction', default=None)


def lane_site_boundary_transaction(access: Any, meter: DiagnosticResourceMeter) -> Any:
    """Bind one synchronous native boundary query to its actual lane-site caller."""
    from clay.layout.route.pattern_compile import _ExactLazyLandingAccess
    from clay.layout.route.resource_lane_site import observe_lane_site
    caller = inspect.currentframe().f_back
    try:
        if (caller.f_code is not observe_lane_site.__code__
                or caller.f_globals is not observe_lane_site.__globals__
                or caller.f_locals.get('access') is not access
                or caller.f_locals.get('meter') is not meter
                or type(access) is not _ExactLazyLandingAccess
                or type(meter) is not DiagnosticResourceMeter
                or access._diagnostic_meter is not meter
                or budget_meter(access._query_context.geometry_budget) is not meter
                or _LANE_SITE_BOUNDARY.get() is not None):
            raise ResourceMeterScopeError('lane-site boundary requires its actual live caller')
    finally:
        del caller
    meter._check_owner()
    scope = meter._scope
    anchors = (access, meter, access._query_context, access.ledger, access.world,
               access.tech, access.conductor_index, access.incidence_index)

    @contextmanager
    def transaction():
        token = _LANE_SITE_BOUNDARY.set(anchors)
        try:
            try:
                yield
            except BaseException as original:
                # A cancelled/expired invocation or meter cannot continue to
                # another net; its original outer cleanup retains priority.
                if (isinstance(original, ExecutionStopped)
                        or isinstance(original, ResourceMeterStopped) and original.reason == 'METER_WALL'):
                    raise
                try:
                    meter.check_boundary()
                    if any(a is not b for a, b in zip(
                            (access, meter, access._query_context, access.ledger, access.world,
                             access.tech, access.conductor_index, access.incidence_index), anchors)):
                        raise ResourceMeterScopeError('lane-site boundary producer operands changed')
                    meter.require_ledger(access.ledger)
                    meter.require_incidence(access.incidence_index, access.ledger, access.conductor_index)
                    from clay.layout.route.resource_world import pdk_provenance_digest
                    meter.check(world_sha256=access.world.sha256(),
                                pdk_sha256=pdk_provenance_digest(access.tech))
                    if (isinstance(original, ResourceMeterStopped)
                            and original.reason == 'METER_NET_LIMIT' and getattr(original, '__notes__', ())):
                        raise ResourceMeterScopeError('net stop has failed boundary cleanup')
                except BaseException as ending:
                    if (isinstance(ending, ExecutionStopped)
                            and (isinstance(original, ResourceMeterStopped) or
                                 isinstance(original, Exception) and _local_budget_error(original))):
                        raise ending from original
                    # A wall reached while rechecking a per-net stop is itself
                    # the terminal budget result.  Leave the scope live for
                    # the outer observer's normal checked stop publication.
                    if (isinstance(original, ResourceMeterStopped) and original.reason == 'METER_NET_LIMIT'
                            and isinstance(ending, ResourceMeterStopped) and ending.reason == 'METER_WALL'):
                        raise ending from original
                    scope._active = False
                    if isinstance(original, ResourceMeterStopped) and original.reason == 'METER_NET_LIMIT':
                        raise ending from original
                    if ending is not original:
                        record_unwind_failure(original, ending)
                raise
        finally:
            _LANE_SITE_BOUNDARY.reset(token)
    return transaction()


def _lane_site_inner_checked(meter: DiagnosticResourceMeter, *, identities: Any, call: Any) -> Any:
    """Keep each inner world/PDK and meter check inside one checked outer call."""
    def check():
        meter.check()
        world, pdk = identities()
        meter.check(world_sha256=world, pdk_sha256=pdk)
    check()
    try:
        value = call()
    except (ExecutionStopped, ResourceMeterStopped):
        raise
    except Exception as original:
        try:
            check()
        except Exception as ending:
            if isinstance(ending, (ExecutionStopped, ResourceMeterStopped)) and _local_budget_error(original):
                raise
            record_unwind_failure(original, ending)
        raise
    check()
    return value


def resource_boundary(*, world_argument: str, tech_argument: str,
                      ledger_world: bool = False) -> Any:
    """Check actual world and PDK at the boundaries of a metered call.

    A query context is itself the meter carrier.  Never silently combine one
    with a different explicit meter, or add a meter to an unmetered context.
    """
    def decorate(function: Any) -> Any:
        signature = inspect.signature(function)

        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            explicit = kwargs.get("resource_meter")
            context = kwargs.get("query_context")
            inherited = budget_meter(getattr(context, "geometry_budget", None)) if context is not None else None
            if explicit is not None and context is not None and explicit is not inherited:
                raise ResourceMeterScopeError("query context belongs to a different meter")
            meter = explicit if explicit is not None else inherited
            if meter is None:
                return function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            world = bound.arguments[world_argument]
            tech = bound.arguments[tech_argument]
            if tech is not meter._tech_object or (not ledger_world and world is not meter._world_object):
                raise ResourceMeterScopeError("consumer requires the actual producer world and PDK")
            ledger = bound.arguments.get("ledger")
            if ledger is not None:
                meter.require_ledger(ledger)

            def identities() -> tuple[str, str]:
                from clay.layout.route.resource_world import pdk_provenance_digest
                digest = (world.routing_world_provenance_digest if ledger_world
                          else world.sha256())
                return digest, pdk_provenance_digest(tech)

            transaction = _LANE_SITE_BOUNDARY.get()
            if transaction is not None:
                from clay.layout.route import pattern_compile as pc, resource_world as rw
                access, owner, query_context, source_ledger, source_world, source_tech, conductors, _index = transaction
                boundary_frame = inspect.currentframe().f_back
                try:
                    while boundary_frame is not None and not (
                            boundary_frame.f_globals is pc.__dict__
                            and boundary_frame.f_code.co_qualname == '_ExactLazyLandingAccess.boundary'
                            and boundary_frame.f_locals.get('self') is access):
                        boundary_frame = boundary_frame.f_back
                    eligible = (boundary_frame is not None and owner is meter and context is query_context
                        and ledger is source_ledger and world is source_world and tech is source_tech
                        and bound.arguments.get('owned_conductor_index') is conductors
                        and function in (rw._query_exact_landing_sites_for_component_lane.__wrapped__,
                                         rw._query_exact_landing_sites_at_points.__wrapped__))
                finally:
                    del boundary_frame
                if eligible:
                    return _lane_site_inner_checked(meter, identities=identities,
                                                    call=lambda: function(*args, **kwargs))
            return checked_call(meter, identities=identities, call=lambda: function(*args, **kwargs))
        return wrapper
    return decorate


def lazy_access_boundary(function: Any) -> Any:
    """Keep lazy caches within their original meter, world and session."""
    @wraps(function)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        context = getattr(self, "_query_context", None)
        meter = budget_meter(getattr(context, "geometry_budget", None))
        if meter is not getattr(self, "_diagnostic_meter", None):
            raise ResourceMeterScopeError("lazy access was moved to another meter scope")
        budget = kwargs.get("budget")
        if meter is None and budget_meter(budget) is not None:
            raise ResourceMeterScopeError("metered incidence budget requires a metered query context")
        if meter is None:
            return function(self, *args, **kwargs)
        if budget is not None and budget_meter(budget) is not meter:
            raise ResourceMeterScopeError("lazy incidence budget belongs to another meter")
        if function.__name__ == "component_lanes" and budget is None:
            raise ResourceMeterScopeError("component/lane access requires a metered budget")
        meter.require_incidence(self.incidence_index, self.ledger, self.conductor_index)
        if self.world is not meter._world_object or self.tech is not meter._tech_object:
            raise ResourceMeterScopeError("lazy access requires the actual producer world and PDK")

        def identities() -> tuple[str, str]:
            from clay.layout.route.resource_world import pdk_provenance_digest
            if (self.ledger.routing_world_provenance_digest != self.world.sha256()
                    or self.conductor_index.world_digest != self.world.sha256()
                    or self.incidence_index.routing_world_provenance_digest != self.world.sha256()
                    or self.incidence_index.routing_resource_ledger_digest != self.ledger.sha256()
                    or self.incidence_index.owned_conductor_index_digest != self.conductor_index.sha256()):
                raise ResourceMeterScopeError("lazy access ledger/index belongs to foreign inputs")
            return self.world.sha256(), pdk_provenance_digest(self.tech)

        def invoke() -> Any:
            meter.note_unmetered("lazy access/cache-sensitive entry")
            return function(self, *args, **kwargs)

        return checked_call(meter, identities=identities, call=invoke)
    return wrapper


def lane_index_boundary(function: Any) -> Any:
    """Prevent a cached lane index from crossing diagnostic meter scopes."""
    @wraps(function)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        meter = budget_meter(kwargs.get("budget"))
        bound = getattr(self, "_diagnostic_meter", None)
        if bound is not None and meter is not bound:
            raise ResourceMeterScopeError("lane index belongs to another meter scope")
        if meter is None:
            return function(self, *args, **kwargs)
        meter.check()
        meter.require_incidence(self._incidence_index)
        if self._incidence_index.routing_world_provenance_digest != meter._world:
            raise ResourceMeterScopeError("lane index belongs to a foreign world")
        if bound is None:
            if self._lanes_by_layer is not None or self._intersection_cache:
                raise ResourceMeterScopeError("cannot meter a previously populated lane index")
            self._diagnostic_meter = meter
        meter.note_unmetered("lane index/cache-sensitive entry")
        return checked_call(meter, identities=lambda: (
                                self._incidence_index.routing_world_provenance_digest, meter._pdk),
                            call=lambda: function(self, *args, **kwargs))
    return wrapper
