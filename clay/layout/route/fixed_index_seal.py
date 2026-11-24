"""Opt-in, invocation-scoped fixed-index authority for a source route.

This is a cooperative same-thread contract, not a Python security boundary.  It
does not interpret obstacles or authorize routing, and is disabled by default.
"""
from contextlib import contextmanager
from threading import Lock
from types import MappingProxyType
import time

from clay.flow.execution_budget import CancellationToken
from clay.flow.execution_session import (ExecutionSession, current_session,
                                         _SESSION_CLOCK_PRODUCERS)
from clay.flow import execution_session as _session_module
from clay.layout.route import engine_gridless as _gridless


class FixedIndexSealStop(RuntimeError):
    """A bounded seal cannot establish its requested invocation scope."""

    def __init__(self, reason, *, used=0, limit=0, solve_used=0, solve_limit=0):
        self.reason, self.used, self.limit = reason, used, limit
        self.solve_used, self.solve_limit = solve_used, solve_limit
        super().__init__(reason)

    def flow_error_data(self):
        return {'reason': self.reason, 'scope': 'fixed_index_seal',
                'used': self.used, 'limit': self.limit,
                'solve_used': self.solve_used, 'solve_limit': self.solve_limit}


_SETUP_LOCK = Lock()
_WALL_CLOCK, _CPU_CLOCK = _SESSION_CLOCK_PRODUCERS


def _qualified_callable(fn, expected_qualname, expected_module=None):
    return (
        callable(fn)
        and getattr(fn, '__qualname__', None) == expected_qualname
        and (expected_module is None or getattr(fn, '__module__', None) == expected_module)
    )


def configured_budget(engine):
    budget = getattr(engine, '_experimental_fixed_index_seal_budget', None)
    if budget is not None and (type(budget) is not int or budget <= 0):
        raise TypeError('fixed index seal budget must be a positive integer')
    return budget


class _Scope:
    def __init__(self, engine, budget, session, lock):
        self.engine, self.budget, self.session, self.lock = engine, budget, session, lock
        self.used, self.rows, self.published, self.route_entered = 0, (), False, False

    def enter_route(self):
        self._check_session()
        if self.route_entered:
            raise FixedIndexSealStop('FIXED_INDEX_SEAL_UNSUPPORTED')
        self.route_entered = True

    def _check_session(self):
        from clay.layout.route.engine_gridless import GridlessEngine
        session = current_session()
        reason = getattr(CancellationToken, 'reason', None)
        index_type = getattr(_gridless, '_SpatialIndex', None)
        if (type(self.engine) is not GridlessEngine or session is not self.session
                or current_session is not _session_module.current_session
                or _gridless.current_session is not _session_module.current_session
                or not _qualified_callable(current_session, 'current_session', 'clay.flow.execution_session')
                or type(session) is not ExecutionSession or type(session.cancellation) is not CancellationToken
                or session._wall_clock is not _WALL_CLOCK or session._cpu_clock is not _CPU_CLOCK
                or time.monotonic is not _WALL_CLOCK or time.process_time is not _CPU_CLOCK
                or any(name in vars(session) for name in ('checkpoint', '_check_owner', '_budget_states', '_stop'))
                or any(name in vars(self.engine) for name in ('_watchdog_check', 'route'))
                or type(reason) is not property
                or not _qualified_callable(reason.fget, 'CancellationToken.reason')
                or not _qualified_callable(ExecutionSession.checkpoint, 'ExecutionSession.checkpoint')
                or not _qualified_callable(ExecutionSession._check_owner, 'ExecutionSession._check_owner')
                or not _qualified_callable(_gridless.GridlessEngine.route, 'GridlessEngine.route')
                or not _qualified_callable(_gridless.GridlessEngine._watchdog_check, 'GridlessEngine._watchdog_check')
                or index_type is None or getattr(index_type, '__qualname__', None) != '_SpatialIndex'):
            raise FixedIndexSealStop('FIXED_INDEX_SEAL_UNSUPPORTED')
        session._check_owner()

    def charge(self, phase):
        self._check_session()
        engine = self.engine
        if engine._solve_expansions_total >= engine._solve_expansion_budget:
            raise FixedIndexSealStop('SOLVE_WORK_BUDGET', used=self.used, limit=self.budget,
                                     solve_used=engine._solve_expansions_total,
                                     solve_limit=engine._solve_expansion_budget)
        if self.used >= self.budget:
            raise FixedIndexSealStop('FIXED_INDEX_SEAL_SUB_BUDGET', used=self.used, limit=self.budget,
                                     solve_used=engine._solve_expansions_total,
                                     solve_limit=engine._solve_expansion_budget)
        self.used += 1
        engine._solve_expansions_total += 1
        engine._watchdog_check('<fixed-index>', phase)

    def seal(self):
        from collections import OrderedDict
        from clay.layout.route.engine_gridless import _SpatialIndex, _norm
        engine = self.engine
        if self.rows or self.published:
            raise FixedIndexSealStop('FIXED_INDEX_SEAL_UNSUPPORTED')
        self.charge('fixed_index_seal')
        if (type(engine._obs_idx) is not dict
                or type(engine._wire_static_legality_cache) not in (dict, OrderedDict)
                or type(engine._via_static_legality_cache) not in (dict, OrderedDict)
                or engine._wire_static_legality_cache or engine._via_static_legality_cache):
            raise FixedIndexSealStop('FIXED_INDEX_SEAL_UNSUPPORTED')
        method_token = (_SpatialIndex.__init__, _SpatialIndex.add, _SpatialIndex.query, _norm)
        mapping, staged = engine._obs_idx, []
        for layer, index in mapping.items():
            self.charge('fixed_index_seal')
            if (type(layer) is not int or type(index) is not _SpatialIndex
                    or type(index.cell) is not int or index.cell <= 0 or type(index.buckets) is not dict):
                raise FixedIndexSealStop('FIXED_INDEX_SEAL_UNSUPPORTED')
            old, built, snapshots = index.buckets, {}, []
            for key, rows in old.items():
                self.charge('fixed_index_seal')
                if (type(key) is not tuple or len(key) != 2 or any(type(v) is not int for v in key)
                        or type(rows) is not list):
                    raise FixedIndexSealStop('FIXED_INDEX_SEAL_UNSUPPORTED')
                copied = []
                for row in rows:
                    self.charge('fixed_index_seal')
                    if type(row) is not tuple or len(row) != 2:
                        raise FixedIndexSealStop('FIXED_INDEX_SEAL_UNSUPPORTED')
                    owner, rect = row
                    if ((owner is not None and type(owner) is not str) or type(rect) is not tuple
                            or len(rect) != 4 or any(type(v) is not int for v in rect)):
                        raise FixedIndexSealStop('FIXED_INDEX_SEAL_UNSUPPORTED')
                    copied.append(row)
                frozen = tuple(copied)
                built[key] = frozen
                snapshots.append((key, rows, frozen))
            staged.append((layer, index, index.cell, old, MappingProxyType(built), snapshots))
        for layer, index, cell, old, _view, snapshots in staged:
            self.charge('fixed_index_seal')
            if mapping.get(layer) is not index or index.cell != cell or index.buckets is not old or len(old) != len(snapshots):
                raise FixedIndexSealStop('FIXED_INDEX_SEAL_UNSUPPORTED')
            for key, rows, expected in snapshots:
                self.charge('fixed_index_seal')
                if old.get(key) is not rows or len(rows) != len(expected):
                    raise FixedIndexSealStop('FIXED_INDEX_SEAL_UNSUPPORTED')
                for actual, wanted in zip(rows, expected):
                    self.charge('fixed_index_seal')
                    if actual is not wanted:
                        raise FixedIndexSealStop('FIXED_INDEX_SEAL_UNSUPPORTED')
        # Pay publication and eventual restoration before either can occur.
        for _ in staged:
            self.charge('fixed_index_seal')
            self.charge('fixed_index_seal')
        if (engine._obs_idx is not mapping or len(mapping) != len(staged)
                or (_SpatialIndex.__init__, _SpatialIndex.add, _SpatialIndex.query, _norm) != method_token):
            raise FixedIndexSealStop('FIXED_INDEX_SEAL_UNSUPPORTED')
        self.rows = tuple((index, old, view) for _layer, index, _cell, old, view, _snapshots in staged)
        try:
            for index, _old, view in self.rows:
                index.buckets = view
            self.published = True
            self.charge('fixed_index_seal')
            for layer, index, cell, _old, view, _snapshots in staged:
                self.charge('fixed_index_seal')
                if mapping.get(layer) is not index or index.cell != cell or index.buckets is not view:
                    raise FixedIndexSealStop('FIXED_INDEX_SEAL_UNSUPPORTED')
        except BaseException:
            for index, old, _view in self.rows:
                index.buckets = old
            self.published = False
            raise

    def close(self):
        try:
            if self.published:
                for index, old, _view in self.rows:
                    index.buckets = old
                self.published = False
        finally:
            self.engine._fixed_index_seal_scope = None
            self.lock.release()


@contextmanager
def fixed_index_seal_scope(engine):
    budget = configured_budget(engine)
    if budget is None:
        yield
        return
    with _SETUP_LOCK:
        lock = getattr(engine, '_fixed_index_seal_lock', None)
        if lock is None:
            lock = Lock()
            engine._fixed_index_seal_lock = lock
    if not lock.acquire(blocking=False):
        raise FixedIndexSealStop('FIXED_INDEX_SEAL_UNSUPPORTED')
    scope = _Scope(engine, budget, current_session(), lock)
    try:
        scope._check_session()
        if getattr(engine, '_fixed_index_seal_scope', None) is not None:
            raise FixedIndexSealStop('FIXED_INDEX_SEAL_UNSUPPORTED')
        engine._fixed_index_seal_scope = scope
    except BaseException:
        lock.release()
        raise
    try:
        yield scope
    except BaseException as original:
        try:
            scope.close()
        except BaseException as cleanup:
            from clay.flow.control_stop import record_unwind_failure
            try:
                record_unwind_failure(original, cleanup)
            except BaseException:
                pass
        raise
    else:
        scope.close()


def seal_if_enabled(engine):
    if configured_budget(engine) is None:
        return
    scope = getattr(engine, '_fixed_index_seal_scope', None)
    if type(scope) is not _Scope or scope.engine is not engine:
        raise FixedIndexSealStop('FIXED_INDEX_SEAL_UNSUPPORTED')
    scope.seal()
