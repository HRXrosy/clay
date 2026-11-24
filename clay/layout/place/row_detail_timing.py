"""Opt-in inclusive timings for one synchronous row-snapshot observation.

No operand or replay result is cached, and the default path simply calls the
original operation.  Measurements are diagnostic and never grant admission.
"""
from contextvars import ContextVar
from threading import get_ident
import time

_CURRENT = ContextVar('clay_row_detail_timing', default=None)
STAGES = ('row.program', 'row.materialization', 'row.projection', 'row.geometry',
          'projection.witness', 'projection.fields', 'witness.operands',
          'witness.source_rows', 'witness.overlay_compare', 'source.canonical',
          'source.chains', 'source.receipt_validation', 'source.gate_receipt',
          'source.geometry_snapshot', 'source.row_replay', 'source.row_digest',
          'source.file_hash')


class RowDetailTiming:
    """Bounded, owner-thread, one-shot diagnostic scope for existing calls."""
    def __init__(self, *, event_limit=100000):
        if type(event_limit) is not int or not 1 <= event_limit <= 100000:
            raise ValueError('row detail event limit must be in 1..100000')
        self.limit = event_limit
        self.seen = 0
        self.recorded = 0
        self.truncated = False
        self.stages = {name: {'calls': 0, 'error_calls': 0, 'wall_ns': 0, 'cpu_ns': 0}
                       for name in STAGES}
        self.unwind_failures = []
        self._used = False
        self._token = None

    def __enter__(self):
        if self._used or _CURRENT.get() is not None:
            raise ValueError('row detail timing is one-shot and cannot nest')
        self._used = True
        self.owner = get_ident()
        try:
            self._token = _CURRENT.set(self)
            return self
        except BaseException as primary:
            try:
                if _CURRENT.get() is self:
                    if self._token is None:
                        _CURRENT.set(None)
                    else:
                        _CURRENT.reset(self._token)
            except BaseException as ending:
                self.unwind_failures.append((primary, ending))
                try:
                    from clay.flow.control_stop import record_unwind_failure
                    record_unwind_failure(primary, ending)
                except BaseException:
                    pass
            raise

    def __exit__(self, exc_type, exc, traceback):
        try:
            _CURRENT.reset(self._token)
        except BaseException as ending:
            if exc is None:
                raise
            self.unwind_failures.append((exc, ending))
            try:
                from clay.flow.control_stop import record_unwind_failure
                record_unwind_failure(exc, ending)
            except BaseException:
                pass
        finally:
            self._token = None
        return False

    def payload(self):
        return {'physical_admission': False, 'scope': 'inclusive; overlapping stages; one owner',
                'event_limit': self.limit, 'events_seen': self.seen, 'events_recorded': self.recorded,
                'truncated': self.truncated, 'stages': {k: dict(v) for k, v in self.stages.items()}}


def measured(name, function, /, *args, **kwargs):
    """Call the original once, measuring only when explicitly activated."""
    scope = _CURRENT.get()
    if scope is None:
        return function(*args, **kwargs)
    if scope.owner != get_ident() or name not in scope.stages:
        raise ValueError('row detail timing owner or stage differs')
    scope.seen += 1
    if scope.seen > scope.limit:
        scope.truncated = True
        return function(*args, **kwargs)
    wall, cpu = time.perf_counter_ns(), time.process_time_ns()
    primary = None
    try:
        return function(*args, **kwargs)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            elapsed_wall, elapsed_cpu = time.perf_counter_ns()-wall, time.process_time_ns()-cpu
            if elapsed_wall < 0 or elapsed_cpu < 0:
                raise ValueError('row detail timing clock moved backwards')
            metric = scope.stages[name]
            metric['calls'] += 1
            metric['error_calls'] += int(primary is not None)
            metric['wall_ns'] += elapsed_wall
            metric['cpu_ns'] += elapsed_cpu
            scope.recorded += 1
        except BaseException as ending:
            if primary is None:
                raise
            scope.unwind_failures.append((primary, ending))
            try:
                from clay.flow.control_stop import record_unwind_failure
                record_unwind_failure(primary, ending)
            except BaseException:
                pass
