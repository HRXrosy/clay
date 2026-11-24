"""Opt-in, bounded timing scopes for a terminal diagnostic invocation.

This helper records no routing facts and grants no admission.  The terminal
caller is responsible for binding its actual diagnostic capability before
entering the scope.  Outside an active owned ExecutionSession it fails closed.
"""
from contextlib import contextmanager
from contextvars import ContextVar

from clay.flow.execution_session import current_session
from clay.layout.place.native_stage_observation import _execution_owner


_CURRENT = ContextVar('terminal_stage_timing', default=None)
_MAX_CHECKS = 8
_MAX_STAGES = 128
_STAGES = frozenset({
    'terminal.inspect', 'terminal.snapshot', 'terminal.engine_initialize_to_group_stop',
    'terminal.world_facts', 'terminal.region_constructor', 'terminal.schedule',
    'terminal.pending_verify', 'terminal.input.identity', 'terminal.input.topology',
    'terminal.input.source', 'terminal.input.policy', 'terminal.input.compact',
    'terminal.input.context', 'terminal.input.kwargs', 'terminal.input.prepared',
    'terminal.input.pdk', 'region.check.input', 'region.check.facts',
    'region.check.consumers', 'region.consumer.values', 'region.consumer.indexes',
    'region.consumer.tech_wide', 'region.hierarchy.precheck', 'region.hierarchy.mapping',
    'region.hierarchy.postcheck', 'region.hierarchy.publish', 'region.inventory',
    'region.first_planar', 'region.first_via', 'region.first_endpoint',
    'publication.serialize', 'publication.atomic_publish',
})


class _TimingRedactedFailure(BaseException):
    """Do not copy a measured operation's exception arguments to the journal."""


class TimingObservationError(RuntimeError):
    """A phase could not publish a terminal observation for a body failure."""


class TerminalStageTiming:
    """Instrument a finite number of complete checks using existing phases.

    Default ``ExecutionSession.phase`` behavior still applies to its own
    admission/exit checkpoints, including cancellation details it records.
    Exceptions raised by the measured body are journaled with a fixed type and
    re-raised as the original object.  No phase name or details come from the
    measured design, source, exception or external callback.
    """

    def __init__(self, *, max_checks=8):
        if type(max_checks) is not int or not 1 <= max_checks <= _MAX_CHECKS:
            raise ValueError('terminal timing check limit must be an integer in 1..8')
        self.max_checks = max_checks
        self._owner = None
        self._session = None
        self._token = None
        self._used = False
        self._active = False
        self._in_check = False
        self._checks = 0
        self._stages = 0
        self._truncated = False

    def __reduce_ex__(self, protocol):
        raise TypeError('terminal timing cannot be serialized')

    def __enter__(self):
        if self._used:
            raise ValueError('terminal timing is one-shot')
        session = current_session()
        if session is None:
            raise ValueError('terminal timing requires the active execution session')
        session._check_owner()
        if _CURRENT.get() is not None:
            raise ValueError('terminal timing cannot nest')
        self._owner, self._session = _execution_owner(), session
        self._used = self._active = True
        self._token = _CURRENT.set(self)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._require_owner()
        if self._in_check:
            raise ValueError('terminal timing check still active')
        self._active = False
        _CURRENT.reset(self._token)
        self._token = None
        return False

    def _require_owner(self):
        if (not self._active or _CURRENT.get() is not self or self._owner != _execution_owner()
                or current_session() is not self._session):
            raise ValueError('terminal timing scope is closed or foreign')
        self._session._check_owner()

    @contextmanager
    def _phase(self, name, *, details):
        self._require_owner()
        original = None
        original_tb = None
        try:
            with self._session.phase(name, details=details):
                try:
                    yield
                except BaseException as error:
                    original, original_tb = error, error.__traceback__
                    raise _TimingRedactedFailure() from None
        except _TimingRedactedFailure as marker:
            if original is None:
                raise
            # The existing phase preserves terminal-write failures as notes on
            # the marker.  Retain an actual typed unwind failure as well as a
            # fixed note, so a CheckedControlStop cannot be consumed cleanly.
            if getattr(marker, '__notes__', None):
                try:
                    from clay.flow.control_stop import record_unwind_failure
                    record_unwind_failure(original, TimingObservationError(
                        'Terminal timing phase-error observation failed'))
                except BaseException:
                    pass
        # Re-raise only after leaving the marker's exception handler.  Raising
        # from None would erase an explicit original cause, while raising
        # inside that handler would replace the original implicit context with
        # the private timing marker.  The original exception remains the one
        # observed by the caller, including its cause/suppression semantics.
        if original is not None:
            raise original.with_traceback(original_tb)

    @contextmanager
    def check(self):
        """Measure at most the first N *complete-check entries*."""
        self._require_owner()
        if self._in_check:
            raise ValueError('terminal timing checks cannot nest')
        self._checks += 1
        self._in_check = True
        try:
            if self._checks == self.max_checks + 1:
                self._truncate('check_limit')
            if self._checks <= self.max_checks and not self._truncated:
                with self._phase('terminal.input.check', details={'ordinal': self._checks}):
                    yield
            else:
                yield
        finally:
            self._in_check = False

    @contextmanager
    def stage(self, name):
        """Time a fixed named stage; details contain only an ordinal."""
        self._require_owner()
        if type(name) is not str or name not in _STAGES:
            raise ValueError('terminal timing stage is not authorized')
        if not self._truncated and self._stages == _MAX_STAGES:
            self._truncate('stage_limit')
        if self._truncated:
            yield
        else:
            self._stages += 1
            with self._phase(name, details={'ordinal': self._checks}):
                yield

    def _truncate(self, reason):
        if not self._truncated:
            self._truncated = True
            with self._phase('terminal.timing.truncated', details={
                    'reason': reason, 'max_checks': self.max_checks, 'max_stages': _MAX_STAGES}):
                pass

    def summary(self):
        self._require_owner()
        return {'checks_seen': self._checks, 'checks_limit': self.max_checks,
                'stages_recorded': self._stages, 'stages_limit': _MAX_STAGES,
                'truncated': self._truncated, 'physical_admission': False}
