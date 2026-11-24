"""Invocation-owned terminal and final-candidate completion for batches."""
from contextlib import contextmanager
from contextvars import ContextVar
import inspect
import re

from clay.flow.execution_session import current_session
from clay.flow.run_admission import current_admission
from clay.layout.place.native_stage_observation import _execution_owner

_ACTIVE=ContextVar('candidate_batch_completion',default=None)


class _TerminalCompletion:
    def __init__(self,candidate_sha256,callback):
        session=current_session();admission=current_admission()
        if (type(candidate_sha256) is not str or re.fullmatch(r'[0-9a-f]{64}',candidate_sha256) is None
                or not callable(callback) or session is None or admission is None
                or session.invocation_id!=admission.invocation_id):
            raise ValueError('candidate completion requires the actual invocation and candidate')
        self.candidate_sha256=candidate_sha256
        self.callback=callback
        self.session=session;self.admission=admission
        self.owner=_execution_owner()
        self.active=True
        self._issued=None
        self._consumed=False

    def _check(self):
        if (not self.active or self.owner!=_execution_owner() or _ACTIVE.get() is not self
                or current_session() is not self.session or current_admission() is not self.admission):
            raise ValueError('candidate completion is closed or foreign')

    def issue(self,*,cap,callback,stop):
        from clay.layout.route.live_diagnostic import DiagnosticRoutingObserved
        self._check()
        if (self._issued is not None or callback is not self.callback or cap._active
                or type(stop) is not DiagnosticRoutingObserved):
            raise ValueError('candidate completion producer identity differs')
        self._issued=(object(),cap,stop)

    def consume(self,*,cap,stop,candidate_sha256):
        self._check()
        if (self._consumed or self._issued is None or self._issued[1] is not cap
                or self._issued[2] is not stop or candidate_sha256!=self.candidate_sha256
                or cap._active or getattr(stop,'__notes__',())):
            raise ValueError('candidate completion is missing, foreign, reused or changed')
        self._consumed=True


@contextmanager
def terminal_completion_scope(*,candidate_sha256,callback):
    if _ACTIVE.get() is not None:raise ValueError('candidate completion scopes cannot nest')
    scope=_TerminalCompletion(candidate_sha256,callback);token=_ACTIVE.set(scope)
    try:
        yield scope
    finally:
        scope.active=False;_ACTIVE.reset(token)


def publish_terminal_completion(*,cap,callback,stop):
    """Called only by the terminal observer after normal final checks/close."""
    scope=_ACTIVE.get()
    if scope is not None:
        from clay.layout.route.live_diagnostic import observe_prepared_routing
        caller=inspect.currentframe().f_back
        try:
            if caller.f_code is not observe_prepared_routing.__code__ or caller.f_globals is not observe_prepared_routing.__globals__:
                raise ValueError('candidate completion lacks terminal producer origin')
        finally:del caller
        scope.issue(cap=cap,callback=callback,stop=stop)


class _CandidateCompletion:
    def __init__(self,*,candidate_sha256,cap,stage,terminal,stop):
        session=current_session();admission=current_admission()
        if (session is None or admission is None or session is not terminal.session
                or admission is not terminal.admission or terminal.active or not terminal._consumed
                or terminal.candidate_sha256!=candidate_sha256 or terminal._issued is None
                or terminal._issued[1] is not cap or terminal._issued[2] is not stop
                or cap._active or stage._active):
            raise ValueError('final candidate completion lacks closed current scopes')
        self.session=session;self.admission=admission;self.owner=_execution_owner()
        self.candidate_sha256=candidate_sha256;self.cap=cap;self.stage=stage
        self.terminal=terminal;self.stop=stop;self.consumed=False

    def consume(self,*,candidate_sha256,cap,stage,terminal,stop):
        if (self.consumed or self.owner!=_execution_owner() or current_session() is not self.session
                or current_admission() is not self.admission or self.candidate_sha256!=candidate_sha256
                or self.cap is not cap or self.stage is not stage or self.terminal is not terminal
                or self.stop is not stop or cap._active or stage._active):
            raise ValueError('final candidate completion is foreign, reused or changed')
        self.consumed=True


def issue_candidate_completion(*,candidate_sha256,cap,stage,terminal,stop,callback_returned):
    """Mint only from the public batch coordinator after all boundary checks."""
    from clay.layout.place.candidate_batch_diagnostic import run_candidate_batch_diagnostic
    caller=inspect.currentframe().f_back
    try:
        if (caller.f_code is not run_candidate_batch_diagnostic.__code__
                or caller.f_globals is not run_candidate_batch_diagnostic.__globals__
                or callback_returned is not True):
            raise ValueError('final candidate completion lacks coordinator origin')
    finally:del caller
    return _CandidateCompletion(candidate_sha256=candidate_sha256,cap=cap,stage=stage,terminal=terminal,stop=stop)
