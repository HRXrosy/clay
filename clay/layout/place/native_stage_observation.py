"""Invocation-owned selection of the existing native frame acquisition observer.

This scope conveys no source, model or admission authority. The native builder
still validates its actual frame and requires its controlled acquisition stop.
Callback selection and an explicit complete-master acquisition request live
here; solver methods and placer globals stay in their implementation modules.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
from threading import current_thread
from typing import Callable, Mapping


NativeStageObserver = Callable[[Mapping[str, object]], None]


def _execution_owner():
    import asyncio
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return os.getpid(), current_thread(), task


@dataclass(slots=True)
class _Observation:
    callback: NativeStageObserver
    owner: tuple
    complete_master_only: bool = False
    active: bool = True

    def require_active_owner(self):
        if not self.active:
            raise ValueError('native stage observation scope has expired')
        if self.owner != _execution_owner():
            raise ValueError('native stage observer belongs to another invocation owner/task')


_CURRENT_OBSERVATION: ContextVar[_Observation | None] = ContextVar(
    'clay_native_stage_observation', default=None)


@contextmanager
def native_stage_observation_scope(observer: NativeStageObserver, *, complete_master_only=False):
    """Select one observer for synchronous native calls in this invocation.

    A copied context cannot outlive the lexical scope or transfer acquisition to
    another thread/task. Nested scopes and explicit competing observers reject.
    """
    if not callable(observer):
        raise TypeError('native stage observation scope requires a callable')
    if type(complete_master_only) is not bool:
        raise TypeError('complete master acquisition selection must be bool')
    if _CURRENT_OBSERVATION.get() is not None:
        raise ValueError('nested native stage observation scope is not supported')
    observation = _Observation(observer, _execution_owner(), complete_master_only)
    token = _CURRENT_OBSERVATION.set(observation)
    try:
        yield
    finally:
        observation.active = False
        _CURRENT_OBSERVATION.reset(token)


def requires_complete_native_master():
    """Skip optional seed decomposition for an owned complete-frame request."""
    observation = _CURRENT_OBSERVATION.get()
    if observation is None:
        return False
    observation.require_active_owner()
    return observation.complete_master_only


def resolve_native_stage_observer(explicit: NativeStageObserver | None):
    """Resolve the current call's observer while preserving the old explicit API."""
    observation = _CURRENT_OBSERVATION.get()
    if observation is not None:
        observation.require_active_owner()
        if explicit is not None:
            raise ValueError('explicit native stage observer conflicts with active scope')
        return observation.callback
    if explicit is not None and not callable(explicit):
        raise TypeError('native stage observer must be callable')
    return explicit


__all__ = ['NativeStageObserver', 'native_stage_observation_scope']
