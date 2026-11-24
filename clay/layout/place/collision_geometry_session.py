"""Own bounded, exact derived collision geometry during one materialization.

Keys contain every rectangle, layer, clearance and computation identity. Values
are displacement unions, never candidate legality or certificates. Nested
stages share their owner's scope; a later invocation starts empty.
Each distinct complete operand tuple is owned once, with one reference per
entry using it; each entry owns its cover rectangles separately. Their total
has a separate count budget. This is not a byte limit: a cache miss still uses
data-dependent working storage under the invocation's external RSS guard.
"""
import asyncio
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from threading import get_ident

from clay.flow.execution_session import current_session

_CURRENT = ContextVar('placement_collision_queries', default=None)


def _invocation():
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return get_ident(), task


class _Queries:
    def __init__(self, capacity, max_rectangles):
        self.owner = _invocation()
        self.capacity = capacity
        self.max_rectangles = max_rectangles
        self.retained_rectangles = 0
        self.values = OrderedDict()
        self._operands = {}
        self.closed = False
        self.lookups = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.raw_atom_pairs = 0
        self.unique_forbidden_boxes = 0
        self.cover_rectangles = 0

    def check(self):
        if self.closed or self.owner != _invocation():
            raise RuntimeError('collision query invocation differs or is closed')

    def _evict_oldest(self):
        key, (_, cover_charge) = self.values.popitem(last=False)
        self.retained_rectangles -= cover_charge
        # Equal left/right operands are one dependency of this entry.
        for operand in dict.fromkeys(key[1:3]):
            canonical, references = self._operands[operand]
            if references == 1:
                del self._operands[canonical]
                self.retained_rectangles -= len(canonical)
            else:
                self._operands[canonical] = (canonical, references - 1)

    def retain(self, key, result):
        self.check()
        operands = tuple(dict.fromkeys(key[1:3]))
        cover_charge = len(result[0])
        # An entry that cannot fit alone must not disturb any retained entry.
        if sum(map(len, operands)) + cover_charge > self.max_rectangles:
            return
        while True:
            additional = cover_charge + sum(
                len(operand) for operand in operands if operand not in self._operands)
            if (len(self.values) < self.capacity and
                    self.retained_rectangles + additional <= self.max_rectangles):
                break
            self._evict_oldest()
            # Eviction can remove the last reference to an input this new
            # entry needs. Recompute its charge before admitting the entry.
        canonical_inputs = {}
        for operand in operands:
            previous = self._operands.get(operand)
            if previous is None:
                canonical, references = operand, 0
                self.retained_rectangles += len(operand)
            else:
                canonical, references = previous
            self._operands[canonical] = (canonical, references + 1)
            canonical_inputs[operand] = canonical
        canonical_key = (key[0], canonical_inputs[key[1]],
                         canonical_inputs[key[2]], key[3])
        self.values[canonical_key] = (result, cover_charge)
        self.retained_rectangles += cover_charge


def _session_stopped(session) -> bool:
    if session.cancellation.reason is not None:
        return True
    rem_wall = session.remaining_wall_seconds()
    if rem_wall is not None and rem_wall <= 0.0:
        return True
    rem_det = session.remaining_deterministic_seconds()
    if rem_det is not None and rem_det <= 0.0:
        return True
    return False


@contextmanager
def collision_query_scope(*, max_entries=2048, max_rectangles=250_000):
    """Reuse an enclosing scope or own a new cache through exception-safe exit."""
    if type(max_entries) is not int or max_entries <= 0:
        raise ValueError('collision cache capacity must be a positive integer')
    if type(max_rectangles) is not int or max_rectangles <= 0:
        raise ValueError('collision cache rectangle budget must be a positive integer')
    current = _CURRENT.get()
    if current is not None:
        current.check()
        yield
        return
    current = _Queries(max_entries, max_rectangles)
    token = _CURRENT.set(current)
    try:
        yield
    except BaseException:
        raise
    else:
        session = current_session()
        if current.lookups > 0 and session is not None and not _session_stopped(session):
            with session.phase(
                'placement.collision.forbidden_union',
                details={
                    'max_entries': max_entries,
                    'max_rectangles': max_rectangles,
                    'lookups': current.lookups,
                    'cache_hits': current.cache_hits,
                    'cache_misses': current.cache_misses,
                    'raw_atom_pairs': current.raw_atom_pairs,
                    'unique_forbidden_boxes': current.unique_forbidden_boxes,
                    'cover_rectangles': current.cover_rectangles,
                },
            ):
                pass
    finally:
        current.closed = True
        current.values.clear()
        current._operands.clear()
        current.retained_rectangles = 0
        _CURRENT.reset(token)


def memoized_collision_union(left, right, *, required_nm, compute):
    # Python key equality aliases False/0.0 with 0; validate before any hit.
    if type(required_nm) is not int or required_nm < 0:
        raise ValueError('pair collision clearance must be one non-negative integer')
    current = _CURRENT.get()
    key = (compute, tuple(left), tuple(right), required_nm)
    if current is not None:
        current.check()
    hit = current is not None and key in current.values
    session = current_session()
    if current is not None:
        if session is not None:
            session.checkpoint()
        if hit:
            result, _ = current.values[key]
            # Keep the stored key's canonical input objects on a cache hit.
            current.values.move_to_end(key)
        else:
            result = (compute(left, right, required_nm=required_nm,
                              checkpoint=session.checkpoint) if session is not None else
                      compute(left, right, required_nm=required_nm))
            current.retain(key, result)
        current.lookups += 1
        if hit:
            current.cache_hits += 1
        else:
            current.cache_misses += 1
        current.raw_atom_pairs += int(result[1])
        current.unique_forbidden_boxes += int(result[2])
        current.cover_rectangles += len(result[0])
        return result

    phase = session.phase('placement.collision.forbidden_union', details={
        'left_atoms': len(left), 'right_atoms': len(right),
        'clearance_nm': required_nm, 'cache_hit': hit,
    }) if session else nullcontext()
    with phase as handle:
        result = (compute(left, right, required_nm=required_nm,
                          checkpoint=session.checkpoint) if session else
                  compute(left, right, required_nm=required_nm))
        if handle:
            handle.annotate(raw_atom_pairs=result[1], unique_forbidden_boxes=result[2],
                            cover_rectangles=len(result[0]))
        return result
