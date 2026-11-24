"""Explicit, source-bound implementation choice for one native row experiment."""
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from pathlib import Path
from threading import current_thread
import os


LVT_440 = 'sky130_lvt_single_440_output_input_m12_v1'
BUFFER_COMPACT_SOURCE = 'sky130_buffer_compact_source_m1_v1'
PRECHARGE_SPLIT_GATE = 'sky130_precharge_split_gate_v1'
_VERIFICATION_SCOPE = ContextVar('native_row_candidate_verification', default=None)


@dataclass(frozen=True, slots=True)
class NativeRowCandidate:
    group: str
    implementation: str
    source_sha256: str
    source_identity: str


def candidate_for_source(cell, group, implementation):
    """Bind an explicit candidate to the original canonical source and chain."""
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.native_diffusion_source import declared_native_chains, _require_canonical_native_arrays
    source_sha, canonical = _require_canonical_native_arrays(cell)
    chains = dict(declared_native_chains(canonical, layout_intent_from_cell_def(canonical)))
    if type(group) is not str or group not in chains or implementation not in (LVT_440, BUFFER_COMPACT_SOURCE, PRECHARGE_SPLIT_GATE):
        raise ValueError('unsupported native row candidate')
    chain = chains[group]
    if implementation == LVT_440:
        if (chain.generation_profile or chain.access_profile or chain.coverage_profile is not None
                or (chain.model, chain.width_nm, chain.length_nm, len(chain.fingers)) !=
                   ('sky130_fd_pr__nfet_01v8_lvt', 2000, 150, 10)):
            raise ValueError('native row candidate differs from canonical source geometry or declaration')
    elif implementation == PRECHARGE_SPLIT_GATE:
        from clay.layout.native_diffusion_pcell import _check_chain, _connected_precharge_profile
        _check_chain(canonical, chain)
        if (chain.generation_profile or chain.access_profile or chain.coverage_profile is not None
                or not _connected_precharge_profile(chain)):
            raise ValueError('native row candidate differs from canonical source geometry or declaration')
    elif (chain.generation_profile or chain.coverage_profile is not None
          or chain.access_profile != 'shared_source_li_m1'
          or (chain.model, chain.width_nm, chain.length_nm, len(chain.fingers)) !=
             ('sky130_fd_pr__nfet_01v8', 500, 150, 3)):
        raise ValueError('native row candidate differs from canonical source geometry or declaration')
    if sha256(Path(cell.source_span['file']).read_bytes()).hexdigest() != source_sha:
        raise ValueError('native row candidate source changed')
    return NativeRowCandidate(group, implementation, source_sha, chain.source_identity)


def _require_candidate_full(cell, chain, candidate):
    if type(candidate) is not NativeRowCandidate:
        raise TypeError('native row candidate context required')
    expected = candidate_for_source(cell, candidate.group, candidate.implementation)
    if candidate != expected:
        raise ValueError('native row candidate source differs')
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.native_diffusion_source import declared_native_chains
    if dict(declared_native_chains(cell, layout_intent_from_cell_def(cell)))[candidate.group] != chain:
        raise ValueError('native row candidate chain differs')
    return candidate.implementation


def _owner():
    import asyncio
    try:
        task=asyncio.current_task()
    except RuntimeError:
        task=None
    return os.getpid(),current_thread(),task


class _CandidateVerification:
    def __init__(self,cell,chain,candidate):
        from clay.layout.place.row_replay_snapshot import freeze_row_values
        self.cell,self.chain,self.candidate=cell,chain,candidate
        self.owner=_owner()
        self.profile=_require_candidate_full(cell,chain,candidate)
        self.source_path=Path(cell.source_span['file'])
        self.source_sha=sha256(self.source_path.read_bytes()).hexdigest()
        self.cell_values=freeze_row_values(vars(cell))
        self.chain_values=freeze_row_values(chain)
        self.candidate_values=freeze_row_values(candidate)
        self.check(cell,chain,candidate)

    def check(self,cell,chain,candidate):
        from clay.layout.place.row_replay_snapshot import freeze_row_values
        if (self.owner!=_owner() or cell is not self.cell or chain is not self.chain or
                candidate is not self.candidate or
                freeze_row_values(vars(cell))!=self.cell_values or
                freeze_row_values(chain)!=self.chain_values or
                freeze_row_values(candidate)!=self.candidate_values or
                sha256(self.source_path.read_bytes()).hexdigest()!=self.source_sha):
            raise ValueError('scoped native row candidate source or operands changed')
        return self.profile


@contextmanager
def candidate_verification_scope(cell,chain,candidate):
    """Reuse one fully checked choice only inside one synchronous receipt replay."""
    if _VERIFICATION_SCOPE.get() is not None:
        raise ValueError('native row candidate verification cannot nest')
    scope=_CandidateVerification(cell,chain,candidate)
    token=_VERIFICATION_SCOPE.set(scope)
    primary=None
    try:
        yield
    except BaseException as error:
        primary=error
        raise
    finally:
        _VERIFICATION_SCOPE.reset(token)
        # A failed replay cannot publish a result; as before, propagate its
        # original exception without adding another expensive source replay.
        if primary is None:
            scope.check(cell,chain,candidate)
            if _require_candidate_full(cell,chain,candidate)!=scope.profile:
                raise ValueError('native row candidate final verification differs')


def require_candidate(cell, chain, candidate):
    scope=_VERIFICATION_SCOPE.get()
    if scope is not None:
        return scope.check(cell,chain,candidate)
    return _require_candidate_full(cell,chain,candidate)


def witness_implementation(cell, chain, witness, group):
    """Return source declaration or its separately witnessed candidate choice."""
    choices=[c for c in getattr(witness,'row_candidates',()) if c.group==group]
    if len(choices)>1:
        raise ValueError('ambiguous native row candidate')
    if choices:
        return require_candidate(cell,chain,choices[0]),choices[0]
    return chain.generation_profile,None
