"""Unique occurrence ownership in a caller-supplied, exact terminal graph.

Parameter classes, physical connectivity, occurrence completeness and port
anchors are caller evidence, not observations made here. In particular, an
omitted body/bulk role is never inferred or certified. A successful result
proves owner uniqueness only within this supplied graph; an explicit optional
third solve can additionally prove the entire used-net mapping unique. No reference file, circuit identity,
geometry, placement authority or physical signoff is consumed or issued.
The record constructors validate immutable structure, not the truth of a
manually supplied witness. Constructing a record does not run the solver.
"""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, replace
import hashlib
from time import monotonic
from typing import Literal

from clay.flow.execution_budget import _validate_seconds
from clay.flow.execution_session import current_session

TerminalRoles = tuple[tuple[str, tuple[str, ...]], ...]
Pairs = tuple[tuple[str, str], ...]
BindingStatus = Literal['UNIQUE', 'AMBIGUOUS', 'INFEASIBLE', 'UNKNOWN']
_ROLE_ARITY = {'gate': 1, 'bulk': 1, 'sd': 2, 'e': 1, 'b': 1, 'c': 1, 'ends': 2}


def _text(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError('binding identifiers must be non-empty exact strings')
    return value


def _pairs(value: Pairs) -> None:
    if type(value) is not tuple:
        raise TypeError('binding pairs must be an immutable tuple')
    for row in value:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError('binding pair rows must be exact immutable pairs')
        for text in row:
            _text(text)
    if len({key for key, _ in value}) != len(value):
        raise ValueError('binding pair keys must be unique')


def _terminals(value: TerminalRoles) -> TerminalRoles:
    if type(value) is not tuple or not value:
        raise TypeError('terminal roles must be a non-empty immutable tuple')
    rows = []
    for row in value:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError('terminal role rows must be exact pairs')
        role, nodes = row
        _text(role)
        if type(nodes) is not tuple or not nodes:
            raise TypeError('terminal nodes must be a non-empty immutable tuple')
        for node in nodes:
            _text(node)
        if role in _ROLE_ARITY and len(nodes) != _ROLE_ARITY[role]:
            raise ValueError(f'terminal role cardinality differs for {role}')
        rows.append((role, tuple(sorted(nodes))))
    if len({role for role, _ in rows}) != len(rows):
        raise ValueError('duplicate terminal role')
    return tuple(sorted(rows))


@dataclass(frozen=True, slots=True)
class PhysicalOccurrence:
    occurrence_id: str
    parameter_class: str
    terminals: TerminalRoles

    def __post_init__(self) -> None:
        _text(self.occurrence_id)
        _text(self.parameter_class)
        object.__setattr__(self, 'terminals', _terminals(self.terminals))


@dataclass(frozen=True, slots=True)
class LogicalOwner:
    owner_id: str
    parameter_class: str
    terminals: TerminalRoles
    multiplicity: int = 1

    def __post_init__(self) -> None:
        _text(self.owner_id)
        _text(self.parameter_class)
        object.__setattr__(self, 'terminals', _terminals(self.terminals))
        if type(self.multiplicity) is not int or self.multiplicity <= 0:
            raise ValueError('logical multiplicity must be an exact positive integer')


@dataclass(frozen=True, slots=True)
class BindingSolution:
    owners: Pairs
    net_mapping: Pairs

    def __post_init__(self) -> None:
        _pairs(self.owners)
        _pairs(self.net_mapping)
        if len({net for _, net in self.net_mapping}) != len(self.net_mapping):
            raise ValueError('binding net mapping must be a bijection')


@dataclass(frozen=True, slots=True)
class BindingWitness:
    input_sha256: str
    code: str
    stage: str
    solutions: tuple[BindingSolution, ...] = ()
    core: tuple[str, ...] = ()
    details: Pairs = ()

    def __post_init__(self) -> None:
        if (type(self.input_sha256) is not str or len(self.input_sha256) != 64 or
                any(char not in '0123456789abcdef' for char in self.input_sha256)):
            raise ValueError('binding witness requires an exact input SHA-256')
        _text(self.code)
        _text(self.stage)
        if type(self.solutions) is not tuple or any(type(row) is not BindingSolution for row in self.solutions):
            raise TypeError('binding solutions must be an immutable typed tuple')
        if type(self.core) is not tuple:
            raise TypeError('binding core must be an immutable tuple')
        for row in self.core:
            _text(row)
        _pairs(self.details)


@dataclass(frozen=True, slots=True)
class NetMappingProof:
    """Optional supplied-graph proof, distinct from logical owner uniqueness.

    Witness mappings use physical-node -> logical-net pairs. ``net_to_node``
    is their explicit inverse and is exposed only after uniqueness is proved.
    Constructors validate immutable structure, not a manually asserted proof.
    """
    status: Literal['UNIQUE', 'AMBIGUOUS', 'UNKNOWN', 'NOT_CHECKED']
    input_sha256: str
    code: str
    net_to_node: Pairs = ()
    witnesses: tuple[BindingSolution, ...] = ()

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in ('UNIQUE', 'AMBIGUOUS', 'UNKNOWN', 'NOT_CHECKED'):
            raise ValueError('unknown net mapping proof status')
        BindingWitness(self.input_sha256, self.code, 'net_uniqueness', self.witnesses)
        _pairs(self.net_to_node)
        if len({node for _, node in self.net_to_node}) != len(self.net_to_node):
            raise ValueError('net mapping proof must expose a bijection')
        if self.status == 'UNIQUE':
            if (not self.net_to_node or len(self.witnesses) != 1 or self.net_to_node !=
                    tuple(sorted((net, node) for node, net in self.witnesses[0].net_mapping))):
                raise ValueError('UNIQUE net mapping must invert one nonempty witness')
        elif self.net_to_node:
            raise ValueError('only UNIQUE net mapping proofs may expose a mapping')
        if self.status == 'NOT_CHECKED' and self.witnesses:
            raise ValueError('unchecked net mapping cannot carry a net proof witness')
        if len(self.witnesses) > 2 or (self.status == 'AMBIGUOUS' and (
                len(self.witnesses) != 2 or self.witnesses[0].net_mapping == self.witnesses[1].net_mapping)):
            raise ValueError('ambiguous net proof requires exactly two different mappings')
        if self.witnesses and any(row.owners != self.witnesses[0].owners for row in self.witnesses):
            raise ValueError('net mapping witnesses must retain the same proven owners')


@dataclass(frozen=True, slots=True)
class IndividualOwnerProof:
    """One occurrence's owner proof within the complete supplied graph."""
    occurrence_id: str
    input_sha256: str
    status: Literal['UNIQUE', 'AMBIGUOUS', 'UNKNOWN']
    code: str
    owner: str | None = None
    witnesses: tuple[BindingSolution, ...] = ()

    def __post_init__(self) -> None:
        _text(self.occurrence_id)
        BindingWitness(self.input_sha256, self.code, 'individual_owner', self.witnesses)
        if self.status not in ('UNIQUE', 'AMBIGUOUS', 'UNKNOWN') or type(self.status) is not str:
            raise ValueError('unknown individual owner proof status')
        if self.status == 'UNKNOWN':
            if self.owner is not None or self.witnesses:
                raise ValueError('unknown individual owner cannot expose an assignment')
            return
        if len(self.witnesses) != (1 if self.status == 'UNIQUE' else 2):
            raise ValueError('individual owner proof has the wrong witness count')
        owners = tuple(dict(row.owners).get(self.occurrence_id) for row in self.witnesses)
        if any(value is None for value in owners):
            raise ValueError('individual owner witness omits occurrence')
        if self.status == 'UNIQUE':
            if type(self.owner) is not str or self.owner != owners[0]:
                raise ValueError('unique individual owner differs from witness')
        elif self.owner is not None or owners[0] == owners[1]:
            raise ValueError('ambiguous individual owner needs distinct witnesses')


@dataclass(frozen=True, slots=True)
class OccurrenceBindingResult:
    status: BindingStatus
    bindings: Pairs
    witness: BindingWitness
    net_mapping_proof: NetMappingProof | None = None
    individual_owner_proofs: tuple[IndividualOwnerProof, ...] = ()

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in ('UNIQUE', 'AMBIGUOUS', 'INFEASIBLE', 'UNKNOWN'):
            raise ValueError('unknown occurrence binding status')
        _pairs(self.bindings)
        if type(self.witness) is not BindingWitness:
            raise TypeError('occurrence binding requires an immutable typed witness')
        proofs = self.individual_owner_proofs
        if type(proofs) is not tuple or any(type(row) is not IndividualOwnerProof for row in proofs):
            raise TypeError('individual owner proofs must be an immutable typed tuple')
        if (tuple(row.occurrence_id for row in proofs) != tuple(sorted({row.occurrence_id for row in proofs}))
                or any(row.input_sha256 != self.witness.input_sha256 for row in proofs)):
            raise ValueError('individual owner proofs have foreign identity or order')
        if self.status == 'UNIQUE':
            if (not self.bindings or len(self.witness.solutions) != 1 or
                    self.bindings != self.witness.solutions[0].owners):
                raise ValueError('UNIQUE bindings must match one non-empty witness')
        elif self.bindings:
            raise ValueError('only UNIQUE results may expose bindings')
        if self.status == 'AMBIGUOUS' and (
                len(self.witness.solutions) != 2 or
                self.witness.solutions[0].owners == self.witness.solutions[1].owners):
            raise ValueError('AMBIGUOUS requires two distinct owner witnesses')
        if self.status in ('INFEASIBLE', 'UNKNOWN') and any(row.status != 'UNKNOWN' for row in proofs):
            raise ValueError('unresolved full graph cannot carry individual proofs')
        if self.status == 'UNIQUE':
            owners = dict(self.bindings)
            if any(row.status == 'AMBIGUOUS' or
                   (row.status == 'UNIQUE' and owners.get(row.occurrence_id) != row.owner)
                   for row in proofs):
                raise ValueError('individual proof contradicts unique full graph')
        proof = self.net_mapping_proof
        if proof is not None:
            if type(proof) is not NetMappingProof:
                raise TypeError('net mapping proof must be an immutable typed record')
            if proof.input_sha256 != self.witness.input_sha256:
                raise ValueError('net mapping proof binds another input graph')
            if proof.status != 'NOT_CHECKED' and self.status != 'UNIQUE':
                raise ValueError('net mapping proof requires already unique owners')
            if any(row.owners != self.bindings for row in proof.witnesses):
                raise ValueError('net mapping witnesses differ from the proven owner assignment')


class _LocalBudgetExhausted(Exception):
    pass


def solve_reference_occurrence_binding(
    physical: tuple[PhysicalOccurrence, ...],
    logical: tuple[LogicalOwner, ...],
    *,
    port_anchors: Pairs = (),
    max_deterministic_seconds: float = 5.0,
    max_wall_seconds: float = 30.0,
    prove_net_mapping: bool = False,
    prove_individual_owners: bool = False,
) -> OccurrenceBindingResult:
    """Prove supplied owner assignments, with a second excluded-owner solve.

Each terminal role is an unordered multiset, so repeated nodes retain their
counts. Named MOS, BJT and resistor roles also enforce their known arities.
Other role names remain exact caller-defined multisets. Anchors are pairs of
logical net and physical node. Both native solves share one finite wall and
deterministic allowance; active session limits can further restrict them.
Cancellation and programming errors propagate. UNKNOWN and AMBIGUOUS results
never expose a binding, even if their diagnostic contains a candidate witness.
Empty inventories are rejected. Expiry after a native return is UNKNOWN;
any already acquired solution remains diagnostic and cannot grant a binding.
With ``prove_net_mapping=True``, a third solve excludes the complete used-net
mapping while keeping the already proved owners. All three solves share the
same allowance. A third-stage UNKNOWN or local expiry preserves owner UNIQUE
and separately reports net UNKNOWN; cancellation still propagates.
With ``prove_individual_owners=True``, each occurrence is also checked against
the same complete graph, excluding its baseline owner. All checks share the
original budget. Unfinished checks remain UNKNOWN; global ambiguity does not
prevent independently proven individual owners from being reported.
    """
    if type(prove_net_mapping) is not bool:
        raise TypeError('prove_net_mapping must be an exact bool')
    if type(prove_individual_owners) is not bool:
        raise TypeError('prove_individual_owners must be an exact bool')
    deterministic_left = _validate_seconds(max_deterministic_seconds, 'max_deterministic_seconds')
    deadline = monotonic() + _validate_seconds(max_wall_seconds, 'max_wall_seconds')
    for rows, expected, label, key in (
        (physical, PhysicalOccurrence, 'physical occurrence', 'occurrence_id'),
        (logical, LogicalOwner, 'logical owner', 'owner_id'),
    ):
        if type(rows) is not tuple or any(type(row) is not expected for row in rows):
            raise TypeError(f'{label} inventory must be an immutable typed tuple')
        if len({getattr(row, key) for row in rows}) != len(rows):
            raise ValueError(f'duplicate {label} identifier')
    physical = tuple(sorted(physical, key=lambda row: row.occurrence_id))
    logical = tuple(sorted(logical, key=lambda row: row.owner_id))
    if type(port_anchors) is not tuple:
        raise TypeError('port anchors must be an immutable tuple')
    for row in port_anchors:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError('port anchors must contain exact pairs')
        for value in row:
            _text(value)
    if len({net for net, _ in port_anchors}) != len(port_anchors):
        raise ValueError('duplicate port anchor net')
    anchors = tuple(sorted(port_anchors))
    digest = hashlib.sha256(repr((
        'clay.reference-occurrence-binding/v1', physical, logical, anchors,
    )).encode()).hexdigest()
    session = current_session()
    solutions: tuple[BindingSolution, ...] = ()
    stage = 'preflight'
    owner_proof: OccurrenceBindingResult | None = None
    net_witnesses: tuple[BindingSolution, ...] = ()

    def result(status, code, *, core=(), details=()):
        return OccurrenceBindingResult(status,
            solutions[0].owners if status == 'UNIQUE' else (),
            BindingWitness(digest, code, stage, solutions, tuple(sorted(core)), details),
            NetMappingProof('NOT_CHECKED', digest, 'OWNER_' + status)
            if prove_net_mapping else None,
            tuple(IndividualOwnerProof(row.occurrence_id, digest, 'UNKNOWN', 'NOT_PROVEN')
                  for row in physical) if prove_individual_owners else ())

    def net_result(status, code):
        return replace(owner_proof, net_mapping_proof=NetMappingProof(status, digest, code,
            tuple(sorted((net, node) for node, net in net_witnesses[0].net_mapping))
            if status == 'UNIQUE' else (), net_witnesses))

    def checkpoint():
        if session is not None:
            session.checkpoint()
        if monotonic() >= deadline or deterministic_left <= 0:
            raise _LocalBudgetExhausted()

    phase = session.phase('reference.occurrence_binding', details={
        'input_sha256': digest, 'physical_occurrences': len(physical),
        'logical_owners': len(logical), 'scope': 'supplied_terminal_graph_only',
    }) if session else nullcontext()
    with phase:
        try:
            checkpoint()
            if not physical or not logical:
                return result('INFEASIBLE', 'EMPTY_INVENTORY', details=(
                    ('physical', str(len(physical))), ('logical', str(len(logical))),))
            if len(physical) != sum(row.multiplicity for row in logical):
                return result('INFEASIBLE', 'OCCURRENCE_DENOMINATOR_MISMATCH', details=(
                    ('physical', str(len(physical))),
                    ('logical', str(sum(row.multiplicity for row in logical))),))
            physical_classes = Counter(row.parameter_class for row in physical)
            logical_classes = Counter()
            for row in logical:
                logical_classes[row.parameter_class] += row.multiplicity
            if physical_classes != logical_classes:
                return result('INFEASIBLE', 'PARAMETER_CLASS_DENOMINATOR_MISMATCH', details=(
                    ('physical', repr(sorted(physical_classes.items()))),
                    ('logical', repr(sorted(logical_classes.items()))),))
            nodes = tuple(sorted({node for row in physical for _, role in row.terminals for node in role}))
            nets = tuple(sorted({net for row in logical for _, role in row.terminals for net in role}))
            if len(nodes) != len(nets):
                return result('INFEASIBLE', 'NET_DENOMINATOR_MISMATCH', details=(
                    ('physical_nodes', repr(nodes)), ('logical_nets', repr(nets)),))
            if any(net not in nets or node not in nodes for net, node in anchors):
                return result('INFEASIBLE', 'ANCHOR_OUTSIDE_USED_GRAPH', details=(
                    ('anchors', repr(anchors)), ('physical_nodes', repr(nodes)), ('logical_nets', repr(nets))))

            from ortools.sat.python import cp_model
            model = cp_model.CpModel()
            net_vars = {}
            for node in nodes:
                checkpoint()
                for index, net in enumerate(nets):
                    if index % 256 == 0:
                        checkpoint()
                    net_vars[node, net] = model.NewBoolVar(f'net_{len(net_vars)}')
            for node in nodes:
                model.AddExactlyOne(net_vars[node, net] for net in nets)
            for net in nets:
                model.AddExactlyOne(net_vars[node, net] for node in nodes)
            assumptions = {}
            def assume(label, constraint):
                variable = model.NewBoolVar(f'assumption_{len(assumptions)}')
                constraint.OnlyEnforceIf(variable)
                model.AddAssumption(variable)
                assumptions[variable.Index()] = label
            for net, node in anchors:
                assume(f'anchor:{net}:{node}', model.Add(net_vars[node, net] == 1))

            owner_vars = {}
            for occurrence in physical:
                checkpoint()
                physical_roles = dict(occurrence.terminals)
                variables = []
                for owner in logical:
                    checkpoint()
                    logical_roles = dict(owner.terminals)
                    if (occurrence.parameter_class != owner.parameter_class or
                        tuple((r, len(v)) for r, v in occurrence.terminals) !=
                        tuple((r, len(v)) for r, v in owner.terminals)):
                        continue
                    variable = model.NewBoolVar(f'owner_{len(owner_vars)}')
                    owner_vars[occurrence.occurrence_id, owner.owner_id] = variable
                    variables.append(variable)
                    for role, physical_nodes in physical_roles.items():
                        counts = Counter(physical_nodes)
                        for net, count in Counter(logical_roles[role]).items():
                            model.Add(sum(multiplicity * net_vars[node, net]
                                for node, multiplicity in counts.items()) == count).OnlyEnforceIf(variable)
                if not variables:
                    return result('INFEASIBLE', 'NO_COMPATIBLE_OWNER', details=(
                        ('physical_occurrence', occurrence.occurrence_id),))
                assume(f'occurrence:{occurrence.occurrence_id}', model.Add(sum(variables) == 1))
            for owner in logical:
                checkpoint()
                assume(f'owner:{owner.owner_id}', model.Add(sum(variable
                    for (occurrence_id, owner_id), variable in owner_vars.items()
                    if owner_id == owner.owner_id) == owner.multiplicity))

            def solve_native():
                nonlocal deterministic_left
                checkpoint()
                native_phase = session.phase(f'reference.occurrence_binding.{stage}') if session else nullcontext()
                with native_phase as handle:
                    if handle:
                        proto = model.Proto()
                        handle.annotate(model_sha256=hashlib.sha256(
                            proto.SerializeToString(deterministic=True)).hexdigest(),
                            variables=len(proto.variables), constraints=len(proto.constraints))
                    checkpoint()
                    solver = cp_model.CpSolver()
                    wall_left = deadline - monotonic()
                    deterministic_limit = deterministic_left
                    if session:
                        session_wall = session.remaining_wall_seconds()
                        session_deterministic = session.remaining_deterministic_seconds()
                        if session_wall is not None:
                            wall_left = min(wall_left, session_wall)
                        if session_deterministic is not None:
                            deterministic_limit = min(deterministic_limit, session_deterministic)
                    checkpoint()
                    solver.parameters.num_search_workers = 1
                    solver.parameters.random_seed = 0
                    solver.parameters.max_deterministic_time = deterministic_limit
                    solver.parameters.max_time_in_seconds = wall_left
                    if handle:
                        handle.annotate(native_deterministic_seconds=deterministic_limit,
                            native_wall_seconds=wall_left)
                    checkpoint()
                    status = solver.Solve(model)
                    used = _validate_seconds(solver.ResponseProto().deterministic_time,
                        'native deterministic usage', allow_zero=True)
                    deterministic_left = max(0.0, deterministic_left - used)
                    if handle:
                        handle.annotate(solver_status=solver.StatusName(status), solver_deterministic_seconds=used)
                        session.charge_deterministic(used)
                    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE, cp_model.INFEASIBLE, cp_model.UNKNOWN):
                        raise RuntimeError(f'reference occurrence model returned {solver.StatusName(status)}')
                    return solver, status

            def solution(solver):
                return BindingSolution(
                    tuple((occurrence.occurrence_id, owner.owner_id)
                        for occurrence in physical for owner in logical
                        if (occurrence.occurrence_id, owner.owner_id) in owner_vars and
                        solver.BooleanValue(owner_vars[occurrence.occurrence_id, owner.owner_id])),
                    tuple((node, net) for node in nodes for net in nets
                        if solver.BooleanValue(net_vars[node, net])))

            stage = 'feasibility'
            solver, status = solve_native()
            if status == cp_model.UNKNOWN:
                checkpoint()
                return result('UNKNOWN', 'SOLVER_UNKNOWN')
            if status == cp_model.INFEASIBLE:
                checkpoint()
                return result('INFEASIBLE', 'CONNECTIVITY_INFEASIBLE', core=tuple(
                    assumptions[int(index)] for index in solver.SufficientAssumptionsForInfeasibility()))
            solutions = (solution(solver),)
            checkpoint()
            # Preserve the full feasible model before adding the owner-exclusion
            # query. Reusing that infeasible query would forge net uniqueness.
            net_model = model.Clone() if prove_net_mapping or prove_individual_owners else None
            # Exclude owners only: a different internal net bijection is permitted.
            model.Add(sum(owner_vars[pair] for pair in solutions[0].owners) <= len(physical) - 1)
            stage = 'uniqueness'
            solver, status = solve_native()
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                solutions += (solution(solver),)
            checkpoint()
            if status == cp_model.UNKNOWN:
                return result('UNKNOWN', 'SOLVER_UNKNOWN')
            if status == cp_model.INFEASIBLE:
                owner_proof = result('UNIQUE', 'OWNER_ASSIGNMENT_UNIQUE')
                if prove_individual_owners:
                    owner_proof = replace(owner_proof, individual_owner_proofs=tuple(
                        IndividualOwnerProof(occurrence, digest, 'UNIQUE', 'GLOBAL_OWNER_UNIQUE', owner, (solutions[0],))
                        for occurrence, owner in solutions[0].owners))
                if not prove_net_mapping:
                    return owner_proof
                net_witnesses = solutions
                stage = 'net_uniqueness'
                model = net_model
                for pair in owner_proof.bindings:
                    checkpoint()
                    model.Add(model.GetBoolVarFromProtoIndex(owner_vars[pair].Index()) == 1)
                chosen = net_witnesses[0].net_mapping
                model.Add(sum(model.GetBoolVarFromProtoIndex(net_vars[pair].Index())
                              for pair in chosen) <= len(chosen) - 1)
                solver, status = solve_native()
                if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    net_witnesses += (solution(solver),)
                checkpoint()
                if status == cp_model.UNKNOWN:
                    return net_result('UNKNOWN', 'SOLVER_UNKNOWN')
                if status == cp_model.INFEASIBLE:
                    return net_result('UNIQUE', 'NET_MAPPING_UNIQUE')
                return net_result('AMBIGUOUS', 'NET_MAPPING_AMBIGUOUS')
            ambiguous = result('AMBIGUOUS', 'OWNER_ASSIGNMENT_AMBIGUOUS')
            if not prove_individual_owners:
                return ambiguous
            baseline, alternative = (dict(row.owners) for row in solutions)
            proofs = []
            exhausted = False
            for occurrence in physical:
                if session is not None:
                    session.checkpoint()
                name = occurrence.occurrence_id
                owner = baseline[name]
                if alternative[name] != owner:
                    proofs.append(IndividualOwnerProof(name, digest, 'AMBIGUOUS', 'ALTERNATIVE_OWNER',
                                                       witnesses=solutions))
                    continue
                if exhausted:
                    proofs.append(IndividualOwnerProof(name, digest, 'UNKNOWN', 'BUDGET_EXHAUSTED'))
                    continue
                try:
                    checkpoint()
                    model = net_model.Clone()
                    model.Add(model.GetBoolVarFromProtoIndex(owner_vars[name, owner].Index()) == 0)
                    stage = 'individual_owner'
                    solver, individual_status = solve_native()
                    checkpoint()
                    if individual_status == cp_model.INFEASIBLE:
                        proofs.append(IndividualOwnerProof(name, digest, 'UNIQUE', 'OWNER_EXCLUSION_INFEASIBLE',
                                                           owner, (solutions[0],)))
                    elif individual_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                        individual_solution = solution(solver)
                        checkpoint()
                        proofs.append(IndividualOwnerProof(name, digest, 'AMBIGUOUS', 'ALTERNATIVE_OWNER',
                                                           witnesses=(solutions[0], individual_solution)))
                    else:
                        proofs.append(IndividualOwnerProof(name, digest, 'UNKNOWN', 'SOLVER_UNKNOWN'))
                except _LocalBudgetExhausted:
                    exhausted = True
                    proofs.append(IndividualOwnerProof(name, digest, 'UNKNOWN', 'BUDGET_EXHAUSTED'))
            return replace(ambiguous, individual_owner_proofs=tuple(proofs))
        except _LocalBudgetExhausted:
            if owner_proof is not None:
                return net_result('UNKNOWN', 'BUDGET_EXHAUSTED')
            return result('UNKNOWN', 'BUDGET_EXHAUSTED')
