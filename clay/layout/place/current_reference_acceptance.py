"""Join one live full reference producer to actual native and final placement.

This checked adapter establishes placement-time source/reference correspondence.
Quality arithmetic remains independent, and the outer producer must finish its
input checks before a host publishes success. No receipt or caller status can
open this scope. Body nets remain source-declared, not physically traced.
"""
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from clay.layout.place.row_replay_snapshot import freeze_row_values

_SEAL = object()
_CURRENT = ContextVar('current_reference_structure_inputs', default=None)
_IMPLEMENTATION_SEEDS = (
    'clay/layout/place/current_reference_acceptance.py',
    'clay/layout/place/native_reference_quality.py',
    'clay/layout/flow_placement.py',
    'clay/layout/generic_placement.py',
    'clay/layout/current_reference_mim.py',
    'clay/layout/current_reference_mos_placement.py',
    'clay/layout/source_resistor_pcell_assembly.py',
    'clay/layout/source_resistor_placement.py',
    'clay/layout/place/current_structure_operands.py',
    'clay/layout/place/current_structure_consumption.py',
    'clay/layout/place/final_quality_observation.py',
    'clay/layout/place/final_quality_evaluation.py',
)


def _require(value, message):
    if not value:
        raise ValueError('current reference structure '+message)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _implementation_state(repo_root):
    from clay.flow.implementation_closure import capture_static_local_implementation_state
    return capture_static_local_implementation_state(repo_root, _IMPLEMENTATION_SEEDS)


def _complete_population(records, population, contract):
    """Pure denominator comparison; this helper grants no input authority."""
    expected = {name: (family, count) for name, family, count, _kind in population}
    _require(len(expected) == len(population) and bool(expected), 'source population is repeated or empty')
    aliases = {'nmos': 'mos', 'pmos': 'mos', 'capacitor': 'mim', 'resistor': 'resistor'}
    counts = Counter(row.logical_id for row in records)
    identities = tuple(row.occurrence_id for row in records)
    _require(len(set(identities)) == len(identities), 'raw occurrence denominator is repeated')
    _require(counts == Counter({name: count for name, (_family, count) in expected.items()}),
             'complete reference/source population denominator differs')
    _require(all(aliases.get(row.family) == expected[row.logical_id][0] for row in records),
             'reference/native source family differs or is unsupported')
    targets = {row.logical_id: tuple(row.occurrence_ids) for row in contract.logical_population}
    actual = {name: tuple(sorted(row.occurrence_id for row in records if row.logical_id == name))
              for name in expected}
    _require(len(targets) == len(contract.logical_population) and targets == actual,
             'contract complete raw occurrence population differs')
    return dict(logical=len(expected), expanded=len(records), raw_occurrence_ids=sorted(identities))


def _require_native_cover(native, reference, population):
    expected = {name: (family, count) for name, family, count, _kind in population}
    source = native['source_population']
    _require(source['owners'] == [list(row) for row in population]
        and source['logical'] == len(expected) and source['expanded'] == sum(row[1] for row in expected.values()),
        'native complete source population denominator differs')
    identity = native['technology_identity']
    _require(native['source_sha256'] == reference['source_sha256']
        and tuple(identity[index] for index in (0, 2, 3)) == tuple(reference['pdk_identity']),
        'native current source or PDK differs from original reference')
    _require(not native['uncovered_logical_owners'], 'native complete family cover has uncovered source owners')
    covered = []
    families = {family for family, _count in expected.values()}
    _require(set(native['families']) == families, 'native family denominator differs')
    for family, value in native['families'].items():
        names = sorted(name for name, (selected, _count) in expected.items() if selected == family)
        _require(value['logical_owners'] == names
            and value['units'] == sum(expected[name][1] for name in names),
            'native family owner/unit denominator differs')
        covered.extend(value['logical_owners'])
    _require(len(covered) == len(set(covered)) and set(covered) == set(expected),
             'native complete family cover is repeated or incomplete')


def _require_same_observation(source_cell, compact, observation):
    from clay.layout.place.final_quality_observation import (
        FinalQualityGeometryObservation, observe_materialized_source,
    )
    _require(type(observation) is FinalQualityGeometryObservation,
             'same candidate requires an exact current final observation')
    current = observe_materialized_source(cell=source_cell, compact=compact)
    _require(current.canonical_data() == observation.canonical_data(),
             'quality observation differs from actual same candidate geometry')
    return current


def _mim_values(value):
    from clay.layout.place.current_structure_operands import _native_values
    return _native_values(tuple(getattr(value, name) for name in (
        'source_member', 'source_sha256', 'source_inventory_sha256', 'recipe',
        'site_ledger', 'terminal_layers', 'terminal_accesses', 'geometry_snapshot')))


def _mos_identity(mos, placement, intent, reference, source_cell, tech):
    from clay.frontend.current_reference_mos_inputs import CurrentReferenceMosInputs
    from clay.layout.current_reference_mos_placement import CurrentReferenceMosPlacement
    _require(type(mos) is CurrentReferenceMosPlacement, 'requires current complete MOS placement inputs')
    inputs = mos.reference_inputs
    _require(type(inputs) is CurrentReferenceMosInputs and inputs._reference is reference
        and inputs.source_cell is source_cell and inputs.tech is tech and intent is mos.intent,
        'MOS reference/source/PDK/intent invocation identity differs')
    mos.require_current()
    materialization = placement.row_materialization
    _require(materialization is not None and materialization.source_projection is mos.row_source_projection
        and materialization.source_projection.reference_overlay is inputs.overlay,
        'actual MOS compiler consumed a different original source/reference projection')
    rows = materialization.realizations
    _require(len(rows) == len(mos.row_geometry_overrides)
        and {row.owner for row in rows} == set(mos.row_geometry_overrides)
        and all(row.geometry is mos.row_geometry_overrides[row.owner] for row in rows),
        'actual MOS compiler geometry differs from original native reference inputs')


def _capture_reference_join(*, source_cell, tech, placement, intent, reference, mos,
                            mims, resistors, contract, repo_root):
    from clay.frontend.reference_device_artwork import CurrentReferenceDeviceArtwork
    from clay.frontend.source_revision import reopen_complete_source
    from clay.layout.current_reference_quality import build_current_reference_quality
    from clay.layout.current_reference_mim import materialize_current_reference_mims
    from clay.layout.current_reference_mos_placement import capture_current_reference_mos_placement
    from clay.layout.reference_quality_contract import ReferenceQualityContract
    from clay.layout.source_resistor_pcell_assembly import CurrentResistorPCellAssembly
    from clay.layout.source_mim_materialization import SourceMimMaterialization
    from clay.layout.place.current_structure_operands import _source_population
    from clay.layout.place.row_replay_session import row_replay_phase
    _require(type(reference) is CurrentReferenceDeviceArtwork, 'requires an actual current reference producer')
    reference.require_current()
    _require(reference._source is source_cell and reference._tech is tech,
             'reference current source/PDK invocation identity differs')
    _require(Path(repo_root).resolve(strict=True) == reference._repo,
             'reference original repository root differs')
    implementation = _implementation_state(repo_root)
    _require(type(contract) is ReferenceQualityContract, 'requires an exact complete reference contract')
    source = reopen_complete_source(source_cell)
    data = reference.canonical_data()
    _require(source['source_sha256'] == data['source_sha256'] and source['pdk_id'] == tech.pdk_id,
             'complete current source differs from original reference')
    with row_replay_phase('reference_structure.rebuild_complete_targets'):
        expected_contract, plan = build_current_reference_quality(source_cell, reference=reference, repo_root=repo_root)
    _require(contract.canonical_data() == expected_contract.canonical_data(),
             'contract differs from freshly rebuilt complete current reference targets')
    population = _source_population(source_cell)
    denominator = _complete_population(reference.records, population, contract)
    if any(family == 'mos' for _name, family, _count, _kind in population):
        _mos_identity(mos, placement, intent, reference, source_cell, tech)
        # Public dataclass construction and a self-consistent snapshot alone do
        # not prove that native raw parts/sites came from the current producer.
        with row_replay_phase('reference_structure.rebuild_original_mos_native'):
            expected_mos = capture_current_reference_mos_placement(source_cell, reference=reference, tech=tech)
        _require(mos._values() == expected_mos._values()
            and mos.reference_inputs._values() == expected_mos.reference_inputs._values(),
            'MOS complete original raw reference/native reconstruction differs')
    else:
        _require(mos is None, 'unexpected MOS reference input population')
    _require(type(mims) is tuple and all(type(value) is SourceMimMaterialization for value in mims),
             'MIM requires complete exact current native inputs')
    with row_replay_phase('reference_structure.rebuild_original_mim_native'):
        expected_mims = materialize_current_reference_mims(source_cell, reference=reference, tech=tech)
    _require(tuple(value.source_member[0] for value in mims) == tuple(sorted(expected_mims))
        and all(_mim_values(value) == _mim_values(expected_mims[value.source_member[0]]) for value in mims),
        'MIM original reference graph/master/site/frame native inputs differ')
    _require(type(resistors) is tuple and all(type(value) is CurrentResistorPCellAssembly
        and value.producer is reference for value in resistors),
        'resistor inputs must retain the same original live reference producer')
    for value in resistors:
        value.validate()
    reference.require_current()
    _require(_implementation_state(repo_root) == implementation,
             'current native/reference implementation changed during reconstruction')
    return source, data, population, denominator, plan.sha256, implementation


class CurrentReferenceStructureInputs:
    """One lexical current native/reference join, with no public constructor."""
    def __init__(self, *args, **kwargs):
        raise TypeError('current reference structure inputs require actual checked capture')

    def _values(self):
        return (tuple(id(getattr(self, name)) for name in ('reference', 'native', 'mos', 'source_cell',
            'tech', 'placement', 'intent', 'contract')), tuple(id(value) for value in self.mims),
            tuple(id(value) for value in self.resistors), self.reference_sha256, self.plan_sha256,
            self.contract.sha256, freeze_row_values(self.source_data), self.population,
            str(self.repo_root), self.implementation,
            None if self.mos is None else (id(self.mos.reference_inputs), self.mos._values(),
                self.mos.reference_inputs._values()), tuple(_mim_values(value) for value in self.mims))

    def _require_current(self):
        from clay.layout.place.row_replay_session import _owner
        _require(getattr(self, '_seal', None) is _SEAL and self._active
            and _CURRENT.get() is self and self._owner == _owner(), 'input capture lifetime is closed or foreign')
        _require(self._values() == self._snapshot, 'original current source/reference native inputs changed')
        self.reference.require_current()
        _require(_sha(self.reference.canonical_data()) == self.reference_sha256,
                 'original current reference report changed')
        _require(_implementation_state(self.repo_root) == self.implementation,
                 'current native/reference implementation changed during checked lifetime')
        if self.mos is not None:
            _mos_identity(self.mos, self.placement, self.intent, self.reference, self.source_cell, self.tech)
        self.native._require_current()


@contextmanager
def current_reference_structure_inputs(*, source_cell, tech, placement, intent, reference, mos,
                                       mims, resistors, contract, repo_root):
    from clay.layout.place.current_structure_operands import capture_current_structure_operands
    from clay.layout.place.row_replay_session import _owner, row_replay_phase
    if isinstance(mims, Mapping):
        _require(all(getattr(value, 'source_member', (None,))[0] == name for name, value in mims.items()),
                 'MIM mapping source owner differs')
        mims = tuple(mims[name] for name in sorted(mims))
    source, data, population, denominator, plan_sha, implementation = _capture_reference_join(
        source_cell=source_cell, tech=tech, placement=placement, intent=intent, reference=reference,
        mos=mos, mims=mims, resistors=resistors, contract=contract, repo_root=repo_root)
    with row_replay_phase('reference_structure.capture_complete_native'):
        native = capture_current_structure_operands(source_cell=source_cell, tech=tech, placement=placement,
            intent=intent, mim_materializations=mims, resistor_assemblies=resistors)
    _require_native_cover(native.canonical_data(), data, population)
    _require(all(native.canonical_data()[key] == source[key]
        for key in ('source_sha256', 'source_inventory_sha256', 'electrical_sha256')),
        'native complete source revision differs from reference invocation source')
    value = object.__new__(CurrentReferenceStructureInputs)
    value._seal, value._owner, value._active = _SEAL, _owner(), False
    value.source_cell, value.tech, value.placement, value.intent = source_cell, tech, placement, intent
    value.reference, value.mos, value.mims, value.resistors = reference, mos, mims, resistors
    value.contract, value.native, value.population = contract, native, population
    value.source_data, value.denominator = source, denominator
    value.repo_root, value.implementation = Path(repo_root).resolve(strict=True), implementation
    value.reference_sha256, value.plan_sha256 = _sha(data), plan_sha
    value._snapshot = value._values()
    with native.checked():
        token = _CURRENT.set(value)
        value._active = True
        try:
            value._require_current()
            try:
                yield value
            except BaseException as error:
                try:
                    value._require_current()
                except BaseException as ending:
                    from clay.flow.control_stop import record_unwind_failure
                    record_unwind_failure(error, ending)
                raise
            else:
                value._require_current()
        finally:
            value._active = False
            _CURRENT.reset(token)


@dataclass(frozen=True, slots=True, init=False)
class CurrentReferenceStructureEvidence:
    _data: bytes
    _seal: object

    def __init__(self, *args, **kwargs):
        raise TypeError('current reference structure evidence requires actual final observation')

    def canonical_data(self):
        _require(self._seal is _SEAL, 'current final observation capture is missing')
        return json.loads(self._data)

    @property
    def sha256(self):
        return hashlib.sha256(self._data).hexdigest()


def observe_complete_current_reference_structure(inputs, *, compact, source_cell, tech, intent,
                                                 contract, observation):
    from clay.layout.place.current_structure_consumption import observe_final_structure_consumption
    from clay.layout.place.final_placement_state import final_placement_state_digest
    from clay.layout.route.exact_artifact_evidence import _emitted_device_snapshot_sha256
    _require(type(inputs) is CurrentReferenceStructureInputs, 'requires current captured reference/native inputs')
    inputs._require_current()
    _require(source_cell is inputs.source_cell and tech is inputs.tech
        and intent is inputs.intent and contract is inputs.contract,
        'final current source/PDK/intent/contract invocation differs')
    actual = _require_same_observation(source_cell, compact, observation)
    native = observe_final_structure_consumption(inputs.native, compact=compact, source_cell=source_cell,
        tech=tech, intent=intent, observation=actual)
    data = native.canonical_data()
    reference = inputs.reference.canonical_data()
    _require_native_cover(data, reference, inputs.population)
    denominator = _complete_population(inputs.reference.records, inputs.population, contract)
    _require({row.logical_id: len(row.occurrences) for row in actual.logical_geometry}
        == {name: count for name, _family, count, _kind in inputs.population},
        'same candidate complete quality occurrence denominator differs')
    state = final_placement_state_digest(compact.final_placement_certificate)
    positions, geometry = compact.ring_inclusive
    artwork = _emitted_device_snapshot_sha256(geometry, positions)
    _require(state == data['final_placement_state_sha256'] == actual.final_placement_state_sha256
        and artwork == data['final_artwork_sha256'], 'final native/reference evidence belongs to another candidate')
    _require_same_observation(source_cell, compact, actual)
    inputs._require_current()
    result = object.__new__(CurrentReferenceStructureEvidence)
    object.__setattr__(result, '_seal', _SEAL)
    object.__setattr__(result, '_data', _json(dict(schema='clay.current-complete-structure-measurement/v2',
        scope='same_candidate_placement_structure_pending_outer_input_binding_completion',
        required_structure_status='pass', current_structure_status='pass', physical_admission=False,
        quality_admission='REQUIRES_COMPLETE_SAME_CANDIDATE_QUALITY_EVALUATION',
        source_sha256=data['source_sha256'], source_inventory_sha256=data['source_inventory_sha256'],
        electrical_sha256=data['electrical_sha256'], source_population=data['source_population'],
        reference_denominator=denominator, technology_identity=data['technology_identity'],
        final_placement_state_sha256=state, final_artwork_sha256=artwork, observation_sha256=actual.sha256,
        contract_sha256=contract.sha256, structure_plan_sha256=inputs.plan_sha256,
        reference_artwork_sha256=inputs.reference_sha256, native_operands_sha256=inputs.native.sha256,
        current_implementation=inputs.implementation, current_implementation_sha256=_sha(inputs.implementation),
        native_consumption_sha256=native.sha256, reference_artwork=reference, native_consumption=data,
        limitations=['Placement-time source/native preservation and current reference correspondence only.',
            'Body nets are source-declared, not independently physically traced.',
            'No routing, exact, DRC, LVS or independent signoff conclusion.'])))
    return result


__all__ = ['CurrentReferenceStructureInputs', 'CurrentReferenceStructureEvidence',
           'current_reference_structure_inputs', 'observe_complete_current_reference_structure']
