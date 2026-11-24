"""Immutable route-time configuration used by source-output replay."""
from dataclasses import dataclass
import json

_CAPABILITIES = ("consumes_fixed_multilayer_joint_terminals",
                 "consumes_artwork_conductors", "strict_device_m1_provenance")

@dataclass(frozen=True, slots=True)
class SourceEngineRecord:
    # A canonical JSON string is deeply immutable and directly serializable.
    canonical_json: str

    def capability(self, name):
        if name not in _CAPABILITIES:
            raise ValueError("unknown source engine capability")
        return json.loads(self.canonical_json)["capabilities"][name] is True

    @property
    def consumes_fixed_multilayer_joint_terminals(self):
        return self.capability("consumes_fixed_multilayer_joint_terminals")


def freeze_source_engine(engine):
    from clay.layout.route.canonical_input_capture import (route_engine_configuration,
                                                            experimental_routing_configuration)
    if engine is None:
        raise ValueError("source output requires an explicitly selected engine")
    caps = {}
    for name in _CAPABILITIES:
        # Distinguish an absent capability from an explicit false capability.
        value = getattr(engine, name, None)
        if value is not None and type(value) is not bool:
            raise TypeError("source engine capability must be a bool: " + name)
        caps[name] = value
    data = {"schema": "clay.source-output-engine/v1",
            "configuration": route_engine_configuration(engine), "capabilities": caps}
    experimental = experimental_routing_configuration(engine)
    if experimental is not None:
        data['schema'] = ('clay.source-output-engine/v3' if 'fixed_index_seal' in experimental
                          else 'clay.source-output-engine/v2')
        data['experimental_routing'] = experimental
    return SourceEngineRecord(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False))

def require_source_engine(record, engine):
    if type(record) is not SourceEngineRecord or record != freeze_source_engine(engine):
        raise ValueError("actual route engine differs from source configuration or capabilities")
    if not all(record.capability(name) for name in _CAPABILITIES):
        raise ValueError("source engine lacks the fixed-stack capabilities")
    return engine


def require_source_engine_with_recorded_experiments(record, source_engine):
    """Compare a source-selected engine under the explicitly recorded options.

    The supplied engine is a fresh reconstruction of the declared source
    policy, not the live routing engine. Reapply only supported recorded
    experiments so that the final exact comparison still checks the complete
    configuration, capabilities and experiment versions.
    """
    if type(record) is not SourceEngineRecord:
        raise ValueError('source engine record required')
    payload = json.loads(record.canonical_json)
    experiment = payload.get('experimental_routing')
    if experiment is None:
        return require_source_engine(record, source_engine)
    if type(experiment) is not dict:
        raise ValueError('source engine experiment differs')
    from clay.layout.route.engine_gridless import GridlessEngine
    if type(source_engine) is not GridlessEngine:
        raise ValueError('source engine experiment requires Gridless')
    keys = set(experiment)
    if keys == {'fixed_index_seal'}:
        seal = experiment['fixed_index_seal']
        if (type(seal) is not dict or set(seal) != {'version', 'work_budget'}
                or seal['version'] != 'cooperative-fixed-index/v1'
                or type(seal['work_budget']) is not int or seal['work_budget'] <= 0):
            raise ValueError('source engine fixed-index experiment differs')
        source_engine._experimental_fixed_index_seal_budget = seal['work_budget']
    elif keys and keys <= {'group_demand_guidance', 'group_full_feedback'}:
        if 'group_demand_guidance' in keys:
            from clay.layout.route.group_demand_guidance import GUIDE_VERSION
            if experiment['group_demand_guidance'] != GUIDE_VERSION:
                raise ValueError('source engine demand guidance differs')
            source_engine.set_experimental_group_demand_guidance(True)
        if 'group_full_feedback' in keys:
            if experiment['group_full_feedback'] != 'same-invocation-group-prefix/v1':
                raise ValueError('source engine group feedback differs')
            source_engine.set_experimental_group_full_feedback(True)
    else:
        raise ValueError('source engine experiment differs')
    return require_source_engine(record, source_engine)
