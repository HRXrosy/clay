"""Bind group fallback to the same routing invocation and its physical evidence.

The fixed scene, physical checks and evidence directory are constructed lazily.
Accepted group-prefix evidence remains separate from final routing and signoff.
"""
from pathlib import Path
from uuid import uuid4
import json

from clay.layout.route.canonical_input_capture import _json_value, route_engine_configuration
from clay.layout.route.group_fallback_adapter import OverlayAdapter
from clay.layout.route.group_fallback_feedback import SameInvocationFeedback
from clay.layout.route.group_fallback_scene import lazy_same_invocation_context
from clay.layout.route.group_fallback_controller import LocalExhaustion, LocalInconclusive
from clay.layout.route.engine_gridless import GroupRouteLandingUnavailable
import clay.layout.route.group_fallback_checkpoint as checkpoints
import clay.layout.route.via_enclosure_check as router_via_checker


def install_group_fallback(engine, *, geometry, placement_um, pins_by_net, tech,
                           routing_world_sha256, evidence_root):
    """Bind this actual route invocation; return the exact method restorer."""
    original_config = route_engine_configuration(engine)
    had_method = '_reserve_group_route_landings_once' in vars(engine)
    prior_method = vars(engine).get('_reserve_group_route_landings_once')
    context_factory = lazy_same_invocation_context(
        geometry=geometry, placement_um=placement_um, pins_by_net=pins_by_net,
        tech=tech, route_engine=engine, routing_world_sha256=routing_world_sha256)
    physical = None
    evidence_dir = None

    def ensure_evidence_dir():
        nonlocal evidence_dir
        if evidence_dir is None:
            evidence_dir = Path(evidence_root) / uuid4().hex
            evidence_dir.mkdir(parents=True, exist_ok=False)
        return evidence_dir

    def journal(event):
        engine._watchdog_check('', 'group_fallback_journal')
        with (ensure_evidence_dir() / 'branch-events.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(_json_value(event), sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
            stream.flush()
        engine._watchdog_check('', 'group_fallback_journal')

    def feedback(active_engine, local, checkpoint):
        nonlocal physical
        if physical is None:
            physical = SameInvocationFeedback(
                context_factory, routing_world_sha256, ensure_evidence_dir(),
                router_via_checker=router_via_checker)
        try:
            result = physical(active_engine, local, checkpoint)
        except BaseException as exc:
            try:
                journal({'phase':'physical_feedback_error','error_type':type(exc).__name__})
            except BaseException as journal_error:
                try:
                    exc.add_note('group fallback journal failed: ' + type(journal_error).__name__)
                except BaseException:
                    # Some exception types (for example frozen dataclasses)
                    # cannot store notes. Preserve the original exception.
                    pass
            raise
        journal({'phase':'physical_feedback','accepted':result,
                 'feedback':physical.events[-1],
                 'accepted_ledgers':physical.accepted_ledgers})
        return result

    def factory(active_engine, book, local, regenerate, close_port):
        def observe(port, key, rejected):
            cause = getattr(rejected, 'original', rejected)
            diagnostic = getattr(cause, 'flow_error_data', None)
            journal({'phase':'branch_rejected','group':port.group,'net':port.net,
                     'candidate_key':key,'reason':type(rejected).__name__,
                     'original_error_type':type(cause).__name__,
                     'diagnostic':diagnostic() if callable(diagnostic) else None,
                     'edge_fail':getattr(rejected,'edge_fail',None),
                     'details':getattr(rejected,'details',None),
                     'blocker_samples':getattr(rejected,'samples',None)})
        class JournaledAdapter(OverlayAdapter):
            def record_causal_reservation(self, port, key, reservation):
                journal({'phase':'causal_reservation_trial', 'group':port.group,
                         'net':port.net, 'candidate_key':key, 'reservation':reservation})

            def record_negotiation_round(self, summary):
                journal({'phase':'negotiation_round', **summary})

            def last_physical_failure(self):
                if physical is None or not physical.events:
                    return None
                event = physical.events[-1]
                return _json_value(event) if not event.get('accepted') else None

            def candidates(self, port):
                try:
                    return super().candidates(port)
                except GroupRouteLandingUnavailable as error:
                    journal({'phase':'landing_unavailable', 'group':port.group,
                             'net':port.net, 'error_type':type(error).__name__,
                             'diagnostic':(error.flow_error_data()
                                 if type(error) is GroupRouteLandingUnavailable else None)})
                    raise
            def route(self, port, selected):
                try:
                    return super().route(port, selected)
                except LocalExhaustion as error:
                    journal({'phase':'local_exhaustion', 'group':port.group,
                             'net':port.net, 'edge_fail':error.edge_fail,
                             'details':error.details, 'blocker_samples':error.samples})
                    raise
                except LocalInconclusive as error:
                    journal({'phase':'local_inconclusive', 'group':port.group,
                             'net':port.net, 'edge_fail':error.edge_fail,
                             'details':error.details, 'blocker_samples':error.samples})
                    raise
        return JournaledAdapter(active_engine, book, local, regenerate, close_port,
                                checkpoint=checkpoints, physical_feedback=feedback,
                                observer=observe)

    def restore():
        if had_method:
            engine._reserve_group_route_landings_once = prior_method
        else:
            del engine._reserve_group_route_landings_once
        if route_engine_configuration(engine) != original_config:
            raise ValueError('selected routing policy changed during group fallback')

    try:
        def allocate_static(*args, **kwargs):
            return engine._reserve_group_route_landings_transactional(
                *args, **kwargs, adapter_factory=factory)
        engine._reserve_group_route_landings_once = allocate_static
        if route_engine_configuration(engine) != original_config:
            raise ValueError('selected routing policy changed during group fallback setup')
    except BaseException:
        if had_method:
            engine._reserve_group_route_landings_once = prior_method
        else:
            vars(engine).pop('_reserve_group_route_landings_once', None)
        raise
    return restore

