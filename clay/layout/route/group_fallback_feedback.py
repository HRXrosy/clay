"""Check same-invocation group-prefix physical evidence before accepting a retry."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import os

import klayout.db as db
from clay.layout.route.canonical_input_capture import _json_value
from clay.layout.route.deck_isolation import check_deck_net_isolation, compare_probe_connectivity
from clay.layout.route.engine_gridless import _group_route_pin_key, _pin_xy_nm, _via_cut_space_nm
from clay.layout.route.metal_min_spacing_check import _metal_layers, check_metal_min_spacing
from clay.layout.route.via_redundancy_check import _via_layers
from clay.layout.route.wide_metal_spacing import wide_spacing_checks

def _edges(errors):
    def edge_key(edge): return tuple(sorted(((edge.x1,edge.y1),(edge.x2,edge.y2))))
    return frozenset(tuple(sorted((edge_key(pair.first),edge_key(pair.second))))
                     for pair in errors.each())

def _checks(path,tech,max_metal):
    if type(max_metal) is not int or max_metal < 1:raise ValueError('routing metal inventory is unavailable')
    spacing=check_metal_min_spacing(str(path),tech,max_metal=max_metal)
    if spacing.status=='unmeasurable': raise ValueError('metal spacing unavailable: '+spacing.reason)
    layout=db.Layout();layout.read(str(path));top=layout.top_cell();result={}
    for _number,name,key in _metal_layers(tech,max_metal):
        idx=layout.find_layer(*key);minimum=tech.min_space(name)
        if idx is None or not minimum: continue
        metal=db.Region(top.begin_shapes_rec(idx)).merged()
        if metal.is_empty(): continue
        checks={'space':metal.space_check(int(minimum)),
                'width':metal.width_check(int(tech.min_width(name)))}
        for check in wide_spacing_checks(metal,tech,name):
            checks[f'wide:{check.kind}:{check.spacing_nm}']=check.errors
        for rule,errors in checks.items():result[f'{name}:{rule}']=_edges(errors)
    for name,key in _via_layers(tech,max_metal-1):
        idx=layout.find_layer(*key)
        if idx is None:continue
        cuts=db.Region(top.begin_shapes_rec(idx)).merged()
        if cuts.is_empty():continue
        n=int(name[3:]);space=_via_cut_space_nm(tech,name,tech.min_space(f'metal{n}'))
        result[f'{name}:cut_space']=_edges(cuts.space_check(int(space)))
    return result


class SameInvocationFeedback:
    def __init__(self, context_factory, routing_world_sha256, evidence_dir, *, router_via_checker):
        self.context_factory=context_factory
        self.routing_world_sha256=routing_world_sha256
        self.evidence_dir=Path(evidence_dir)
        if not self.evidence_dir.is_dir(): raise ValueError('evidence directory is unavailable')
        if router_via_checker is None: raise ValueError('router-owned via checker is unavailable')
        self.router_via_checker=router_via_checker
        self.bound=None;self.events=[];self.accepted_ledgers=[]

    def _observe_rejected_physical_result(self, *, engine, local, checkpoint,
                                          before_path, after_path, event,
                                          emitted_ledger):
        """Optional in-process diagnostic hook; ordinary feedback does nothing.

        The paths are valid only during the hook and are removed as usual by
        this feedback's existing cleanup. This hook does not decide admission.
        """
        return None

    def _wire_obligations(self,engine,local):
        """Keep anchors pending unless an exact routed centerline ends there."""
        known=getattr(engine,'_required_wire_endpoints',None)
        if known is None: raise ValueError('engine wire-endpoint obligations unavailable')
        routes=local['baselines'];grouped=local['grouped_pin_keys']
        statuses=[];missing_grouped=[]
        for net,pins in self.bound.context.endpoint_ledger:
            engine._watchdog_check(net,'group_feedback_wire_endpoints')
            for pin in pins:
                if not pin.requires_wire_endpoint:continue
                point=tuple(_pin_xy_nm(pin))
                states={tuple(node) for owner,node in known if owner==net and tuple(node[:2])==point}
                if len(states)!=1:raise ValueError('wire-endpoint authority does not bind uniquely')
                node=next(iter(states)); route=routes.get(net)
                ends=set() if route is None else {
                    (x,y,int(seg.gi)) for seg in route.segs
                    for x,y in ((seg.ax,seg.ay),(seg.bx,seg.by))}
                complete=node in ends
                grouped_pin=_group_route_pin_key(net,pin) in grouped
                row={'net':net,'device':pin.device,'terminal':pin.terminal,
                     'node':node,'status':'satisfied' if complete else 'pending',
                     'grouped_access':grouped_pin}
                statuses.append(row)
                if grouped_pin and not complete:missing_grouped.append(row)
        return statuses,missing_grouped

    def _scene(self, engine, baselines):
        layout=self.bound.layout.dup();top=layout.cell(self.bound.top.name)
        ledger=[];via_manifest=[]
        for net in sorted(baselines):
            engine._watchdog_check(net,'group_feedback_emit')
            route=baselines[net]
            for via in route.vias:
                engine._require_policy_via(via)
                polygons=engine._via_polys(via)
                cuts=[tuple((int(p.x),int(p.y)) for p in shape.polygon.vertices) for shape in polygons[2:]]
                allowed_cuts = (
                    (1, engine.via_redundancy ** 2)
                    if bool(getattr(engine, "_single_via_escape_allowed", lambda: False)())
                    else (engine.via_redundancy ** 2,)
                )
                if len(cuts) not in allowed_cuts or len(set(cuts)) != len(cuts):
                    raise ValueError('router via array is incomplete')
                cut_layer=engine._tech.via_key_for(via.lo)
                if cut_layer is None:raise ValueError('router via lacks a PDK cut layer')
                via_manifest.append({'net':str(net),'site':_json_value(via),
                    'cut_layer':tuple(cut_layer),'cuts':cuts})
            for shape in engine._emit_net(route).wires:
                points=tuple((int(p.x),int(p.y)) for p in shape.polygon.vertices)
                top.shapes(layout.layer(shape.layer,shape.datatype)).insert(db.Polygon([db.Point(*p) for p in points]))
                ledger.append((str(net),int(shape.layer),int(shape.datatype),points))
        engine._watchdog_check('', 'group_feedback_emit')
        return layout,top,tuple(ledger),tuple(via_manifest)

    def __call__(self,engine,local,checkpoint):
        engine._watchdog_check('', 'group_feedback_begin')
        if self.bound is None:
            context=self.context_factory()
            self.bound=context.bind(engine,routing_world_sha256=self.routing_world_sha256)
        _,state=checkpoint
        before_endpoints,_=self._wire_obligations(engine,state['local'])
        endpoint_status,missing_grouped=self._wire_obligations(engine,local)
        def identity(row):return (row['net'],row['device'],row['terminal'],tuple(row['node']))
        before_satisfied={identity(row) for row in before_endpoints if row['status']=='satisfied'}
        now_satisfied={identity(row) for row in endpoint_status if row['status']=='satisfied'}
        lost_satisfied=before_satisfied-now_satisfied
        if missing_grouped or lost_satisfied:
            self.events.append({'accepted':False,'reason':'grouped_wire_endpoint_missing',
                                'missing_grouped_wire_endpoints':missing_grouped,
                                'lost_satisfied_wire_endpoints':sorted(lost_satisfied)})
            return False
        before,before_top,_,_=self._scene(engine,state['local']['baselines'])
        after,after_top,ledger,via_manifest=self._scene(engine,local['baselines'])
        paths=[]
        try:
            for scene in (before,after):
                fd,name=tempfile.mkstemp(prefix='group-fallback-',suffix='.gds',dir=self.evidence_dir)
                os.close(fd);path=Path(name);paths.append(path)
                options=db.SaveLayoutOptions();options.gds2_write_timestamps=False
                scene.write(str(path),options)
            engine._watchdog_check('', 'group_feedback_rules')
            tech=engine._tech.pdk
            max_metal=max(engine._tech._gi_to_pdk)
            old=_checks(paths[0],tech,max_metal);new=_checks(paths[1],tech,max_metal)
            additions={key:sorted(new.get(key,frozenset())-old.get(key,frozenset()))
                       for key in set(old)|set(new)}
            old_report=check_deck_net_isolation(layout=before,cell=before_top,stack=self.bound.stack,probes=self.bound.probes,include_probe_partition=True)
            new_report=check_deck_net_isolation(layout=after,cell=after_top,stack=self.bound.stack,probes=self.bound.probes,include_probe_partition=True)
            connectivity=compare_probe_connectivity(old_report,new_report)
            shorts=engine._detect_shorts(local['baselines'])
            added_unresolved=set(new_report.unresolved_probes)-set(old_report.unresolved_probes)
            added_ambiguous=set(new_report.ambiguous_nets)-set(old_report.ambiguous_nets)
            manifest=self.router_via_checker.actual_router_cut_manifest(engine,local['baselines'])
            via_report=self.router_via_checker.check_router_cuts(paths[1],tech,manifest)
            if via_report.artifact_sha256!=sha256(paths[1].read_bytes()).hexdigest():
                raise ValueError('router via checker observed another GDS')
            if via_report.status=='unmeasurable':
                raise ValueError('router via checking unavailable: '+via_report.reason)
            manifest_sha256=sha256(json.dumps(manifest,sort_keys=True,separators=(',',':'),
                                             allow_nan=False).encode()).hexdigest()
            if via_report.manifest_sha256!=manifest_sha256 or via_report.cuts!=manifest:
                raise ValueError('router via checker observed another cut manifest')
            engine._watchdog_check('', 'group_feedback_complete')
            accepted=not (any(additions.values()) or via_report.status!='pass' or shorts or new_report.merged_net_groups
                          or connectivity.lost_probe_groups
                          or added_unresolved)
            event={'accepted':accepted,'before_sha256':sha256(paths[0].read_bytes()).hexdigest(),
                   'after_sha256':sha256(paths[1].read_bytes()).hexdigest(),
                   'added_rule_witnesses':{key:value for key,value in additions.items() if value},
                   'router_via_status':via_report.status,'router_cuts_checked':via_report.checked,
                   'router_cut_manifest_sha256':manifest_sha256,
                   'router_via_violations':via_report.violations,'router_via_scope':via_report.scope,
                   'shorts':shorts,'lost_probe_groups':connectivity.lost_probe_groups,
                   'newly_resolved_isolated_groups':connectivity.newly_resolved_isolated_groups,
                   'before_probe_groups':old_report.resolved_probe_groups,
                   'after_probe_groups':new_report.resolved_probe_groups,
                   'merged_net_groups':new_report.merged_net_groups,
                   'added_unresolved':sorted(added_unresolved),'added_ambiguous':sorted(added_ambiguous)}
            self.events.append(event)
            if not accepted:
                # Diagnostic observers must never turn the original physical
                # rejection into acceptance or hide it with an ordinary
                # observation error. Cancellation and the source watchdog
                # still propagate through the original execution session.
                try:
                    self._observe_rejected_physical_result(
                        engine=engine, local=local, checkpoint=checkpoint,
                        before_path=paths[0], after_path=paths[1], event=event,
                        emitted_ledger=ledger)
                except Exception as error:
                    from clay.flow.execution_budget import ExecutionStopped
                    from clay.layout.route.engine_gridless import NegotiationWatchdogStop
                    if isinstance(error, (ExecutionStopped, NegotiationWatchdogStop)):
                        raise
                    self.observation_error = type(error).__name__
            if accepted:
                target=self.evidence_dir/('accepted-group-prefix-'+event['after_sha256']+'.gds')
                if target.exists():
                    if sha256(target.read_bytes()).hexdigest()!=event['after_sha256']:
                        raise ValueError('existing accepted evidence digest differs')
                else:
                    paths[1].replace(target)
                engine._watchdog_check('', 'group_feedback_accept')
                literal_ledger={**self.bound.literal_ledger(), 'emitted_route_literals':_json_value(ledger),
                    'router_via_manifest':_json_value(via_manifest),
                    'materialized_rails':[(str(net),int(gi),tuple(engine._tech.gi_to_metal_key(gi)),tuple(rect))
                        for net,rows in sorted(engine._rail_by_net.items()) for gi,rect in rows],
                    'actual_gds_sha256':event['after_sha256'], 'feedback':_json_value(event),
                    'actual_gds_name':target.name,
                    'wire_endpoint_status':_json_value(endpoint_status),
                    'pending_wire_endpoints':_json_value([row for row in endpoint_status if row['status']=='pending']),
                    'scope':'same-invocation partial group feedback; full routing/signoff pending'}
                encoded=(json.dumps(literal_ledger,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
                ledger_hash=sha256(encoded).hexdigest()
                ledger_path=self.evidence_dir/('accepted-group-ledger-'+ledger_hash+'.json')
                if ledger_path.exists():
                    if sha256(ledger_path.read_bytes()).hexdigest()!=ledger_hash:
                        raise ValueError('existing accepted ledger digest differs')
                else:
                    ledger_path.write_bytes(encoded)
                engine._watchdog_check('', 'group_feedback_ledger')
                self.accepted_ledgers.append({'ledger_sha256':ledger_hash,
                    'ledger_name':ledger_path.name,'actual_gds_sha256':event['after_sha256'],
                    'actual_gds_name':target.name})
            return accepted
        finally:
            for path in paths:path.unlink(missing_ok=True)
