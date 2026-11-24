"""Bind group-prefix feedback to placed artwork, routing policy and endpoint authority."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import klayout.db as db
from clay.layout.owned_artwork import _stamp_records
from clay.layout.route.canonical_input_capture import _json_value, route_engine_configuration
from clay.layout.route.deck_isolation import DeckConnectStack, NetProbe, check_deck_net_isolation
from clay.layout.route.engine_gridless import _pin_xy_nm
from clay.layout.route.resource_world import canonical_pdk_routing_json


def _hash(value):
    data=json.dumps(_json_value(value),sort_keys=True,separators=(',',':'),allow_nan=False)
    return sha256(data.encode()).hexdigest()


class FeedbackUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class PlacedSceneContext:
    stamps: tuple
    endpoint_ledger: tuple
    routing_world_sha256: str
    pdk_routing_json: str
    engine_configuration_json: str
    source_artwork_sha256: str

    @classmethod
    def capture(cls, *, geometry, placement_um, pins_by_net, tech,
                route_engine, routing_world_sha256):
        if set(geometry) - set(placement_um):
            raise FeedbackUnavailable('placed source has artwork without placement')
        stamps=tuple((name, _stamp_records(stamp, placement_um[name]))
                     for name,stamp in sorted(geometry.items()))
        return cls._from_ledger(stamps,pins_by_net,tech,route_engine,routing_world_sha256)

    @classmethod
    def replay_captured_records(cls, *, captured_stamps, pins_by_net, tech,
                                route_engine, routing_world_sha256):
        """Diagnostic replay of the existing prepared-input polygon ledger."""
        stamps=[]
        for name,stamp in sorted(captured_stamps.items()):
            records=[]
            for layer,datatype,points in stamp['polygons_nm']:
                if (type(layer) is not int or type(datatype) is not int
                    or any(type(x) is not int or type(y) is not int for x,y in points)):
                    raise FeedbackUnavailable('captured polygon ledger is not exact')
                records.append((layer,datatype,tuple((x,y) for x,y in points)))
            stamps.append((name,tuple(records)))
        return cls._from_ledger(tuple(stamps),pins_by_net,tech,route_engine,routing_world_sha256)

    @classmethod
    def _from_ledger(cls, stamps, pins_by_net, tech, route_engine, routing_world_sha256):
        if (type(routing_world_sha256) is not str or len(routing_world_sha256) != 64
            or any(c not in '0123456789abcdef' for c in routing_world_sha256)):
            raise FeedbackUnavailable('routing-world identity unavailable')
        # Preserve every actual endpoint, including absent exact-nm coordinates
        # and wire-endpoint obligations.  Do not infer that pending anchors are
        # connected simply because a partial scene has been rendered.
        ledger=tuple((str(net), tuple(pins_by_net[net])) for net in sorted(pins_by_net))
        if not stamps or not any(rows for _,rows in ledger):
            raise FeedbackUnavailable('placed artwork or endpoint ledger is empty')
        config=json.dumps(route_engine_configuration(route_engine),sort_keys=True,separators=(',',':'))
        return cls(stamps,ledger,routing_world_sha256,canonical_pdk_routing_json(tech),config,_hash(stamps))

    def _placed_layout(self, engine):
        layout=db.Layout(); layout.dbu=0.001; top=layout.create_cell('GROUP_FEEDBACK')
        for _name,records in self.stamps:
            for layer,datatype,points in records:
                top.shapes(layout.layer(layer,datatype)).insert(
                    db.Polygon([db.Point(int(x),int(y)) for x,y in points]))
        # The engine has already consumed actual caller rails through its normal
        # materialization path.  Union semantics tolerate rails already stamped.
        for _net,rows in engine._rail_by_net.items():
            for gi,rect in rows:
                top.shapes(layout.layer(*engine._tech.gi_to_metal_key(gi))).insert(db.Box(*rect))
        return layout,top

    def bind(self, engine, *, routing_world_sha256):
        if routing_world_sha256 != self.routing_world_sha256:
            raise FeedbackUnavailable('routing-world identity changed')
        config=json.dumps(route_engine_configuration(engine),sort_keys=True,separators=(',',':'))
        if config!=self.engine_configuration_json:
            raise FeedbackUnavailable('selected engine configuration changed')
        if canonical_pdk_routing_json(engine._tech.pdk)!=self.pdk_routing_json:
            raise FeedbackUnavailable('PDK routing authority changed')
        if _hash(self.stamps)!=self.source_artwork_sha256:
            raise FeedbackUnavailable('placed artwork changed')
        stack=DeckConnectStack.from_tech(engine._tech.pdk)
        layout,top=self._placed_layout(engine)
        layers=set(stack.conductors)|{connection.cut for connection in stack.connections}|{carve.top_marker for carve in stack.mim_carves}
        projection={}
        for key in sorted(layers):
            index=layout.find_layer(*key)
            source=db.Region() if index is None else db.Region(top.begin_shapes_rec(index)).merged()
            fixed=db.Region()
            for layer,_owner,rect in engine._fixed_deck_literal_rects:
                if tuple(layer)==key: fixed.insert(db.Box(*rect))
            fixed.merge()
            missing,extra=source-fixed,fixed-source
            projection[key]={'missing_area_nm2':missing.area(),'extra_area_nm2':extra.area()}
            if not missing.is_empty() or not extra.is_empty():
                raise FeedbackUnavailable(f'fixed literal projection differs for {key}')
        probes=[]; pending=[]
        for net,pins in self.endpoint_ledger:
            for pin in pins:
                if pin.net != net:
                    raise FeedbackUnavailable('endpoint net changed')
                point=_pin_xy_nm(pin)
                gi=engine._tech.resolve_layer_to_gi(pin.layer)
                layer=tuple(pin.layer) if type(pin.layer) is tuple else (
                    None if gi is None else tuple(engine._tech.gi_to_metal_key(gi)))
                if layer not in stack.conductors:
                    raise FeedbackUnavailable('endpoint layer outside connectivity stack')
                probes.append(NetProbe(net,layer,*point))
                if pin.requires_wire_endpoint:
                    pending.append((net,pin.device,pin.terminal,point,layer))
        return BoundPlacedScene(self,layout,top,stack,tuple(probes),tuple(pending),projection)


@dataclass
class BoundPlacedScene:
    context: PlacedSceneContext
    layout: object
    top: object
    stack: object
    probes: tuple
    pending_wire_endpoints: tuple
    projection: dict

    def observe(self):
        return check_deck_net_isolation(layout=self.layout,cell=self.top,
                                        stack=self.stack,probes=self.probes)

    def literal_ledger(self):
        return {'schema':'clay.group-fallback-placed-scene.v1',
                'source_artwork_sha256':self.context.source_artwork_sha256,
                'routing_world_sha256':self.context.routing_world_sha256,
                'pdk_routing_sha256':sha256(self.context.pdk_routing_json.encode()).hexdigest(),
                'selected_engine':json.loads(self.context.engine_configuration_json),
                'source_stamps':_json_value(self.context.stamps),
                'endpoint_ledger':_json_value(self.context.endpoint_ledger),
                'pending_wire_endpoints':_json_value(self.pending_wire_endpoints),
                'projection':{str(key):value for key,value in sorted(self.projection.items())}}


def lazy_same_invocation_context(**kwargs):
    """Keep the default path free of placed-scene construction until invoked."""
    def build():
        return PlacedSceneContext.capture(**kwargs)
    return build
