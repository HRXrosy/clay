"""Conservative attribution of an actual rejected group-prefix spacing marker.

This diagnostic consumes the existing same-invocation physical feedback.  It
does not select candidates, change the Book, or authorize a route.  Ambiguous
geometry and identities remain UNKNOWN.
"""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import klayout.db as db

from clay.layout.route.canonical_input_capture import _json_value
from clay.layout.route.group_fallback_checkpoint import _canonical, digest as checkpoint_digest
from clay.layout.route.group_fallback_feedback import SameInvocationFeedback
from clay.layout.route.group_ports import (GroupRouteLandingCertificate, GroupRouteRoutingContract,
                                            verify_group_route_landing_certificate)


def _hash(value):
    return sha256(json.dumps(_canonical(value, set()), ensure_ascii=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _region(layout, top, layer):
    idx = layout.find_layer(*layer)
    return db.Region() if idx is None else db.Region(top.begin_shapes_rec(idx)).merged()


def _rect(points):
    if len(points) < 4:
        return None
    poly = db.Polygon([db.Point(int(x), int(y)) for x, y in points])
    box = poly.bbox()
    if poly.area() != box.area() or box.width() <= 0 or box.height() <= 0:
        return None
    return (box.left, box.bottom, box.right, box.top)


def _edge_pair(pair):
    if len(pair) != 2 or any(len(edge) != 2 for edge in pair):
        return None
    (a, b), (c, d) = pair
    if any(len(pt) != 2 or any(type(v) is not int for v in pt) for pt in (a, b, c, d)):
        return None
    if a[1] == b[1] and c[1] == d[1] and a[1] != c[1]:
        lo, hi = max(min(a[0], b[0]), min(c[0], d[0])), min(max(a[0], b[0]), max(c[0], d[0]))
        return ('h', lo, hi, a[1], c[1]) if lo < hi else None
    if a[0] == b[0] and c[0] == d[0] and a[0] != c[0]:
        lo, hi = max(min(a[1], b[1]), min(c[1], d[1])), min(max(a[1], b[1]), max(c[1], d[1]))
        return ('v', lo, hi, a[0], c[0]) if lo < hi else None
    return None


def _touches(rect, axis, lo, hi, line):
    x0, y0, x1, y1 = rect
    if axis == 'h':
        return (line in (y0, y1) and max(x0, lo) < min(x1, hi), x0 <= lo and x1 >= hi)
    return (line in (x0, x1) and max(y0, lo) < min(y1, hi), y0 <= lo and y1 >= hi)


class AssemblyConflictObserver(SameInvocationFeedback):
    """Opt-in live observer of one feedback rejection, with no admission role.

    The route caller supplies its actual contract and certificate before calling
    feedback. The accepted/rejected result remains that of the original checker.
    """

    def __init__(self, *args, contract, certificate, **kwargs):
        super().__init__(*args, **kwargs)
        if type(contract) is not GroupRouteRoutingContract or type(certificate) is not GroupRouteLandingCertificate:
            raise TypeError('exact same-invocation group contract and certificate required')
        if certificate.routing_contract_sha256 != contract.sha256:
            raise ValueError('certificate and contract differ')
        if {(x.group, x.net) for x in certificate.landings} != {(x.group, x.net) for x in contract.ports}:
            raise ValueError('certificate does not cover all source ports')
        self.contract, self.certificate = contract, certificate
        self._contract_object, self._certificate_object = contract, certificate
        self._contract_sha256, self._certificate_sha256 = contract.sha256, certificate.sha256
        self._live = None
        self._physical = None
        self.observation = {'status': 'UNKNOWN', 'reason': 'feedback_not_observed'}

    def __call__(self, engine, local, checkpoint):
        if self._live is not None:
            raise ValueError('assembly observation cannot be nested')
        self._live = (engine, local, checkpoint, engine._book)
        try:
            return super().__call__(engine, local, checkpoint)
        finally:
            self._live = None

    def _observe_rejected_physical_result(self, *, engine, local, checkpoint,
                                          before_path, after_path, event, emitted_ledger):
        try:
            self._physical = (before_path, after_path, event, emitted_ledger,
                _hash(event), _hash(emitted_ledger),
                _hash({name: getattr(engine._book, name) for name in ('metal','via_pads','cuts','persistent')}),
                _hash(local['group_contributions']), checkpoint_digest(checkpoint))
            self.observation = self._capture(engine, local, checkpoint, before_path,
                                             after_path, event, emitted_ledger)
        except Exception as error:
            from clay.flow.execution_budget import ExecutionStopped
            from clay.layout.route.engine_gridless import NegotiationWatchdogStop
            if isinstance(error, (ExecutionStopped, NegotiationWatchdogStop)):
                raise
            self.observation = {'status': 'UNKNOWN', 'reason': type(error).__name__, 'detail': str(error)}
        finally:
            self._physical = None

    def _capture(self, engine, local, checkpoint, before_path, after_path, event, emitted_ledger):
        engine._watchdog_check('', 'group_assembly_observe')
        if (self._live is None or any(a is not b for a, b in zip(
                self._live, (engine, local, checkpoint, engine._book))) or
                self.contract is not self._contract_object or self.certificate is not self._certificate_object or
                self.contract.sha256 != self._contract_sha256 or
                self.certificate.sha256 != self._certificate_sha256 or
                not self.events or self.events[-1] is not event or
                self._physical is None or any(a is not b for a, b in zip(
                    self._physical[:4], (before_path, after_path, event, emitted_ledger))) or
                self._physical[4:] != (_hash(event), _hash(emitted_ledger),
                    _hash({name: getattr(engine._book, name) for name in ('metal','via_pads','cuts','persistent')}),
                    _hash(local['group_contributions']), checkpoint_digest(checkpoint)) or
                event.get('accepted') is not False or engine._book is None):
            raise ValueError('actual rejected feedback and Book required')
        verify_group_route_landing_certificate(
            self.contract,
            self.certificate,
            expected_via_redundancy=engine.via_redundancy,
            allow_single_escape=bool(getattr(engine, "_single_via_escape_allowed", lambda: False)()),
        )
        if (engine._group_route_landing_baselines is not local['baselines'] or
                engine._group_route_required_states is not local['required_states'] or
                Counter(map(id, local['landings'])) != Counter(map(id, self.certificate.landings))):
            raise ValueError('certificate does not bind actual live landing state')
        before_bytes, after_bytes = Path(before_path).read_bytes(), Path(after_path).read_bytes()
        if sha256(before_bytes).hexdigest() != event.get('before_sha256') or sha256(after_bytes).hexdigest() != event.get('after_sha256'):
            raise ValueError('feedback GDS bytes differ')
        before, after = db.Layout(), db.Layout()
        before.read(str(before_path)); after.read(str(after_path))
        if before.dbu != 0.001 or after.dbu != 0.001 or before.top_cell() is None or after.top_cell() is None:
            raise ValueError('feedback layout units or top differ')
        before_top, after_top = before.top_cell(), after.top_cell()
        book = engine._book
        counts = Counter((int(gi), str(net), tuple(rect)) for gi, rows in book.metal.items() for net, rect in rows)
        contributions = local['group_contributions']
        group_rows = {}
        for key, value in contributions.items():
            if type(key) is not tuple or len(key) != 2 or key not in {(p.group, p.net) for p in self.contract.ports}:
                raise ValueError('unbound group contribution')
            for gi, rect in value['incremental']['metal']:
                row = (int(gi), str(key[1]), tuple(rect))
                group_rows.setdefault(row, set()).add(key)
        emitted = Counter()
        layers = set()
        for net, layer, datatype, points in emitted_ledger:
            key = (int(layer), int(datatype)); layers.add(key)
            rect = _rect(points)
            if rect is not None:
                emitted[(str(net), key, rect)] += 1
        layers.update((info.layer, info.datatype) for info in before.layer_infos())
        layers.update((info.layer, info.datatype) for info in after.layer_infos())
        # Compare actual emitted geometry with the same feedback's actual GDS.
        for layer in layers:
            engine._watchdog_check('', 'group_assembly_layers')
            expected = _region(before, before_top, layer)
            for index, (_net, ll, dd, points) in enumerate(emitted_ledger):
                if index % 128 == 0:
                    engine._watchdog_check('', 'group_assembly_ledger')
                if (ll, dd) == layer:
                    expected.insert(db.Polygon([db.Point(int(x), int(y)) for x, y in points]))
            if not (expected.merged() ^ _region(after, after_top, layer)).is_empty():
                raise ValueError('actual emitted ledger and GDS differ')
        witnesses = []
        for rule, pairs in event.get('added_rule_witnesses', {}).items():
            if not (rule.startswith('metal') and rule.endswith(':space') and rule[5:-6].isdigit()):
                continue
            pdk_n = int(rule[5:-6]); layer = tuple(engine._tech.pdk.metal_key(pdk_n))
            gis = [gi for gi in range(engine._n_layers) if tuple(engine._tech.gi_to_metal_key(gi)) == layer]
            if len(gis) != 1:
                raise ValueError('marker layer has no unique graph layer')
            gi = gis[0]
            for pair in pairs:
                engine._watchdog_check('', 'group_assembly_marker')
                geometry = _edge_pair(pair)
                if geometry is None:
                    continue
                axis, lo, hi, first, second = geometry
                owners = []
                for edge in (first, second):
                    band = db.Box(lo, edge-1, hi, edge+1) if axis == 'h' else db.Box(edge-1, lo, edge+1, hi)
                    if not (_region(before, before_top, layer) & db.Region(band)).is_empty():
                        owners = []; break
                    found, covering, evidence = set(), False, []
                    for index, ((row_gi, net, rect), groups) in enumerate(group_rows.items()):
                        if index % 128 == 0:
                            engine._watchdog_check('', 'group_assembly_owners')
                        hit, full = _touches(rect, axis, lo, hi, edge)
                        if row_gi != gi or not hit:
                            continue
                        if len(groups) != 1 or counts[(row_gi, net, rect)] != 1 or emitted[(net, layer, rect)] != 1:
                            raise ValueError('group metal cannot be uniquely bound to emitted wire')
                        from clay.layout.route.engine_gridless import _seg_bbox
                        segments = [seg for seg in local['baselines'][net].segs if seg.gi == gi and
                            tuple(_seg_bbox(seg.ax, seg.ay, seg.bx, seg.by,
                                engine._segment_half_width(net, seg))) == rect]
                        if len(segments) != 1:
                            raise ValueError('emitted wire has no unique native centerline')
                        found.update(groups); covering |= full
                        evidence.append({'group': next(iter(groups))[0], 'net': net, 'rect_nm': rect,
                                         'segment': _json_value(segments[0])})
                    if len(found) != 1 or not covering:
                        owners = []; break
                    # No unbound emitted polygon may participate in this edge.
                    for index, (net, ll, dd, points) in enumerate(emitted_ledger):
                        if index % 128 == 0:
                            engine._watchdog_check('', 'group_assembly_ambiguity')
                        if (ll, dd) != layer:
                            continue
                        polygon = db.Region(db.Polygon([db.Point(int(x), int(y)) for x, y in points]))
                        if (polygon & db.Region(band)).is_empty():
                            continue
                        rect = _rect(points)
                        if rect is None or (gi, net, rect) not in group_rows or group_rows[(gi, net, rect)] != found:
                            raise ValueError('marker edge includes unbound emitted geometry')
                    owners.append((next(iter(found)), evidence))
                if len(owners) == 2 and owners[0][0][1] != owners[1][0][1]:
                    witnesses.append({'rule': rule, 'layer': layer, 'marker_edges': pair,
                                      'groups_and_nets': [row[0] for row in owners],
                                      'emitted_wires': [row[1] for row in owners],
                                      'overlap_nm': (lo, hi)})
        if len(witnesses) != 1:
            raise ValueError('no unique attributable spacing marker')
        # Detached, byte-bound witness; retain the exact GDS which the feedback
        # is about to remove. These artifacts never replace its rejection.
        for name, contents in (('before.gds', before_bytes), ('after.gds', after_bytes)):
            path = self.evidence_dir / name
            with path.open('xb') as stream:
                stream.write(contents)
        book_value = {name: getattr(book, name) for name in ('metal','via_pads','cuts','persistent')}
        result = {'status': 'OBSERVED_GROUP_SPACING_CONFLICT', 'physical_admission': False,
                  'coordination_admission': False, 'source_identity': 'UNKNOWN_NOT_BOUND',
                  'world_origin': 'CALLER_SUPPLIED_NOT_INDEPENDENTLY_BOUND',
                  'contract_origin': 'CALLER_SUPPLIED_NOT_INDEPENDENTLY_BOUND',
                  'routing_world_sha256': self.routing_world_sha256,
                  'contract_sha256': self.contract.sha256, 'certificate_sha256': self.certificate.sha256,
                  'checkpoint_sha256': checkpoint_digest(checkpoint),
                  'book_sha256': _hash(book_value),
                  'group_contributions_sha256': _hash(contributions),
                  'emitted_ledger_sha256': _hash(emitted_ledger), 'before_sha256': event['before_sha256'],
                  'after_sha256': event['after_sha256'], 'feedback': _json_value(event),
                  'contract': self.contract.canonical_data(), 'certificate': self.certificate.canonical_data(),
                  'placed_scene': self.bound.literal_ledger(),
                  'checkpoint_canonical': _canonical(checkpoint[1], set()),
                  'book_canonical': _canonical(book_value, set()),
                  'group_contributions_canonical': _canonical(contributions, set()),
                  'emitted_ledger_canonical': _canonical(emitted_ledger, set()),
                  'witness': _json_value(witnesses[0])}
        path = self.evidence_dir / 'assembly-conflict.json'
        with path.open('x', encoding='utf8') as stream:
            json.dump(result, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        return result
