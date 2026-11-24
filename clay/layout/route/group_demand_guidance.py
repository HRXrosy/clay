"""Opt-in experimental 2-D guidance from source-materialized obligations.

Trees are Manhattan sketches of real endpoint/rail obligations.  They are not
legal routes, capacity, geometry evidence, or permission to bypass any native
candidate, closure, certificate or physical check.  A guide is scoped to one
group reservation and uses each freshly produced candidate batch and Book.
"""
from collections import Counter
from dataclasses import dataclass
import time

GUIDE_VERSION = 'source-demand-2d-mst/v3'


class UnsupportedGuide(ValueError):
    """The experimental proxy cannot represent this initialized obligation."""


def _rect(row):
    if len(row) != 4 or any(type(v) is not int for v in row):
        raise UnsupportedGuide('rectangle must contain four exact integers')
    x0, y0, x1, y1 = row
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _axis_nearest(a0, a1, b0, b1):
    if a1 < b0:
        return a1, b0
    if b1 < a0:
        return a0, b1
    common = max(a0, b0)
    return common, common


def _nearest(a, b):
    ax, bx = _axis_nearest(a[0], a[2], b[0], b[2])
    ay, by = _axis_nearest(a[1], a[3], b[1], b[3])
    return (ax, ay), (bx, by), abs(ax-bx)+abs(ay-by)


def _segments(a, b, horizontal_first):
    bend = (b[0], a[1]) if horizontal_first else (a[0], b[1])
    return tuple((p, q) for p, q in ((a, bend), (bend, b)) if p != q)


def _box(segment):
    (x0, y0), (x1, y1) = segment
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _touch(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


@dataclass(frozen=True)
class Demand:
    net: str
    # Each entry is (kind, occurrence ordinal, source identity, exact rectangle).
    obligations: tuple
    # Each segment is a rectilinear sketch of one MST edge, not real metal.
    tree_segments: tuple


class Guide:
    def __init__(self, net_nodes, net_node_pins, net_layers, rail_by_net, contract, engine,
                 topology_plan, pins_by_net):
        from clay.layout.route.engine_gridless import _group_route_pin_key
        from clay.layout.route.resource_plan import TopologyPlan
        from clay.layout.route.topology_compile import _terminal_from_pin
        self.engine, self.contract = engine, contract
        if type(topology_plan) is not TopologyPlan:
            raise UnsupportedGuide('same-invocation topology plan is absent')
        self.topology_sha256 = topology_plan.sha256()
        plan_demands = {row.net: row for row in topology_plan.demands}
        for net, pins in pins_by_net.items():
            self.engine._watchdog_check(net, 'experimental_demand_plan')
            source_terminals = Counter(_terminal_from_pin(pin) for pin in pins)
            # The canonical compiler deliberately omits unstructured leaf nets.
            # They remain explicit endpoint occurrences in our own ledger.
            if net not in plan_demands and len(pins) > 1:
                raise UnsupportedGuide('topology plan omits a multi-terminal net')
            if net in plan_demands and source_terminals != Counter(plan_demands[net].terminals):
                raise UnsupportedGuide('topology terminals differ from original pins')
            if Counter(_terminal_from_pin(pin) for pin in net_node_pins.get(net, ())) != source_terminals:
                raise UnsupportedGuide('initialized endpoints differ from topology terminals')
        for net, demand in plan_demands.items():
            if any(goal.net != net or goal.target is None for goal in demand.required_goals):
                raise UnsupportedGuide('unresolved or foreign topology goal')
            if net not in net_nodes:
                raise UnsupportedGuide('topology demand has no initialized net')
        if set(net_nodes) != set(net_node_pins) or set(net_nodes) != set(net_layers):
            raise UnsupportedGuide('incomplete all-net endpoint inventory')
        if set(rail_by_net) - set(net_nodes):
            raise UnsupportedGuide('rail not present in the complete net inventory')
        self.demands = {}
        self.ledger = Counter()
        self.ledger['plan_nets'] = len(plan_demands)
        self.ledger['plan_candidate_options'] = len(topology_plan.candidates)
        self.ledger['leaf_nets_omitted_by_plan'] = len(set(pins_by_net) - set(plan_demands))
        self.group_keys = {}
        for port in contract.ports:
            if port.net not in net_nodes:
                raise UnsupportedGuide('group references absent net')
            keys = tuple(_group_route_pin_key(port.net, access) for access in port.accesses)
            self.group_keys[(port.group, port.net)] = keys
        for net in sorted(net_nodes):
            self.engine._watchdog_check(net, 'experimental_demand_build')
            nodes, pins = tuple(net_nodes[net]), tuple(net_node_pins[net])
            layers = tuple(net_layers[net])
            if len(nodes) != len(pins) or not layers:
                raise UnsupportedGuide('incomplete endpoint or layer inventory')
            obligations = []
            actual_keys = []
            for i, (node, pin) in enumerate(zip(nodes, pins)):
                self._step(net, 'experimental_demand_build')
                if (type(node) is not tuple or len(node) != 3 or
                        any(type(v) is not int for v in node)):
                    raise UnsupportedGuide('unsupported endpoint node')
                key = _group_route_pin_key(net, pin)
                actual_keys.append(key)
                obligations.append(('endpoint', i, key, (node[0],node[1],node[0],node[1])))
                self.ledger['endpoint_occurrences'] += 1
            for i, (gi, rect) in enumerate(rail_by_net.get(net, ())):
                self._step(net, 'experimental_demand_build')
                if type(gi) is not int:
                    raise UnsupportedGuide('unsupported rail layer')
                obligations.append(('rail', i, gi, _rect(tuple(rect))))
                self.ledger['rail_rectangles'] += 1
            for i, goal in enumerate(plan_demands[net].required_goals if net in plan_demands else ()):
                self._step(net, 'experimental_demand_build')
                target = goal.target
                x, y = target.xy_nm
                obligations.append(('goal', i, goal.declaration_ref, (x, y, x, y)))
                self.ledger['required_goals'] += 1
            for (group, owner), keys in self.group_keys.items():
                if owner == net and any(actual_keys.count(key) != 1 for key in keys):
                    raise UnsupportedGuide('group access does not map uniquely to an endpoint')
            # A duplicate endpoint remains a separate AND occurrence. Prim's
            # complete-graph MST uses actual point/whole-rectangle obligations.
            points = tuple(row[3] for row in obligations)
            segments = []
            if points:
                reached = {0}
                best = {}
                def relax(i):
                    for j in range(len(points)):
                        if j not in reached:
                            self._step(net, 'experimental_demand_tree')
                            a, b, distance = _nearest(points[i], points[j])
                            edge = (distance, i, j, a, b)
                            if j not in best or edge < best[j]:
                                best[j] = edge
                            self.ledger['tree_pair_checks'] += 1
                relax(0)
                while len(reached) < len(points):
                    self.engine._watchdog_check(net, 'experimental_demand_tree')
                    _, _, j, a, b = min(best.values())
                    segments.extend(_segments(a, b, True))
                    reached.add(j)
                    del best[j]
                    relax(j)
            self.demands[net] = Demand(net, tuple(obligations), tuple(segments))
        self.ledger['nets'] = len(self.demands)
        self.ledger['tree_segments'] = sum(len(row.tree_segments) for row in self.demands.values())
        self.selections = []
        self.unknown = []

    def report_demands(self):
        """Preserve every input occurrence and whole rail for traceability."""
        return {net: [{'kind':kind, 'ordinal':ordinal, 'identity':identity,
                       'rectangle_nm':rect} for kind, ordinal, identity, rect in demand.obligations]
                for net, demand in self.demands.items()}

    def _step(self, net, phase):
        self.ledger['operation_units'] += 1
        if self.ledger['operation_units'] % 128 == 0:
            self.engine._watchdog_check(net, phase)

    def choose(self, port, rows, book, landings=()):
        """Score original live rows; return an original row, without changes.

        Unsupported scope keeps native min. Scores are explicitly a 2-D proxy;
        spatial crossings are not physical conflicts or capacity.
        """
        from clay.layout.route.engine_gridless import _group_route_pin_key
        started = time.perf_counter()
        original = min(rows)
        if port not in self.contract.ports or port.net not in self.demands:
            self.unknown.append({'group': port.group, 'net': port.net, 'reason': 'port absent from demand'})
            return original
        grouped = {key for (group, net), keys in self.group_keys.items()
                   if net == port.net for key in keys}
        own = self.demands[port.net]
        targets = []
        for row in own.obligations:
            self._step(port.net, 'experimental_demand_rank')
            if row[0] != 'endpoint' or row[2] not in grouped:
                targets.append((row[0], row[2], row[3]))
        # Group-local pins are replaced by a single portal in the global route.
        # Use the actual closed portal when available; otherwise the current
        # contract envelope is the best available interface region.  Landings
        # are supplied by the live caller, including after transactional restore.
        closed = {(landing.group, landing.net): landing for landing in landings}
        for other in self.contract.ports:
            if other.net != port.net or other.group == port.group:
                continue
            landing = closed.get((other.group, other.net))
            if landing is None:
                targets.append(('group_envelope', other.group, other.envelope_nm))
            else:
                x, y = landing.portal_xy_nm
                targets.append(('group_portal', other.group, (x, y, x, y)))
        if not targets:
            self.unknown.append({'group': port.group, 'net': port.net, 'reason': 'no external demand'})
            return original
        declared_trunks = getattr(self.engine, '_declared_trunk_gi_by_net', None) or {}
        own_trunk = declared_trunks.get(port.net)
        promoted = getattr(self.engine, '_narrow_corridor_portal_gi', None) or {}
        if (port.group, port.net) in promoted:
            own_trunk = promoted[(port.group, port.net)]
        pref = getattr(self.engine, '_prototype_portal_preference', None)
        if pref is not None and hasattr(getattr(self.engine, '_tech', None), '_gi_to_pdk'):
            gi_to_pdk = self.engine._tech._gi_to_pdk
            if isinstance(pref, dict) and (port.group, port.net) in pref:
                target_lvl = pref[(port.group, port.net)]
                own_trunk = next((gi for gi, lvl in enumerate(gi_to_pdk) if lvl == target_lvl), own_trunk)
            elif isinstance(pref, tuple) and len(pref) >= 3 and (port.group, port.net) == pref[:2]:
                target_lvl = pref[2]
                own_trunk = next((gi for gi, lvl in enumerate(gi_to_pdk) if lvl == target_lvl), own_trunk)
        foreign = []
        for net, demand in self.demands.items():
            if net != port.net:
                f_trunk = declared_trunks.get(net)
                for segment in demand.tree_segments:
                    self._step(port.net, 'experimental_demand_rank')
                    foreign.append((net, _box(segment), f_trunk))
        book_layers = {}
        scored = []
        for ordinal, row in enumerate(rows):
            if ordinal % 64 == 0:
                self.engine._watchdog_check(port.net, 'experimental_demand_rank')
            portal = row[10]
            if type(portal) is not tuple or len(portal) != 3 or any(type(v) is not int for v in portal):
                raise UnsupportedGuide('candidate portal is unsupported')
            point = (portal[0], portal[1], portal[0], portal[1])
            nearest = None
            for _kind, _identity, target in targets:
                self._step(port.net, 'experimental_demand_rank')
                candidate = _nearest(point, target)
                if nearest is None or (candidate[2],candidate[1]) < (nearest[2],nearest[1]):
                    nearest = candidate
            a, b, distance = nearest
            if portal[2] not in book_layers:
                book_layers[portal[2]] = []
                for collection in (book.metal, book.via_pads):
                    for owner, rect in collection.get(portal[2], ()):
                        self._step(port.net, 'experimental_demand_rank')
                        if owner != port.net:
                            book_layers[portal[2]].append((owner, _rect(tuple(rect))))
            actual = book_layers[portal[2]]
            same_layer_foreign = tuple(
                (net, rect)
                for net, rect, f_trunk in foreign
                if f_trunk is None or int(f_trunk) == int(portal[2])
            )
            trunk_delta = (
                abs(int(portal[2]) - int(own_trunk))
                if type(own_trunk) is int
                else 0
            )
            variants = []
            for first in (True, False):
                boxes = tuple(_box(segment) for segment in _segments(a, b, first))
                touched = set()
                for owner, rect in (*same_layer_foreign, *actual):
                    for box in boxes:
                        self._step(port.net, 'experimental_demand_rank')
                        self.ledger['proxy_intersection_checks'] += 1
                        if _touch(box, rect):
                            touched.add(owner)
                variants.append(
                    (len(touched), trunk_delta, distance, int(not first))
                    if type(own_trunk) is int
                    else (len(touched), distance, int(not first))
                )
            proxy = min(variants)
            scored.append((proxy, row))
            self.ledger['candidate_rows_scored'] += 1
        proxy, selected = min(scored)
        if not any(selected is row for row in rows):
            raise ValueError('choice is not a current producer row')
        self.selections.append({'group':port.group, 'net':port.net, 'candidate_count':len(rows),
            'proxy_score':proxy, 'changed_from_native': selected != original,
            'selected_portal':selected[10], 'native_portal':original[10],
            'external_targets':targets,
            'elapsed_seconds':time.perf_counter()-started,
            'scope':'experimental 2-D endpoint demand sketch; not routing or capacity'})
        return selected


def build_guide(net_nodes, net_node_pins, net_layers, rail_by_net, contract, engine,
                topology_plan, pins_by_net):
    return Guide(net_nodes, net_node_pins, net_layers, rail_by_net, contract, engine,
                 topology_plan, pins_by_net)
