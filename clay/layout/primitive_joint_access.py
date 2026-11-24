"""Source-bound AND/OR terminals on a finite legal routing graph.

This research adapter eliminates access choices by contracting *proven existing*
joint conductivity. It never enumerates products of terminal alternatives.
Graph edges are supplied by a geometry backend; this module grants neither DRC
nor LVS authority. The current production Pin-only route exit remains guarded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import heapq
import json

import klayout.db as kdb


def _sha(value):
    from clay.layout.route.canonical_input_capture import _json_value
    return hashlib.sha256(json.dumps(_json_value(value), sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class JointAccessDemand:
    net: str
    stamp_id: str
    component_id: str
    source_refs: tuple[tuple[str, str, str], ...]
    # Every routing component and all its actual absolute polygons, with holes.
    alternatives: tuple

    @property
    def identity(self):
        return self.stamp_id, self.component_id


@dataclass(frozen=True)
class JointAccessIR:
    demands: tuple[JointAccessDemand, ...]
    face_sha256: str
    scope: str = 'replayed_current_joint_face_finite_graph_only/v1'

    @property
    def sha256(self):
        return _sha(asdict(self))


def _from_replayed_face(face):
    from clay.layout.physical_terminal_inventory import PhysicalTerminalPolygonNM
    demands = []
    for row in face.joint_faces:
        inventory = row.joint_inventory
        inventory.__post_init__()
        sites = {site.site_id: site for site in inventory.expected_sites}
        routing = {component.component_id: component for component in inventory.routing_components}
        dx, dy = row.delta_nm
        for joint in inventory.joint_components:
            refs = tuple(sorted((site_id + '/' + role, sites[site_id].owner, role)
                for site_id in joint.site_ids for role in sites[site_id].allowed_roles))
            alternatives = []
            for identity in joint.routing_component_ids:
                access = routing[identity]
                assert access.joint_component_id == joint.component_id
                geometry = tuple((layer, tuple(PhysicalTerminalPolygonNM(
                    tuple((x + dx, y + dy) for x, y in polygon.hull_nm),
                    tuple(tuple((x + dx, y + dy) for x, y in hole)
                          for hole in polygon.holes_nm)) for polygon in polygons))
                    for layer, polygons in access.geometry_by_layer_nm)
                alternatives.append((identity, geometry))
            demands.append(JointAccessDemand(joint.source_net, row.stamp_id,
                joint.component_id, refs, tuple(alternatives)))
    demands.sort(key=lambda item: (item.net, item.identity))
    if len({item.identity for item in demands}) != len(demands):
        raise ValueError('joint access demand identity is duplicated')
    return JointAccessIR(tuple(demands), _sha(asdict(face)))


def compile_current_joint_access_ir(**current_operands):
    """The source/PDK/native/current-artwork consumer, not a JSON PASS flag."""
    from clay.layout.joint_placement_terminals import build_placement_terminal_face
    return _from_replayed_face(build_placement_terminal_face(**current_operands))


def require_current_joint_access_ir(ir, **current_operands):
    if type(ir) is not JointAccessIR or ir != compile_current_joint_access_ir(**current_operands):
        raise ValueError('joint access IR differs from the complete current source face')


def _graph(nodes, edges):
    if type(nodes) is not dict or not 0 < len(nodes) <= 100_000:
        raise ValueError('joint access graph needs a bounded nonempty node dictionary')
    for name, state in nodes.items():
        if (type(name) is not str or not name or type(state) is not tuple or len(state) != 4
                or any(type(v) is not int for v in state)
                or any(abs(v) > 1_000_000_000 for v in state[:2])
                or any(not 0 <= v <= 65535 for v in state[2:])):
            raise ValueError('joint access graph states must be exact (x,y,layer,datatype)')
    if len(set(nodes.values())) != len(nodes):
        raise ValueError('joint access graph states are duplicated')
    if type(edges) is not tuple or len(edges) > 1_000_000:
        raise ValueError('joint access graph edges must be a bounded exact tuple')
    result = {}
    for row in edges:
        if (type(row) is not tuple or len(row) != 3 or row[0] not in nodes or row[1] not in nodes
                or row[0] == row[1] or type(row[2]) is not int or row[2] < 0):
            raise ValueError('joint access graph edge is invalid')
        key = tuple(sorted(row[:2]))
        if key in result:
            raise ValueError('joint access graph edge is duplicated')
        result[key] = row[2]
    return result


def graph_access_domains(ir, nodes, *, net, checkpoint=lambda: None):
    """Retain all graph events covered by each real mask; no bbox-only accesses."""
    if type(ir) is not JointAccessIR or ir.scope != 'replayed_current_joint_face_finite_graph_only/v1':
        raise TypeError('joint access needs the exact typed current-face IR')
    _graph(nodes, ())
    demands = tuple(row for row in ir.demands if row.net == net)
    if not demands or len(demands) > 4096:
        raise ValueError('joint access net has no bounded complete demand inventory')
    if len(demands) * len(nodes) > 2_000_000:
        raise ValueError('joint access graph projection exceeds its pair budget')
    domains = []
    for demand in demands:
        checkpoint()
        regions = {}
        for _identity, geometry in demand.alternatives:
            for layer, polygons in geometry:
                region = regions.setdefault(layer, kdb.Region())
                for polygon in polygons:
                    region += polygon.to_region()
        members = []
        for index, (name, (x, y, layer, datatype)) in enumerate(sorted(nodes.items())):
            if index % 128 == 0:
                checkpoint()
            region = regions.get((layer, datatype))
            if region is not None and (kdb.Region(kdb.Box(x, y, x+1, y+1)) - region).is_empty():
                members.append(name)
        domains.append((demand.identity, tuple(members)))
    return tuple(domains)


def route_joint_access_graph(ir, nodes, edges, *, net, max_expansions=1_000_000,
                             checkpoint=lambda: None):
    """Grow a tree on the quotient graph using one multi-source Dijkstra/pass.

    This is a feasibility strategy, not a minimum-Steiner-tree claim. An exhausted
    complete graph proves graph-scope disconnection; a budget stop is UNKNOWN.
    Existing FEOL connections create virtual equivalence, never emitted wires.
    """
    if type(max_expansions) is not int or max_expansions < 0:
        raise ValueError('joint access expansion budget must be a nonnegative integer')
    weights = _graph(nodes, edges)
    domains = graph_access_domains(ir, nodes, net=net, checkpoint=checkpoint)
    binding = _sha(dict(ir=ir.sha256, net=net, nodes=nodes, edges=sorted((*key, cost)
        for key, cost in weights.items())))
    base = dict(scope='finite_input_graph_only', physical_admission=False, input_sha256=binding,
                expansions=0, searches=0, selected_edges=(), cost=0,
                demand_count=len(domains), access_event_count=sum(len(row) for _, row in domains))
    if any(not members for _, members in domains):
        return dict(base, status='UNSAT', reason='a required joint component has no graph access')
    parent = {node: node for node in nodes}
    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    for _identity, members in domains:
        checkpoint()
        origin = find(members[0])
        for node in members[1:]:
            other = find(node)
            if origin != other:
                parent[other] = origin
    roots = {node: find(node) for node in nodes}
    required = {roots[members[0]] for _, members in domains}
    adjacency = {root: [] for root in roots.values()}
    for index, ((a, b), cost) in enumerate(sorted(weights.items())):
        if index % 128 == 0:
            checkpoint()
        x, y = roots[a], roots[b]
        if x != y:
            adjacency[x].append((y, cost, (a, b)))
            adjacency[y].append((x, cost, (a, b)))
    tree = {min(required)}
    selected = set()
    while not required <= tree:
        checkpoint()
        base['searches'] += 1
        distances = {node: 0 for node in tree}
        queue = [(0, node) for node in sorted(tree)]
        heapq.heapify(queue)
        previous = {}
        target = None
        while queue:
            checkpoint()
            cost, node = heapq.heappop(queue)
            if cost != distances[node]:
                continue
            if base['expansions'] >= max_expansions:
                return dict(base, status='UNKNOWN', reason='expansion budget exhausted')
            base['expansions'] += 1
            if node in required and node not in tree:
                target = node
                break
            for other, price, edge in adjacency[node]:
                candidate = cost + price
                if candidate < distances.get(other, float('inf')):
                    distances[other] = candidate
                    previous[other] = node, edge
                    heapq.heappush(queue, (candidate, other))
        if target is None:
            return dict(base, status='UNSAT', reason='required components are disconnected in the full input graph')
        while target not in tree:
            tree.add(target)
            target, edge = previous[target]
            selected.add(edge)
    return dict(base, status='FEASIBLE', selected_edges=tuple(sorted(selected)),
                cost=sum(weights[edge] for edge in selected))


def replay_joint_access_witness(ir, nodes, edges, selected_edges, *, net):
    """Independent graph witness replay on original vertices and virtual hubs."""
    weights = _graph(nodes, edges)
    domains = graph_access_domains(ir, nodes, net=net)
    if type(selected_edges) is not tuple or len(set(selected_edges)) != len(selected_edges):
        raise ValueError('joint access witness has repeated or malformed edges')
    adjacency = {('node', name): set() for name in nodes}
    for edge in selected_edges:
        if type(edge) is not tuple or edge not in weights:
            raise ValueError('joint access witness contains a foreign edge')
        a, b = (('node', name) for name in edge)
        adjacency[a].add(b)
        adjacency[b].add(a)
    for identity, members in domains:
        hub = ('joint', identity)
        adjacency[hub] = {('node', member) for member in members}
        for member in members:
            adjacency[('node', member)].add(hub)
    start = ('joint', domains[0][0])
    seen, pending = {start}, [start]
    while pending:
        for node in adjacency[pending.pop()] - seen:
            seen.add(node)
            pending.append(node)
    return all(members and ('joint', identity) in seen for identity, members in domains)
