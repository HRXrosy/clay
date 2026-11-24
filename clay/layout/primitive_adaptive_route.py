"""Cheap exact lower-bound witness before compiling coupled route constraints.

Every net uses a shortest path with all foreign fixed pads reserved. If these
independent minima meet the complete hard contract together, their sum is a
global lower bound and feasible witness: no CP/CBS model is needed. Failure to
meet the bound is UNKNOWN, followed by exact coupled search over the full graph.
This is a solver selection mechanism, not a new shortest-path algorithm.
"""
from __future__ import annotations

import heapq
import math
import time

from .primitive_search import boxes_clear, path_length, wire_box


def shortest_path_certificate(problem, *, seconds=10):
    started = time.perf_counter()
    deadline = started+seconds
    result = dict(status='UNKNOWN', paths={}, objective_nm=None, lower_bound_nm=0,
                  shortest_path_calls=0, geometry_probes=0, solver_calls=0,
                  optimality_certificate=None)
    pads = {net: [wire_box(problem.nodes[node], problem.nodes[node],
                           problem.terminal_pad_widths.get(node, problem.wire_width))
                  for node in terminals] for net, terminals in problem.terminals.items()}
    nets = tuple(sorted(problem.terminals))
    for i, a in enumerate(nets):
        for b in nets[i+1:]:
            if any(not boxes_clear(left, right, problem.minimum_gap) for left in pads[a] for right in pads[b]):
                return {**result, 'status': 'UNSAT', 'reason': 'fixed terminal pads conflict'}
    adjacency = problem.adjacency()
    edge_boxes = {frozenset((a,b)): wire_box(problem.nodes[a], problem.nodes[b], problem.wire_width)
                  for a,b in problem.edges}
    paths, minima = {}, {}
    for net in nets:
        if time.perf_counter() >= deadline:
            return {**result, 'reason': 'shortest-path budget exhausted'}
        start, target = problem.terminals[net]
        distances, previous, queue = {start: 0}, {}, [(0, start)]
        while queue:
            if time.perf_counter() >= deadline:
                return {**result, 'reason': 'shortest-path budget exhausted'}
            distance, node = heapq.heappop(queue)
            if distance != distances[node]: continue
            if node == target: break
            for other in adjacency[node]:
                box = edge_boxes[frozenset((node, other))]
                foreign_pads = [pad for foreign in nets if foreign != net for pad in pads[foreign]]
                result['geometry_probes'] += len(foreign_pads)
                if any(not boxes_clear(box, pad, problem.minimum_gap) for pad in foreign_pads): continue
                p, q = problem.nodes[node], problem.nodes[other]
                value = distance+abs(p[0]-q[0])+abs(p[1]-q[1])
                if value < distances.get(other, math.inf):
                    distances[other], previous[other] = value, node
                    heapq.heappush(queue, (value, other))
        result['shortest_path_calls'] += 1
        if target not in distances or distances[target] > problem.length_caps[net]:
            return {**result, 'status': 'UNSAT', 'reason': 'individual net has no cap-feasible path with fixed pads reserved'}
        path = [target]
        while path[-1] != start: path.append(previous[path[-1]])
        paths[net], minima[net] = tuple(reversed(path)), distances[target]
    result['lower_bound_nm'] = sum(minima.values())
    for a,b in problem.length_pairs:
        if abs(minima[a]-minima[b]) > problem.tol_rel*max(minima[a], minima[b]):
            return {**result, 'reason': 'independent minima violate a length relation'}
    shapes = {net: [edge_boxes[frozenset((a,b))] for a,b in zip(path,path[1:])] for net,path in paths.items()}
    for net, boxes in shapes.items():
        # Paths are simple in node identity, but long graph edges can physically
        # cross or retrace. Reject this witness without removing any graph edge.
        for i,a in enumerate(boxes):
            for b in boxes[i+2:]:
                result['geometry_probes'] += 1
                if not boxes_clear(a,b,0):
                    return {**result, 'reason': 'shortest witness self-intersects'}
        points = [problem.nodes[node] for node in paths[net]]
        for p,q,r in zip(points,points[1:],points[2:]):
            if (p[0]-q[0])*(r[0]-q[0])+(p[1]-q[1])*(r[1]-q[1]) > 0:
                return {**result, 'reason': 'shortest witness retraces an adjacent segment'}
    for i,a in enumerate(nets):
        for b in nets[i+1:]:
            for left in shapes[a]:
                for right in shapes[b]:
                    result['geometry_probes'] += 1
                    if not boxes_clear(left,right,problem.minimum_gap):
                        return {**result, 'reason': 'independent shortest witnesses conflict'}
    return {**result, 'status': 'OPTIMAL', 'paths': paths, 'objective_nm': sum(minima.values()),
            'optimality_certificate': {'kind': 'feasible_sum_of_individual_shortest_path_lower_bounds',
                                       'individual_minima_nm': minima, 'scope': 'full input graph'},
            'seconds': time.perf_counter()-started}


def adaptive_route(problem, *, seconds=10, shortest_paths=True, fallback='monolithic', lazy=False):
    if fallback not in {'monolithic', 'decomposed'}:
        raise ValueError('Unknown coupled routing backend')
    started = time.perf_counter()
    certificate = shortest_path_certificate(problem, seconds=seconds) if shortest_paths else None
    if certificate is not None and certificate['status'] != 'UNKNOWN':
        return {**certificate, 'backend': 'shortest_path_certificate', 'fallback_calls': 0}
    remaining = max(0, seconds-(time.perf_counter()-started))
    if remaining <= 0:
        return {**(certificate or {'status': 'UNKNOWN', 'paths': {}, 'objective_nm': None, 'lower_bound_nm': 0}),
                'backend': 'budget_exhausted', 'fallback_calls': 0}
    from .primitive_decomposed_route import arc_flow_route, conflict_decomposed_route
    kernel = arc_flow_route if fallback == 'monolithic' else conflict_decomposed_route
    row = kernel(problem, seconds=max(0, seconds-(time.perf_counter()-started)),
                 **({'lazy': lazy} if fallback == 'monolithic' else {}))
    return {**row, 'backend': fallback, 'fallback_calls': 1,
            'shortest_path_attempt': certificate, 'seconds': time.perf_counter()-started}
