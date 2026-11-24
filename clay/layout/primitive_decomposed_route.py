"""Finite static routing: exact arc flow, lazy geometry and conflict decomposition.

Research backend. All original edges remain available unless a proved shortest
distance/cap condition excludes them. Wires persist; no MAPF time dimension.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import heapq
import itertools
import math
import time

from ortools.sat.python import cp_model

from .primitive_search import GraphProblem, boxes_clear, path_length, wire_box


def shortest_distances(problem, origins):
    distances = {node: math.inf for node in problem.nodes}
    queue = []
    for node in origins:
        distances[node] = 0
        heapq.heappush(queue, (0, node))
    adjacency = problem.adjacency()
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distances[node]:
            continue
        for other in adjacency[node]:
            a, b = problem.nodes[node], problem.nodes[other]
            new = cost + abs(a[0] - b[0]) + abs(a[1] - b[1])
            if new < distances[other]:
                distances[other] = new
                heapq.heappush(queue, (new, other))
    return distances


def _self_conflict(problem, edges, boxes, i, j):
    shared = set(edges[i]) & set(edges[j])
    if not shared:
        return not boxes_clear(boxes[i], boxes[j], 0)
    center = next(iter(shared))
    ends = [next(node for node in edges[k] if node != center) for k in (i, j)]
    p = problem.nodes[center]
    a, b = [problem.nodes[node] for node in ends]
    return (a[0] - p[0]) * (b[0] - p[0]) + (a[1] - p[1]) * (b[1] - p[1]) > 0


def geometric_conflicts(problem, paths):
    """Return actual selected-resource violations, including fixed foreign pads."""
    edges = list(problem.edges)
    index = {frozenset(edge): i for i, edge in enumerate(edges)}
    boxes = [wire_box(problem.nodes[a], problem.nodes[b], problem.wire_width) for a, b in edges]
    selected = {net: [index[frozenset(edge)] for edge in zip(path, path[1:])]
                for net, path in paths.items()}
    conflicts = []
    probes = 0
    for net, ids in selected.items():
        for i, j in itertools.combinations(ids, 2):
            probes += 1
            if _self_conflict(problem, edges, boxes, i, j):
                conflicts.append(((net, i), (net, j)))
        for foreign, terminals in problem.terminals.items():
            if foreign == net:
                continue
            for node in terminals:
                pad = wire_box(problem.nodes[node], problem.nodes[node],
                               problem.terminal_pad_widths.get(node, problem.wire_width))
                for i in ids:
                    probes += 1
                    if not boxes_clear(boxes[i], pad, problem.minimum_gap):
                        conflicts.append(((net, i),))
    for a, b in itertools.combinations(selected, 2):
        for i, j in itertools.product(selected[a], selected[b]):
            probes += 1
            if not boxes_clear(boxes[i], boxes[j], problem.minimum_gap):
                conflicts.append(((a, i), (b, j)))
    return conflicts, probes


def _fixed_pads_clear(problem):
    for a, b in itertools.combinations(problem.terminals, 2):
        for left, right in itertools.product(problem.terminals[a], problem.terminals[b]):
            boxes = [wire_box(problem.nodes[node], problem.nodes[node],
                             problem.terminal_pad_widths.get(node, problem.wire_width))
                     for node in (left, right)]
            if not boxes_clear(*boxes, problem.minimum_gap):
                return False
    return True


def arc_flow_route(problem: GraphProblem, *, nets=None, forbidden=(), lazy=True,
                   seconds=10, max_refinements=1000):
    """Exact group oracle in a fixed full world; outside-group pads stay reserved."""
    started = time.perf_counter()
    deadline = started + seconds
    nets = tuple(sorted(problem.terminals if nets is None else nets))
    if not nets or not set(nets) <= set(problem.terminals):
        raise ValueError("A nonempty declared group is required")
    if len({frozenset(edge) for edge in problem.edges}) != len(problem.edges):
        raise ValueError("Canonical undirected edges must be unique")
    metrics = dict(solver_calls=0, refinements=0, geometry_clauses=0, geometry_probes=0,
                   solver_branches=0, solver_conflicts=0, compile_seconds=0.0)
    result = dict(status="UNKNOWN", paths={}, objective_nm=None, lower_bound_nm=0, **metrics)
    if not _fixed_pads_clear(problem):
        return {**result, "status": "UNSAT", "reason": "fixed terminal pads conflict"}
    model = cp_model.CpModel()
    edges = list(problem.edges)
    boxes = [wire_box(problem.nodes[a], problem.nodes[b], problem.wire_width) for a, b in edges]
    weights = [abs(problem.nodes[a][0]-problem.nodes[b][0])+abs(problem.nodes[a][1]-problem.nodes[b][1])
               for a, b in edges]
    used, directed, active, lengths = {}, {}, {}, {}
    for net in nets:
        start, target = problem.terminals[net]
        ds, dt = shortest_distances(problem, (start,)), shortest_distances(problem, (target,))
        active[net] = [i for i, (a, b) in enumerate(edges)
                       if min(ds[a]+weights[i]+dt[b], ds[b]+weights[i]+dt[a]) <= problem.length_caps[net]]
        used[net] = {i: model.NewBoolVar(f"use_{net}_{i}") for i in active[net]}
        nodes = {node for i in active[net] for node in edges[i]} | {start, target}
        order = {node: model.NewIntVar(0, len(problem.nodes)-1, f"order_{net}_{node}") for node in nodes}
        incoming, outgoing = {node: [] for node in nodes}, {node: [] for node in nodes}
        directed[net] = {}
        for i in active[net]:
            a, b = edges[i]
            arcs = []
            for u, v in ((a, b), (b, a)):
                arc = model.NewBoolVar(f"arc_{net}_{i}_{u}")
                directed[net][u, v] = arc
                outgoing[u].append(arc)
                incoming[v].append(arc)
                model.Add(order[v] >= order[u]+1).OnlyEnforceIf(arc)
                arcs.append(arc)
            model.Add(sum(arcs) == used[net][i])
        for node in nodes:
            supply = int(node == start)
            demand = int(node == target)
            model.Add(sum(outgoing[node])-sum(incoming[node]) == supply-demand)
            model.Add(sum(incoming[node])+supply <= 1)
            model.Add(sum(outgoing[node])+demand <= 1)
        lengths[net] = sum(weights[i]*used[net][i] for i in active[net])
        model.Add(lengths[net] <= problem.length_caps[net])
    p, q = problem.tol_rel.numerator, problem.tol_rel.denominator
    for a, b in problem.length_pairs:
        if a in nets and b in nets:
            model.Add((q-p)*lengths[a] <= q*lengths[b])
            model.Add((q-p)*lengths[b] <= q*lengths[a])
        elif a in nets or b in nets:
            raise ValueError("Cannot split an attribute factor across exact groups")
    model.Minimize(sum(lengths.values()))
    clauses = set()

    def add_clause(conflict):
        key = tuple(sorted(conflict))
        if key in clauses:
            return False
        clauses.add(key)
        model.Add(sum(used[net][i] for net, i in key) <= len(key)-1)
        result["geometry_clauses"] += 1
        return True

    for net, i in forbidden:
        if net in used and i in used[net]:
            model.Add(used[net][i] == 0)
    # Cheap fixed-pad restrictions are present in both formulations.
    for net in nets:
        for foreign, terminals in problem.terminals.items():
            if foreign == net:
                continue
            for node in terminals:
                pad = wire_box(problem.nodes[node], problem.nodes[node],
                               problem.terminal_pad_widths.get(node, problem.wire_width))
                for i in active[net]:
                    result["geometry_probes"] += 1
                    if not boxes_clear(boxes[i], pad, problem.minimum_gap):
                        add_clause(((net, i),))
    if not lazy:
        for net in nets:
            for i, j in itertools.combinations(active[net], 2):
                result["geometry_probes"] += 1
                if _self_conflict(problem, edges, boxes, i, j):
                    add_clause(((net, i), (net, j)))
        for a, b in itertools.combinations(nets, 2):
            for i, j in itertools.product(active[a], active[b]):
                result["geometry_probes"] += 1
                if not boxes_clear(boxes[i], boxes[j], problem.minimum_gap):
                    add_clause(((a, i), (b, j)))
    result.update(variables=len(model.Proto().variables), constraints=len(model.Proto().constraints),
                  active_edges=sum(map(len, active.values())), compile_seconds=time.perf_counter()-started)
    for iteration in range(max_refinements+1):
        remaining = deadline-time.perf_counter()
        if remaining <= 0:
            break
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_time_in_seconds = remaining
        status = solver.Solve(model)
        result["solver_calls"] += 1
        result["solver_branches"] += solver.NumBranches()
        result["solver_conflicts"] += solver.NumConflicts()
        if status == cp_model.MODEL_INVALID:
            raise ValueError(solver.ResponseStats())
        if status == cp_model.INFEASIBLE:
            return {**result, "status": "UNSAT", "seconds": time.perf_counter()-started,
                    "constraints": len(model.Proto().constraints)}
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            result["lower_bound_nm"] = max(result["lower_bound_nm"], 0,
                                           math.floor(solver.BestObjectiveBound()-1e-7))
            paths = {}
            for net in nets:
                start, target = problem.terminals[net]
                following = {u: v for (u, v), arc in directed[net].items() if solver.Value(arc)}
                path = [start]
                while path[-1] != target:
                    path.append(following[path[-1]])
                    if len(path) > len(problem.nodes):
                        raise AssertionError("Flow encoding permitted a cycle")
                paths[net] = tuple(path)
            conflicts, probes = geometric_conflicts(problem, paths)
            result["geometry_probes"] += probes
            if not conflicts:
                value = sum(path_length(path, problem.nodes) for path in paths.values())
                result.update(paths=paths, objective_nm=value)
                if status == cp_model.OPTIMAL:
                    result.update(status="OPTIMAL", lower_bound_nm=value)
                return {**result, "seconds": time.perf_counter()-started,
                        "constraints": len(model.Proto().constraints)}
            if not lazy:
                raise AssertionError("Eager model omitted a geometric constraint")
            added = sum(add_clause(conflict) for conflict in conflicts)
            if not added:
                raise AssertionError("Geometry refinement made no progress")
            result["refinements"] += 1
    return {**result, "seconds": time.perf_counter()-started,
            "constraints": len(model.Proto().constraints)}


def attribute_groups(problem):
    parent = {net: net for net in problem.terminals}
    def find(net):
        while parent[net] != net:
            net = parent[net]
        return net
    for a, b in problem.length_pairs:
        parent[find(a)] = find(b)
    groups = {}
    for net in parent:
        groups.setdefault(find(net), []).append(net)
    return tuple(sorted(tuple(sorted(group)) for group in groups.values()))


def conflict_decomposed_route(problem, *, seconds=10, merge_after=3,
                              node_limit=10000, oracle=arc_flow_route):
    """Best-first permanent-resource conflict splitting, with exact attribute groups."""
    started = time.perf_counter()
    deadline = started + seconds
    groups = attribute_groups(problem)
    result = dict(status="UNKNOWN", paths={}, objective_nm=None, lower_bound_nm=0,
                  high_level_nodes=0, oracle_calls=0, cache_hits=0, merges=0,
                  initial_groups=len(groups), maximum_group_size=max(map(len, groups)),
                  refinements=0, geometry_clauses=0, solver_branches=0, geometry_probes=0)
    cache, seen, queue, unresolved, hits = {}, set(), [], [], Counter()
    serial = itertools.count()
    incumbent = math.inf
    root_lb = 0

    def evaluate(partition, forbidden):
        nonlocal root_lb, incumbent
        key = (partition, forbidden)
        if key in seen:
            return
        seen.add(key)
        paths, bound, exact, impossible = {}, 0, True, False
        for group in partition:
            local_forbid = tuple(sorted(resource for resource in forbidden if resource[0] in group))
            local_key = group, local_forbid
            if local_key in cache:
                local = cache[local_key]
                result["cache_hits"] += 1
            else:
                local = oracle(problem, nets=group, forbidden=local_forbid, lazy=True,
                               seconds=max(0, deadline-time.perf_counter()))
                if local["status"] != "UNKNOWN":
                    cache[local_key] = local
                result["oracle_calls"] += 1
                for field in ("refinements", "geometry_clauses", "solver_branches", "geometry_probes"):
                    result[field] += local.get(field, 0)
            result["maximum_group_size"] = max(result["maximum_group_size"], len(group))
            if local["status"] == "UNSAT":
                impossible = True
                break
            paths.update(local["paths"])
            bound += local["lower_bound_nm"]
            exact &= local["status"] == "OPTIMAL"
        if impossible:
            return
        if not forbidden:
            root_lb = bound
        if not exact:
            unresolved.append(bound)
            if set(paths) == set(problem.terminals) and not geometric_conflicts(problem, paths)[0]:
                value = sum(path_length(path, problem.nodes) for path in paths.values())
                if value < incumbent:
                    incumbent = value
                    result.update(paths=paths, objective_nm=value)
            return
        heapq.heappush(queue, (bound, next(serial), partition, forbidden, paths))

    evaluate(groups, frozenset())
    while queue and result["high_level_nodes"] < node_limit and time.perf_counter() < deadline:
        bound, _, partition, forbidden, paths = heapq.heappop(queue)
        if bound >= incumbent:
            continue
        result["high_level_nodes"] += 1
        conflicts, probes = geometric_conflicts(problem, paths)
        result["geometry_probes"] += probes
        if not conflicts:
            incumbent = bound
            result.update(paths=paths, objective_nm=bound)
            continue
        conflict = conflicts[0]
        if len(conflict) != 2:
            raise AssertionError("Group oracle failed its own fixed-pad check")
        (a, i), (b, j) = conflict
        ga, gb = [next(group for group in partition if net in group) for net in (a, b)]
        if ga == gb:
            raise AssertionError("Group oracle returned an internal conflict")
        pair = tuple(sorted((ga, gb)))
        hits[pair] += 1
        if merge_after is not None and hits[pair] >= merge_after:
            merged = tuple(sorted((*ga, *gb)))
            new_partition = tuple(sorted((*(group for group in partition if group not in (ga, gb)), merged)))
            result["merges"] += 1
            evaluate(new_partition, forbidden)
        else:
            evaluate(partition, forbidden | {(a, i)})
            evaluate(partition, forbidden | {(b, j)})
    bounds = [row[0] for row in queue] + unresolved
    if incumbent < math.inf and (not bounds or min(bounds) >= incumbent):
        result.update(status="OPTIMAL", lower_bound_nm=incumbent)
    elif not bounds and incumbent == math.inf:
        result.update(status="UNSAT")
    else:
        result["lower_bound_nm"] = min([incumbent, *bounds]) if bounds else root_lb
    result["seconds"] = time.perf_counter()-started
    return result
