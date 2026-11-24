"""Compact primitive master and guarded static-routing Benders feedback.

Finite OrbitConstructor separator domain only. Geometry and rule signatures
scope every proof; no certificate is reused across calls/worlds implicitly.
"""
from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import itertools
import math
import time

from ortools.sat.python import cp_model

from .primitive_relations import _validate_separator, scoped_interface
from .primitive_search import boxes_clear, path_length, wire_box
from .primitive_decomposed_route import conflict_decomposed_route, shortest_distances


def guarded_capacity(problem, domains, selection=None):
    """A cut proof over concrete edges and *all* opposite-side start choices.

    Foreign variable source pads, lengths and source guards are relaxed, making
    capacity an upper bound. Fixed foreign target pads remain obstacles.
    """
    edges = list(problem.edges)
    shapes = [wire_box(problem.nodes[a], problem.nodes[b], problem.wire_width) for a, b in edges]
    for axis in (0, 1):
        coordinates = sorted({p[axis] for p in problem.nodes.values()})
        for left, right in zip(coordinates, coordinates[1:]):
            twice = left+right
            side = {node: 2*p[axis] < twice for node, p in problem.nodes.items()}
            guards = {net: tuple(site for site in sites if side[site] != side[problem.terminals[net][1]])
                      for net, sites in domains.items()}
            if selection is None:
                demand = [net for net in domains if guards[net] and set(guards[net]) == set(domains[net])]
            else:
                demand = [net for net in domains if selection[net] in guards[net]]
            if len(demand) < 2:
                continue
            allowed = {}
            for net in demand:
                usable = []
                for i, edge in enumerate(edges):
                    clear = True
                    for foreign, (_, target) in problem.terminals.items():
                        if net == foreign:
                            continue
                        pad = wire_box(problem.nodes[target], problem.nodes[target],
                                       problem.terminal_pad_widths.get(target, problem.wire_width))
                        if not boxes_clear(shapes[i], pad, problem.minimum_gap):
                            clear = False
                    if clear:
                        usable.append(edge)
                optimistic = replace(problem, edges=tuple(usable))
                ds = shortest_distances(optimistic, guards[net])
                dt = shortest_distances(optimistic, (problem.terminals[net][1],))
                allowed[net] = {i for i, (a, b) in enumerate(edges) if (a, b) in usable
                               and ds[a] < math.inf and dt[b] < math.inf}
                # Undirected reachability: both ends belong to the same component.
            crossing = [i for i, (a, b) in enumerate(edges) if side[a] != side[b]
                        and any(i in allowed[net] for net in demand)]
            bins = []
            for i in crossing:
                bucket = next((b for b in bins if all(not boxes_clear(shapes[i], shapes[j], problem.minimum_gap)
                                                     for j in b)), None)
                if bucket is None:
                    bins.append([i])
                else:
                    bucket.append(i)
            eligible = {net: [i for i, bucket in enumerate(bins) if set(bucket) & allowed[net]] for net in demand}
            owners = {}
            def augment(net, seen):
                for bucket in eligible[net]:
                    if bucket in seen:
                        continue
                    seen.add(bucket)
                    if bucket not in owners or augment(owners[bucket], seen):
                        owners[bucket] = net
                        return True
                return False
            rank = sum(augment(net, set()) for net in demand)
            if rank < len(demand):
                return dict(status="UNSAT", kind="guarded-cut-capacity", axis=axis,
                            twice_cut_coordinate_nm=twice, guard={net: list(guards[net]) for net in demand},
                            demand=len(demand), matching_rank=rank, bins=bins,
                            world={"nodes": deepcopy(problem.nodes), "edges": deepcopy(problem.edges),
                                   "width": problem.wire_width, "gap": problem.minimum_gap,
                                   "fixed_targets": {net: nodes[1] for net, nodes in problem.terminals.items()},
                                   "terminal_pad_widths": deepcopy(problem.terminal_pad_widths)},
                            covered_interface_combinations=math.prod(len(guards[net]) for net in demand))
    return dict(status="NO_CERTIFICATE")


def solve_primitive_flow(problem, constructor, domains, *, seconds=10, capacity=True,
                         routing=conflict_decomposed_route, initial_proofs=()):
    """Source choices + ports stay compact; route proof/value cuts select the winner."""
    started = time.perf_counter()
    deadline = started+seconds
    _validate_separator(problem, constructor)
    model = cp_model.CpModel()
    source = [model.NewBoolVar(f"role_a_{i}") for i in range(constructor.n)]
    model.Add(sum(source) == constructor.a_orbits)
    if constructor.first_role:
        model.Add(source[0] == int(constructor.first_role == "a"))
    fixed, groups, factors = constructor.local_factors()
    costs = []
    for indices, factor in zip(groups, factors):
        cost = model.NewIntVar(min(factor.values()), max(factor.values()), f"local_cost_{indices[0]}")
        model.AddAllowedAssignments([*(source[i] for i in indices), cost],
                                    [(int(a == "a"), int(b == "a"), value) for (a, b), value in factor.items()])
        costs.append(cost)
    ports = {net: {site: model.NewBoolVar(f"port_{net}_{site}") for site in sites}
             for net, sites in domains.items()}
    for net, options in ports.items():
        model.Add(sum(options.values()) == 1)
        for site, chosen in options.items():
            guard = constructor.access_guard(net, site)
            if guard is None:
                model.Add(chosen == 0)
            else:
                for i, role in guard.items():
                    model.Add(source[i] == int(role == "a")).OnlyEnforceIf(chosen)
    # Only local pair exclusions; never a complete joint-interface table.
    for a, b in itertools.combinations(ports, 2):
        for left, right in itertools.product(ports[a], ports[b]):
            if not boxes_clear(wire_box(constructor.sites[left], constructor.sites[left], constructor.local_width_nm),
                               wire_box(constructor.sites[right], constructor.sites[right], constructor.local_width_nm),
                               max(400, problem.minimum_gap)):
                model.Add(ports[a][left]+ports[b][right] <= 1)
    lower_lengths = {}
    for net, options in ports.items():
        distances = shortest_distances(problem, (problem.terminals[net][1],))
        lower_lengths[net] = model.NewIntVar(0, problem.length_caps[net], f"lower_length_{net}")
        for site, chosen in options.items():
            if distances[site] == math.inf:
                model.Add(chosen == 0)
            else:
                model.Add(lower_lengths[net] >= int(distances[site])).OnlyEnforceIf(chosen)
    w, pad = problem.wire_width, constructor.local_width_nm
    constant = len(ports)*(w*w+pad*pad-w*(pad+w)//2)
    constant += sum((3-constructor.layers[net]-1)*pad*pad for net in ports)
    theta = model.NewIntVar(constant, constant+w*sum(problem.length_caps.values()), "boundary_area")
    model.Add(theta >= constant+w*sum(lower_lengths.values()))
    model.Minimize(fixed+sum(costs)+theta)
    result = dict(status="UNKNOWN", objective_area_nm2=None, assignment=None, selection=None, paths={},
                  lower_bound_area_nm2=0, master_calls=0, route_calls=0, certificates=[],
                  cost_cuts=0, point_nogoods=0, interface_rows_materialized=0, materializations=0,
                  rejected_stale_proofs=0,
                  source_variables=len(source), port_variables=sum(map(len, ports.values())),
                  variables=len(model.Proto().variables), constraints=len(model.Proto().constraints))

    known_capacity = set()
    def add_capacity(proof):
        if proof["status"] != "UNSAT":
            return False
        guard = proof["guard"]
        key = tuple(sorted((net, tuple(sites)) for net, sites in guard.items()))
        if key in known_capacity:
            return False
        known_capacity.add(key)
        model.Add(sum(ports[net][site] for net, sites in guard.items() for site in sites) <= len(guard)-1)
        result["certificates"].append(proof)
        return True

    for proof in initial_proofs:
        if certificate_applies(proof, problem, domains):
            add_capacity(proof)
        else:
            result["rejected_stale_proofs"] += 1
    if capacity:
        add_capacity(guarded_capacity(problem, domains))
        # Discover a conditional cut even when an extra right-side port bypasses
        # the cut. The optimistic proof expands the seed to all same-side ports.
        seed = {net: sites[0] for net, sites in domains.items()}
        add_capacity(guarded_capacity(problem, domains, seed))
    incumbent = math.inf
    while time.perf_counter() < deadline:
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_time_in_seconds = max(0, deadline-time.perf_counter())
        status = solver.Solve(model)
        result["master_calls"] += 1
        if status == cp_model.INFEASIBLE:
            result["status"] = "OPTIMAL" if incumbent < math.inf else "UNSAT"
            break
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        bound = max(0, math.floor(solver.BestObjectiveBound()-1e-7))
        if status == cp_model.OPTIMAL:
            bound = int(round(solver.ObjectiveValue()))
        result["lower_bound_area_nm2"] = max(result["lower_bound_area_nm2"], bound)
        if incumbent < math.inf and bound >= incumbent:
            result["status"] = "OPTIMAL"
            break
        selection = {net: next(site for site, variable in options.items() if solver.Value(variable))
                     for net, options in ports.items()}
        if capacity and add_capacity(guarded_capacity(problem, domains, selection)):
            continue
        selected = [ports[net][site] for net, site in selection.items()]
        scoped = scoped_interface(problem, constructor, selection)
        routed = routing(scoped, seconds=max(0, deadline-time.perf_counter()))
        result["route_calls"] += 1
        if routed["paths"]:
            assignment = tuple("a" if solver.Value(variable) else "b" for variable in source)
            local = fixed+sum(solver.Value(cost) for cost in costs)
            value = local+constant+w*routed["objective_nm"]
            if value < incumbent:
                incumbent = value
                result.update(objective_area_nm2=value, assignment=assignment, selection=selection, paths=routed["paths"])
        if routed["status"] == "UNSAT":
            model.Add(sum(selected) <= len(selected)-1)
            result["point_nogoods"] += 1
        elif routed["status"] == "OPTIMAL":
            model.Add(theta >= constant+w*routed["objective_nm"]).OnlyEnforceIf(selected)
            result["cost_cuts"] += 1
        else:
            # Do not retry forever or manufacture a cut from incomplete search.
            break
    if result["assignment"] is not None:
        # Materialize only the winning witness; external checker replays the union.
        result["mode"] = constructor.materialize(result["assignment"]).name
        result["materializations"] = 1
    if result["status"] == "OPTIMAL":
        result["lower_bound_area_nm2"] = result["objective_area_nm2"]
    result["seconds"] = time.perf_counter()-started
    return result


def certificate_applies(proof, problem, domains):
    """Validity-domain gate for trusted locally generated proof envelopes."""
    if proof.get("kind") != "guarded-cut-capacity":
        return False
    world = {"nodes": problem.nodes, "edges": problem.edges, "width": problem.wire_width,
             "gap": problem.minimum_gap,
             "fixed_targets": {net: nodes[1] for net, nodes in problem.terminals.items()},
             "terminal_pad_widths": problem.terminal_pad_widths}
    # Normalize tuple/list representations, including receipts read from JSON.
    import json
    if json.dumps(world, sort_keys=True) != json.dumps(proof["world"], sort_keys=True):
        return False
    return all(net in domains and set(sites) <= set(domains[net])
               for net, sites in proof["guard"].items())
