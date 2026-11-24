"""Finite multi-primitive composition, pose/interface master and rank feedback.

Each CC instance keeps its own conserved source inventory; poses are explicit
input choices. Internal M1/M2 conductors fit declared nonoverlapping outlines,
M3 routing sees only selected 600nm interface pads. Exactness is confined to
this physical separator contract, not a generic PDK primitive compiler.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import math
import time

from ortools.sat.python import cp_model

from .primitive_relations import OrbitConstructor
from .primitive_search import GraphProblem, boxes_clear, wire_box
from .primitive_decomposed_route import shortest_distances
from .primitive_adaptive_route import adaptive_route
from .primitive_proof_flow import guarded_capacity
from .primitive_capacity_index import RankCapacityIndex


@dataclass(frozen=True)
class PrimitiveInstance:
    name: str
    constructor: OrbitConstructor
    poses: dict[str, tuple[int, int]]
    # local roles -> global nets; local role -> local candidate site names
    nets: dict[str, str]
    sites: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class Composition:
    instances: tuple[PrimitiveInstance, ...]
    problem: GraphProblem
    # (instance id, pose id, local site id) -> global graph node
    bindings: dict[tuple[str, str, str], str]

    def interfaces(self):
        domains, metadata = {}, {}
        for instance in self.instances:
            c = instance.constructor
            if set(instance.nets) != {"a", "b"} or set(instance.sites) != {"a", "b"}:
                raise ValueError("Each primitive requires its two declared roles")
            for role, net in instance.nets.items():
                if net in domains:
                    raise ValueError("Shared or multi-terminal nets need a different separator contract")
                domains[net] = []
                for pose in instance.poses:
                    for site in instance.sites[role]:
                        node = self.bindings[instance.name, pose, site]
                        if node in metadata:
                            raise ValueError("Each candidate interface must have a unique owner")
                        domains[net].append(node)
                        metadata[node] = (instance.name, pose, role, site)
                        px, py = instance.poses[pose]
                        x, y = c.sites[site]
                        if self.problem.nodes[node] != (x+px, y+py):
                            raise ValueError("Port geometry differs from its primitive pose")
        if set(domains) != set(self.problem.terminals):
            raise ValueError("Every global net must belong to exactly one primitive role")
        return domains, metadata

    def validate_separator(self):
        domains, _ = self.interfaces()
        p = self.problem
        if len({i.name for i in self.instances}) != len(self.instances):
            raise ValueError("Duplicate primitive id")
        if p.wire_width > 600:
            raise ValueError("Routing exceeds the interface landing contract")
        for axis in (0, 1):
            values = sorted({point[axis] for point in p.nodes.values()})
            if any(b-a < 1200 for a, b in zip(values, values[1:])):
                raise ValueError("Affine area requires separated coordinate lines")
        for a, b in p.edges:
            wire_box(p.nodes[a], p.nodes[b], p.wire_width)
            if sum(abs(p.nodes[a][axis]-p.nodes[b][axis]) for axis in (0, 1)) < 1200:
                raise ValueError("Short edges require polygon-union optimization")
        for net, (_, target) in p.terminals.items():
            if target in {node for sites in domains.values() for node in sites}:
                raise ValueError("Fixed targets cannot be variable primitive interfaces")
            if p.terminal_pad_widths.get(target, p.wire_width) != p.wire_width:
                raise ValueError("Additional target geometry requires polygon optimization")
        if any(width != p.wire_width for width in p.terminal_pad_widths.values()):
            raise ValueError("Input pads must be ordinary wire-width fixed pads")
        return domains

    def scoped(self, selection):
        return replace(self.problem,
            terminals={net: (selection[net], pair[1]) for net, pair in self.problem.terminals.items()},
            terminal_pad_widths={**self.problem.terminal_pad_widths, **{node: 600 for node in selection.values()}})

    def outline(self, instance, pose):
        x, y = instance.poses[pose]
        a, b, c, d = instance.constructor.outline
        return a+x, b+y, c+x, d+y

    def boundary_constant(self):
        w = self.problem.wire_width
        # Per-net M3 pad/route union plus A's additional M2 via landing.
        return len(self.problem.terminals)*(w*w+600*600-w*(600+w)//2)+len(self.instances)*600*600


def compile_master(composition, *, matching_bound=True):
    domains = composition.validate_separator()
    domains, metadata = composition.interfaces()
    p = composition.problem
    model = cp_model.CpModel()
    labels, poses, local_costs, fixed = {}, {}, [], 0
    for instance in composition.instances:
        c = instance.constructor
        labels[instance.name] = [model.NewBoolVar(f"source_{instance.name}_{i}") for i in range(c.n)]
        model.Add(sum(labels[instance.name]) == c.a_orbits)
        if c.first_role:
            model.Add(labels[instance.name][0] == int(c.first_role == "a"))
        poses[instance.name] = {pose: model.NewBoolVar(f"pose_{instance.name}_{pose}") for pose in instance.poses}
        model.AddExactlyOne(poses[instance.name].values())
        base, groups, factors = c.local_factors()
        fixed += base
        for group, table in zip(groups, factors):
            cost = model.NewIntVar(min(table.values()), max(table.values()), f"cost_{instance.name}_{group[0]}")
            model.AddAllowedAssignments([*(labels[instance.name][i] for i in group), cost],
                [(int(a == "a"), int(b == "a"), value) for (a, b), value in table.items()])
            local_costs.append(cost)
    ports = {net: {node: model.NewBoolVar(f"port_{net}_{node}") for node in nodes} for net, nodes in domains.items()}
    lookup = {instance.name: instance for instance in composition.instances}
    for net, options in ports.items():
        model.AddExactlyOne(options.values())
        for node, var in options.items():
            owner, pose, role, site = metadata[node]
            model.Add(var <= poses[owner][pose])
            guard = lookup[owner].constructor.access_guard(role, site)
            if guard is None:
                model.Add(var == 0)
            else:
                for index, value in guard.items():
                    model.Add(labels[owner][index] == int(value == "a")).OnlyEnforceIf(var)
    # Pose exclusions concern the declared footprint contract, not placement hints.
    for a, b in itertools.combinations(composition.instances, 2):
        for pa, pb in itertools.product(a.poses, b.poses):
            if not boxes_clear(composition.outline(a, pa), composition.outline(b, pb), p.minimum_gap):
                model.Add(poses[a.name][pa]+poses[b.name][pb] <= 1)
    for a, b in itertools.combinations(ports, 2):
        for left, right in itertools.product(ports[a], ports[b]):
            if not boxes_clear(wire_box(p.nodes[left], p.nodes[left], 600), wire_box(p.nodes[right], p.nodes[right], 600), p.minimum_gap):
                model.Add(ports[a][left]+ports[b][right] <= 1)
    lower_lengths = {}
    for net, options in ports.items():
        distances = shortest_distances(p, (p.terminals[net][1],))
        length = model.NewIntVar(0, p.length_caps[net], f"lower_{net}")
        for node, var in options.items():
            if distances[node] == math.inf:
                model.Add(var == 0)
            else:
                model.Add(length >= int(distances[node])).OnlyEnforceIf(var)
        lower_lengths[net] = length
    # These variables are relaxed route lengths, not the shortest distances
    # themselves. Feasible paths can assign their ACTUAL lengths, so imposing
    # declared matching here remains a sound coupled lower-bound relaxation.
    for a, b in p.length_pairs if matching_bound else ():
        num, den = p.tol_rel.numerator, p.tol_rel.denominator
        model.Add((den-num)*lower_lengths[a] <= den*lower_lengths[b])
        model.Add((den-num)*lower_lengths[b] <= den*lower_lengths[a])
    constant = composition.boundary_constant()
    theta = model.NewIntVar(constant, constant+p.wire_width*sum(p.length_caps.values()), "external_area")
    model.Add(theta >= constant+p.wire_width*sum(lower_lengths.values()))
    local = fixed+sum(local_costs)
    model.Minimize(local+theta)
    return model, labels, poses, ports, local, theta


def solve_composition(composition, *, seconds=10, feedback="rank", routing=adaptive_route, capacity_index=True, matching_bound=True):
    """Rank feedback generalizes failed selections to a cardinality inequality.

    A capacity rank of r over the full guarded set upper-bounds every subset:
    feasible crossing nets inject into pairwise-conflicting edge bins, hence
    their number cannot exceed the maximum matching rank of the full set.
    Unknown routing results never generate infeasibility or exact value cuts.
    """
    if feedback not in {"rank", "nogood", "none"}:
        raise ValueError("Unknown feedback mode")
    started = time.perf_counter()
    deadline = started+seconds
    model, labels, poses, ports, local, theta = compile_master(composition, matching_bound=matching_bound)
    domains, _ = composition.interfaces()
    p = composition.problem
    indexed_capacity = RankCapacityIndex(p, domains) if feedback != "none" and capacity_index else None
    result = dict(status="UNKNOWN", objective_area_nm2=None, lower_bound_area_nm2=0,
        assignments=None, poses=None, selection=None, paths={}, master_calls=0, route_calls=0,
        rank_cuts=0, capacity_nogoods=0, point_nogoods=0, cost_cuts=0, certificates=[],
        materializations=0, feedback=feedback, matching_bound=matching_bound, compile_seconds=time.perf_counter()-started,
        variables=len(model.Proto().variables), constraints=len(model.Proto().constraints))
    incumbent, seen = math.inf, set()
    while time.perf_counter() < deadline:
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_time_in_seconds = max(0, deadline-time.perf_counter())
        status = solver.Solve(model)
        result["master_calls"] += 1
        if status == cp_model.MODEL_INVALID:
            raise RuntimeError(solver.ResponseStats())
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
        selected = {net: next(node for node, var in options.items() if solver.Value(var)) for net, options in ports.items()}
        if feedback != "none":
            proof = indexed_capacity.certificate(selected) if indexed_capacity else guarded_capacity(p, domains, selected)
            if proof["status"] == "UNSAT":
                guard, rank = proof["guard"], proof["matching_rank"]
                limit = rank if feedback == "rank" else len(guard)-1
                key = (tuple(sorted((net, tuple(nodes)) for net, nodes in guard.items())), limit)
                if key not in seen:
                    seen.add(key)
                    model.Add(sum(ports[net][node] for net, nodes in guard.items() for node in nodes) <= limit)
                    result["certificates"].append({**proof, "inequality_upper_bound": limit})
                    result["rank_cuts" if feedback == "rank" else "capacity_nogoods"] += 1
                    continue
        chosen_vars = [ports[net][node] for net, node in selected.items()]
        route = routing(composition.scoped(selected), seconds=max(0, deadline-time.perf_counter()))
        result["route_calls"] += 1
        if route["paths"]:
            value = solver.Value(local)+composition.boundary_constant()+p.wire_width*route["objective_nm"]
            if value < incumbent:
                incumbent = value
                result.update(objective_area_nm2=value,
                    assignments={owner: tuple("a" if solver.Value(var) else "b" for var in variables) for owner, variables in labels.items()},
                    poses={owner: next(pose for pose, var in options.items() if solver.Value(var)) for owner, options in poses.items()},
                    selection=selected, paths=route["paths"])
        if route["status"] == "UNSAT":
            model.Add(sum(chosen_vars) <= len(chosen_vars)-1)
            result["point_nogoods"] += 1
        elif route["status"] == "OPTIMAL":
            model.Add(theta >= composition.boundary_constant()+p.wire_width*route["objective_nm"]).OnlyEnforceIf(chosen_vars)
            result["cost_cuts"] += 1
        else:
            break
    if result["status"] == "OPTIMAL":
        result["lower_bound_area_nm2"] = result["objective_area_nm2"]
    if result["assignments"] is not None:
        for instance in composition.instances:
            instance.constructor.materialize(result["assignments"][instance.name])
        result["materializations"] = len(composition.instances)
    result["seconds"] = time.perf_counter()-started
    result["capacity_index_stats"] = indexed_capacity.stats if indexed_capacity else None
    return result
