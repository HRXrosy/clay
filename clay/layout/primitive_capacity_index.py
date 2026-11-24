"""Immutable-world index for conditional cut capacity and rank inequalities.

Fixed-target obstacles and undirected target components are compiled once.
For any guard, either it has a member in the target component and all its edges
are optimistically reachable, or it has none. This reproduces the two-sided
reachability relaxation without re-running weighted shortest paths at each cut.
Variable source pads, source guards and length constraints remain relaxed.
"""
from copy import deepcopy
import itertools
import json
import math

from .primitive_search import boxes_clear, wire_box


def world(problem):
    return dict(nodes=problem.nodes, edges=problem.edges, width=problem.wire_width,
        gap=problem.minimum_gap, fixed_targets={net: pair[1] for net, pair in problem.terminals.items()},
        terminal_pad_widths=problem.terminal_pad_widths)


class RankCapacityIndex:
    def __init__(self, problem, domains):
        self.problem = problem
        self.domains = {net: tuple(nodes) for net, nodes in domains.items()}
        self.world = deepcopy(world(problem))
        self.signature = json.dumps(self.world, sort_keys=True)
        self.edges = tuple(problem.edges)
        self.boxes = [wire_box(problem.nodes[a], problem.nodes[b], problem.wire_width) for a, b in self.edges]
        self.stats = dict(queries=0, cuts_examined=0, fixed_pad_probes=0, bin_geometry_probes=0, bin_cache_hits=0)
        pads = {net: wire_box(problem.nodes[pair[1]], problem.nodes[pair[1]],
            problem.terminal_pad_widths.get(pair[1], problem.wire_width)) for net, pair in problem.terminals.items()}
        adjacency = {node: [] for node in problem.nodes}
        for i, (a, b) in enumerate(self.edges):
            adjacency[a].append((b, i))
            adjacency[b].append((a, i))
        self.components, self.allowed = {}, {}
        for net, (_, target) in problem.terminals.items():
            usable = set()
            for i, box in enumerate(self.boxes):
                clear = True
                for foreign, pad in pads.items():
                    if foreign == net:
                        continue
                    self.stats["fixed_pad_probes"] += 1
                    if not boxes_clear(box, pad, problem.minimum_gap):
                        clear = False
                        break
                if clear:
                    usable.add(i)
            component, pending = {target}, [target]
            while pending:
                node = pending.pop()
                for other, i in adjacency[node]:
                    if i in usable and other not in component:
                        component.add(other)
                        pending.append(other)
            self.components[net] = frozenset(component)
            self.allowed[net] = sum(1 << i for i in usable if self.edges[i][0] in component and self.edges[i][1] in component)
        self.cuts = []
        for axis in (0, 1):
            values = sorted({point[axis] for point in problem.nodes.values()})
            for left, right in zip(values, values[1:]):
                twice = left+right
                side = {node: 2*point[axis] < twice for node, point in problem.nodes.items()}
                guard = {net: tuple(node for node in nodes if side[node] != side[problem.terminals[net][1]])
                         for net, nodes in self.domains.items()}
                crossing = sum(1 << i for i, (a, b) in enumerate(self.edges) if side[a] != side[b])
                self.cuts.append((axis, twice, guard, crossing))
        self.bin_cache = {}

    def certificate(self, selection):
        if json.dumps(world(self.problem), sort_keys=True) != self.signature:
            raise ValueError("Capacity index world changed; rebuild before using any proof")
        if set(selection) != set(self.domains) or any(selection[net] not in nodes for net, nodes in self.domains.items()):
            raise ValueError("Selection is outside the compiled interface domain")
        self.stats["queries"] += 1
        for axis, twice, guard, crossing in self.cuts:
            demand = [net for net in self.domains if selection[net] in guard[net]]
            if len(demand) < 2:
                continue
            self.stats["cuts_examined"] += 1
            allowed = {net: self.allowed[net] if any(node in self.components[net] for node in guard[net]) else 0 for net in demand}
            union = 0
            for value in allowed.values():
                union |= value
            relevant = crossing & union
            cache_key = (axis, twice, relevant)
            if cache_key in self.bin_cache:
                bins = self.bin_cache[cache_key]
                self.stats["bin_cache_hits"] += 1
            else:
                bins = []
                for i in range(len(self.edges)):
                    if not relevant & (1 << i):
                        continue
                    for bucket in bins:
                        compatible = True
                        for j in bucket:
                            self.stats["bin_geometry_probes"] += 1
                            if boxes_clear(self.boxes[i], self.boxes[j], self.problem.minimum_gap):
                                compatible = False
                                break
                        if compatible:
                            bucket.append(i)
                            break
                    else:
                        bins.append([i])
                self.bin_cache[cache_key] = bins
            masks = [sum(1 << i for i in bucket) for bucket in bins]
            eligible = {net: [j for j, mask in enumerate(masks) if mask & allowed[net]] for net in demand}
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
                    twice_cut_coordinate_nm=twice, guard={net: list(guard[net]) for net in demand},
                    demand=len(demand), matching_rank=rank, bins=deepcopy(bins), world=deepcopy(self.world),
                    covered_interface_combinations=math.prod(len(guard[net]) for net in demand),
                    reachability_compiler="fixed-pad target-component index")
        return dict(status="NO_CERTIFICATE")
