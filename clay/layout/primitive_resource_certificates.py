"""Sound interface conflicts from compulsory conductors and cut capacity.

The graph is an optimistic superset of legal routes: only foreign fixed pads
are obstacles. A source-target bridge must be used by every completion. Two
compulsory edges with conflicting artwork prove this interface row UNSAT.
Separated nets also need distinct bins in a physical conflict-clique cover
of the crossing edges; deficient matching supplies a capacity certificate.
No full source construction or complete path pool is required.
"""
from __future__ import annotations
import itertools
from .primitive_search import boxes_clear, wire_box


def mandatory_resource_conflict(problem, *, capacity_support=True):
    resources={}
    allowed_edges={}
    probes=0
    # Canonical resources prevent opposite edge listings from being counted twice.
    edges=tuple(sorted({tuple(sorted(edge)) for edge in problem.edges}))
    shapes={edge:wire_box(problem.nodes[edge[0]],problem.nodes[edge[1]],problem.wire_width) for edge in edges}
    for net,(start,target) in problem.terminals.items():
        obstacles=[wire_box(problem.nodes[node],problem.nodes[node],problem.terminal_pad_widths.get(node,problem.wire_width))
                   for other,pins in problem.terminals.items() if other!=net for node in pins]
        adjacency={node:set() for node in problem.nodes}
        allowed_edges[net]=set()
        for a,b in edges:
            probes+=1
            shape=shapes[(a,b)]
            if all(boxes_clear(shape,pad,problem.minimum_gap) for pad in obstacles):
                adjacency[a].add(b);adjacency[b].add(a)
                allowed_edges[net].add((a,b))
        # Iterative low-link traversal; no recursion limit or path enumeration.
        entered={start:0};low={start:0};parent={start:None}
        contains_target={start:start==target};forced=[]
        stack=[(start,iter(sorted(adjacency[start])))]
        while stack:
            node,neighbours=stack[-1]
            other=next(neighbours,None)
            if other is None:
                stack.pop();above=parent[node]
                if above is not None:
                    if contains_target[node] and low[node]>entered[above]:
                        forced.append((above,node))
                    low[above]=min(low[above],low[node])
                    contains_target[above] |= contains_target[node]
                continue
            if other==parent[node]:continue
            if other in entered:
                low[node]=min(low[node],entered[other]);continue
            parent[other]=node;entered[other]=len(entered);low[other]=entered[other]
            contains_target[other]=other==target
            stack.append((other,iter(sorted(adjacency[other]))))
        if target not in entered:
            return {'status':'UNSAT','kind':'fixed-pad-disconnection','net':net,'edge_probes':probes}
        resources[net]=forced
    for left,right in itertools.combinations(resources,2):
        for a,b in resources[left]:
            for c,d in resources[right]:
                if not boxes_clear(wire_box(problem.nodes[a],problem.nodes[b],problem.wire_width),
                                   wire_box(problem.nodes[c],problem.nodes[d],problem.wire_width),problem.minimum_gap):
                    return {'status':'UNSAT','kind':'mandatory-edge-conflict','net_pair':[left,right],
                            'compulsory_edges':[[a,b],[c,d]],'minimum_gap_nm':problem.minimum_gap,'edge_probes':probes}
    if capacity_support:
        proof=_cut_capacity_conflict(problem,edges,shapes,allowed_edges)
        if proof is not None:
            return {**proof,'edge_probes':probes}
    return {'status':'NO_CERTIFICATE','edge_probes':probes}


def _cut_capacity_conflict(problem,edges,shapes,allowed_edges):
    """Hall deficiency on a conservative physical clique cover of a graph cut.

    Every separated net needs a cut edge. Each bin consists of pairwise
    conflicting conductor shapes, so at most one distinct net can use it.
    A legal routing therefore induces an injective net-to-bin assignment.
    Failure of maximum bipartite matching is a sound necessary-condition
    certificate. A cover with too many bins only loses pruning power.
    """
    for axis in (0,1):
        coordinates=sorted({point[axis] for point in problem.nodes.values()})
        for left,right in zip(coordinates,coordinates[1:]):
            # Integer doubled coordinate avoids floating-point partitions.
            twice_cut=left+right
            side={node:2*point[axis]<twice_cut for node,point in problem.nodes.items()}
            demand=[net for net,(start,target) in problem.terminals.items() if side[start]!=side[target]]
            if len(demand)<2:continue
            crossing=[edge for edge in edges if side[edge[0]]!=side[edge[1]]
                      and any(edge in allowed_edges[net] for net in demand)]
            bins=[]
            for edge in crossing:
                bucket=next((bucket for bucket in bins if all(
                    not boxes_clear(shapes[edge],shapes[other],problem.minimum_gap) for other in bucket)),None)
                if bucket is None:
                    bins.append([edge])
                else:
                    bucket.append(edge)
            eligible={net:[index for index,bucket in enumerate(bins)
                           if any(edge in allowed_edges[net] for edge in bucket)] for net in demand}
            owner={}
            def augment(net,seen):
                for index in eligible[net]:
                    if index in seen:continue
                    seen.add(index)
                    if index not in owner or augment(owner[index],seen):
                        owner[index]=net
                        return True
                return False
            rank=sum(augment(net,set()) for net in demand)
            if rank<len(demand):
                return {'status':'UNSAT','kind':'cut-capacity-conflict','axis':axis,
                        'twice_cut_coordinate_nm':twice_cut,'separated_nets':demand,
                        'conflict_cliques':[[list(edge) for edge in bucket] for bucket in bins],
                        'eligible_cliques':eligible,'matching_rank':rank,'required_crossings':len(demand),
                        'minimum_gap_nm':problem.minimum_gap}
    return None
