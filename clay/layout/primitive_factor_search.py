"""Exact local factor elimination and reusable guarded primitive suffixes."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math


class GuardedSuffixPlanner:
    """Physical column costs with conserved inventory; no complete source expansion.

    Only guards on remaining variables enter the cache. Interface identities
    and boundary costs belong to each query, never to the internal suffix key.
    """
    def __init__(self, constructor, *, share=True):
        self.constructor = constructor
        self.fixed, self.groups, self.factors = constructor.local_factors()
        self.share = share
        self.cache = {}
        self.transitions = self.hits = self.states = 0
        self.future = [set(index for group in self.groups[depth:] for index in group)
                       for depth in range(len(self.groups)+1)]

    def solve(self, guard):
        if any(not 0 <= i < self.constructor.n or role not in {"a", "b"} for i, role in guard.items()):
            raise ValueError("Invalid primitive role guard")
        if not self.share:
            self.cache = {}

        def visit(depth, inventory, residual):
            key = depth, inventory, residual
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            self.states += 1
            remaining = 2*(len(self.groups)-depth)
            if inventory > self.constructor.a_orbits or inventory+remaining < self.constructor.a_orbits:
                value = math.inf, 0, None
            elif depth == len(self.groups):
                value = (0, 1, None) if inventory == self.constructor.a_orbits else (math.inf, 0, None)
            else:
                conditions = dict(residual)
                best, count, reconstruction = math.inf, 0, None
                for labels in itertools.product("ab", repeat=2):
                    self.transitions += 1
                    if self.constructor.first_role and 0 in self.groups[depth] and labels[self.groups[depth].index(0)] != self.constructor.first_role:
                        continue
                    if any(i in conditions and conditions[i] != role for i, role in zip(self.groups[depth], labels)):
                        continue
                    child_residual = tuple((i, role) for i, role in residual if i in self.future[depth+1])
                    child_key = depth+1, inventory+labels.count("a"), child_residual
                    cost, ways, _ = visit(*child_key)
                    count += ways
                    cost += self.factors[depth][labels]
                    if cost < best:
                        best, reconstruction = cost, (labels, child_key)
                value = best, count, reconstruction
            self.cache[key] = value
            return value

        root = 0, 0, tuple(sorted(guard.items()))
        cost, count, _ = visit(*root)
        if not count:
            return dict(status="UNSAT", count=0, cost_nm2=None, assignment=None)
        assignment = [None]*self.constructor.n
        state = root
        for indices in self.groups:
            labels, state = self.cache[state][2]
            for index, label in zip(indices, labels):
                assignment[index] = label
        return dict(status="OPTIMAL", count=count, cost_nm2=self.fixed+cost, assignment=tuple(assignment))


@dataclass(frozen=True)
class Factor:
    scope: tuple[str, ...]
    table: dict[tuple, int]


def eliminate_factors(domains, factors):
    """Exact min-sum elimination; enumerate only each induced local bucket."""
    domains = {name: tuple(values) for name, values in domains.items()}
    live = list(factors)
    pending = set(domains)
    records = []
    absent = object()
    metrics = dict(bucket_entries=0, maximum_bucket_entries=0, maximum_scope=0,
                   generated_message_entries=0, elimination_order=[])
    while pending:
        def scope_of(name):
            return set().union(*(set(factor.scope) for factor in live if name in factor.scope)) or {name}
        def weight(name):
            scope = scope_of(name)
            return math.prod(len(domains[v]) for v in scope), len(scope), name
        variable = min(pending, key=weight)
        bucket = [factor for factor in live if variable in factor.scope]
        scope = tuple(sorted(scope_of(variable)-{variable}))
        entries = len(domains[variable])*math.prod(len(domains[name]) for name in scope)
        metrics["maximum_bucket_entries"] = max(metrics["maximum_bucket_entries"], entries)
        metrics["maximum_scope"] = max(metrics["maximum_scope"], len(scope)+1)
        metrics["elimination_order"].append(variable)
        message, choices = {}, {}
        for values in itertools.product(*(domains[name] for name in scope)):
            assignment = dict(zip(scope, values))
            best, chosen = math.inf, absent
            for value in domains[variable]:
                metrics["bucket_entries"] += 1
                assignment[variable] = value
                cost = sum(factor.table.get(tuple(assignment[name] for name in factor.scope), math.inf)
                           for factor in bucket)
                if cost < best:
                    best, chosen = cost, value
            if chosen is not absent:
                message[values] = best
                choices[values] = chosen
        metrics["generated_message_entries"] += len(message)
        records.append((variable, scope, choices))
        live = [factor for factor in live if variable not in factor.scope]
        live.append(Factor(scope, message))
        pending.remove(variable)
    cost = sum(factor.table.get((), math.inf) for factor in live)
    if cost == math.inf:
        return dict(status="UNSAT", objective=None, assignment={}, **metrics)
    assignment = {}
    for variable, scope, choices in reversed(records):
        assignment[variable] = choices[tuple(assignment[name] for name in scope)]
    return dict(status="OPTIMAL", objective=cost, assignment=assignment, **metrics)
