"""Primitive-family decision diagrams with guarded physical interfaces.

Research implementation: a conserved two-row CC component constructor, exact
local metal factors, relational boundary support, and delayed materialization.
The interface is an exact separator for this constructor: internal conductors
use M1/M2, external paths use M3, and all M3 boundary pads are interface-owned.
Broader constructors must compile their own guards and boundary geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import itertools
import math

from .primitive_search import (FamilyMode, GraphProblem, boxes_clear, wire_box,
                               enumerate_accesses, search_group, _residual_domains,
                               _spectra_supported, path_length, SearchResult)
from .primitive_resource_certificates import mandatory_resource_conflict


def region_area(boxes):
    import klayout.db as db
    region = db.Region()
    for box in boxes:
        region.insert(db.Box(*box))
    region.merge()
    return region.area()


@dataclass(frozen=True)
class OrbitConstructor:
    """Source units assigned to reflection orbits, with geometric components.

    Orbit i comprises (i*pitch,0) and ((n-1-i)*pitch,row_height).
    A units connect on M1 to the upper collector; B on M2 to the lower
    collector. Component endpoints follow assigned unit positions; no phase
    conditional changes the electrical recipe.
    """
    n: int
    a_orbits: int
    first_role: str = ""
    pitch_nm: int = 4000
    row_height_nm: int = 4000
    collector_offset_nm: int = 2000
    local_width_nm: int = 600

    def __post_init__(self):
        if type(self.n) is not int or self.n < 2 or self.n % 2:
            raise ValueError("An even positive reflection-orbit inventory is required")
        if type(self.a_orbits) is not int or not 0 < self.a_orbits < self.n:
            raise ValueError("Both source roles need conserved units")
        if self.first_role not in {"", "a", "b"}:
            raise ValueError("Unknown first source role")
        if any(type(v) is not int or v <= 0 for v in (self.pitch_nm, self.row_height_nm, self.collector_offset_nm, self.local_width_nm)):
            raise ValueError("Positive integer construction geometry required")
        if self.local_width_nm % 2 or self.pitch_nm <= self.local_width_nm or self.collector_offset_nm <= self.local_width_nm:
            raise ValueError("Components must occupy separated columns and collectors")
        if self.local_width_nm != 600:
            raise ValueError("The current stack program has a 600 nm landing contract")

    @property
    def unit_counts(self):
        return {"a": 2*self.a_orbits, "b": 2*(self.n-self.a_orbits)}

    @property
    def slots(self):
        return tuple((i*self.pitch_nm, y) for y in (0, self.row_height_nm) for i in range(self.n))

    @property
    def sites(self):
        return {f"p{i}": (i*self.pitch_nm, self.row_height_nm) for i in range(self.n)}

    @property
    def layers(self):
        return {"a": 1, "b": 2}

    @property
    def outline(self):
        margin = 1000
        return (-margin, -self.collector_offset_nm-margin,
                (self.n-1)*self.pitch_nm+margin, self.row_height_nm+self.collector_offset_nm+margin)

    def orbit_points(self, index):
        return ((index*self.pitch_nm, 0), ((self.n-1-index)*self.pitch_nm, self.row_height_nm))

    def collector_y(self, role):
        return self.row_height_nm+self.collector_offset_nm if role == "a" else -self.collector_offset_nm

    def fixed_conductors(self, role):
        return (wire_box((0, self.collector_y(role)), ((self.n-1)*self.pitch_nm, self.collector_y(role)), self.local_width_nm),)

    def component(self, index, role):
        return tuple(wire_box(point, (point[0], self.collector_y(role)), self.local_width_nm) for point in self.orbit_points(index))

    def access_guard(self, net, site, gap=400):
        """Compile actual component coverage and blockers to role literals.

        A site has one designated unit landing component. All fixed and
        conditional foreign artwork on intervening layers must remain clear.
        No complete assignment or conductor family is generated here.
        """
        if site not in self.sites or net not in self.layers:
            raise ValueError("Unknown boundary role/site")
        pad = wire_box(self.sites[site], self.sites[site], self.local_width_nm)
        contributors = []
        for index in range(self.n):
            if any(box[0] <= pad[0] and box[1] <= pad[1] and box[2] >= pad[2] and box[3] >= pad[3] for box in self.component(index, net)):
                contributors.append(index)
        if not contributors:
            return None
        guard = {}
        for foreign, layer in self.layers.items():
            if foreign == net or not self.layers[net] < layer < 3:
                continue
            if any(not boxes_clear(pad, box, gap) for box in self.fixed_conductors(foreign)):
                return None
            for index in range(self.n):
                if any(not boxes_clear(pad, box, gap) for box in self.component(index, foreign)):
                    required = "b" if foreign == "a" else "a"
                    if index in guard and guard[index] != required:
                        return None
                    guard[index] = required
        if not any(guard.get(index) == net for index in contributors):
            possible = [index for index in contributors if index not in guard or guard[index] == net]
            if not possible:
                return None
            if len(possible) != 1:
                raise ValueError("Unresolved disjunctive component coverage requires an OR interface guard")
            guard[possible[0]] = net
        return guard

    def local_factors(self):
        """Exact metal factors after grouping reflected column incidences.

        Orbit i and orbit n-1-i share the same two physical columns. Distinct
        groups are disjoint outside the fixed collector trunks. Compute each
        factor from polygon union, not centreline or duplicated segment sums.
        """
        fixed = sum(region_area(self.fixed_conductors(role)) for role in self.layers)
        groups = tuple((i, self.n-1-i) for i in range(self.n//2))
        factors = []
        for left, right in groups:
            costs = {}
            for labels in itertools.product("ab", repeat=2):
                value = 0
                for role in self.layers:
                    base = self.fixed_conductors(role)
                    parts = tuple(box for index, label in zip((left, right), labels) if label == role for box in self.component(index, role))
                    value += region_area((*base, *parts)) - region_area(base)
                costs[labels] = value
            factors.append(costs)
        return fixed, groups, tuple(factors)

    def materialize(self, assignment):
        if len(assignment) != self.n or assignment.count("a") != self.a_orbits or any(role not in self.layers for role in assignment):
            raise ValueError("Assignment violates the conserved source inventory")
        if self.first_role and assignment[0] != self.first_role:
            raise ValueError("Assignment violates the hard first member")
        roles = {point: role for index, role in enumerate(assignment) for point in self.orbit_points(index)}
        occurrence = {"a": 0, "b": 0}
        units = []
        for point in self.slots:
            role = roles[point]
            units.append((f"{role}{occurrence[role]}", role, point))
            occurrence[role] += 1
        conductors = {role: (*self.fixed_conductors(role), *(box for index, label in enumerate(assignment) if label == role for box in self.component(index, role))) for role in self.layers}
        first = "MA" if assignment[0] == "a" else "MB"
        return FamilyMode("orbit_"+"".join(assignment), first, tuple(units), conductors, self.layers, self.outline)


class RelationDiagram:
    """Weighted shared DAG over inventory and surviving joint interface rows.

    Each row is a conjunction of orbit-role guards. A state retains the set of
    still-compatible rows as a bitmask; their marginal projections are never
    substituted for the relation. Complexity depends on interface width.
    """
    def __init__(self, constructor, guards):
        self.constructor = constructor
        self.guards = tuple(dict(guard) for guard in guards)
        if any(type(index) is not int or not 0 <= index < constructor.n or role not in {"a", "b"}
               for guard in self.guards for index, role in guard.items()):
            raise ValueError("Interface guards must refer to source orbits and declared roles")
        self.fixed_area, self.groups, self.factors = constructor.local_factors()
        # Compile row compatibility once; each transition intersects bitsets.
        self.compatibility = tuple({labels: sum(1 << bit for bit, guard in enumerate(self.guards)
            if all(index not in guard or guard[index] == label for index, label in zip(indices, labels)))
            for labels in itertools.product("ab", repeat=2)} for indices in self.groups)
        self.nodes, self.transitions = 0, 0
        self.edges = {}
        self._counts = {}
        self.root = (0, 0, (1 << len(self.guards))-1)
        self._build(*self.root)

    def _build(self, depth, used_a, live):
        state = (depth, used_a, live)
        if state in self._counts:
            return self._counts[state]
        self.nodes += 1
        remaining = 2*(len(self.groups)-depth)
        if not live or used_a > self.constructor.a_orbits or used_a+remaining < self.constructor.a_orbits:
            self._counts[state] = 0
            return 0
        if depth == len(self.groups):
            self._counts[state] = int(used_a == self.constructor.a_orbits)
            return self._counts[state]
        indices = self.groups[depth]
        arcs, count = [], 0
        for labels in itertools.product("ab", repeat=2):
            if self.constructor.first_role and 0 in indices and labels[indices.index(0)] != self.constructor.first_role:
                continue
            next_live = live & self.compatibility[depth][labels]
            child = (depth+1, used_a+labels.count("a"), next_live)
            self.transitions += 1
            ways = self._build(*child)
            if ways:
                arcs.append((labels, child, self.factors[depth][labels]))
                count += ways
        self.edges[(depth, used_a, live)] = tuple(arcs)
        self._counts[state] = count
        return count

    @property
    def count(self):
        return self._build(*self.root)

    def supports_assignment(self, assignment):
        if len(assignment) != self.constructor.n or any(role not in {"a", "b"} for role in assignment):
            return False
        state = self.root
        for indices in self.groups:
            labels = tuple(assignment[index] for index in indices)
            arc = next((arc for arc in self.edges.get(state, ()) if arc[0] == labels), None)
            if arc is None:
                return False
            state = arc[1]
        return bool(self._build(*state))

    def count_with_rows(self, active_mask):
        """Cofactor existing DAG after certified row deletion, without rebuilding."""
        self.cofactor_visits = 0
        @lru_cache(maxsize=None)
        def count(state):
            self.cofactor_visits += 1
            live = state[2]
            if not live & active_mask:
                return 0
            if live & active_mask == live:
                return self._counts[state]
            if state[0] == len(self.groups):
                return self._counts[state]
            return sum(count(child) for labels, child, cost in self.edges.get(state, ()))
        return count(self.root)

    def role_domains(self):
        domains = [set() for _ in range(self.constructor.n)]
        seen = set()
        def visit(state):
            if state in seen:
                return
            seen.add(state)
            for labels, child, cost in self.edges.get(state, ()):
                for index, label in zip(self.groups[state[0]], labels):
                    domains[index].add(label)
                visit(child)
        visit(self.root)
        return tuple(frozenset(domain) for domain in domains)

    def supported_rows(self, active_mask=None):
        rows = 0
        seen = set()
        def visit(state):
            nonlocal rows
            if state in seen:
                return
            seen.add(state)
            if state[0] == len(self.groups) and self._build(*state):
                rows |= state[2]
            for labels, child, cost in self.edges.get(state, ()):
                visit(child)
        visit(self.root)
        if active_mask is not None:
            rows &= active_mask
        return tuple(bit for bit in range(len(self.guards)) if rows & (1 << bit))

    def cheapest(self, boundary_costs):
        """Min-plus elimination of internal source decisions and boundary rows."""
        @lru_cache(maxsize=None)
        def best(state):
            if state[0] == len(self.groups):
                choices = [(cost, bit) for bit, cost in boundary_costs.items() if state[2] & (1 << bit)]
                if not choices or not self._build(*state):
                    return None
                cost, bit = min(choices)
                return cost, bit
            options = []
            for labels, child, cost in self.edges.get(state, ()):
                suffix = best(child)
                if suffix is not None:
                    options.append((cost+suffix[0], labels, child, suffix[1]))
            if not options:
                return None
            value, labels, child, bit = min(options)
            backtrack[state] = (labels, child)
            return value, bit
        backtrack = {}
        winner = best(self.root)
        if winner is None:
            return None
        assignment = [""] * self.constructor.n
        state = self.root
        for indices in self.groups:
            labels, state = backtrack[state]
            for index, role in zip(indices, labels):
                assignment[index] = role
        return {"objective_area_nm2": self.fixed_area+winner[0], "assignment": tuple(assignment), "row": winner[1]}

    def minimum_local_costs(self):
        """One shared min-plus pass: exact internal minimum for every row.

        No source assignments are enumerated or materialized. Prefixes with
        identical inventory/live-row states share all remaining decisions.
        """
        costs={self.root:self.fixed_area}
        result={}
        for state in sorted(self._counts,key=lambda state:state[0]):
            if state not in costs:continue
            value=costs[state]
            if state[0]==len(self.groups) and self._counts[state]:
                for bit in range(len(self.guards)):
                    if state[2] & (1<<bit):
                        result[bit]=min(result.get(bit,math.inf),value)
            for labels,child,cost in self.edges.get(state,()):
                costs[child]=min(costs.get(child,math.inf),value+cost)
        return result

    def payload(self, active_mask=None):
        count = self.count if active_mask is None else self.count_with_rows(active_mask)
        return {"nodes": self.nodes, "transitions": self.transitions, "accepted_assignments": count,
                "interface_rows": len(self.guards), "supported_rows": self.supported_rows(active_mask),
                "cofactor_visits": getattr(self, "cofactor_visits", 0),
                "state": "(column-pair depth, conserved A inventory, surviving interface-row mask)"}


def merge_guards(*guards):
    result = {}
    for guard in guards:
        if guard is None:
            return None
        for index, role in guard.items():
            if index in result and result[index] != role:
                return None
            result[index] = role
    return result


def scoped_interface(problem, constructor, selection):
    return replace(problem, terminals={net: (site, problem.terminals[net][1]) for net, site in selection.items()},
                   terminal_pad_widths={**problem.terminal_pad_widths, **{site: constructor.local_width_nm for site in selection.values()}})


def measure_total_metal(constructor, mode, family, scoped, paths):
    """All signal metal on M1/M2/M3, including local artwork and via landings."""
    local = sum(region_area(boxes) for boxes in mode.conductors.values())
    extra_intermediate = 0
    for row in family["via_program"]:
        net = row["net"]
        pad = wire_box(constructor.sites[row["site"]], constructor.sites[row["site"]], row["landing_width_nm"])
        for layer in row["landing_layers"]:
            if mode.layers[net] < layer < 3:
                extra_intermediate += region_area((pad,))
    external = 0
    for net, path in paths.items():
        shapes = [wire_box(scoped.nodes[a], scoped.nodes[b], scoped.wire_width) for a, b in zip(path, path[1:])]
        for node in scoped.terminals[net]:
            shapes.append(wire_box(scoped.nodes[node], scoped.nodes[node], scoped.terminal_pad_widths.get(node, scoped.wire_width)))
        external += region_area(shapes)
    return {"local_area_nm2": local, "intermediate_landing_area_nm2": extra_intermediate,
            "external_area_nm2": external, "total_area_nm2": local+extra_intermediate+external,
            "via_cut_count": sum(len(row["landing_layers"])-1 for row in family["via_program"])}


def interface_metal_cost(constructor, scoped, paths):
    """Boundary-only part of the exact area objective, independent of labels."""
    external = 0
    for net, path in paths.items():
        shapes = [wire_box(scoped.nodes[a], scoped.nodes[b], scoped.wire_width) for a, b in zip(path, path[1:])]
        for node in scoped.terminals[net]:
            shapes.append(wire_box(scoped.nodes[node], scoped.nodes[node], scoped.terminal_pad_widths.get(node, scoped.wire_width)))
        external += region_area(shapes)
    intermediate = sum((3-layer-1)*constructor.local_width_nm**2 for layer in constructor.layers.values())
    return external+intermediate


def _validate_separator(problem, constructor):
    if set(problem.terminals) != {"a", "b"}:
        raise ValueError("This constructor exports exactly two role networks")
    # Area is affine in centreline length when positive edges extend beyond
    # terminal pads and nonadjacent self-contact is forbidden. This permits the
    # existing group inner search to minimize the boundary metal objective.
    if any(abs(problem.nodes[a][0]-problem.nodes[b][0])+abs(problem.nodes[a][1]-problem.nodes[b][1]) < 2*constructor.local_width_nm for a, b in problem.edges):
        raise ValueError("Area separator requires edges longer than terminal pad extents")
    if problem.wire_width > constructor.local_width_nm:
        raise ValueError("Routing wire must fit the boundary pad")
    if any(width != problem.wire_width for node, width in problem.terminal_pad_widths.items() if node not in constructor.sites):
        raise ValueError("Additional non-boundary terminal shapes need an area-aware inner search")
    if any(target in constructor.sites for start, target in problem.terminals.values()):
        raise ValueError("External targets must lie outside the unit boundary-site inventory")
    for axis in (0, 1):
        coordinates = sorted({point[axis] for point in problem.nodes.values()})
        if any(b-a < 2*constructor.local_width_nm for a, b in zip(coordinates, coordinates[1:])):
            raise ValueError("Area-affine routing requires separated coordinate lines")
    for site, point in constructor.sites.items():
        if site in problem.nodes and problem.nodes[site] != point:
            raise ValueError("Boundary site and graph geometry differ")


def _boundary_area_lower_bound(problem, constructor, distances, domains):
    """Safe affine area lower bound under the validated physical separator."""
    lengths={net:min(domains[net]) if domains.get(net) else distances[net][start]
             for net,(start,target) in problem.terminals.items()}
    # Any matched completion has L_a >= ceil((1-tol)*L_b), and vice versa.
    changed=True
    while changed:
        changed=False
        for a,b in problem.length_pairs:
            for left,right in ((a,b),(b,a)):
                value=(1-problem.tol_rel)*lengths[right]
                bound=-(-value.numerator//value.denominator)
                if bound>lengths[left]:
                    lengths[left]=bound;changed=True
    width=problem.wire_width
    area=width*sum(lengths.values())
    for terminals in problem.terminals.values():
        area+=width**2
        for node in terminals:
            pad=problem.terminal_pad_widths.get(node,width)
            area+=pad**2-width*(pad+width)//2
    area+=sum((3-layer-1)*constructor.local_width_nm**2 for layer in constructor.layers.values())
    return area


def search_relational_family(problem, constructor, *, candidate_sites=None, joint_support=True, resource_support=True,
                             capacity_support=True, objective_bound=True, state_limit=200_000):
    """Route/layout support exchange without full internal-family enumeration.

    Compile component guards -> relaxed joint interface rows -> source decision
    diagram -> detailed route certificates -> source diagram contraction ->
    min-plus choice -> one complete physical materialization.
    """
    _validate_separator(problem, constructor)
    candidate_sites = candidate_sites or {net: tuple(site for site in constructor.sites if site in problem.nodes) for net in problem.terminals}
    rows, guards, relaxed_work = [], [], {"edge_probes": 0, "walk_states": 0}
    for combination in itertools.product(*(candidate_sites[net] for net in problem.terminals)):
        selection = dict(zip(problem.terminals, combination))
        if len(set(combination)) != len(combination):
            continue
        if not all(boxes_clear(wire_box(constructor.sites[a], constructor.sites[a], constructor.local_width_nm),
                               wire_box(constructor.sites[b], constructor.sites[b], constructor.local_width_nm), max(400, problem.minimum_gap))
                   for a, b in itertools.combinations(combination, 2)):
            continue
        guard = merge_guards(*(constructor.access_guard(net, site) for net, site in selection.items()))
        if guard is None:
            continue
        scoped = scoped_interface(problem, constructor, selection)
        initial = {net: (start,) for net, (start, target) in scoped.terminals.items()}
        domains, distances = _residual_domains(scoped, initial, scoped.adjacency(), spectra=joint_support and bool(scoped.length_pairs), metrics=relaxed_work)
        feasible = domains is not None and (not joint_support or _spectra_supported(scoped, domains))
        rows.append({"selection": selection, "guard": guard, "relaxed_status": "SUPPORTED" if feasible else "UNSAT", "relaxed_lengths_nm": {net: sorted(values) if values is not None else None for net, values in (domains or {}).items()},
                     "boundary_lower_area_nm2":_boundary_area_lower_bound(scoped,constructor,distances,domains) if feasible and objective_bound else None})
        if feasible:
            guards.append(guard)
    live_rows = [row for row in rows if row["relaxed_status"] == "SUPPORTED"]
    diagram = RelationDiagram(constructor, guards)
    initial_diagram = diagram.payload()
    routed_rows, boundary_costs = [], {}
    local_bounds=diagram.minimum_local_costs() if objective_bound else {}
    lower_bounds={bit:local_bounds[bit]+live_rows[bit]['boundary_lower_area_nm2'] for bit in local_bounds}
    ordered_rows=sorted(diagram.supported_rows(),key=lambda bit:(lower_bounds.get(bit,0),bit))
    active_mask = (1 << len(guards))-1
    feedback = []
    route_calls = 0
    incumbent=None
    bound_feedback=[]
    unknown_rows=[]
    for bit in ordered_rows:
        row = live_rows[bit]
        scoped = scoped_interface(problem, constructor, row["selection"])
        certificate = mandatory_resource_conflict(scoped,capacity_support=capacity_support) if resource_support else {'status':'NO_CERTIFICATE'}
        if certificate['status'] == 'UNSAT':
            routed = SearchResult('UNSAT', reason='Compulsory physical resources conflict in the optimistic interface graph')
        elif objective_bound and incumbent is not None and lower_bounds[bit]>=incumbent:
            # Dominated is an objective fact, not an infeasibility fact. Keep
            # the row and every source assignment it supports in the relation.
            bound_feedback.append({'row':bit,'lower_area_nm2':lower_bounds[bit],
                                   'incumbent_area_nm2':incumbent,'selection':row['selection']})
            continue
        else:
            route_calls += 1
            routed = search_group(scoped, residual_pruning=True, spectrum_pruning=joint_support and bool(scoped.length_pairs), state_limit=state_limit)
        record = {"row": bit, "selection": row["selection"], **routed.payload(scoped)}
        routed_rows.append(record)
        if certificate['status'] == 'UNSAT':
            record['resource_certificate'] = certificate
        if routed.status == "UNKNOWN":unknown_rows.append(bit)
        if routed.status == "UNSAT":
            before = diagram.count_with_rows(active_mask)
            active_mask &= ~(1 << bit)
            after = diagram.count_with_rows(active_mask)
            feedback.append({"row": bit, "assignments_before": before, "assignments_after": after,
                             "cofactor_visits": diagram.cofactor_visits, "certificate": "UNSAT",
                             "resource_witness": certificate if certificate['status']=='UNSAT' else None})
        if routed.paths:
            boundary_costs[bit] = interface_metal_cost(constructor, scoped, routed.paths)
            if objective_bound:
                value=local_bounds[bit]+boundary_costs[bit]
                incumbent=value if incumbent is None else min(incumbent,value)
    # Proven route UNSAT removes a guarded row; UNKNOWN keeps its potential.
    best = diagram.cheapest(boundary_costs)
    unresolved=[bit for bit in unknown_rows if not best or not objective_bound or lower_bounds[bit]<best['objective_area_nm2']]
    unknown=bool(unresolved)
    result = {"status": "UNKNOWN" if unknown else ("OPTIMAL" if best else "UNSAT"),
              "objective_scope": "Total M1/M2/M3 signal-metal area, including internal conductors and via landings",
              "source_assignment_count": math.comb(constructor.n-(1 if constructor.first_role else 0), constructor.a_orbits-(1 if constructor.first_role == "a" else 0)),
              "initial_relation": initial_diagram, "final_relation": diagram.payload(active_mask),
              "interface_trials": rows, "route_trials": routed_rows, "relaxed_analysis_work": relaxed_work,
              "feedback": feedback,
              "materializations": 0, "detailed_route_calls": route_calls,
              "resource_certificates": sum('resource_certificate' in row for row in routed_rows),
              "objective_bounds_nm2":lower_bounds,"objective_bound_feedback":bound_feedback,
              "objective_bound_prunes":len(bound_feedback),"unresolved_rows":unresolved,
              "expanded": sum(record["expanded"] for record in routed_rows),
              "reason": "Guarded component interface is an exact routing separator for the declared layer construction"}
    if best:
        bit = best["row"]
        row = live_rows[bit]
        mode = constructor.materialize(best["assignment"])
        family = next((family for family in enumerate_accesses(mode, constructor.sites) if family["selected_sites"] == row["selection"]), None)
        if family is None:
            raise AssertionError("Compiled interface guard failed physical replay")
        routed = next(record for record in routed_rows if record["row"] == bit)
        scoped = scoped_interface(problem, constructor, row["selection"])
        measured = measure_total_metal(constructor, mode, family, scoped, routed["paths"])
        if measured["total_area_nm2"] != best["objective_area_nm2"]:
            raise AssertionError("Decision-diagram objective disagrees with polygon union replay")
        result.update({**best, "family": family, "route": routed, "measured_cost": measured, "materializations": 1})
    return result


def enumerate_orbit_assignments(constructor):
    """Same-domain eager baseline; combinations enumerated only in this control."""
    for positions in itertools.combinations(range(constructor.n), constructor.a_orbits):
        chosen = set(positions)
        assignment = tuple("a" if index in chosen else "b" for index in range(constructor.n))
        if not constructor.first_role or assignment[0] == constructor.first_role:
            yield assignment


def search_eager_family(problem, constructor, *, candidate_sites=None, state_limit=200_000, cache_interfaces=False):
    """Correct same-domain baseline: full materialization + shared group routing."""
    _validate_separator(problem, constructor)
    candidate_sites = candidate_sites or {net: tuple(site for site in constructor.sites if site in problem.nodes) for net in problem.terminals}
    sites = {site: point for site, point in constructor.sites.items() if site in problem.nodes}
    result = {"status": "UNSAT", "materializations": 0, "detailed_route_calls": 0, "expanded": 0, "trials": []}
    unknown = False
    cache = {}
    for assignment in enumerate_orbit_assignments(constructor):
        mode = constructor.materialize(assignment)
        result["materializations"] += 1
        for family in enumerate_accesses(mode, sites):
            selection = family["selected_sites"]
            if any(selection[net] not in candidate_sites[net] for net in selection):
                continue
            scoped = scoped_interface(problem, constructor, selection)
            if not all(boxes_clear(wire_box(scoped.nodes[a], scoped.nodes[a], constructor.local_width_nm),
                                   wire_box(scoped.nodes[b], scoped.nodes[b], constructor.local_width_nm), max(400, problem.minimum_gap))
                       for a, b in itertools.combinations(selection.values(), 2)):
                continue
            key = tuple(sorted(selection.items()))
            if cache_interfaces and key in cache:
                routed = cache[key]
            else:
                routed = search_group(scoped, residual_pruning=True, spectrum_pruning=bool(scoped.length_pairs), state_limit=state_limit)
                result["detailed_route_calls"] += 1
                result["expanded"] += routed.expanded
                if cache_interfaces:
                    cache[key] = routed
            result["trials"].append({"assignment": assignment, "selection": selection, "status": routed.status})
            unknown |= routed.status == "UNKNOWN"
            if routed.paths:
                measured = measure_total_metal(constructor, mode, family, scoped, routed.paths)
                if "objective_area_nm2" not in result or measured["total_area_nm2"] < result["objective_area_nm2"]:
                    result.update({"assignment": assignment, "family": family, "route": routed.payload(scoped), "measured_cost": measured, "objective_area_nm2": measured["total_area_nm2"]})
    result["status"] = "UNKNOWN" if unknown else ("OPTIMAL" if "family" in result else "UNSAT")
    return result
