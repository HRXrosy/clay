"""Experimental primitive-family and attributed group search for tiny cases.

Coordinates are integer nanometres. This module is a research kernel; existing
production engines are not redirected to it. Paths are simple paths in an
explicit finite rectilinear graph, with one wire width and one routing layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
import heapq
import itertools
from typing import Iterable

Point = tuple[int, int]
Rect = tuple[int, int, int, int]


def wire_box(a: Point, b: Point, width: int) -> Rect:
    if a[0] != b[0] and a[1] != b[1]:
        raise ValueError("Only rectilinear edges are supported")
    half = width // 2
    return min(a[0], b[0]) - half, min(a[1], b[1]) - half, max(a[0], b[0]) + half, max(a[1], b[1]) + half


def boxes_clear(a: Rect, b: Rect, gap: int) -> bool:
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    # Contact is forbidden even with a requested zero clearance.
    return (dx > 0 or dy > 0) and dx * dx + dy * dy >= gap * gap


def path_length(path: tuple[str, ...], nodes: dict[str, Point]) -> int:
    return sum(abs(nodes[a][0] - nodes[b][0]) + abs(nodes[a][1] - nodes[b][1]) for a, b in zip(path, path[1:]))


@dataclass(frozen=True)
class GraphProblem:
    nodes: dict[str, Point]
    edges: tuple[tuple[str, str], ...]
    terminals: dict[str, tuple[str, str]]
    length_caps: dict[str, int]
    length_pairs: tuple[tuple[str, str], ...] = ()
    tol_rel: Fraction = Fraction(1, 10)
    wire_width: int = 400
    minimum_gap: int = 300
    terminal_pad_widths: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.wire_width <= 0 or self.wire_width % 2 or self.minimum_gap < 0:
            raise ValueError("Positive even width and nonnegative spacing required")
        if not 0 <= self.tol_rel < 1:
            raise ValueError("Relative length tolerance must lie in [0,1)")
        for a, b in self.edges:
            if a == b or a not in self.nodes or b not in self.nodes:
                raise ValueError("Invalid graph edge")
            if self.nodes[a] == self.nodes[b]:
                raise ValueError("Graph edges must have positive physical length")
            wire_box(self.nodes[a], self.nodes[b], self.wire_width)
        for net, (a, b) in self.terminals.items():
            if a == b or a not in self.nodes or b not in self.nodes or self.length_caps[net] <= 0:
                raise ValueError("Invalid terminal/cap")
        if any(a not in self.terminals or b not in self.terminals for a, b in self.length_pairs):
            raise ValueError("Length factor references an unknown network")
        if any(node not in self.nodes or width <= 0 or width % 2 for node, width in self.terminal_pad_widths.items()):
            raise ValueError("Invalid physical terminal pad")

    def adjacency(self) -> dict[str, tuple[str, ...]]:
        result = {key: [] for key in self.nodes}
        for a, b in self.edges:
            result[a].append(b)
            result[b].append(a)
        return {key: tuple(sorted(values)) for key, values in result.items()}


@dataclass
class SearchResult:
    status: str
    paths: dict[str, tuple[str, ...]] = field(default_factory=dict)
    objective_nm: int | None = None
    expanded: int = 0
    geometry_rejections: int = 0
    attribute_prunes: int = 0
    residual_prunes: int = 0
    spectrum_prunes: int = 0
    memo_prunes: int = 0
    residual_edge_probes: int = 0
    spectrum_walk_states: int = 0
    complete_assignments: int = 0
    state_limit: int = 0
    reason: str = ""

    def payload(self, problem: GraphProblem) -> dict:
        return {**self.__dict__, "paths": {net: list(path) for net, path in self.paths.items()},
                "lengths_nm": {net: path_length(path, problem.nodes) for net, path in self.paths.items()}}


def _distances(problem: GraphProblem, target: str) -> dict[str, int]:
    adjacency = problem.adjacency()
    distance = {target: 0}
    queue = [(0, target)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distance[node]:
            continue
        for other in adjacency[node]:
            new = cost + abs(problem.nodes[node][0] - problem.nodes[other][0]) + abs(problem.nodes[node][1] - problem.nodes[other][1])
            if new < distance.get(other, 10**30):
                distance[other] = new
                heapq.heappush(queue, (new, other))
    return distance


def _extension_legal(problem: GraphProblem, paths: dict[str, tuple[str, ...]], net: str, node: str) -> bool:
    path = paths[net]
    if node in path:
        return False
    if len(path) >= 2:
        previous, tip, following = (problem.nodes[key] for key in (path[-2], path[-1], node))
        incoming = (tip[0]-previous[0], tip[1]-previous[1])
        outgoing = (following[0]-tip[0], following[1]-tip[1])
        if incoming[0]*outgoing[0] + incoming[1]*outgoing[1] < 0:
            return False  # Adjacent collinear reversal duplicates physical wire.
    box = wire_box(problem.nodes[path[-1]], problem.nodes[node], problem.wire_width)
    for other, other_path in paths.items():
        if other == net:
            continue
        # Foreign terminals remain reserved, including targets not yet reached.
        points = set(other_path) | set(problem.terminals[other])
        for point in points:
            if not boxes_clear(box, wire_box(problem.nodes[point], problem.nodes[point], problem.terminal_pad_widths.get(point, problem.wire_width)), problem.minimum_gap):
                return False
        for a, b in zip(other_path, other_path[1:]):
            if not boxes_clear(box, wire_box(problem.nodes[a], problem.nodes[b], problem.wire_width), problem.minimum_gap):
                return False
    # Reject nonadjacent self-contact: otherwise graph labels can conceal a loop
    # or shortcut in the physical centreline and corrupt the length attribute.
    for a, b in zip(path[:-2], path[1:-1]):
        if not boxes_clear(box, wire_box(problem.nodes[a], problem.nodes[b], problem.wire_width), 0):
            return False
    return True


def _lengths_satisfied(problem: GraphProblem, paths: dict[str, tuple[str, ...]]) -> bool:
    lengths = {net: path_length(path, problem.nodes) for net, path in paths.items()}
    return all(abs(lengths[a] - lengths[b]) <= problem.tol_rel * max(lengths[a], lengths[b]) for a, b in problem.length_pairs)


def _ceil(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _attribute_possible(problem: GraphProblem, paths: dict[str, tuple[str, ...]], distance: dict[str, dict[str, int]]) -> bool:
    intervals = {}
    for net, path in paths.items():
        length = path_length(path, problem.nodes)
        if path[-1] == problem.terminals[net][1]:
            intervals[net] = [length, length]
        else:
            remaining = distance[net].get(path[-1])
            if remaining is None:
                return False
            intervals[net] = [length + remaining, problem.length_caps[net]]
        if intervals[net][0] > intervals[net][1]:
            return False
    # Exact integer rounding. Pair tolerances do NOT imply exact transitive
    # equality; propagate the actual relative-tolerance interval relation.
    factor = 1 - problem.tol_rel
    changed = True
    while changed:
        changed = False
        for a, b in problem.length_pairs:
            for source, target in ((a, b), (b, a)):
                lo = max(intervals[target][0], _ceil(intervals[source][0] * factor))
                hi_value = Fraction(intervals[source][1]) / factor
                hi = min(intervals[target][1], hi_value.numerator // hi_value.denominator)
                if lo > hi:
                    return False
                if [lo, hi] != intervals[target]:
                    intervals[target] = [lo, hi]
                    changed = True
    return True


def _residual_domains(problem, paths, adjacency, *, spectra=False, spectrum_limit=20_000, metrics=None):
    """Optimistic suffix graphs: retain all edges of every legal continuation.

    Reserve foreign history/terminals and own visited nodes. Own final edge is
    omitted from obstacle checks for ALL suffix edges: a deliberate relaxation,
    needed because the first suffix edge can touch it. Walks may revisit suffix
    nodes; their attainable lengths therefore overapproximate simple paths.
    """
    domains, distances = {}, {}
    for net, path in paths.items():
        tip, target = path[-1], problem.terminals[net][1]
        used = path_length(path, problem.nodes)
        if tip == target:
            domains[net], distances[net] = {used}, {tip: 0}
            continue
        blocked_nodes = set(path[:-1])
        allowed = {node: [] for node in adjacency if node not in blocked_nodes}
        for a, b in problem.edges:
            if metrics is not None:
                metrics["edge_probes"] = metrics.get("edge_probes", 0) + 1
            if a not in allowed or b not in allowed:
                continue
            box = wire_box(problem.nodes[a], problem.nodes[b], problem.wire_width)
            clear = True
            for other, history in paths.items():
                if other == net:
                    obstacles = [(wire_box(problem.nodes[x], problem.nodes[y], problem.wire_width), 0)
                                 for x, y in zip(path[:-2], path[1:-1])]
                else:
                    obstacles = [(wire_box(problem.nodes[node], problem.nodes[node], problem.terminal_pad_widths.get(node, problem.wire_width)), problem.minimum_gap)
                                 for node in set(history) | set(problem.terminals[other])]
                    obstacles += [(wire_box(problem.nodes[x], problem.nodes[y], problem.wire_width), problem.minimum_gap)
                                  for x, y in zip(history, history[1:])]
                if any(not boxes_clear(box, obstacle, gap) for obstacle, gap in obstacles):
                    clear = False
                    break
            if clear:
                cost = abs(problem.nodes[a][0] - problem.nodes[b][0]) + abs(problem.nodes[a][1] - problem.nodes[b][1])
                if cost <= 0:
                    raise ValueError("Residual length analysis requires positive graph edges")
                allowed[a].append((b, cost))
                allowed[b].append((a, cost))
        distance, queue = {target: 0}, [(0, target)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != distance[node]:
                continue
            for other, edge_cost in allowed.get(node, ()):
                new = cost + edge_cost
                if new < distance.get(other, 10**30):
                    distance[other] = new
                    heapq.heappush(queue, (new, other))
        distances[net] = distance
        cap = problem.length_caps[net] - used
        if tip not in distance or distance[tip] > cap:
            return None, distances
        if not spectra:
            continue
        # Bounded positive-cost walk DP; target is absorbing. On resource limit
        # fall back to no spectrum for this role, never to a false empty set.
        pending, seen, lengths = [(0, tip)], {(tip, 0)}, set()
        overflow = False
        while pending:
            cost, node = heapq.heappop(pending)
            if metrics is not None:
                metrics["walk_states"] = metrics.get("walk_states", 0) + 1
            if node == target:
                lengths.add(used + cost)
                continue
            for other, edge_cost in allowed.get(node, ()):
                new = cost + edge_cost
                state = (other, new)
                if new + distance.get(other, 10**30) <= cap and state not in seen:
                    seen.add(state)
                    heapq.heappush(pending, (new, other))
            if len(seen) > spectrum_limit:
                overflow = True
                break
        domains[net] = None if overflow else lengths
    return domains, distances


def _spectra_supported(problem, domains):
    """Arc consistency on declared pair factors, with actual integer lengths."""
    domains = {net: None if values is None else set(values) for net, values in domains.items()}
    if any(values == set() for values in domains.values()):
        return False
    changed = True
    while changed:
        changed = False
        for a, b in problem.length_pairs:
            for source, target in ((a, b), (b, a)):
                if domains[source] is None or domains[target] is None:
                    continue
                supported = {x for x in domains[target] if any(abs(x-y) <= problem.tol_rel * max(x, y) for y in domains[source])}
                if not supported:
                    return False
                if supported != domains[target]:
                    domains[target] = supported
                    changed = True
    return True


def search_group(problem: GraphProblem, *, attribute_pruning: bool = True, residual_pruning: bool = False,
                 spectrum_pruning: bool = False, memoization: str = "history", order: Iterable[str] | None = None,
                 state_limit: int = 200_000, spectrum_limit: int = 20_000) -> SearchResult:
    """Grow related paths in one reversible state; optimize total wire length.

    One edge is grown per action; all physical histories and shared net identities
    remain in the state. The scheduler interleaves unfinished roles. No complete
    route candidates are generated before this search.
    """
    nets = tuple(order or problem.terminals)
    if state_limit < 1 or spectrum_limit < 0:
        raise ValueError("Positive search budget and nonnegative spectrum budget required")
    if memoization not in {"none", "history", "endpoints_unsafe"}:
        raise ValueError("Unknown state equivalence")
    if set(nets) != set(problem.terminals) or len(nets) != len(problem.terminals):
        raise ValueError("Order must contain each network exactly once")
    adjacency = problem.adjacency()
    distance = {net: _distances(problem, problem.terminals[net][1]) for net in nets}
    result = SearchResult("UNSAT", state_limit=state_limit)
    exhausted = False
    initial = {net: (problem.terminals[net][0],) for net in nets}
    seen = set()

    def visit(paths: dict[str, tuple[str, ...]], cursor: int) -> None:
        nonlocal exhausted
        if exhausted:
            return
        if result.expanded >= state_limit:
            exhausted = True
            return
        result.expanded += 1
        lengths = {net: path_length(path, problem.nodes) for net, path in paths.items()}
        if memoization != "none":
            key = (cursor, tuple((net, paths[net] if memoization == "history" else (paths[net][-1], lengths[net])) for net in nets))
            if key in seen:
                result.memo_prunes += 1
                return
            seen.add(key)
        if any(lengths[net] > problem.length_caps[net] for net in nets):
            return
        if attribute_pruning and not _attribute_possible(problem, paths, distance):
            result.attribute_prunes += 1
            return
        active_distance = distance
        if residual_pruning or spectrum_pruning:
            metrics = {}
            domains, active_distance = _residual_domains(problem, paths, adjacency, spectra=spectrum_pruning, spectrum_limit=spectrum_limit, metrics=metrics)
            result.residual_edge_probes += metrics.get("edge_probes", 0)
            result.spectrum_walk_states += metrics.get("walk_states", 0)
            if domains is None:
                result.residual_prunes += 1
                return
            if attribute_pruning and not _attribute_possible(problem, paths, active_distance):
                result.attribute_prunes += 1
                return
            if spectrum_pruning and not _spectra_supported(problem, domains):
                result.spectrum_prunes += 1
                return
        unfinished = {net for net in nets if paths[net][-1] != problem.terminals[net][1]}
        lower = sum(lengths[net] + (active_distance[net].get(paths[net][-1], 10**30) if net in unfinished else 0) for net in nets)
        if result.objective_nm is not None and lower >= result.objective_nm:
            return
        if not unfinished:
            result.complete_assignments += 1
            if _lengths_satisfied(problem, paths):
                result.paths = dict(paths)
                result.objective_nm = sum(lengths.values())
            return
        index = next((cursor + offset) % len(nets) for offset in range(len(nets)) if nets[(cursor + offset) % len(nets)] in unfinished)
        net = nets[index]
        current = paths[net][-1]
        next_nodes = sorted(adjacency[current], key=lambda node: (abs(problem.nodes[node][0] - problem.nodes[current][0]) + abs(problem.nodes[node][1] - problem.nodes[current][1]) + distance[net].get(node, 10**30), node))
        for node in next_nodes:
            if node in paths[net]:
                continue
            if not _extension_legal(problem, paths, net, node):
                result.geometry_rejections += 1
                continue
            updated = {**paths, net: (*paths[net], node)}
            visit(updated, (index + 1) % len(nets))

    visit(initial, 0)
    result.status = "UNKNOWN" if exhausted else ("OPTIMAL" if result.paths else "UNSAT")
    result.reason = "State budget reached; incumbent may exist" if exhausted else "Exhausted declared finite simple-path domain"
    if memoization == "endpoints_unsafe":
        result.reason = "UNSOUND ablation: endpoint/length merging can discard feasible histories; status is not a certificate"
    return result


def greedy_route(problem: GraphProblem, *, order: Iterable[str] | None = None) -> SearchResult:
    """Same geometry/graph/domain, with irreversible shortest-path commitment."""
    nets = tuple(order or problem.terminals)
    paths = {net: (problem.terminals[net][0],) for net in problem.terminals}
    result = SearchResult("BLOCKED")
    for net in nets:
        queue = [(0, paths[net])]
        winner = None
        while queue:
            length, path = heapq.heappop(queue)
            result.expanded += 1
            if length > problem.length_caps[net]:
                continue
            if path[-1] == problem.terminals[net][1]:
                winner = path
                break
            for node in problem.adjacency()[path[-1]]:
                if node not in path and _extension_legal(problem, {**paths, net: path}, net, node):
                    grown = (*path, node)
                    heapq.heappush(queue, (path_length(grown, problem.nodes), grown))
        if winner is None:
            result.paths = {key: value for key, value in paths.items() if len(value) > 1}
            result.reason = f"Irreversible commitment blocked {net}"
            return result
        paths[net] = winner
    result.paths = paths
    result.objective_nm = sum(path_length(path, problem.nodes) for path in paths.values())
    result.status = "FEASIBLE" if _lengths_satisfied(problem, paths) else "INVALID"
    result.reason = "Final length factors evaluated after individual commitments"
    return result


@dataclass(frozen=True)
class FamilyMode:
    name: str
    first_member: str
    units: tuple[tuple[str, str, Point], ...]
    conductors: dict[str, tuple[Rect, ...]]
    layers: dict[str, int]
    outline: Rect = (-1000, -1000, 13000, 5000)


def generate_cc_assignments(unit_counts: dict[str, int], slots: tuple[Point, ...], *, first_role: str = ""):
    """Construct assignments by conserved reflection orbits, not canned modes.

    Scope: equal-weight units, point-reflected slots, no fixed centre slot.
    Assign each two-slot orbit to one role and decrement its remaining count.
    Every emitted role has the shared centroid by construction. This enumerates
    the declared point-mirrored family, not all possible common-centroid arrays.
    """
    if not slots or len(set(slots)) != len(slots) or sum(unit_counts.values()) != len(slots):
        raise ValueError("Slot inventory must equal the source unit inventory")
    if any(type(count) is not int or count <= 0 or count % 2 for count in unit_counts.values()) or first_role and first_role not in unit_counts:
        raise ValueError("Reflection pairs require positive even role counts")
    center = (min(x for x, y in slots) + max(x for x, y in slots), min(y for x, y in slots) + max(y for x, y in slots))
    indices = {point: index for index, point in enumerate(slots)}
    orbits = []
    for index, (x, y) in enumerate(slots):
        mate = indices.get((center[0]-x, center[1]-y))
        if mate is None or mate == index:
            raise ValueError("Every slot needs a distinct reflected mate")
        if index < mate:
            orbits.append((index, mate))
    labels = [""] * len(slots)
    remaining = dict(unit_counts)
    def assign(index):
        if index == len(orbits):
            yield tuple(labels)
            return
        left, right = orbits[index]
        for role in sorted(remaining):
            if remaining[role] < 2 or first_role and left == 0 and role != first_role:
                continue
            labels[left] = labels[right] = role
            remaining[role] -= 2
            yield from assign(index + 1)
            remaining[role] += 2
    yield from assign(0)


def materialize_matching_family(unit_counts=None, slots=None, *, first_member="", escape_extensions_nm=(0,)):
    """Emit local artwork lazily for the generated equal-weight two-row family."""
    unit_counts = {"a": 2, "b": 2} if unit_counts is None else unit_counts
    slots = ((0, 0), (4000, 0), (0, 4000), (4000, 4000)) if slots is None else tuple(slots)
    if not slots:
        raise ValueError("Explicit empty slot family is invalid")
    if not escape_extensions_nm or any(type(value) is not int or value < 0 for value in escape_extensions_nm) or len(set(escape_extensions_nm)) != len(escape_extensions_nm):
        raise ValueError("Escape extensions must be distinct nonnegative integer lengths")
    if set(unit_counts) != {"a", "b"} or {y for x, y in slots} != {0, 4000}:
        raise ValueError("This local conductor constructor supports two roles/two rows")
    first_role = {"MA": "a", "MB": "b"}.get(first_member, "")
    if first_member and not first_role:
        raise ValueError("Unknown source first member")
    endpoint_b = max(x for x, y in slots) + 6000
    for index, labels in enumerate(generate_cc_assignments(unit_counts, tuple(slots), first_role=first_role)):
        first = "MA" if labels[0] == "a" else "MB"
        occurrence = {"a": 0, "b": 0}
        units = []
        for label, point in zip(labels, slots):
            units.append((f"{label}{occurrence[label]}", label, point))
            occurrence[label] += 1
        polygons = {"a": [], "b": []}
        for _identity, net, (x, y) in units:
            endpoint = endpoint_b + (2000 if net == "a" and first == "MB" else 0)
            polygons[net].append(wire_box((x, y), (x, 2000), 600))
            polygons[net].append(wire_box((x, 2000), (endpoint, 2000), 600))
        suffix = "" if len(slots) == 4 else "_" + "".join(labels)
        base_endpoint_a = endpoint_b + (2000 if first == "MB" else 0)
        # Exposure extensions are explicit construction variables; source CC and
        # first_member constrain units, not an artificially frozen local bus.
        for extension in escape_extensions_nm:
            conductors = {net: tuple(boxes) for net, boxes in polygons.items()}
            if extension:
                conductors["a"] += (wire_box((base_endpoint_a, 2000), (base_endpoint_a + extension, 2000), 600),)
            name = f"phase_{first}{suffix}" + (f"_escape{extension}" if extension else "")
            yield FamilyMode(name, first, tuple(units), conductors, {"a": 1, "b": 2},
                             (-1000, -1000, endpoint_b + max(2000, max(escape_extensions_nm)+2000) + 1000, 5000))


def matching_family() -> tuple[FamilyMode, ...]:
    return tuple(materialize_matching_family())


def enumerate_accesses(mode: FamilyMode, sites: dict[str, Point], *, pose: Point = (0, 0), gap: int = 400):
    """Derive availability from emitted lower-layer polygons, not a site table."""
    import klayout.db as db
    availability = {}
    for net, boxes in mode.conductors.items():
        region = db.Region()
        for box in boxes:
            region.insert(db.Box(*box))
        available = []
        for site, (x, y) in sites.items():
            lower_pad = db.Region(db.Box(x - 300, y - 300, x + 300, y + 300))
            pad_box = (x - 300, y - 300, x + 300, y + 300)
            # A stacked transition to M3 also materializes intermediate metal.
            # Foreign fixed artwork there must be checked before admitting access.
            intermediate_clear = all(boxes_clear(pad_box, obstacle, gap)
                                     for foreign, obstacles in mode.conductors.items() if foreign != net
                                     and mode.layers[net] < mode.layers[foreign] < 3 for obstacle in obstacles)
            if (lower_pad - region).is_empty() and intermediate_clear:
                available.append(site)
        availability[net] = available
    for chosen in itertools.product(*(availability[net] for net in availability)):
        if len(set(chosen)) != len(chosen):
            continue
        pads = [wire_box(sites[site], sites[site], 600) for site in chosen]
        if all(boxes_clear(a, b, gap) for a, b in itertools.combinations(pads, 2)):
            yield {"mode": mode.name, "pose_nm": pose, "selected_sites": dict(zip(availability, chosen)), "availability": availability,
                    "units": [(identity, net, (point[0] + pose[0], point[1] + pose[1])) for identity, net, point in mode.units],
                    "conductors": {net: [(x0 + pose[0], y0 + pose[1], x1 + pose[0], y1 + pose[1]) for x0, y0, x1, y1 in boxes] for net, boxes in mode.conductors.items()},
                    "layers": mode.layers, "outline_nm": (mode.outline[0] + pose[0], mode.outline[1] + pose[1], mode.outline[2] + pose[0], mode.outline[3] + pose[1]),
                    "via_program": [{"net": net, "site": site, "landing_layers": list(range(mode.layers[net], 4)),
                                     "landing_width_nm": 600, "cut_width_nm": 300, "enclosure_nm": 150} for net, site in zip(availability, chosen)],
                    "sites_nm": {key: (x + pose[0], y + pose[1]) for key, (x, y) in sites.items()}}


def allocate_accesses(mode: FamilyMode, sites: dict[str, Point], *, pose: Point = (0, 0), gap: int = 400) -> dict | None:
    return next(enumerate_accesses(mode, sites, pose=pose, gap=gap), None)


def search_family(sites: dict[str, Point], *, first_member: str = "", fixed_mode: bool = False, bbox_only_dominance: bool = False, poses: tuple[Point, ...] = ((0, 0), (6000, 0)), modes=None) -> dict:
    modes = (mode for mode in (modes if modes is not None else materialize_matching_family(first_member=first_member)) if not first_member or mode.first_member == first_member)
    if fixed_mode:
        modes = itertools.islice(modes, 1)
    if bbox_only_dominance:
        # Deliberately unsound ablation: equal outlines do not imply equivalent
        # electrical access interfaces. The case demonstrates the lost solution.
        representatives = {}
        for mode in modes:
            representatives.setdefault(mode.outline, mode)
        modes = list(representatives.values())
    attempts = 0
    generated = 0
    for mode in modes:
        generated += 1
        for pose in poses:
            attempts += 1
            chosen = allocate_accesses(mode, sites, pose=pose)
            if chosen is not None:
                return {"status": "FEASIBLE", "attempts": attempts, "generated_modes": generated, **chosen}
    return {"status": "UNSAT", "attempts": attempts, "generated_modes": generated, "reason": "Exhausted declared local mode/pose/access domain"}


def search_coupled_family(problem: GraphProblem, sites: dict[str, Point], *, first_member: str = "", fixed_mode: bool = False,
                          greedy: bool = False, attribute_pruning: bool = True, state_limit: int = 200_000,
                          first_access_only: bool = False, interface_pruning: bool = True,
                          residual_pruning: bool = False, spectrum_pruning: bool = False,
                          escape_extensions_nm=(0,)) -> dict:
    """Joint local mode, actual escape and attributed route search.

    Site names are graph node identities; targets stay fixed. Every legal access
    assignment is retained until its route obligations have been solved.
    """
    if any(problem.nodes.get(site) != point for site, point in sites.items()):
        raise ValueError("Site geometry must match the routing graph")
    modes = materialize_matching_family(first_member=first_member, escape_extensions_nm=escape_extensions_nm)
    if fixed_mode:
        modes = itertools.islice(modes, 1)
    best, trials, unknown = None, [], False
    interface_work = {"edge_probes": 0, "walk_states": 0}
    for mode in modes:
        accesses = enumerate_accesses(mode, sites)
        if first_access_only:
            accesses = itertools.islice(accesses, 1)
        has_access = False
        for family in accesses:
            has_access = True
            scoped = replace(problem, terminals={net: (family["selected_sites"][net], target) for net, (_start, target) in problem.terminals.items()},
                             terminal_pad_widths={**problem.terminal_pad_widths, **{site: 600 for site in sites}})
            initial = {net: (start,) for net, (start, _target) in scoped.terminals.items()}
            if interface_pruning and not greedy and not first_access_only:
                domains, distances = _residual_domains(scoped, initial, scoped.adjacency(), spectra=spectrum_pruning, metrics=interface_work)
                lower = sum(distances.get(net, {}).get(initial[net][0], 10**30) for net in initial)
                impossible = domains is None or attribute_pruning and not _attribute_possible(scoped, initial, distances)
                impossible |= domains is not None and spectrum_pruning and not _spectra_supported(scoped, domains)
                if impossible or best is not None and lower >= best["objective_nm"]:
                    trials.append({"mode": mode.name, "accesses": family["selected_sites"], "stage": "interface_bound",
                                   "status": "PRUNED", "lower_bound_nm": lower, "reason": "Relaxed suffix infeasible" if impossible else "Incumbent bound", "expanded": 0})
                    continue
            routed = greedy_route(scoped) if greedy else search_group(scoped, attribute_pruning=attribute_pruning, state_limit=state_limit,
                                                                       residual_pruning=residual_pruning, spectrum_pruning=spectrum_pruning)
            trials.append({"mode": mode.name, "accesses": family["selected_sites"], "stage": "routing", **routed.payload(scoped)})
            unknown |= routed.status == "UNKNOWN"
            if routed.status not in {"OPTIMAL", "FEASIBLE", "UNKNOWN"} or not routed.paths:
                continue
            if best is None or routed.objective_nm < best["objective_nm"]:
                best = {"family": family, "route": routed.payload(scoped), "objective_nm": routed.objective_nm}
        if not has_access:
            trials.append({"mode": mode.name, "status": "UNSAT", "stage": "access", "scope": "Local access domain"})
    return {"status": "UNKNOWN" if unknown else (("FEASIBLE" if greedy else "OPTIMAL") if best else ("BLOCKED" if greedy else "UNSAT")),
            **(best or {}), "trials": trials, "expanded": sum(trial.get("expanded", 0) for trial in trials),
            "interface_prunes": sum(trial["stage"] == "interface_bound" for trial in trials),
            "interface_analysis_work": interface_work,
            "objective_scope": "External graph centreline length; local conductor cost not included",
            "domain_scope": "First legal access only (restricted ablation)" if first_access_only else "Full declared mode/access/simple-path domain",
            "reason": "No source condition relaxed; unresolved search budgets retain UNKNOWN"}
