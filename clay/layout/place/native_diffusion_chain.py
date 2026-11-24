"""Compile source-bound native MOS groups into shared-diffusion net chains.

The compiler produces a topology candidate, not PDK artwork.  In particular, it
does not assert that a reference used the chosen orientation or finger order.
It consumes the existing SKY130 device contract for dimensions and preserves
each source device's gate, body and finger multiplicity.
"""

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class NativeDiffusionFinger:
    owner: str
    occurrence: int
    replica: int
    finger: int
    gate: str
    body: str
    left: str
    right: str


@dataclass(frozen=True)
class NativeDiffusionChain:
    source_identity: str
    model: str
    width_nm: int
    length_nm: int
    body: str
    fingers: tuple[NativeDiffusionFinger, ...]
    columns: tuple[str, ...]
    coverage_profile: tuple[str, str] | None = None
    access_profile: str = ""
    generation_profile: str = ""
    synthesis: str = ""
    tap_rail: str = ""


def compile_native_diffusion_chains(
    cell,
    members,
    *,
    synthesis="",
    tap_rail="",
    _specs=None,
    _source_identity=None,
):
    """Compile named SKY130 native MOS into connected source-net trails.

    Each MOS finger is an undirected edge between its actual D and S nets.
    Equal model, per-finger geometry and body form a compatible physical class;
    connected components with Euler trails form candidate continuous OD rows.
    Traversal is linear in fingers and does not search device permutations.
    Components with more than two odd-degree nets require an explicit break and
    are rejected rather than silently changing source connectivity.
    """
    from clay.layout.native_sky130 import native_source_identity
    from clay.pdk.sky130_native_geometry import native_cell_geometry_specs

    names = tuple(members)
    if not names or len(set(names)) != len(names):
        raise ValueError("native diffusion members must be nonempty and unique")
    devices = {item.instance_name: item for item in cell.devices}
    if not set(names) <= devices.keys():
        raise ValueError("native diffusion group contains an unknown source device")
    specs = native_cell_geometry_specs(cell) if _specs is None else _specs
    identity = native_source_identity(cell) if _source_identity is None else _source_identity
    classes = defaultdict(list)
    nf_by_owner = {}
    for name in names:
        device, spec = devices[name], specs[name]
        if spec["family"] != "mos":
            raise ValueError("native diffusion group contains a non-MOS device")
        pins = device.pins
        if any(not pins.get(k) for k in ("d", "g", "s", "b")):
            raise ValueError("native diffusion member requires all four source nets")
        key = (spec["model"], spec["w_nm"] // spec["nf"], spec["l_nm"], pins["b"])
        nf_by_owner[name] = spec["nf"]
        for index in range(spec["occurrences"]):
            classes[key].append((name, index, pins["d"], pins["s"], pins["g"]))

    result = []
    for (model, width, length, body), edges in sorted(classes.items()):
        if synthesis in ("parameterized", "cmos_gate"):
            # Place non-self-loop edges before self-loop (d == s) dummy edges on
            # interior traversals, while anchoring self-loop dummies at trail ends.
            edges.sort(key=lambda edge: (edge[2] == edge[3], edge[0], edge[1]))
        else:
            edges.sort(key=lambda edge: (edge[0], edge[1]))
        adjacency = defaultdict(list)
        for index, (_, _, drain, source, _) in enumerate(edges):
            adjacency[drain].append(index)
            adjacency[source].append(index)
        used = set()
        # Start at an odd node when one exists.  A stable source-derived order
        # makes each result reproducible without claiming reference identity.
        remaining = set(range(len(edges)))
        while remaining:
            seed = min(remaining)
            stack = [edges[seed][2]]
            nodes, component = set(), set()
            while stack:
                node = stack.pop()
                if node in nodes:
                    continue
                nodes.add(node)
                for edge in adjacency[node]:
                    component.add(edge)
                    _, _, left, right, _ = edges[edge]
                    stack.append(right if node == left else left)
            odd = sorted(node for node in nodes if len(adjacency[node]) % 2)
            if len(odd) not in (0, 2):
                raise ValueError("source diffusion component needs multiple OD chains")
            start = odd[0] if odd else min(nodes)
            cursors = {node: 0 for node in nodes}
            walk = [(start, None)]
            reverse = []
            while walk:
                node, incoming = walk[-1]
                options = adjacency[node]
                while cursors[node] < len(options) and options[cursors[node]] in used:
                    cursors[node] += 1
                if cursors[node] == len(options):
                    reverse.append(walk.pop())
                    continue
                edge = options[cursors[node]]
                used.add(edge)
                _, _, left, right, _ = edges[edge]
                walk.append((right if node == left else left, edge))
            trail = tuple(reversed(reverse))
            if len(trail) != len(component) + 1:
                raise ValueError("source diffusion trail did not cover its component")
            columns = tuple(node for node, _ in trail)
            fingers = tuple(NativeDiffusionFinger(
                owner=edges[edge][0], occurrence=edges[edge][1],
                replica=edges[edge][1] // nf_by_owner[edges[edge][0]],
                finger=edges[edge][1] % nf_by_owner[edges[edge][0]],
                gate=edges[edge][4], body=body,
                left=columns[i - 1], right=columns[i],
            ) for i, (_, edge) in enumerate(trail) if edge is not None)
            if synthesis in ("parameterized", "cmos_gate"):
                fingers, columns = _anchor_self_loop_dummies_at_ends(fingers, columns, body)
            result.append(NativeDiffusionChain(identity, model, width, length,
                                               body, fingers, columns,
                                               synthesis=synthesis, tap_rail=tap_rail))
            remaining -= component
    return tuple(result)


def _anchor_self_loop_dummies_at_ends(fingers, columns, body):
    """Rotate closed Euler circuits to ``body`` and balance end-net self-loop dummies onto outer ends."""
    if len(fingers) >= 2 and columns[0] == columns[-1] and columns[0] != body and body in columns[:-1]:
        rot = columns[:-1].index(body)
        fingers = fingers[rot:] + fingers[:rot]
        columns = columns[rot:-1] + columns[:rot] + (body,)
    if not any(f.left != f.right for f in fingers):
        return fingers, columns
    end_nets = {columns[0], columns[-1]}
    core_fingers = []
    core_columns = [columns[0]]
    extracted = []
    for f in fingers:
        if f.left == f.right and f.left in end_nets:
            extracted.append(f)
        else:
            core_fingers.append(f)
            core_columns.append(f.right)
    if not extracted:
        return fingers, columns
    left_dummies = []
    right_dummies = []
    place_left_next = True
    for d in extracted:
        net = d.left
        if core_columns[0] == net and core_columns[-1] == net:
            if place_left_next:
                left_dummies.append(d)
            else:
                right_dummies.append(d)
            place_left_next = not place_left_next
        elif core_columns[0] == net:
            left_dummies.append(d)
        else:
            right_dummies.append(d)
    out_fingers = tuple(left_dummies) + tuple(core_fingers) + tuple(right_dummies)
    out_columns = (
        (core_columns[0],) * len(left_dummies)
        + tuple(core_columns)
        + (core_columns[-1],) * len(right_dummies)
    )
    return out_fingers, out_columns


def order_native_diffusion_chain(chain, owners, *, left_net=None):
    """Apply an observed owner order without searching finger permutations.

    ``owners`` is a complete, ordered occurrence ledger, for example one
    obtained from a uniquely bound reference.  Each device's fingers retain its
    source D/S pair and gate.  At most the two orientations of the first finger
    are tried; each subsequent column follows directly from net continuity.
    """
    if not isinstance(chain, NativeDiffusionChain):
        raise TypeError("expected a compiled native diffusion chain")
    ordered = tuple(owners)
    if not ordered:
        raise ValueError("observed owner order must not be empty")
    buckets = defaultdict(list)
    for finger in chain.fingers:
        buckets[finger.owner].append(finger)
    for values in buckets.values():
        values.sort(key=lambda item: item.occurrence, reverse=True)
    arranged = []
    for owner in ordered:
        if not buckets.get(owner):
            raise ValueError("observed owner order differs from source finger inventory")
        arranged.append(buckets[owner].pop())
    if any(buckets.values()) or len(ordered) != len(chain.fingers):
        raise ValueError("observed owner order differs from source finger inventory")
    first = arranged[0]
    choices = (left_net,) if left_net is not None else tuple(sorted({first.left, first.right}))
    for start in choices:
        columns = [start]
        fingers = []
        for source in arranged:
            left, right = source.left, source.right
            if columns[-1] == left:
                nxt = right
            elif columns[-1] == right:
                nxt = left
            else:
                break
            fingers.append(NativeDiffusionFinger(source.owner, source.occurrence,
                source.replica, source.finger, source.gate, source.body, columns[-1], nxt))
            columns.append(nxt)
        else:
            return NativeDiffusionChain(chain.source_identity, chain.model, chain.width_nm,
                                        chain.length_nm, chain.body, tuple(fingers), tuple(columns),
                                        chain.coverage_profile, chain.access_profile, chain.generation_profile,
                                        chain.synthesis, chain.tap_rail)
    raise ValueError("observed finger order cannot share the source diffusion nets")


def order_native_diffusion_occurrences(chain, occurrences, *, left_net=None):
    """Replay an exact source occurrence ledger on the existing diffusion trail."""
    from clay.layout.spatial.intent_projection import MosaicOccurrence

    if type(occurrences) is not tuple or not occurrences:
        raise TypeError("native occurrence ledger must be one nonempty exact tuple")
    if any(type(item) is not MosaicOccurrence or item.transform != "R0"
           for item in occurrences):
        raise ValueError("native occurrence ledger has a foreign type or transform")
    for item in occurrences:
        item.__post_init__()
    expected = tuple((item.logical_owner, item.occurrence_ordinal) for item in occurrences)
    if len(expected) != len(set(expected)):
        raise ValueError("native occurrence ledger contains duplicate source occurrences")
    ordered = order_native_diffusion_chain(
        chain, tuple(item.logical_owner for item in occurrences), left_net=left_net
    )
    if tuple((finger.owner, finger.occurrence) for finger in ordered.fingers) != expected:
        raise ValueError("native occurrence ledger differs from the source occurrence cover")
    return ordered
