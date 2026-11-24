"""Bind actual unsimplified GDS resistor segments to surviving LVS anchors.

Source segment names are only labels for a graph that the raw extraction must
independently contain. Neither a summed resistance nor a source ordinal proves
that a physical midpoint exists.
"""
from collections import defaultdict
import hashlib
import itertools
import math

from .joint_extraction import _parse_extracted_resistor, _spice_logical_cards
from .uncombined_mos import _body, _name


def raw_device_snapshot_ruby(path_expression, *, prefix="uncombined"):
    """Export measured secondary resistor dimensions without changing LVS.

    KLayout extracts L/W on ordinary resistors but its default SPICE writer
    omits secondary parameters. Enable them on a deep netlist copy only: the
    comparison and device-combination policy of the original stays unchanged.
    ``path_expression`` is Ruby code supplied by the two existing deck callers.
    """
    return "\n".join((
        f"{prefix}_netlist = netlist.dup",
        f"{prefix}_netlist.each_device_class do |dc|",
        "  if dc.is_a?(RBA::DeviceClassResistor) || dc.is_a?(RBA::DeviceClassResistorWithBulk)",
        '    dc.enable_parameter("L", true)',
        '    dc.enable_parameter("W", true)',
        "  end",
        "end",
        f"{prefix}_writer = RBA::NetlistSpiceWriter::new",
        f"{prefix}_writer.use_net_names = true",
        f'{prefix}_netlist.write({path_expression}, {prefix}_writer, "Actual GDS devices before combine_devices/simplify; measured resistor L/W retained")',
    ))


def _resistor_conductance(rows, anchors):
    """Exact resistive Schur elimination; no midpoint or unit is synthesized."""
    graph = defaultdict(dict)
    for resistor, *_ in rows:
        if resistor.a == resistor.b:
            raise ValueError("Actual resistor is shorted to itself")
        conductance = 1.0 / resistor.resistance_ohm
        graph[resistor.a][resistor.b] = graph[resistor.a].get(resistor.b, 0.) + conductance
        graph[resistor.b][resistor.a] = graph[resistor.b].get(resistor.a, 0.) + conductance
    while any(node not in anchors for node in graph):
        node = min((node for node in graph if node not in anchors), key=lambda n: (len(graph[n]), n))
        neighbors = tuple(graph[node].items())
        total = sum(value for _, value in neighbors)
        if not total > 0:
            raise ValueError("Actual resistor component has no surviving LVS anchor")
        for (left, gl), (right, gr) in itertools.combinations(neighbors, 2):
            addition = gl * gr / total
            graph[left][right] = graph[left].get(right, 0.) + addition
            graph[right][left] = graph[right].get(left, 0.) + addition
        for neighbor, _ in neighbors:
            del graph[neighbor][node]
        del graph[node]
    output = defaultdict(float)
    for left, neighbors in graph.items():
        for right, conductance in neighbors.items():
            if left < right:
                a, b = sorted((anchors[left], anchors[right]))
                if a == b:
                    raise ValueError("Distinct raw resistor nodes map to one LVS anchor")
                output[(a, b)] += conductance
    return dict(output)


def bind_actual_resistor_graph(raw_text, reference_text, *, cell_name, net_mapping,
                              aggregate_rows=None, relative_tolerance=0.02):
    """Bind a complete physical reference graph by unique anchored isomorphism.

    Reference labels are expectations only. Every element, parallel count,
    intermediate node and R value is independently present in the raw GDS
    extraction. A second possible midpoint mapping is a blocking ambiguity.
    """
    import networkx as nx

    raw = _resistors(raw_text, cell_name)
    reference = _resistors(reference_text, cell_name)
    if len(raw) != len(reference):
        raise ValueError(f"Actual resistor graph segment count differs: raw={len(raw)}, reference={len(reference)}")
    raw_nodes = {node for row, *_ in raw for node in (row.a, row.b)}
    reference_nodes = {node for row, *_ in reference for node in (row.a, row.b)}
    mapping = {}
    for key, value in net_mapping.items():
        key, value = _name(key), _name(value)
        if key in mapping and mapping[key] != value:
            raise ValueError("Ambiguous actual resistor LVS anchor")
        mapping[key] = value
    for name in set(mapping.values()) & raw_nodes:
        mapping.setdefault(name, name)
    if any(key in raw_nodes and value not in reference_nodes for key, value in mapping.items()):
        raise ValueError('Actual resistor LVS anchor is absent from the physical reference')
    anchors = {key: value for key, value in mapping.items()
               if key in raw_nodes and value in reference_nodes}
    if len(set(anchors.values())) != len(anchors):
        raise ValueError("Actual resistor graph anchors are not one-to-one")
    required_anchors = set()
    for card in _spice_logical_cards(reference_text):
        tokens = card.split()
        if len(tokens) >= 2 and tokens[0].casefold() == '.subckt' and tokens[1].casefold() == cell_name.casefold():
            required_anchors.update(_name(token) for token in tokens[2:] if '=' not in token)
            break
    required_anchors &= reference_nodes
    if aggregate_rows is not None:
        required_anchors.update(node for row in aggregate_rows for node in (row.a, row.b))
    if required_anchors - set(anchors.values()):
        raise ValueError('Actual resistor graph is missing a surviving LVS boundary anchor')

    def graph(rows, anchor_names):
        result = nx.Graph()
        for index, (row, *_rest) in enumerate(rows):
            result.add_node(row.a, anchor=anchor_names.get(row.a))
            result.add_node(row.b, anchor=anchor_names.get(row.b))
            if not result.has_edge(row.a, row.b):
                result.add_edge(row.a, row.b, members=[])
            result[row.a][row.b]['members'].append((index, row))
        return result

    raw_graph = graph(raw, anchors)
    reference_graph = graph(reference, {name: name for name in anchors.values()})
    def edge_equal(left, right):
        lrows, rrows = left['members'], right['members']
        if len(lrows) != len(rrows):
            return False
        def key(entry):
            row = entry[1]
            return (row.resistance_ohm, row.model or '', row.length or 0., row.width or 0.)
        for (_, actual), (_, expected) in zip(sorted(lrows, key=key), sorted(rrows, key=key)):
            if not math.isclose(actual.resistance_ohm, expected.resistance_ohm, rel_tol=min(relative_tolerance, 1e-6)):
                return False
            for field in ('model', 'length', 'width'):
                wanted, measured = getattr(expected, field), getattr(actual, field)
                if wanted is not None and (measured is None or
                        (measured != wanted if field == 'model' else
                         not math.isclose(measured, wanted, rel_tol=1e-9, abs_tol=1e-15))):
                    return False
        return True
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        raw_graph, reference_graph,
        node_match=lambda left, right: left['anchor'] == right['anchor'], edge_match=edge_equal)
    candidates = matcher.isomorphisms_iter()
    selected = next(candidates, None)
    if selected is None:
        raise ValueError("Actual resistor graph differs from physical reference topology/model/geometry")
    if next(candidates, None) is not None:
        raise ValueError("Actual resistor midpoint graph mapping is ambiguous")
    conductance_receipt = None
    if aggregate_rows is not None:
        aggregate = [(row, (), '') for row in aggregate_rows]
        surviving = {node for row in aggregate_rows for node in (row.a, row.b)}
        raw_anchors = {node: value for node, value in selected.items() if value in surviving}
        actual_g = _resistor_conductance(raw, raw_anchors)
        aggregate_g = _resistor_conductance(aggregate, {name: name for name in surviving})
        if (set(actual_g) != set(aggregate_g) or any(not math.isclose(actual_g[key], aggregate_g[key],
                rel_tol=1e-6, abs_tol=1e-24) for key in actual_g)):
            raise ValueError("Actual resistor graph conductance differs from same-LVS aggregate")
        conductance_receipt = [dict(endpoints=list(key), actual_siemens=value,
                                   aggregate_siemens=aggregate_g[key]) for key, value in sorted(actual_g.items())]
    records, cards = [], []
    for index, (row, _tokens, raw_card) in enumerate(raw):
        a, b = selected[row.a], selected[row.b]
        cards.append(f"RPEX_ACTUAL_{index} {a} {b} {row.resistance_ohm:.12g}")
        records.append(dict(raw_card=raw_card, raw_nodes=[row.a,row.b],mapped_nodes=[a,b],
                            resistance_ohm=row.resistance_ohm,model=row.model,
                            length_m=row.length,width_m=row.width))
    proof = dict(verified=True, raw_resistor_count=len(raw), pex_resistor_count=len(raw),
                 input_pex_resistor_count=len(reference), mapped_nodes=selected, segments=records,
                 resistor_graph_unique=True, physical_reference_sha256=hashlib.sha256(reference_text.encode()).hexdigest(),
                 actual_aggregate_conductance=conductance_receipt,
                 source_ratios_are_not_measurements=True,
                 resistance_authority='Same-GDS actual unsimplified device extraction',
                 midpoint_authority='Unique physical graph isomorphism fixed by surviving same-LVS anchors')
    return cards, proof


def _resistors(text, cell):
    result = []
    for card in _body(text, cell):
        tokens = card.split()
        if not tokens or not tokens[0].upper().startswith("R"):
            continue
        resistor = _parse_extracted_resistor(tokens, card_index=len(result), node=_name)
        if not math.isfinite(resistor.resistance_ohm):
            raise ValueError("Actual resistor has nonfinite resistance")
        result.append((resistor, tokens, card))
    return result


def _anchored_chains(rows, anchors):
    incident = defaultdict(list)
    for index, (resistor, _tokens, _card) in enumerate(rows):
        incident[resistor.a].append(index)
        incident[resistor.b].append(index)
    for node, edges in incident.items():
        if node not in anchors and len(edges) != 2:
            raise ValueError(f"unbound actual resistor anchor/degree at {node}: {len(edges)}")
    visited = set()
    chains = defaultdict(list)
    for start in sorted(set(anchors) & set(incident)):
        for seed in incident[start]:
            if seed in visited:
                continue
            current, edge = start, seed
            ordered, nodes = [], [start]
            while True:
                if edge in visited:
                    raise ValueError("Actual resistor chain closes without a distinct LVS anchor")
                visited.add(edge)
                ordered.append(edge)
                resistor = rows[edge][0]
                other = resistor.b if resistor.a == current else resistor.a
                nodes.append(other)
                if other in anchors:
                    break
                onward = [item for item in incident[other] if item != edge]
                if len(onward) != 1:
                    raise ValueError(f"Actual resistor midpoint {other} is not degree two")
                current, edge = other, onward[0]
            a, b = anchors[start], anchors[other]
            if a == b:
                raise ValueError("Actual resistor chain has ambiguous identical endpoint anchors")
            if a > b:
                a, b = b, a
                ordered.reverse()
                nodes.reverse()
            chains[(a, b)].append((ordered, nodes))
    if len(visited) != len(rows):
        raise ValueError("Actual resistor graph contains an unbound component without LVS anchors")
    return chains


def restore_uncombined_resistors(pex_text, raw_text, *, cell_name, net_mapping,
                                 relative_tolerance=0.02):
    """Prove each actual segment/node, retain measured R, leave wire C unchanged."""
    raw = _resistors(raw_text, cell_name)
    target = _resistors(pex_text, cell_name)
    target_nodes = {node for row, *_ in target for node in (row.a, row.b)}
    raw_nodes = {node for row, *_ in raw for node in (row.a, row.b)}
    mapping = {}
    for key, value in net_mapping.items():
        key, value = _name(key), _name(value)
        if key in mapping and mapping[key] != value:
            raise ValueError(f"Ambiguous actual resistor LVS mapping for {key}")
        mapping[key] = value
    # NetlistSpiceWriter(use_net_names=True) may emit the surviving net's
    # proven schematic name rather than its numerical extracted identifier.
    # Accept only names already present as values of this same-LVS xref.
    for name in set(mapping.values()) & raw_nodes:
        mapping.setdefault(name, name)
    anchors = {key: value for key, value in mapping.items()
               if key in raw_nodes and value in target_nodes}
    incidence = defaultdict(int)
    for row, *_ in raw:
        incidence[row.a] += 1
        incidence[row.b] += 1
    if any(degree > 2 and node not in anchors for node, degree in incidence.items()):
        actual_cards, proof = bind_actual_resistor_graph(
            raw_text, pex_text, cell_name=cell_name, net_mapping=net_mapping,
            relative_tolerance=relative_tolerance)
        output, emitted = [], False
        for card in _spice_logical_cards(pex_text):
            if card[:1].casefold() == 'r':
                if not emitted:
                    output.extend(actual_cards)
                    emitted = True
            else:
                output.append(card)
        caps_before = [card for card in _spice_logical_cards(pex_text) if card[:1].casefold() == 'c']
        caps_after = [card for card in output if card[:1].casefold() == 'c']
        if caps_before != caps_after:
            raise ValueError('Actual resistor graph restoration changed a parasitic capacitor')
        proof.update(route_cap_cards_unchanged=True, retained_cap_card_count=len(caps_after))
        return '\n'.join(output) + '\n', proof
    raw_chains = _anchored_chains(raw, anchors)
    target_chains = _anchored_chains(target, {name: name for name in anchors.values()})
    if set(raw_chains) != set(target_chains):
        raise ValueError("Actual resistor chain endpoint topology differs from PEX")
    replacements = {}
    mapped_nodes = dict(anchors)
    records = []
    parallel_groups = []
    for ends in sorted(raw_chains):
        if len(raw_chains[ends]) != 1 or len(target_chains[ends]) != 1:
            raw_group, target_group = raw_chains[ends], target_chains[ends]
            if any(len(order) != 1 for order, _path in (*raw_group, *target_group)):
                raise ValueError(f"Actual resistor chain mapping is ambiguous at {ends}")
            # Direct parallel elements have no midpoint to assign. Their true
            # raw multiset is uniquely an equivalent two-node conductance;
            # preserve every measured physical element, never infer its count.
            raw_indexes = sorted(order[0] for order, _ in raw_group)
            target_indexes = sorted(order[0] for order, _ in target_group)
            raw_g = sum(1.0 / raw[index][0].resistance_ohm for index in raw_indexes)
            target_g = sum(1.0 / target[index][0].resistance_ohm for index in target_indexes)
            if not math.isclose(raw_g, target_g, rel_tol=1e-6, abs_tol=1e-24):
                raise ValueError(f"Actual parallel resistor conductance differs at {ends}")
            first_name = target[target_indexes[0]][1][0]
            emitted = []
            for ordinal, index in enumerate(raw_indexes):
                measured, _tokens, raw_card = raw[index]
                emitted.append(f"{first_name}_RAW_{ordinal} {ends[0]} {ends[1]} "
                               f"{measured.resistance_ohm:.12g}")
                records.append(dict(raw_card=raw_card,
                                    target_card=target[target_indexes[0]][2],
                                    raw_nodes=[measured.a, measured.b],
                                    mapped_nodes=[anchors[measured.a], anchors[measured.b]],
                                    resistance_ohm=measured.resistance_ohm,
                                    direct_parallel=True))
            for index in target_indexes:
                replacements[target[index][1][0].casefold()] = ""
            replacements[first_name.casefold()] = "\n".join(emitted)
            parallel_groups.append(dict(endpoints=list(ends), raw_count=len(raw_indexes),
                                        input_pex_count=len(target_indexes),
                                        conductance_siemens=raw_g,
                                        equivalent_conductance_verified=True))
            continue
        raw_order, raw_path = raw_chains[ends][0]
        target_order, target_path = target_chains[ends][0]
        if len(raw_order) != len(target_order):
            raise ValueError(f"Actual resistor segment count differs at {ends}: "
                             f"raw={len(raw_order)}, PEX={len(target_order)}")
        for old, new in zip(raw_path, target_path):
            previous = mapped_nodes.setdefault(old, new)
            if previous != new:
                raise ValueError(f"Actual resistor midpoint mapping is ambiguous at {old}")
        for raw_index, target_index in zip(raw_order, target_order):
            measured, raw_tokens, raw_card = raw[raw_index]
            requested, target_tokens, target_card = target[target_index]
            error = abs(measured.resistance_ohm - requested.resistance_ohm) / max(
                measured.resistance_ohm, requested.resistance_ohm)
            if error > relative_tolerance:
                raise ValueError(f"Actual segment resistance differs for {target_tokens[0]}: "
                                 f"raw={measured.resistance_ohm}, target={requested.resistance_ohm}")
            replacements[target_tokens[0].casefold()] = (
                f"{target_tokens[0]} {requested.a} {requested.b} {measured.resistance_ohm:.12g}")
            records.append(dict(raw_card=raw_card, target_card=target_card,
                                raw_nodes=[measured.a, measured.b],
                                mapped_nodes=[mapped_nodes[measured.a], mapped_nodes[measured.b]],
                                resistance_ohm=measured.resistance_ohm))
    cards = _spice_logical_cards(pex_text)
    output = [line for card in cards
              for line in replacements.get(card.split()[0].casefold(), card).splitlines()]
    caps_before = [card for card in cards if card[:1].casefold() == "c"]
    caps_after = [card for card in output if card[:1].casefold() == "c"]
    if caps_before != caps_after:
        raise ValueError("Actual resistor restoration changed a parasitic capacitor card")
    proof = dict(verified=True, raw_resistor_count=len(raw), pex_resistor_count=len(raw),
                 input_pex_resistor_count=len(target), parallel_groups=parallel_groups,
                 actual_chain_count=sum(map(len, raw_chains.values())),
                 mapped_nodes=mapped_nodes, segments=records,
                 resistance_authority="Same-GDS foundry/device extraction before simplify/combine",
                 midpoint_authority="Unique degree-two raw graph between surviving same-LVS xref anchors",
                 source_ratios_are_not_measurements=True,
                 route_cap_cards_unchanged=True, retained_cap_card_count=len(caps_after))
    return "\n".join(output) + "\n", proof
