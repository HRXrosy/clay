"""Restore actual GDS MOS channels after a proven same-extraction LVS fold.

LVS parallel reduction is an equivalence for terminal topology and total W,
not for width-dependent BSIM simulation. Raw MOS channels are accepted only
after same-LVS net naming and per-topology W/junction conservation checks.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

from .joint_extraction import _spice_logical_cards, full_extraction_mos_models_for_pdk
from .magic_parser import _parse_spice_value


def _name(value):
    return str(value).strip().lstrip('\\$').casefold()


def _body(text, cell):
    inside = False
    found = False
    result = []
    for card in _spice_logical_cards(text):
        tokens = card.split()
        if tokens and tokens[0].casefold() == '.subckt':
            inside = len(tokens) >= 2 and tokens[1].casefold() == cell.casefold()
            found |= inside
        elif tokens and tokens[0].casefold() == '.ends':
            inside = False
        elif inside:
            result.append(card)
    if not found:
        raise ValueError(f'Uncombined MOS netlist has no cell {cell}')
    return result


def _mos(card, models=None, mapping=None):
    tokens = card.split()
    if not tokens or tokens[0][0].upper() not in ('M', 'X'):
        return None
    if len(tokens) < 6:
        raise ValueError(f'Malformed extracted MOS: {card}')
    model = tokens[5].casefold()
    prefix = tokens[0][0].upper()
    if models is not None:
        if model not in models:
            raise ValueError(f'Unbound raw MOS model {model}')
        model, prefix = models[model]
    nodes = [_name(token) for token in tokens[1:5]]
    if mapping is not None:
        for i, net in enumerate(nodes):
            if net not in mapping:
                raise ValueError(f'unbound raw-channel LVS net {tokens[i+1]!r}')
            nodes[i] = mapping[net]
    literals = dict((key.casefold(), value) for key, value in
                    re.findall(r'(\w+)\s*=\s*([^\s]+)', ' '.join(tokens[6:])))
    geometry = {key: _parse_spice_value(value) for key, value in literals.items()
                if key in {'w', 'l', 'm', 'mr', 'ad', 'as', 'pd', 'ps'}}
    if any(not math.isfinite(value) or value < 0 for value in geometry.values()):
        raise ValueError('Nonphysical extracted MOS geometry')
    if geometry.get('w', 0) <= 0 or geometry.get('l', 0) <= 0:
        raise ValueError('Extracted MOS must have positive physical W/L')
    count = geometry.get('m', 1) * geometry.get('mr', 1)
    if count <= 0:
        raise ValueError('Extracted MOS multiplicity must be positive')
    d, g, s, b = nodes
    key = (str(model).casefold(), round(geometry['l'], 15), g, b, tuple(sorted((d, s))))
    # KLayout's symmetric MOS combiner can average D/S junction parameters.
    # Its group totals are conserved, but per-terminal attribution is lossy.
    # Check the conserved totals and retain the raw physical D/S parameters
    # verbatim in simulation rather than reusing that averaged distribution.
    vector = [geometry['w'] * count,
              (geometry.get('ad', 0) + geometry.get('as', 0)) * count,
              (geometry.get('pd', 0) + geometry.get('ps', 0)) * count]
    return key, vector, nodes, model, prefix, tokens[6:], geometry


def restore_uncombined_mos(pex_text, raw_text, *, cell_name, net_mapping, mos_models):
    mapping = {}
    for source, target in net_mapping.items():
        key, value = _name(source), _name(target)
        if key in mapping and mapping[key] != value:
            raise ValueError(f'Ambiguous raw-channel net mapping: {source}')
        mapping[key] = value
    models = {key.casefold(): value for key, value in mos_models.items()}
    aggregate = [row for card in _body(pex_text, cell_name) if (row := _mos(card))]
    raw = [row for card in _body(raw_text, cell_name) if (row := _mos(card, models, mapping))]
    if not aggregate or not raw:
        raise ValueError('Both aggregate and actual-channel MOS inventories are required')

    def totals(rows):
        out = {}
        for key, values, *_ in rows:
            current = out.setdefault(key, [0.] * len(values))
            for i, value in enumerate(values):
                current[i] += value
        return out

    old, new = totals(aggregate), totals(raw)
    if old.keys() != new.keys():
        raise ValueError(f'MOS topology conservation failed: missing={old.keys()-new.keys()}, extra={new.keys()-old.keys()}')
    for key in old:
        if any(not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-23)
               for a, b in zip(old[key], new[key])):
            raise ValueError(f'MOS width/junction conservation failed: {key}: {old[key]} != {new[key]}')
    cards = [f'{prefix}PEX_CHANNEL_{i} {" ".join(nodes)} {model} {" ".join(parameters)}'
             for i, (_key, _v, nodes, model, prefix, parameters, _g) in enumerate(raw)]
    output = []
    inserted = False
    for card in _spice_logical_cards(pex_text):
        if _mos(card) is not None:
            if not inserted:
                output.extend(cards)
                inserted = True
        else:
            output.append(card)
    result = '\n'.join(output) + '\n'
    evidence = {
        'raw_mos_count': len(raw), 'aggregated_mos_count': len(aggregate),
        'topology_groups': len(old), 'width_and_junction_conservation': True,
        'width_authority': 'Native GDS MOS channels before combine_devices and simplify',
        'no_arbitrary_WMAX_slicing': True,
        'junction_distribution_source': 'Actual uncombined channel D/S, not LVS-combiner averaged AS/AD',
        'raw_widths_um': sorted({round(row[-1]['w'] * 1e6, 9) for row in raw}),
        'total_channel_width_um': sum(row[1][0] for row in raw) * 1e6,
    }
    return result, evidence


def refine_pex_from_channel_evidence(pex_path, evidence_path, *, output_path=None):
    import klayout.db as kdb
    from clay.layout.route.full_extraction_body_check import _lvs_net_xref_mapping

    pex_path, evidence_path = Path(pex_path), Path(evidence_path)
    record = json.loads(evidence_path.read_text())
    for field, hash_field in (('gds', 'gds_sha256'), ('raw_netlist', 'raw_sha256')):
        if hashlib.sha256(Path(record[field]).read_bytes()).hexdigest() != record[hash_field]:
            raise ValueError(f'Channel extraction evidence changed: {field}')
    verdict = json.loads(Path(record['full_lvs_verdict']).read_text())
    if not verdict['matched'] or verdict['gds_sha256'] != record['gds_sha256']:
        raise ValueError('Channel extraction lacks matching same-GDS full LVS')
    lvsdb = Path(record['lvsdb'])
    if hashlib.sha256(lvsdb.read_bytes()).hexdigest() != verdict['lvsdb_sha256']:
        raise ValueError('Channel extraction LVS cross-reference changed')
    database = kdb.LayoutVsSchematic()
    database.read(str(lvsdb))
    text, proof = restore_uncombined_mos(
        pex_path.read_text(), Path(record['raw_netlist']).read_text(),
        cell_name=record['cell'], net_mapping=_lvs_net_xref_mapping(database),
        mos_models=full_extraction_mos_models_for_pdk(record['pdk']))
    from .uncombined_resistors import restore_uncombined_resistors

    text, resistor_proof = restore_uncombined_resistors(
        text, Path(record['raw_netlist']).read_text(), cell_name=record['cell'],
        net_mapping=_lvs_net_xref_mapping(database))
    proof['resistor_segments'] = resistor_proof
    output = Path(output_path) if output_path else pex_path.with_suffix('.channels.spice')
    output.write_text(text, encoding='utf-8')
    proof.update(record)
    proof.update(pex=str(output.resolve()),
                 input_pex=str(pex_path.resolve()),
                 input_pex_sha256=hashlib.sha256(pex_path.read_bytes()).hexdigest(),
                 pex_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.json').write_text(json.dumps(proof, indent=2)+'\n')
    return output, proof
