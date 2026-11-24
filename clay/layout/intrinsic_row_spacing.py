"""Conservative whole-artwork spacing selected from current PDK declarations.

Same-net selection is an explicit construction input, not proof of electrical
connectivity. The caller owns its source/net justification. All other material
rules remain in the maximum; this policy does not merge wells or certify DRC.
"""
from collections.abc import Mapping
import re


def _require(ok, message):
    if not ok:
        raise ValueError('local assembly ' + message)


def _same_net_rules(tech, raw_layer_keys, selection):
    _require(type(selection) is tuple and all(type(token) is str
        and re.fullmatch(r'[a-z][a-z0-9_]*', token) for token in selection),
        'same-net layers must be an exact tuple of canonical tokens')
    _require(selection == tuple(sorted(set(selection))),
        'same-net layer tokens must be unique and canonically ordered')
    if not selection:
        return {}
    declared = getattr(tech.adapter, 'placement_same_net_layer_spacing_nm', {})
    _require(isinstance(declared, Mapping), 'PDK same-net spacing authority must be a mapping')
    result = {}
    for token in selection:
        key = tech.rules._data.layermap.get(token)
        _require(key is not None and tuple(key) in raw_layer_keys,
            'same-net spacing layer is unmapped or absent from raw artwork: ' + token)
        value = declared.get(token)
        _require(type(value) is int and value > 0,
            'PDK same-net spacing must be declared positive exact integer nm: ' + token)
        result[token] = value
    return result


def placement_row_clearance(tech, raw_layer_keys, grid, *, same_net_layers=()):
    """Return grid-rounded maximum and ordered per-rule provenance rows.

    The empty selection preserves the original rule order and row payloads. An
    explicit token replaces every original row for that token, retaining its old
    value and source in the row. Missing/invalid original rules cannot be repaired
    by selecting a same-net value, and unselected layer aliases stay conservative.
    """
    _require(type(grid) is int and grid > 0, 'manufacturing grid must be positive exact integer nm')
    same_net = _same_net_rules(tech, raw_layer_keys, same_net_layers)
    layermap = tech.rules._data.layermap
    rows = []
    for token in sorted(tech.rules._data.layer_rules):
        key = layermap.get(token)
        if key is None or tuple(key) not in raw_layer_keys:
            continue
        value = tech.min_space(token)
        if value is not None:
            _require(type(value) is int and value > 0, 'PDK layer spacing is not positive exact nm')
            rows.append({'layer': tuple(key), 'token': token, 'spacing_nm': value, 'source': 'current_tech_min_space'})
    for token in tech.via_names():
        rule = tech.via_rule(token)
        if rule is None or tuple(rule.key) not in raw_layer_keys:
            continue
        value = tech.via_spacing(token)
        if value is not None:
            _require(type(value) is int and value > 0, 'PDK cut spacing is not positive exact nm')
            rows.append({'layer': tuple(rule.key), 'token': token, 'spacing_nm': value, 'source': 'current_tech_via_spacing'})
    for token, value in tech.adapter.placement_layer_spacing_nm.items():
        key = layermap.get(token)
        _require(key is not None, 'PDK placement spacing layer is unmapped')
        if tuple(key) in raw_layer_keys:
            _require(type(value) is int and value > 0, 'PDK declared placement spacing is invalid')
            rows.append({'layer': tuple(key), 'token': token, 'spacing_nm': value,
                'source': 'current_adapter_placement_layer_spacing_nm'})
    covered = {row['layer'] for row in rows}
    _require(raw_layer_keys <= covered,
        'spacing rule unavailable for raw artwork layer ' + str(sorted(raw_layer_keys-covered)))
    _require(rows, 'no current PDK row clearance evidence')
    _require(set(same_net) <= {row['token'] for row in rows},
        'same-net spacing selection has no original layer rule')
    for row in rows:
        if row['token'] in same_net:
            row.update(replaced_source=row['source'], replaced_spacing_nm=row['spacing_nm'],
                source='current_adapter_placement_same_net_layer_spacing_nm',
                spacing_nm=same_net[row['token']])
    value = max(row['spacing_nm'] for row in rows)
    return ((value+grid-1)//grid)*grid, rows
