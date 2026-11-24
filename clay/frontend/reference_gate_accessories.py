"""Measure complete leaf gate contacts against the current native terminal graph.

This is geometric observation data. The outer reference producer authenticates
the GDS, source assignment and PDK; these functions do not mint that authority.
"""
from dataclasses import dataclass

import klayout.db as kdb

from clay.frontend.reference_device_shapes import _region, _require, _box


@dataclass(frozen=True, slots=True)
class ReferenceGateAccessory:
    occurrence_path: str
    master_name: str
    parts: tuple
    occurrence_paths: tuple[str, ...]
    node_id: str


def _complete_node(nodes, key, region, checkpoint):
    hits, covered = [], kdb.Region()
    for index,node in enumerate(nodes):
        checkpoint()
        geometry = node.get(key)
        if geometry is not None and not (region & geometry).is_empty():
            hits.append('node:'+str(index))
            covered += geometry
    _require(len(hits) == 1 and (region-covered).is_empty(), 'gate accessory does not resolve to one complete node')
    return hits[0]


def _conductive_touch(left, right):
    if not (left & right).is_empty():
        return True
    common = left.edges() & right.edges()
    return any(edge.p1 != edge.p2 for edge in common.each())


def observe_reference_gate_accessories(*, leaves, observed, nodes, semantic_layers, poly_wire_layer, checkpoint):
    keys = dict(semantic_layers)
    mos = tuple(row for row in observed if row['primitive'].family in ('nmos','pmos'))
    gates = {row['occurrence_path']:_region(next(shapes for role,_layer,shapes
             in row['primitive'].terminals if role == 'gate')) for row in mos}
    node_by_path = {row['occurrence_path']:dict(row['terminal_nodes'])['gate'] for row in mos}
    covered, result, paths = set(), [], set()
    for path,master,parts in leaves:
        checkpoint()
        _require(path not in paths, 'gate accessory hierarchy path is duplicated')
        paths.add(path)
        regions = {key:_region(tuple(shape for layer,shape in parts if layer == key))
                   for key in {layer for layer,_shape in parts}}
        # Metal-to-metal routing stacks do not become intrinsic gate artwork.
        if not all(keys[name] in regions for name in ('poly','contact','metal1')):
            continue
        _require(not any(keys[name] in regions for name in ('diff','res_mk')),
                 'gate accessory contains another active device')
        poly,contacts,metal = (regions[keys[name]] for name in ('poly','contact','metal1'))
        context = f' leaf={path!r}, master={master!r}, poly_bbox_nm={_box(poly)!r}'
        _require(not contacts.is_empty() and (contacts-poly).is_empty() and (contacts-metal).is_empty(),
                 'complete gate accessory contact is outside poly or metal'+context)
        owners = tuple(sorted(path for path,gate in gates.items() if _conductive_touch(gate,poly)))
        if not owners:
            # A complete poly-contact leaf can belong to a non-MOS device.
            # It remains in the full top artwork; no MOS ownership is invented.
            continue
        associated = sum((gates[path] for path in owners),kdb.Region())
        _require((poly-poly.interacting(associated)).is_empty(), 'gate accessory has an unassociated poly component')
        poly_node = _complete_node(nodes,poly_wire_layer,poly,checkpoint)
        metal_node = _complete_node(nodes,keys['metal1'],metal,checkpoint)
        _require(poly_node == metal_node and all(node_by_path[path] == poly_node for path in owners),
                 'gate accessory current node differs from its complete raw gate association')
        covered.update(owners)
        result.append(ReferenceGateAccessory(path,master,parts,owners,poly_node))
    for row in mos:
        checkpoint()
        path = row['occurrence_path']
        if path in covered:
            continue
        parts = row['parts']
        contact = _region(tuple(shape for key,shape in parts if key == keys['contact']))
        metal = _region(tuple(shape for key,shape in parts if key == keys['metal1']))
        cuts = contact.interacting(gates[path])
        _require(not cuts.is_empty() and (cuts-gates[path]).is_empty() and (cuts-metal).is_empty(),
                 'raw MOS gate lacks a complete intrinsic or measured accessory access')
    return tuple(sorted(result,key=lambda row:row.occurrence_path))
