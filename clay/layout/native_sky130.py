"""Source-bound SKY130 native PCells for the existing physical flow.

The installed Magic PCell is the geometry producer.  This adapter does not
stretch a reference GDS, reinterpret native dimensions as portable sizing, or
grant signoff.  Every generated PCell is reopened and its complete extraction,
conductor graph, and source-owned polygon inventory are checked before placement.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid

import klayout.db as kdb

from clay.frontend.native_spice_import import emit_native_spice, validate_native_device
from clay.layout.backend import LayerPolygon, Point, Polygon
from clay.layout.connectivity import ConnectModel, DerivedLayer, extract_nodes
from clay.layout.device_cell import DeviceGeom, Terminal
from clay.pdk.native_spice import NativeSpiceFamily
from clay.pdk.sky130_native_geometry import (NativeSky130GeometryError, native_geometry_spec,
    single_tap_selections as _single_tap_selections, native_cell_geometry_specs as _cell_native_specs)


def _require(ok, message):
    if not ok:
        raise NativeSky130GeometryError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stage_sha(path, checkpoint):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            checkpoint()
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    checkpoint()
    return digest.hexdigest()


def _json(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(data):
    return hashlib.sha256(_json(data).encode()).hexdigest()


def _write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def native_source_identity(cell):
    text = emit_native_spice(cell, pdk_id='sky130')
    data = {'spice': text, 'port_roles': sorted(cell.port_roles.items()),
            'devices': [(d.instance_name, d.source_model, d.native_parameters,
                         sorted(d.pins.items())) for d in cell.devices]}
    selections = _single_tap_selections(cell)
    if selections:
        data['body_taps'] = sorted(selections.items())
    from clay.pdk.sky130_native_geometry import stacked_mim_surround_selections
    stacked_mims = stacked_mim_surround_selections(cell)
    if stacked_mims:
        data['stacked_mims'] = sorted(stacked_mims.items())
    return _digest(data)


def _generation_tcl(spec):
    w, length = spec['w_nm'] / 1000, spec['l_nm'] / 1000
    params = f'w {w:.3f} l {length:.3f} m {spec["replicas"]}'
    if spec['family'] == 'mos':
        params += f' nf {spec["nf"]} conn_gates 1 topc 1 botc 0 doports 1 viagb 100'
    elif spec['family'] == 'varactor':
        params += ' doports 1 conn_gates 1 topc 1 botc 0 viagb 100'
    elif spec['family'] == 'mim' and spec.get('vertical_stack_m3_surround_nm'):
        surround_um = int(spec['vertical_stack_m3_surround_nm']) / 1000
        params += f' bot_surround {surround_um:.3f}'
    # The native varactor PCell sizes its well around diffusion only. varac.5
    # requires the entire connected poly plus 0.15um. Compute its supported
    # sub_surround_dev parameter from the installed PCell rules, keeping W/L.
    repair = ''
    if spec['family'] == 'varactor':
        repair = '''set varwell [expr {[dict get $sky130::ruleset gate_to_polycont] +
    [dict get $sky130::ruleset contact_size]/2.0 + [dict get $sky130::ruleset poly_surround] + 0.15}]
append params " sub_surround_dev $varwell"
'''
    single_tap = spec.get('body_tap') == 'single'
    if single_tap:
        params += ' guard 0'
    tap = ''
    if single_tap and '__pfet_' in spec['model']:
        tap = '''box 1.40um 0um 2.30um 1.23um
paint nwell
box 1.91um 0.5um 1.91um 0.5um
sky130::draw_contact 0.18 0.17 0.12 0.08 0.17 nsd nsc li horz
sky130::draw_contact 0.17 0.17 0.08 0.06 0.17 li mcon metal1 full
label B c nsc
'''
    elif single_tap:
        tap = '''box 1.23um 0.34um 1.23um 0.34um
sky130::draw_contact 0.18 0.17 0.12 0.08 0.17 psd psc li horz
sky130::draw_contact 0.17 0.17 0.08 0.06 0.17 li mcon metal1 full
label B c psc
'''
    return f'''crashbackups stop
drc off
load primitive
box 0 0 0 0
set params {{{params}}}
{repair}if {{[catch {{magic::gencell sky130::{spec['model']} U0 -spice {{*}}$params}} failure]}} {{
    puts stderr "NATIVE_PCELL_FAILURE: $failure"
    exit 2
}}
{tap}
select top cell
gds write primitive.gds
writeall force
set handle [open scale_um.txt w]
puts $handle [cif scale out]
close $handle
drc euclidean on
drc style drc(full)
drc check
drc catchup
set handle [open drc.txt w]
puts $handle [drc listall why]
close $handle
flatten primitive_flat
load primitive_flat
select top cell
extract path .
extract no capacitance
extract no coupling
extract no resistance
extract no length
extract all
ext2spice lvs
ext2spice merge none
ext2spice -p . -o native.spice
puts NATIVE_PCELL_COMPLETED
quit -noprompt
'''


def _runner_script(specs, pdk_root, *, role_trace_keys=(), role_trace_enabled=False):
    script = f'''import hashlib,json,pathlib,subprocess,time
root=pathlib.Path('/work')
pdk=pathlib.Path({pdk_root!r})/'sky130A'
subprocess.run(['python3','/work/toolchain.py'],check=True)
toolkit=pathlib.Path('/foss/tools/magic/lib/magic/tcl/toolkit.tcl')
(root/'pcell-runtime.json').write_text(json.dumps(dict(path=str(toolkit),sha256=hashlib.sha256(toolkit.read_bytes()).hexdigest())))
tech=(pdk/'libs.tech/magic/sky130A.tech').read_text()
assert 'gridlimit 5' in tech and 'grow 150\\n\\tand-not nwell' in tech, 'native grid/varactor rule authority differs'
(root/'sky130A.tech').write_bytes((pdk/'libs.tech/magic/sky130A.tech').read_bytes())
(root/'sky130A.tcl').write_bytes((pdk/'libs.tech/magic/sky130A.tcl').read_bytes())
(root/'sky130A_mr.drc').write_bytes((pdk/'libs.tech/klayout/drc/sky130A_mr.drc').read_bytes())
records=[]
for key in {sorted(specs)!r}:
 folder=root/key
 cmd=['magic','-dnull','-noconsole','-rcfile',str(pdk/'libs.tech/magic/sky130A.magicrc'),str(folder/'generate.tcl')]
 start=time.monotonic()
 p=subprocess.run(cmd,cwd=folder,capture_output=True,timeout=120)
 (folder/'stdout.log').write_bytes(p.stdout); (folder/'stderr.log').write_bytes(p.stderr)
 row=dict(key=key,command=cmd,returncode=p.returncode,seconds=time.monotonic()-start)
 records.append(row)
 (root/'executions.json').write_text(json.dumps(records,indent=2))
 if p.returncode or b'NATIVE_PCELL_COMPLETED' not in p.stdout: raise RuntimeError(row)
'''
    if role_trace_enabled:
        script += '''
lyp=pdk/'libs.tech/klayout/tech/sky130A.lyp'
lyp_bytes=lyp.read_bytes()
(root/'role-trace.sky130A.lyp').write_bytes(lyp_bytes)
(root/'role-trace-runtime.json').write_text(json.dumps(dict(path=str(lyp),sha256=hashlib.sha256(lyp_bytes).hexdigest(),size_bytes=len(lyp_bytes))))
'''
    if role_trace_keys:
        script += f'''
trace_records=[]
for key in {sorted(role_trace_keys)!r}:
 folder=root/key/'trace'
 cmd=['magic','-dnull','-noconsole','-rcfile',str(pdk/'libs.tech/magic/sky130A.magicrc'),str(folder/'generate.tcl')]
 start=time.monotonic()
 p=subprocess.run(cmd,cwd=folder,capture_output=True,timeout=120)
 (folder/'stdout.log').write_bytes(p.stdout); (folder/'stderr.log').write_bytes(p.stderr)
 row=dict(key=key,command=cmd,returncode=p.returncode,seconds=time.monotonic()-start)
 trace_records.append(row)
 (root/'role-trace-executions.json').write_text(json.dumps(trace_records,indent=2))
 if p.returncode or b'NATIVE_PCELL_COMPLETED' not in p.stdout: raise RuntimeError(row)
'''
    return script


_DIFF = (65, 20)
_TAP = (65, 44)
_POLY = (66, 20)
_LI = (67, 20)
_NWELL = (64, 20)
_CAPM = (89, 44)
_SD = (65000, 1)
_NTAP = (65000, 2)
_VIA3_OUT = (65000, 3)
_VIA3_IN = (65000, 4)
_METALS = ((68, 20), (69, 20), (70, 20), (71, 20), (72, 20))


def native_connect_model(family):
    """SKY130 layer semantics, consumed by the existing connectivity engine."""
    derived = (
        DerivedLayer(_SD, (('not', _DIFF, _POLY),)),
        DerivedLayer(_NTAP, (('and', _TAP, (93, 44)),)),
        DerivedLayer(_VIA3_OUT, (('not', (70, 44), _CAPM),)),
        DerivedLayer(_VIA3_IN, (('and', (70, 44), _CAPM),)),
    )
    vias = (((67, 44), _LI, _METALS[0]),
            ((68, 44), _METALS[0], _METALS[1]),
            ((69, 44), _METALS[1], _METALS[2]),
            (_VIA3_OUT, _METALS[2], _METALS[3]),
            (_VIA3_IN, _CAPM, _METALS[3]),
            ((71, 44), _METALS[3], _METALS[4]))
    bridges = ((_NTAP, _TAP), (_NTAP, _NWELL))
    if family == 'varactor':
        bridges += ((_SD, _NWELL),)
    return ConnectModel(metals=(*_METALS, _LI, _SD, _TAP, _POLY, _NWELL, _NTAP, _CAPM),
                        vias=vias, derived=derived, bridges=bridges,
                        extra_connect=tuple(((66, 44), key, _LI) for key in (_SD, _TAP, _POLY)))


def _read_gds(path, *, _bytes=None):
    layout = kdb.Layout()
    if _bytes is None:
        layout.read(str(path))
    else:
        layout.read_bytes(_bytes)
    top = layout.cell('primitive')
    _require(top is not None and len(layout.top_cells()) == 1, 'native primitive top identity differs')
    scale = Decimal(str(layout.dbu)) * 1000
    _require(scale > 0 and scale == scale.to_integral_value(), 'native GDS requires integer nm DBU')
    transform = kdb.ICplxTrans(int(scale), 0, False, 0, 0)
    shapes, polygons = {}, []
    for index in layout.layer_indices():
        info = layout.get_info(index)
        key = (info.layer, info.datatype)
        region = kdb.Region(top.begin_shapes_rec(index)).transformed(transform)
        parts = list(region.each())
        if not parts:
            continue
        shapes[key] = parts
        for part in parts:
            clean = part.resolved_holes()
            points = [(p.x, p.y) for p in clean.each_point_hull()]
            _require((kdb.Region(kdb.Polygon([kdb.Point(*p) for p in points])) ^ kdb.Region(part)).is_empty(),
                     'native polygon conversion lost a hole or contour')
            polygons.append(LayerPolygon(info.layer, info.datatype,
                                          Polygon([Point(x, y) for x, y in points])))
    _require(polygons, 'native primitive GDS contains no polygons')
    return shapes, polygons


def _raw_devices(text, model, terminal_count):
    rows = []
    for line in text.splitlines():
        tokens = line.split()
        if not tokens or not tokens[0].casefold().startswith('x'):
            continue
        _require(len(tokens) > terminal_count + 1 and tokens[terminal_count + 1].casefold() == model.casefold(),
                 'generated PCell contains an unexpected native model or arity')
        params = dict(token.casefold().split('=', 1) for token in tokens[terminal_count + 2:])
        rows.append((tuple(n.casefold() for n in tokens[1:terminal_count + 1]), params))
    _require(rows, 'native primitive extraction contains no devices')
    return rows


def _raw_anchors(text, scale_nm):
    anchors = {}
    kinds = {}
    for line in text.splitlines():
        parts = shlex.split(line)
        if parts and parts[0] in {'node', 'substrate'}:
            _require(len(parts) >= 7, 'truncated native extraction node')
            name = parts[1].casefold()
            anchor = (int(parts[4]) * scale_nm, int(parts[5]) * scale_nm, parts[6])
            _require(name not in anchors or (anchors[name] == anchor and kinds[name] == parts[0]),
                     'conflicting native extraction node: ' + name)
            anchors[name] = anchor
            kinds[name] = parts[0]
    return anchors


def _role_nodes(rows, spec):
    result = defaultdict(set)
    roles = [r for _, r in spec['contract']['terminals']]
    if spec['family'] != 'mos':
        for nodes, _ in rows:
            for role, node in zip(roles, nodes):
                result[role].add(node)
        return result
    sides, edges = {}, []
    for nodes, _ in rows:
        d, g, s, b = nodes
        result['g'].add(g); result['b'].add(b)
        edges.append((d, s))
        for node in (d, s):
            if re.search(r'(?:^|[./])d(?:\d|_|$)', node):
                sides[node] = 'd'
            elif re.search(r'(?:^|[./])s(?:\d|_|$)', node):
                sides[node] = 's'
    for _ in range(len(edges) + 1):
        changed = False
        for a, b in edges:
            if a in sides and b not in sides:
                sides[b] = 's' if sides[a] == 'd' else 'd'; changed = True
            elif b in sides and a not in sides:
                sides[a] = 's' if sides[b] == 'd' else 'd'; changed = True
            elif a in sides and b in sides:
                _require(sides[a] != sides[b], 'native PCell source/drain coloring contradicts its port labels')
        if not changed:
            break
    _require(set(sides) == {n for pair in edges for n in pair}, 'native diffusion occurrence has no port authority')
    for node, role in sides.items():
        result[role].add(node)
    return result


@dataclass(frozen=True)
class NativeDeviceGeometry(DeviceGeom):
    terminal_accesses: dict = field(default_factory=dict)
    terminal_layers: dict = field(default_factory=dict)
    native_spec: dict = field(default_factory=dict)
    native_evidence: dict = field(default_factory=dict)


def orient_native_device(geometry, orientation, tech):
    """Apply a source-declared rotation to the complete observed native artwork.

    The original bundle remains the geometry authority.  This deliberately
    supports observed MOS and varactors with rotations. Terminal accesses are declared dataclass fields on this
    subclass and must be transported along with the generic DeviceGeom artwork.
    """
    _require(type(geometry) is NativeDeviceGeometry and geometry.kind in ('native_varactor', 'nmos', 'pmos'),
             'native orientation supports observed MOS and varactors')
    rotations = ('R0', 'R90', 'R180', 'R270')
    reflections = ('MR0', 'MR90', 'MR180', 'MR270')
    _require(orientation in rotations or (orientation in reflections and geometry.kind in ('nmos', 'pmos')),
             'native orientation must be an orthogonal rotation or an observed MOS reflection')
    if orientation == 'R0':
        return geometry
    from clay.layout.device_cell import _device_grid_nm, _rotate_box
    from clay.layout.passive_device_cell import _transform_devicegeom

    grid = _device_grid_nm(tech)
    _require(grid > 0 and all(v.x % grid == 0 and v.y % grid == 0
                             for p in geometry.polygons for v in p.polygon.vertices),
             'native rotation would snap observed artwork')
    def transform_box(box):
        if orientation.startswith('MR'):
            x0, y0, x1, y1 = box
            box = (x0, -y1, x1, -y0)
            rotation = 'R' + orientation[2:]
        else:
            rotation = orientation
        return _rotate_box(box, rotation)
    rotated = _transform_devicegeom(geometry, orientation, tech)
    raw_box = transform_box(geometry.comp_bbox)
    dx, dy = rotated.comp_bbox[0] - raw_box[0], rotated.comp_bbox[1] - raw_box[1]
    _require(rotated.comp_bbox == (raw_box[0] + dx, raw_box[1] + dy,
                                  raw_box[2] + dx, raw_box[3] + dy),
             'native rotation changed the device extent')
    def access(terminal):
        box = transform_box(terminal.bbox)
        _require(all(value % grid == 0 for value in box) and dx % grid == 0 and dy % grid == 0,
                 'native rotation would snap a terminal access')
        return Terminal(role=terminal.role, net=terminal.net,
                        bbox=(box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy))
    rotated = replace(rotated, terminal_accesses={role: tuple(access(t) for t in rows)
                      for role, rows in geometry.terminal_accesses.items()})
    _require({role: rows[0] for role, rows in rotated.terminal_accesses.items()} == rotated.terminals,
             'native rotation lost its primary terminal binding')
    _require(rotated.terminal_layers == geometry.terminal_layers
             and rotated.native_spec == geometry.native_spec
             and rotated.native_evidence == geometry.native_evidence,
             'native rotation changed source or terminal authority')
    return rotated


def orient_native_varactor(geometry, orientation, tech):
    """Retain the varactor-only public helper for existing callers."""
    _require(type(geometry) is NativeDeviceGeometry and geometry.kind == 'native_varactor',
             'native orientation currently supports only observed varactors')
    return orient_native_device(geometry, orientation, tech)


def _observe_mos_channel_sites(shapes, rows, ext_text, grid_nm, spec, checkpoint):
    """Observe complete local channel sites without assigning electrical ends.

    EXT anchors are producer-local evidence, not reference-layout positions.
    This is a geometric observation only; body, cuts and source-net binding are
    outside its scope. Ambiguous geometry remains UNKNOWN.
    """
    from clay.layout.occurrence_channels import _shape_keys
    from clay.layout.physical_terminal_inventory import PhysicalTerminalPolygonNM
    from clay.layout.terminal_body_geometry import (
        SharedMosChannelUnsupported, observe_shared_mos_channel_nm,
    )
    from clay.verification.native_rc import NativeEvidenceError, parse_magic_ext

    def unknown(reason):
        return {'status': 'UNKNOWN', 'reason': reason, 'occurrences': ()}

    checkpoint()
    active = kdb.Region(shapes.get(_DIFF, [])).merged()
    poly = kdb.Region(shapes.get(_POLY, [])).merged()
    channels = tuple(kdb.Region(p) for p in (active & poly).merged().each())
    try:
        extracted = parse_magic_ext(ext_text)['devices']
    except (NativeEvidenceError, ValueError):
        return unknown('EXT_DEVICE_FORMAT_OR_MODEL')
    records = []
    for row in extracted.values():
        checkpoint()
        fields, nodes = row['tokens'], row['nodes']
        if (fields[1] != 'msubckt' or row['model'] != spec['model'].casefold()
                or len(nodes) != 4):
            return unknown('EXT_DEVICE_FORMAT_OR_MODEL')
        try:
            records.append((int(fields[3]) * grid_nm, int(fields[4]) * grid_nm,
                            nodes[0], nodes[1]))
        except ValueError:
            return unknown('EXT_DEVICE_COORDINATE')
    if len(records) != spec['occurrences'] or len(channels) != spec['occurrences']:
        return unknown('OCCURRENCE_DENOMINATOR')
    if Counter((body, gate) for _, _, body, gate in records) != Counter(
            (nodes[3], nodes[1]) for nodes, _ in rows):
        return unknown('EXT_SPICE_GATE_BODY_MULTISET')
    gates = tuple(kdb.Region(p) for p in poly.each())
    used, result = set(), []
    for x, y, _body, _gate in records:
        checkpoint()
        probe = kdb.Region(kdb.Box(x, y, x + 1, y + 1))
        hits = [i for i, channel in enumerate(channels) if not (channel & probe).is_empty()]
        if len(hits) != 1 or hits[0] in used:
            return unknown('CHANNEL_ANCHOR_AMBIGUOUS')
        index, = hits
        used.add(index)
        channel = channels[index]
        matching_gates = [gate for gate in gates if not (gate & channel).is_empty()]
        if len(matching_gates) != 1:
            return unknown('GATE_COMPONENT_AMBIGUOUS')
        try:
            observation = observe_shared_mos_channel_nm(active, poly, channel)
            if (observation.width_nm, observation.length_nm) != (spec['w_nm'] // spec['nf'], spec['l_nm']):
                return unknown('CHANNEL_DIMENSIONS_DIFFER')
            channel_shapes = tuple(PhysicalTerminalPolygonNM(*item) for item in _shape_keys(channel))
            gate_shapes = tuple(PhysicalTerminalPolygonNM(*item) for item in _shape_keys(matching_gates[0]))
        except (SharedMosChannelUnsupported, ValueError):
            return unknown('CHANNEL_SITE_UNSUPPORTED')
        if len(channel_shapes) != 1 or len(gate_shapes) != 1:
            return unknown('CHANNEL_SITE_AMBIGUOUS')
        result.append((channel_shapes[0], gate_shapes[0], observation.diffusion_sides_nm))
    if len(used) != len(channels):
        return unknown('CHANNEL_ANCHOR_INCOMPLETE')
    return {'status': 'OBSERVED', 'reason': None, 'occurrences': tuple(sorted(result))}


def observe_native_primitive(folder, spec, *, _mos_site_sink=None, _mos_site_checkpoint=None,
                             _bound_inputs=None):
    """Independent GDS reopen and complete native channel/terminal observation."""
    folder = Path(folder)
    if _bound_inputs is None:
        drc = (folder / 'drc.txt').read_text()
        _require(not drc.strip(), 'native primitive failed full Magic DRC')
        shapes, polygons = _read_gds(folder / 'primitive.gds')
        raw = (folder / 'native.spice').read_text()
        scale_text = (folder / 'scale_um.txt').read_text()
        ext_text = (folder / 'primitive_flat.ext').read_text()
    else:
        expected = {'drc.txt', 'primitive.gds', 'native.spice', 'scale_um.txt', 'primitive_flat.ext'}
        _require(type(_bound_inputs) is dict and set(_bound_inputs) == expected
                 and all(type(value) is bytes for value in _bound_inputs.values()),
                 'native primitive bound observation inputs differ')
        drc = _bound_inputs['drc.txt'].decode('utf-8')
        _require(not drc.strip(), 'native primitive failed full Magic DRC')
        shapes, polygons = _read_gds(folder / 'primitive.gds', _bytes=_bound_inputs['primitive.gds'])
        raw = _bound_inputs['native.spice'].decode('utf-8')
        scale_text = _bound_inputs['scale_um.txt'].decode('utf-8')
        ext_text = _bound_inputs['primitive_flat.ext'].decode('utf-8')
    rows = _raw_devices(raw, spec['model'], len(spec['contract']['terminals']))
    _require(len(rows) == spec['occurrences'], 'native physical occurrence denominator differs from nf/m/mult')
    w_nm = spec['w_nm'] // spec['nf']
    for _, params in rows:
        _require(Decimal(params['w']) * 1000 == w_nm and Decimal(params['l']) * 1000 == spec['l_nm'],
                 'native PCell silently changed source W/L or finger split')
    scale_nm = Decimal(scale_text.strip()) * 1000
    # Magic reports its exact integer grid through a single-precision Tcl value
    # (5nm prints as 4.999999888nm). Recover only that bounded representation dust.
    grid_nm = int(scale_nm.to_integral_value())
    _require(grid_nm > 0 and abs(scale_nm - grid_nm) < Decimal('0.000001'),
             'native extraction grid is not integer nm')
    anchors = _raw_anchors(ext_text, grid_nm)
    role_nodes = _role_nodes(rows, spec)
    nodes = extract_nodes(shapes, native_connect_model(spec['family']))
    type_keys = {'ndif': _SD, 'pdif': _SD, 'nnd': _SD, 'ndiff': _SD,
                 'p': _POLY, 'poly': _POLY, 'pc': _POLY, 'ppd': _TAP, 'npd': _TAP,
                 'nw': _NWELL, 'nwell': _NWELL, 'mim': _CAPM,
                 **{f'm{i+1}': key for i, key in enumerate(_METALS)}}
    used_nodes, terminal_layers, accesses = {}, {}, {}
    well_roles = set()
    for role, names in sorted(role_nodes.items()):
        terms, role_layers = [], set()
        for name in sorted(names):
            _require(name in anchors, 'physical terminal has no native extraction coordinate: ' + name)
            x, y, typename = anchors[name]
            _require(typename in type_keys, 'unsupported native anchor material: ' + typename)
            key = type_keys[typename]
            probe = kdb.Region(kdb.Box(x, y, x + 1, y + 1))
            hits = [index for index, node in enumerate(nodes)
                    if key in node and not (node[key] & probe).is_empty()]
            _require(len(hits) == 1, 'native terminal coordinate does not identify one conductor: ' + name)
            index, = hits
            previous = used_nodes.get(index)
            _require(previous in (None, name), 'independent native extraction nodes merged in observed geometry')
            used_nodes[index] = name
            node = nodes[index]
            if _NWELL in node and not node[_NWELL].is_empty():
                well_roles.add(role)
            available = [(layer, node[layer].merged()) for layer in _METALS
                         if layer in node and not node[layer].is_empty()]
            _require(available, 'native terminal has no real routing-metal access: ' + name)
            layer, metal = available[0]
            # Multiple disconnected routing islands need the existing AND/OR
            # domain consumer. Never silently choose one or turn OR into AND.
            _require(len(list(metal.each())) == 1, 'JOINT_ROUTING_ACCESS_UNSUPPORTED: ' + name)
            boxes = [p.bbox() for p in metal.decompose_trapezoids_to_region().each() if p.is_box()]
            _require(boxes, 'native routing conductor has no rectangular access')
            box = max(boxes, key=lambda b: (b.area(), -b.left, -b.bottom))
            terms.append(Terminal(role, (box.left, box.bottom, box.right, box.top)))
            role_layers.add(layer)
        _require(len(role_layers) == 1, 'one native terminal uses heterogeneous entry layers')
        terminal_layers[role] = next(iter(role_layers))
        accesses[role] = tuple(terms)
    routing_nodes = {i for i, node in enumerate(nodes) if any(key in node for key in _METALS)}
    _require(routing_nodes == set(used_nodes), 'native PCell has unowned or unobserved routing conductors')
    well_nodes = {i for i, node in enumerate(nodes)
                  if _NWELL in node and not node[_NWELL].is_empty()}
    _require(well_nodes <= set(used_nodes), 'native PCell has an unowned nwell conductor')
    _require(len(well_roles) <= 1, 'native PCell has multiple independently biased nwell roles')
    all_boxes = [p.bbox() for parts in shapes.values() for p in parts]
    extent = (min(b.left for b in all_boxes), min(b.bottom for b in all_boxes),
              max(b.right for b in all_boxes), max(b.top for b in all_boxes))
    kind = ('pmos' if '__pfet_' in spec['model'] else 'nmos') if spec['family'] == 'mos' else 'native_' + spec['family']
    evidence = {'model': spec['model'], 'source_spec': spec, 'physical_occurrences': len(rows),
                'terminal_nodes': {r: sorted(n) for r, n in role_nodes.items()},
                'nwell_terminal_role': next(iter(well_roles), None),
                'polygons': len(polygons), 'routing_nodes': len(routing_nodes),
                'magic_drc_clean': not drc.strip(),
                'artifacts': {p.name: _sha(p) for p in folder.iterdir() if p.is_file()}}
    if _mos_site_sink is not None and spec['family'] == 'mos':
        checkpoint = _mos_site_checkpoint if _mos_site_checkpoint is not None else lambda: None
        _mos_site_sink.append(_observe_mos_channel_sites(shapes, rows, ext_text, grid_nm, spec, checkpoint))
    return NativeDeviceGeometry(polygons=polygons, terminals={r: ts[0] for r, ts in accesses.items()},
        kind=kind, finger_count=spec['occurrences'], gate_length_nm=spec['l_nm'], channel_width_nm=w_nm,
        comp_bbox=extent, terminal_accesses=accesses, terminal_layers=terminal_layers,
        native_spec=spec, native_evidence=evidence)


def _is_valid_producer_digest(value):
    return isinstance(value, str) and bool(re.fullmatch(r'[0-9a-f]{64}', value)) and value != '0' * 64


class NativeGeometryBundle:
    """Complete, current source/producer/artifact authority, not a PASS flag."""
    def __init__(self, *args, **kwargs):
        raise TypeError('use materialize_native_sky130')

    def require_current(self, cell):
        _require(all(d.device_class == 'native' for d in cell.devices), 'native source contains mixed device classes')
        _require(native_source_identity(cell) == self.source_identity, 'native geometry source changed')
        _require(_is_valid_producer_digest(getattr(self, 'producer_sha256', None)), 'native geometry producer changed')
        live_seal = bool(getattr(self, '_live_session_seal', False))
        manifest = None
        if live_seal:
            _require(_sha(__file__) == self.producer_sha256, 'native geometry producer changed')
        else:
            manifest_path = self.output_dir / 'geometry.json'
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                _require(manifest.get('producer_sha256') == self.producer_sha256, 'native geometry producer changed')
        for path, digest in self.common_artifacts.items():
            _require(_sha(self.output_dir / path) == digest, 'native geometry execution evidence changed: ' + path)
        if live_seal:
            for path, digest in self.dependencies.items():
                _require(_sha(path) == digest, 'native geometry dependency changed: ' + path)
        else:
            _require(isinstance(self.dependencies, dict) and bool(self.dependencies)
                     and (manifest is None or manifest.get('dependencies') == self.dependencies)
                     and all(_is_valid_producer_digest(digest) for digest in self.dependencies.values()),
                     'native geometry dependency changed')
            ref_spice = self.output_dir / 'reference.spice'
            _require(ref_spice.is_file()
                     and ref_spice.read_text(encoding='utf-8').strip() == emit_native_spice(cell, pdk_id='sky130').strip(),
                     'native geometry execution evidence changed: reference.spice')
        _require(set(self.geometries) == {d.instance_name for d in cell.devices}, 'native geometry member denominator changed')
        current_specs = _cell_native_specs(cell)
        for device in cell.devices:
            _require(current_specs[device.instance_name] == self.geometries[device.instance_name].native_spec,
                     'native physical model contract changed: ' + device.instance_name)
        checked_files: dict[Path, str] = {}
        for name, geometry in self.geometries.items():
            folder = self.folders[name]
            for path, digest in geometry.native_evidence['artifacts'].items():
                fpath = folder / path
                actual_sha = checked_files.get(fpath)
                if actual_sha is None:
                    actual_sha = _sha(fpath)
                    checked_files[fpath] = actual_sha
                _require(actual_sha == digest, 'native primitive artifact changed: ' + name + '/' + path)
            _require(_geometry_identity(geometry) == self.geometry_identities[name], 'native primitive geometry/terminals changed: ' + name)
        return self

    def _require_individual_placements(self, placement, cell, names):
        """Replay selected standalone native masks and their exact terminal accesses."""
        from clay.layout.generic_placement import (_place_polys, _placed_terminal_accesses,
                                                   annotated_cell_from_cell_def)
        from clay.layout.annotate_to_solver import _orient_map_from_cell
        from clay.layout.tech import LayoutTech
        annotated, _ = annotated_cell_from_cell_def(cell, native_geometry=self)
        declared = _orient_map_from_cell(annotated)
        _require(set(declared) <= set(self.geometries), 'native orientation names an unknown device')
        for name in sorted(names):
            base = self.geometries[name]
            if name in declared:
                base = orient_native_device(base, declared[name], LayoutTech.resolve('sky130'))
            placed = placement.devices[name]
            extent = base.extent()
            box = placed.bbox
            _require((box[2]-box[0], box[3]-box[1]) == (extent[2]-extent[0], extent[3]-extent[1]),
                     'native placement resized device: ' + name)
            dx, dy = box[0]-extent[0], box[1]-extent[1]
            _require(dx % 5 == 0 and dy % 5 == 0, 'native placement is off the manufacturing grid: ' + name)
            matched = False
            for mirrored in ((False,) if name in declared else (False, True)):
                polygons = _place_polys(base, dx, dy, mirrored, 0)
                accesses = _placed_terminal_accesses(base, dx, dy, mirrored)
                if (_polygon_inventory(polygons) == _polygon_inventory(placed.polys)
                        and accesses == placed.terminal_accesses
                        and {r: ts[0] for r, ts in accesses.items()} == placed.terminals
                        and base.terminal_layers == placed.term_layers):
                    matched = True
                    break
            _require(matched, 'native placement altered masks or terminal ownership: ' + name)

    def require_placement(self, placement, cell):
        """Replay all owned masks and accesses under one rigid transform per device."""
        self.require_current(cell)
        _require(set(placement.devices) == set(self.geometries), 'native placement member denominator changed')
        self._require_individual_placements(placement, cell, set(self.geometries))
        _require(_polygon_inventory(placement.polys) == _polygon_inventory(
            [p for d in placement.devices.values() for p in d.polys]),
            'native placement contains unowned or missing polygons')
        return self

    def require_partitioned_placement(self, placement, cell):
        """Replay source-owned composite rows and every remaining native device.

        The native bundle remains the complete source/PDK authority.  A row is
        admitted only through its retained source projection and final stored-row
        replay; standalone devices retain the exact native artwork check above.
        """
        from clay.layout.generic_placement import (verify_row_source_compiler_authority,
                                                   verify_stored_row_projection)
        from clay.layout.tech import LayoutTech

        self.require_current(cell)
        _require(set(placement.devices) == set(self.geometries), 'native placement member denominator changed')
        program = placement.row_program
        materialization = placement.row_materialization
        _require(program is not None and materialization is not None, 'native row context is missing')
        projection = materialization.source_projection
        _require(projection is not None, 'native row source projection is missing')
        tech = LayoutTech.resolve('sky130')
        # Retain the actual compiler objects and complete member inventory as
        # well as replaying their source. Canonical row bytes alone omit free
        # members and do not prove the pre-solver compiler binding.
        verify_row_source_compiler_authority(placement, cell, tech)
        verify_stored_row_projection(placement)
        members = tuple(name for entry in program.entries for name in entry.consumed_devices)
        _require(bool(members) and len(set(members)) == len(members)
                 and set(members) <= set(self.geometries), 'native row source partition differs')
        self._require_individual_placements(placement, cell, set(self.geometries) - set(members))
        return self


def _polygon_inventory(polygons):
    def canonical(points):
        return min(tuple(points[i:] + points[:i]) for i in range(len(points)))
    return Counter((p.layer, p.datatype,
                    min(canonical([(v.x, v.y) for v in p.polygon.vertices]),
                        canonical([(v.x, v.y) for v in reversed(p.polygon.vertices)])))
                   for p in polygons)


@dataclass(frozen=True)
class NativeRouteDevice:
    """Full native topology consumed alongside already-materialized artwork."""
    name: str
    kind: str
    model: str
    native_terminals: dict
    nets: dict
    W: float
    L: float
    M: int
    native_well_net: str | None


def native_route_devices(cell, bundle):
    bundle.require_current(cell)
    roles = {'s': 'source', 'd': 'drain', 'g': 'gate', 'b': 'body'}
    result = []
    for device in cell.devices:
        geom = bundle.geometries[device.instance_name]
        pins = dict(device.pins)
        _require(set(pins) == set(geom.terminal_accesses), 'native source/physical terminal roles differ')
        well_role = geom.native_evidence['nwell_terminal_role']
        result.append(NativeRouteDevice(device.instance_name, geom.kind, device.source_model,
            pins, {roles.get(r, r): n for r, n in pins.items()},
            geom.native_spec['w_nm'] / 1000, geom.native_spec['l_nm'] / 1000,
            geom.native_spec['occurrences'], pins[well_role] if well_role is not None else None))
    return result


def _geometry_identity(geometry):
    return _digest({'polygons': [(p.layer, p.datatype, [(v.x, v.y) for v in p.polygon.vertices])
                                for p in geometry.polygons],
                    'terminals': {r: [t.bbox for t in ts] for r, ts in geometry.terminal_accesses.items()},
                    'primary_terminals': {r: t.bbox for r, t in geometry.terminals.items()},
                    'layers': geometry.terminal_layers, 'spec': geometry.native_spec,
                    'evidence': geometry.native_evidence, 'kind': geometry.kind,
                    'finger_count': geometry.finger_count, 'gate_length_nm': geometry.gate_length_nm,
                    'channel_width_nm': geometry.channel_width_nm, 'comp_bbox': geometry.comp_bbox,
                    'nwell_bbox': geometry.nwell_bbox, 'placement_extent': geometry.placement_extent,
                    'orient': geometry.orient, 'mirror': geometry.mirror})


def _producer_dependencies(*, checkpoint=None):
    from clay import pdk, port_roles, spice_analysis
    from clay.frontend import native_spice_import
    from clay.layout import (annotations, backend, connectivity, connectivity_queries, device_cell,
                             generic_placement, passive_device_cell)
    from clay.layout.spatial import collision_queries
    from clay.pdk import geometry, native_spice, sky130, sky130_native_geometry
    from clay.verification import native_rc, native_sky130
    # Bind the helpers that interpret source units, extract native connectivity,
    # validate the runtime and replay placed artwork, including transitive calls.
    # This provider's own bytes are bound separately by producer_sha256.
    modules = (native_spice_import, native_spice, sky130, sky130_native_geometry, geometry, native_sky130,
               annotations, backend, connectivity, connectivity_queries, device_cell, passive_device_cell,
               generic_placement, collision_queries, pdk, port_roles, spice_analysis, native_rc)
    return {str(Path(module.__file__).resolve()):
            (_sha(module.__file__) if checkpoint is None else _stage_sha(module.__file__, checkpoint))
            for module in modules}


def materialize_native_sky130(cell_def, *, output_dir, image, pdk_root='/foss/pdks',
                              expected_runtime_files=None, timeout_seconds=600,
                              owned_execution=None, runtime_sha256=None, checkpoint=None,
                              source_runtime=None):
    """Generate all source devices once and return the existing placer's geometry."""
    _require(bool(re.fullmatch(r'sha256:[0-9a-f]{64}', str(image))), 'native geometry requires a pinned image digest')
    _require(all(d.device_class == 'native' for d in cell_def.devices) and bool(cell_def.devices),
             'native geometry requires a complete native source cell')
    if owned_execution is not None:
        _require(callable(checkpoint), 'native primitive execution requires a checkpoint')
    producer_identity = _sha(__file__) if owned_execution is None else _stage_sha(__file__, checkpoint)
    dependencies = (_producer_dependencies() if owned_execution is None
                    else _producer_dependencies(checkpoint=checkpoint))
    if owned_execution is not None:
        from clay.flow import native_owned_container, owned_process, control_stop, resource_budget
        for module in (native_owned_container, owned_process, control_stop, resource_budget):
            dependencies[str(Path(module.__file__).resolve())] = _stage_sha(module.__file__, checkpoint)
    source_identity = native_source_identity(cell_def)
    source_specs = _cell_native_specs(cell_def)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = emit_native_spice(cell_def, pdk_id='sky130')
    (output / 'reference.spice').write_text(source, encoding='utf-8')
    specs, members = {}, {}
    for device in cell_def.devices:
        spec = source_specs[device.instance_name]
        key = 'pcell_' + _digest(spec)[:20]
        specs[key] = spec
        members[device.instance_name] = key
    role_trace_request = getattr(owned_execution, 'role_trace_request', None)
    role_trace_keys = []
    if role_trace_request is not None:
        from clay.flow import primitive_role_trace, primitive_role_trace_capture
        from clay.flow import primitive_mim_role_trace
        from clay.flow.primitive_mim_role_trace import eligible_for_request
        for module in (primitive_role_trace, primitive_role_trace_capture, primitive_mim_role_trace):
            dependencies[str(Path(module.__file__).resolve())] = _stage_sha(module.__file__, checkpoint)
        role_trace_keys = [key for key, spec in specs.items() if eligible_for_request(spec, role_trace_request)]
    for key, spec in sorted(specs.items()):
        folder = output / key
        folder.mkdir()
        _write(folder / 'request.json', spec)
        control_tcl = _generation_tcl(spec)
        if role_trace_request is None:
            (folder / 'generate.tcl').write_text(control_tcl, encoding='utf-8')
        else:
            (folder / 'generate.tcl').write_bytes(control_tcl.encode('utf-8'))
        if key in role_trace_keys:
            from clay.flow.primitive_mim_role_trace import instrument_for_request
            trace = folder / 'trace'
            trace.mkdir()
            (trace / 'generate.tcl').write_bytes(instrument_for_request(spec, control_tcl, role_trace_request).encode('utf-8'))
    from clay.verification.native_sky130 import (
        _toolchain_script, _runtime_expectations, _validate_toolchain)
    source_contracts = {'models': {s['model']: s['contract'] for s in specs.values()}}
    expected = _runtime_expectations(expected_runtime_files, pdk_root)
    toolchain = _toolchain_script(pdk_root, source_contracts, expected)
    if role_trace_request is None:
        (output / 'toolchain.py').write_text(toolchain, encoding='utf-8')
    else:
        (output / 'toolchain.py').write_bytes(toolchain.encode('utf-8'))
    runner = _runner_script(specs, pdk_root, role_trace_keys=role_trace_keys,
                            role_trace_enabled=role_trace_request is not None)
    if role_trace_request is None:
        (output / 'generate.py').write_text(runner, encoding='utf-8')
    else:
        (output / 'generate.py').write_bytes(runner.encode('utf-8'))
    name = 'clay-native-geometry-' + uuid.uuid4().hex[:12]
    command = ['docker', 'run', '--rm', '--network', 'none', '--name', name,
               '-v', f'{output.as_posix()}:/work', '-w', '/work',
               '-e', 'PDK=sky130A', '-e', f'PDK_ROOT={pdk_root}',
               '-e', f'PDKPATH={pdk_root}/sky130A', '--entrypoint', 'bash', image,
               '-lc', 'exec python3 /work/generate.py']
    start = time.monotonic()
    evidence_path = None
    if owned_execution is None:
        _write(output / 'command.json', command)
        try:
            proc = subprocess.run(command, capture_output=True, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=30)
            _write(output / 'execution.json', {'timeout': True, 'seconds': time.monotonic()-start})
            raise
        (output / 'stdout.log').write_bytes(proc.stdout); (output / 'stderr.log').write_bytes(proc.stderr)
        _write(output / 'execution.json', {'returncode': proc.returncode, 'seconds': time.monotonic()-start})
        _require(proc.returncode == 0, 'native PCell generator failed; see ' + str(output / 'stderr.log'))
    else:
        from clay.flow.native_owned_container import run_native_container
        _require(type(runtime_sha256) is str and len(runtime_sha256) == 64 and callable(checkpoint),
                 'native primitive execution requires bound runtime and checkpoint')
        _require(type(source_runtime) is dict and source_runtime.get('image') == image
                 and source_runtime.get('pdk_root') == pdk_root
                 and source_runtime.get('files') == expected_runtime_files
                 and timeout_seconds <= source_runtime.get('timeout_seconds', 0),
                 'native primitive runtime parameters differ from source')
        generated_inputs = [output/'reference.spice', output/'generate.py', output/'toolchain.py']
        generated_inputs.extend(output/key/name for key in sorted(specs)
                                for name in ('request.json', 'generate.tcl'))
        generated_inputs.extend(output/key/'trace'/'generate.tcl' for key in sorted(role_trace_keys))
        generated_hashes = {str(path.resolve(strict=True)): _stage_sha(path, checkpoint) for path in generated_inputs}
        evidence_path, _plan, _plan_identity = run_native_container(
            stage=owned_execution, work_dir=output, image=image, runtime_sha256=runtime_sha256,
            pdk_root=pdk_root, timeout_seconds=timeout_seconds, checkpoint=checkpoint,
            inputs_sha256=generated_hashes, runtime_contract=source_runtime)
        for path, digest in generated_hashes.items():
            checkpoint()
            _require(_stage_sha(path, checkpoint) == digest, 'native primitive generated input changed during execution')
        evidence = json.loads(evidence_path.read_bytes())
        _write(output/'command.json', next(row['argv'] for row in evidence['commands'] if row['argv'][1] == 'create'))
        start_record = next(row for row in evidence['commands'] if row['argv'][1:3] == ['start', '--attach'])
        (output/'stdout.log').write_bytes((output/start_record['stdout']).read_bytes())
        (output/'stderr.log').write_bytes((output/start_record['stderr']).read_bytes())
        _write(output/'execution.json', {'returncode': 0, 'seconds': time.monotonic()-start})
    # The external probe records its observations and the requested inventory;
    # exit status alone does not compare the observed hashes with that inventory.
    # Reuse the complete native-signoff validator before admitting any geometry.
    runtime = _validate_toolchain(output, source_contracts, pdk_root, expected)
    # Fresh runtime copies must match the already validated same-container
    # inventory; these bytes are later consumed by placement, not a new PDK.
    copied = (('sky130A.tech', 'libs.tech/magic/sky130A.tech'),
              ('sky130A_mr.drc', 'libs.tech/klayout/drc/sky130A_mr.drc'))
    if any((output / local).exists() for local, _ in copied):
        inventory = {row['path']: row for row in runtime['files']}
        for local, relative in copied:
            target = output / local
            remote = str(Path(pdk_root) / 'sky130A' / relative).replace('\\', '/')
            row = inventory.get(remote)
            _require(target.is_file() and row is not None
                     and target.stat().st_size == row['bytes']
                     and _sha(target) == row['sha256'],
                     'native PCell copied PDK rule bytes differ from validated toolchain')
    _write(output / 'runtime-validation.json', {
        'passed': True, 'expected_file_count': len(expected),
        'observed_file_count': len(runtime['files']),
        'toolchain_sha256': _sha(output / 'toolchain.json'),
    })
    observed = {key: observe_native_primitive(output / key, spec) for key, spec in specs.items()}
    if role_trace_request is not None:
        from clay.flow.primitive_role_trace_capture import build_role_trace_manifest
        from clay.flow.run_admission import _identity
        from clay.flow.run_workspace import atomic_json
        checkpoint()
        capability = build_role_trace_manifest(output=output, specs=specs, members=members,
            request=role_trace_request, invocation_id=owned_execution.invocation_id,
            source_identity=source_identity, runtime_sha256=runtime_sha256,
            runtime_file_identity=owned_execution._stage_sources['runtime'],
            plan_identity=_plan_identity, evidence_identity=_identity(evidence_path, checkpoint=checkpoint),
            checkpoint=checkpoint)
        atomic_json(output / 'role-trace.json', capability)
    bundle = object.__new__(NativeGeometryBundle)
    bundle.source_identity = source_identity
    bundle.producer_sha256 = producer_identity
    bundle.geometries = {name: observed[key] for name, key in members.items()}
    bundle.folders = {name: output / key for name, key in members.items()}
    bundle.geometry_identities = {name: _geometry_identity(g) for name, g in bundle.geometries.items()}
    bundle.output_dir = output
    bundle.common_artifacts = {p.name: _sha(p) for p in output.iterdir() if p.is_file()}
    bundle.dependencies = dependencies
    bundle._live_session_seal = True
    if evidence_path is not None:
        bundle.primitive_container_evidence = evidence_path
    _write(output / 'geometry.json', {'source_identity': source_identity, 'producer_sha256': bundle.producer_sha256,
                                    'members': members, 'primitives': {k: g.native_evidence for k, g in observed.items()},
                                    'geometry_identities': bundle.geometry_identities,
                                    'dependencies': dependencies, 'execution_artifacts': bundle.common_artifacts})
    bundle.common_artifacts['geometry.json'] = _sha(output / 'geometry.json')
    return bundle.require_current(cell_def)


__all__ = ['NativeSky130GeometryError', 'NativeDeviceGeometry', 'NativeGeometryBundle',
           'native_geometry_spec', 'observe_native_primitive', 'materialize_native_sky130']
