"""Read-only v1 observations of a restricted SKY130 PCell role trace.

This module instruments the existing native generator; it does not define a
PDK, infer source-terminal roles, or certify body, cuts, nets or admission.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

SCHEMA = 'clay.primitive-role-trace/v1'
_MODELS = frozenset(('sky130_fd_pr__nfet_01v8', 'sky130_fd_pr__pfet_01v8'))
_FILES = ('primitive_flat.ext', 'native.spice', 'drc.txt', 'scale_um.txt')
_LIMIT = 256 * 1024 * 1024

# Derived from the recorded v9 execution observer, not PDK source parsing.
_HOOK = r'''
set ::clay_seen {}
proc clay_emit {kind args} {
    set encoded [list CLAY4 $kind]
    foreach item $args {lappend encoded [binary encode hex [encoding convertto utf-8 $item]]}
    puts [join $encoded |]
}
proc clay_trace_label {command operation} {
    set frame [info level -1]
    set arguments [lindex $frame end]
    if {[catch {dict size $arguments}]} {
        clay_emit UNRESOLVED $command $frame [box values]
        return
    }
    set hits {}
    foreach role {bulk drain source gate} {
        if {[dict exists $arguments $role] && [dict get $arguments $role] eq [lindex $command 1]} {
            lappend hits $role
        }
    }
    if {[llength $hits] != 1} {
        clay_emit UNRESOLVED $command $frame [box values]
        return
    }
    set role [lindex $hits 0]
    foreach field {nf m doports guard} {
        if {![dict exists $arguments $field]} {
            clay_emit UNRESOLVED $command $frame [box values]
            return
        }
    }
    lappend ::clay_seen [list $role [lindex $command 1]]
    clay_emit ROLE $role [lindex $frame 0] [lindex $command 0] [lindex $command 1] \
        [lindex $command 3] [join [box values] ,] [dict get $arguments $role] \
        [dict get $arguments nf] [dict get $arguments m] [dict get $arguments doports] \
        [dict get $arguments guard]
}
trace add execution label enter clay_trace_label
'''

_QUERY = r'''
trace remove execution label enter clay_trace_label
select top cell
set clay_children [cellname list children]
clay_emit CHILDREN [llength $clay_children] $clay_children
if {[llength $clay_children] != 1} {exit 3}
load [lindex $clay_children 0]
box 0 0 0 0
clay_emit PORT_RANGE [port first] [port last]
foreach clay_item $::clay_seen {
    lassign $clay_item clay_role clay_name
    findlabel $clay_name
    select clear
    select area labels
    if {[catch {
        clay_emit PORT $clay_role $clay_name [port $clay_name index] [port $clay_name exists] \
            [setlabel text] [setlabel layer] [join [setlabel box] ,] [join [box values] ,]
    } clay_failure]} {
        clay_emit UNRESOLVED $clay_failure
        exit 3
    }
}
load primitive
box 0 0 0 0
puts CLAY4_STYLE_BEFORE_BEGIN
cif ostyle
puts CLAY4_STYLE_BEFORE_END
'''

_STYLE_AFTER = r'''
puts CLAY4_STYLE_AFTER_BEGIN
cif ostyle
puts CLAY4_STYLE_AFTER_END
'''


def eligible_spec(spec):
    """Eligibility only; guard is independently checked in the execution frame."""
    return (type(spec) is dict and spec.get('family') == 'mos' and spec.get('model') in _MODELS
            and all(type(spec.get(name)) is int and spec[name] == 1
                    for name in ('nf', 'm', 'mult', 'replicas'))
            and spec.get('body_tap') is None)


def instrument_generation_tcl(spec, control_tcl):
    """Add trace hooks only to the exact existing producer-generated script."""
    from clay.layout.native_sky130 import _generation_tcl
    if not eligible_spec(spec):
        raise ValueError('native spec is outside role trace v1 scope')
    if not isinstance(control_tcl, str) or control_tcl != _generation_tcl(spec):
        raise ValueError('control script differs from the native producer')
    gen = 'if {[catch {magic::gencell'
    gds = '\nselect top cell\ngds write primitive.gds'
    if control_tcl.count(gen) != 1 or control_tcl.count(gds) != 1:
        raise ValueError('native producer instrumentation anchors differ')
    result = control_tcl.replace(gen, _HOOK + '\n' + gen, 1)
    return result.replace(gds, '\n' + _QUERY + gds + _STYLE_AFTER, 1)


def _box(value):
    if not re.fullmatch(r'\s*[+-]?\d+(?:(?:\s*,\s*|\s+)[+-]?\d+){3}\s*', value, re.ASCII):
        raise ValueError('trace box must have four integers')
    return tuple(map(int, re.findall(r'[+-]?\d+', value, re.ASCII)))


def _style(text, label):
    start, end = 'CLAY4_STYLE_' + label + '_BEGIN', 'CLAY4_STYLE_' + label + '_END'
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) >= text.index(end):
        raise ValueError('trace style marker missing or repeated')
    part = text.split(start, 1)[1].split(end, 1)[0]
    hits = re.findall(r'The current style is "([^"]+)"\.', part)
    if len(hits) != 1:
        raise ValueError('trace style is not unique')
    return hits[0]


def verify_trace_stdout(data):
    """Validate observed PDK frames and Magic ports without assigning source nets."""
    if not isinstance(data, bytes) or len(data) > 16 * 1024 * 1024:
        raise ValueError('invalid trace stdout bytes')
    text = data.decode('utf-8')
    rows = []
    for line in text.splitlines():
        if not line.startswith('CLAY4|'):
            continue
        fields = line.split('|')
        if len(fields) < 2 or fields[1] not in ('ROLE', 'PORT', 'CHILDREN', 'PORT_RANGE'):
            raise ValueError('unresolved or unknown trace record')
        try:
            decoded = tuple(bytes.fromhex(v).decode('utf-8') for v in fields[2:])
        except (ValueError, UnicodeError) as exc:
            raise ValueError('malformed trace record') from exc
        rows.append((fields[1], decoded))
    roles = [x for kind, x in rows if kind == 'ROLE']
    ports = [x for kind, x in rows if kind == 'PORT']
    children = [x for kind, x in rows if kind == 'CHILDREN']
    ranges = [x for kind, x in rows if kind == 'PORT_RANGE']
    expected = {'bulk', 'drain', 'source', 'gate'}
    if (len(roles) != 4 or any(len(x) != 11 for x in roles)
            or len(ports) != 4 or any(len(x) != 8 for x in ports)
            or {x[0] for x in roles} != expected or {x[0] for x in ports} != expected
            or len(children) != 1 or len(children[0]) != 2 or children[0][0] != '1'
            or not children[0][1] or len(ranges) != 1 or len(ranges[0]) != 2):
        raise ValueError('trace role, port or child denominator differs')
    by_role, by_port = {r[0]: r for r in roles}, {p[0]: p for p in ports}
    if len({r[3] for r in roles}) != 4 or len({p[2] for p in ports}) != 4:
        raise ValueError('trace labels or port indices are not unique')
    for role in sorted(expected):
        row, port = by_role[role], by_port[role]
        proc = 'sky130::guard_ring' if role == 'bulk' else 'sky130::mos_device'
        if (row[1] != proc or row[2] != 'label' or row[3] != row[6] or not row[3]
                or row[7:] != ('1', '1', '1', '1') or port[1] != row[3]
                or port[3] != '1' or port[4] != row[3] or not port[5]
                or not re.fullmatch(r'[0-9]+', port[2]) or int(port[2]) <= 0
                or _box(port[6]) != _box(row[5])):
            raise ValueError('executed label or saved port differs')
    if (any(not re.fullmatch(r'[0-9]+', value) for value in ranges[0])
            or int(ranges[0][0]) != min(int(port[2]) for port in ports)
            or int(ranges[0][1]) != max(int(port[2]) for port in ports)):
        raise ValueError('reported trace port range differs')
    if text.index('CLAY4_STYLE_BEFORE_END') >= text.index('CLAY4_STYLE_AFTER_BEGIN'):
        raise ValueError('trace style observations are out of order')
    before, after = _style(text, 'BEFORE'), _style(text, 'AFTER')
    if before != after:
        raise ValueError('trace changed Magic output style')
    return {'schema': 'clay.pdk-role-frames/v1', 'roles': by_role, 'ports': by_port,
            'children': children[0], 'port_range': ranges[0], 'style': before,
            'source_terminal_binding': 'UNKNOWN', 'physical_admission': False}


def _valid_dates(payload):
    if len(payload) != 24:
        raise ValueError('GDS begin record timestamp size differs')
    values = struct.unpack('>12h', payload)
    for i in (0, 6):
        year, month, day, hour, minute, second = values[i:i+6]
        try:
            if year < 0:
                raise ValueError('negative GDS year')
            datetime(1900 + year, month, day, hour, minute, second)
        except (ValueError, OverflowError) as exc:
            raise ValueError('invalid GDS timestamp') from exc


def normalize_gds_timestamps(data, checkpoint=lambda: None):
    """Validate framing and nesting; normalize only legal begin timestamps."""
    if not isinstance(data, bytes) or not data or len(data) > _LIMIT:
        raise ValueError('invalid GDS bytes')
    out = bytearray(data)
    offset = 0
    stage = 'header'
    in_structure = False
    in_element = False
    have_libname = have_units = False
    structure_count = 0
    have_strname = False
    element_types = {0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x15, 0x2d}
    while offset < len(data):
        checkpoint()
        if len(data) - offset < 4:
            raise ValueError('truncated GDS record header')
        size, kind, dtype = struct.unpack_from('>HBB', data, offset)
        if size < 4 or size % 2 or offset + size > len(data) or dtype > 6:
            raise ValueError('invalid GDS record framing')
        payload = data[offset+4:offset+size]
        if stage == 'header':
            if (kind, dtype, size) != (0, 2, 6):
                raise ValueError('GDS HEADER must be first')
            stage = 'bgnlib'
        elif stage == 'bgnlib':
            if (kind, dtype, size) != (1, 2, 28):
                raise ValueError('GDS BGNLIB must follow HEADER')
            _valid_dates(payload)
            out[offset+4:offset+28] = bytes(24)
            stage = 'library'
        elif stage == 'done':
            raise ValueError('GDS data follows ENDLIB')
        elif kind == 1 or kind == 0:
            raise ValueError('repeated GDS library header')
        elif kind == 2:  # LIBNAME
            if in_structure or in_element or have_libname or dtype != 6 or not payload:
                raise ValueError('invalid GDS LIBNAME')
            have_libname = True
        elif kind == 3:  # UNITS
            if in_structure or in_element or have_units or dtype != 5 or size != 20:
                raise ValueError('invalid GDS UNITS')
            have_units = True
        elif kind == 5:  # BGNSTR
            if in_structure or in_element or not have_libname or not have_units or dtype != 2 or size != 28:
                raise ValueError('invalid GDS BGNSTR')
            _valid_dates(payload)
            out[offset+4:offset+28] = bytes(24)
            in_structure, have_strname = True, False
            structure_count += 1
        elif kind == 6:  # STRNAME
            if not in_structure or in_element or have_strname or dtype != 6 or not payload:
                raise ValueError('invalid GDS STRNAME')
            have_strname = True
        elif kind == 7:  # ENDSTR
            if not in_structure or in_element or not have_strname or dtype != 0 or size != 4:
                raise ValueError('invalid GDS ENDSTR')
            in_structure = False
        elif kind in element_types:
            if not in_structure or not have_strname or in_element or dtype != 0 or size != 4:
                raise ValueError('invalid GDS element start')
            in_element = True
        elif kind == 0x11:  # ENDEL
            if not in_structure or not in_element or dtype != 0 or size != 4:
                raise ValueError('invalid GDS ENDEL')
            in_element = False
        elif kind == 4:  # ENDLIB
            if in_structure or in_element or not have_libname or not have_units or not structure_count or dtype != 0 or size != 4:
                raise ValueError('invalid GDS ENDLIB')
            stage = 'done'
        offset += size
    if stage != 'done' or in_structure or in_element:
        raise ValueError('incomplete GDS library')
    checkpoint()
    return bytes(out)


def _read(path, checkpoint):
    chunks = []
    size = 0
    with Path(path).open('rb') as stream:
        while True:
            checkpoint()
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > _LIMIT:
                raise ValueError('role trace artifact exceeds byte limit')
            chunks.append(chunk)
    checkpoint()
    return b''.join(chunks)


def compare_artifacts(control_dir, trace_dir, checkpoint=lambda: None):
    """Compare complete GDS records and exact extraction files for one pair."""
    control, trace = Path(control_dir), Path(trace_dir)
    details = {}
    for name in ('primitive.gds', *_FILES):
        before = _read(control / name, checkpoint)
        after = _read(trace / name, checkpoint)
        equivalent = (normalize_gds_timestamps(before, checkpoint) == normalize_gds_timestamps(after, checkpoint)
                      if name == 'primitive.gds' else before == after)
        if not equivalent:
            raise ValueError('control/trace artifact differs: ' + name)
        details[name] = {'control_sha256': sha256(before).hexdigest(), 'control_size_bytes': len(before),
                         'trace_sha256': sha256(after).hexdigest(), 'trace_size_bytes': len(after)}
    return {'schema': 'clay.primitive-role-noninterference/v1', 'status': 'OBSERVED',
            'scope': 'complete GDS records except legal begin timestamps; exact EXT/SPICE/DRC/scale bytes',
            'files': details, 'physical_admission': False}


def observe_label_components(trace_dir, lyp_bytes, trace_stdout_bytes, checkpoint=lambda: None):
    """Observe traced PDK label sites in native LI components, without net authority.

    The LYP bytes are a same-container PDK observation supplied by the caller;
    they are not treated as a source-explicit expected digest. EXT anchor names
    and the native observer's name-derived role nodes are deliberately unused.
    """
    from clay.layout.connectivity import extract_nodes
    from clay.layout.native_sky130 import _LI, _read_gds, native_connect_model
    import klayout.db as kdb

    if not isinstance(lyp_bytes, bytes) or len(lyp_bytes) > 16 * 1024 * 1024:
        raise ValueError('invalid PDK LYP bytes')
    found = {}
    for entry in ET.fromstring(lyp_bytes).iter('properties'):
        name, source = entry.findtext('name'), entry.findtext('source')
        if not name or not source:
            continue
        short = name.split(' - ', 1)[0]
        if short in ('li1.label', 'li1.drawing'):
            found.setdefault(short, []).append(source.split('@', 1)[0])
    if set(found) != {'li1.label', 'li1.drawing'} or any(len(items) != 1 for items in found.values()):
        raise ValueError('PDK LI drawing/label purposes are missing or ambiguous')
    def key(value):
        items = value.split('/')
        if len(items) != 2 or not all(re.fullmatch(r'[0-9]+', x) for x in items):
            raise ValueError('PDK LI purpose is not an exact GDS pair')
        pair = tuple(map(int, items))
        if any(n > 65535 for n in pair):
            raise ValueError('PDK LI purpose exceeds GDS range')
        return pair
    label_key, li_key = key(found['li1.label'][0]), key(found['li1.drawing'][0])
    model = native_connect_model('mos')
    if li_key != _LI or li_key not in model.metals:
        raise ValueError('PDK LI drawing differs from native connectivity')
    folder = Path(trace_dir)
    gds_path, scale_path = folder / 'primitive.gds', folder / 'scale_um.txt'
    gds_bytes, scale_bytes = _read(gds_path, checkpoint), _read(scale_path, checkpoint)
    normalize_gds_timestamps(gds_bytes, checkpoint)
    layout = kdb.Layout()
    layout.read_bytes(gds_bytes)
    if len(layout.top_cells()) != 1:
        raise ValueError('native GDS top cell differs')
    unit = Decimal(str(layout.dbu)) * 1000
    if unit <= 0 or unit != unit.to_integral_value():
        raise ValueError('native GDS unit is unsupported')
    unit = int(unit)
    scale = Decimal(scale_bytes.decode('ascii').strip()) * 1000
    grid = int(scale.to_integral_value())
    if grid <= 0 or abs(scale-grid) >= Decimal('0.000001'):
        raise ValueError('native extraction scale is unsupported')
    index = layout.find_layer(*label_key)
    if index is None:
        raise ValueError('native GDS has no PDK LI label purpose')
    labels = {}
    iterator = layout.top_cell().begin_shapes_rec(index)
    while not iterator.at_end():
        checkpoint()
        shape = iterator.shape()
        if shape.is_text():
            text = shape.text
            local = kdb.Point(text.trans.disp.x, text.trans.disp.y)
            world = iterator.trans() * local
            labels.setdefault(text.string, []).append((local.x*unit, local.y*unit,
                                                        world.x*unit, world.y*unit))
        iterator.next()
    checkpoint()
    shapes, _ = _read_gds(gds_path, _bytes=gds_bytes)
    nodes = extract_nodes(shapes, model)
    frames = verify_trace_stdout(trace_stdout_bytes)
    roles = frames.get('roles')
    if not isinstance(roles, dict) or set(roles) != {'bulk', 'drain', 'source', 'gate'}:
        raise ValueError('PDK frame roles are incomplete')
    components = {}
    for role, row in sorted(roles.items()):
        checkpoint()
        if not isinstance(row, (list, tuple)) or len(row) != 11:
            raise ValueError('invalid PDK frame role')
        entries = labels.get(row[3], [])
        if len(entries) != 1:
            raise ValueError('PDK label is missing or not unique')
        local_x, local_y, world_x, world_y = entries[0]
        box = _box(row[5])
        if (box[0] != box[2] or box[1] != box[3]
                or (local_x, local_y) != (box[0] * grid, box[1] * grid)):
            raise ValueError('GDS label and PDK execution point differ')
        probe = kdb.Region(kdb.Box(world_x, world_y, world_x+1, world_y+1))
        hits = [i for i, node in enumerate(nodes) if li_key in node and not (node[li_key] & probe).is_empty()]
        if len(hits) != 1:
            raise ValueError('PDK GDS label does not meet one native LI component')
        components[role] = {'port_name': row[3], 'native_component_index': hits[0],
                            'local_nm': [local_x, local_y], 'world_nm': [world_x, world_y]}
    if len({value['native_component_index'] for value in components.values()}) != 4:
        raise ValueError('distinct PDK roles do not identify four LI components')
    if _read(gds_path, checkpoint) != gds_bytes or _read(scale_path, checkpoint) != scale_bytes:
        raise ValueError('PDK label evidence changed during observation')
    return {'schema': 'clay.pdk-label-components/v1', 'status': 'OBSERVED',
            'components': components, 'lyp_observed_sha256': sha256(lyp_bytes).hexdigest(),
            'lyp_observed_size_bytes': len(lyp_bytes),
            'lyp_authority': 'source-image runtime observation',
            'trace_stdout_observed': {'sha256': sha256(trace_stdout_bytes).hexdigest(),
                                      'size_bytes': len(trace_stdout_bytes)},
            'gds_observed': {'sha256': sha256(gds_bytes).hexdigest(), 'size_bytes': len(gds_bytes)},
            'scale_observed': {'sha256': sha256(scale_bytes).hexdigest(), 'size_bytes': len(scale_bytes)},
            'source_terminal_binding': 'UNKNOWN', 'ext_anchor_binding': 'UNKNOWN',
            'body_cut_coverage': 'UNKNOWN', 'physical_admission': False}
