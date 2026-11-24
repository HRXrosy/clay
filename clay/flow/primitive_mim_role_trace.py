"""Opt-in v2 passive role frames; v1 MOS implementation remains unchanged."""
from decimal import Decimal
from hashlib import sha256
import re
import xml.etree.ElementTree as ET

from clay.flow import primitive_role_trace as v1


def eligible_for_request(spec, request):
    if v1.eligible_spec(spec):
        return True
    return (request.get('schema') == 'clay.primitive-role-trace-request/v2'
            and type(spec) is dict and spec.get('family') == 'mim'
            and spec.get('model') == 'sky130_fd_pr__cap_mim_m3_1'
            and all(type(spec.get(k)) is int and spec[k] == 1
                    for k in ('nf', 'm', 'mult', 'replicas'))
            and spec.get('body_tap') is None)

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
    if {[catch {dict size $arguments}]} {clay_emit UNRESOLVED $command $frame; return}
    set hits {}
    foreach role {term_t term_b} {
        if {[dict exists $arguments $role] && [dict get $arguments $role] eq [lindex $command 1]} {lappend hits $role}
    }
    if {[llength $hits] != 1} {clay_emit UNRESOLVED $command $frame; return}
    set role [lindex $hits 0]
    foreach field {nx ny doports top_type bot_type} {
        if {![dict exists $arguments $field]} {clay_emit UNRESOLVED $command $frame; return}
    }
    lappend ::clay_seen [list $role [lindex $command 1]]
    clay_emit ROLE $role [lindex $frame 0] [lindex $command 0] [lindex $command 1] \
        [lindex $command 3] [join [box values] ,] [dict get $arguments $role] \
        [dict get $arguments nx] [dict get $arguments ny] [dict get $arguments doports] \
        [dict get $arguments top_type] [dict get $arguments bot_type]
}
trace add execution label enter clay_trace_label
'''


def instrument_for_request(spec, control, request):
    if not eligible_for_request(spec, request):
        raise ValueError('native spec is outside requested role trace scope')
    if spec['family'] == 'mos':
        return v1.instrument_generation_tcl(spec, control)
    from clay.layout.native_sky130 import _generation_tcl
    if control != _generation_tcl(spec):
        raise ValueError('control script differs from native producer')
    gen = 'if {[catch {magic::gencell'
    gds = '\nselect top cell\ngds write primitive.gds'
    if control.count(gen) != 1 or control.count(gds) != 1:
        raise ValueError('native producer instrumentation anchors differ')
    return control.replace(gen, _HOOK + '\n' + gen, 1).replace(gds, '\n' + v1._QUERY + gds + v1._STYLE_AFTER, 1)


def verify_mim_frames(data):
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
            rows.append((fields[1], tuple(bytes.fromhex(x).decode('utf-8') for x in fields[2:])))
        except (ValueError, UnicodeError) as exc:
            raise ValueError('malformed trace record') from exc
    roles = [x for kind,x in rows if kind == 'ROLE']
    ports = [x for kind,x in rows if kind == 'PORT']
    children = [x for kind,x in rows if kind == 'CHILDREN']
    ranges = [x for kind,x in rows if kind == 'PORT_RANGE']
    expected = {'term_t', 'term_b'}
    if (len(roles) != 2 or any(len(x) != 12 for x in roles) or {x[0] for x in roles} != expected
        or len(ports) != 2 or any(len(x) != 8 for x in ports) or {x[0] for x in ports} != expected
        or len(children) != 1 or len(children[0]) != 2 or children[0][0] != '1' or not children[0][1]
        or len(ranges) != 1 or len(ranges[0]) != 2):
        raise ValueError('MIM role, port or child denominator differs')
    by_role, by_port = {r[0]:r for r in roles}, {p[0]:p for p in ports}
    if len({r[3] for r in roles}) != 2 or len({p[2] for p in ports}) != 2:
        raise ValueError('MIM label or port index is not unique')
    if by_role['term_t'][7:12] != by_role['term_b'][7:12] or by_role['term_t'][10] == by_role['term_t'][11]:
        raise ValueError('MIM execution parameters or conductor layers differ')
    for role in sorted(expected):
        row, port = by_role[role], by_port[role]
        layer = row[10] if role == 'term_t' else row[11]
        if (row[1] != 'sky130::cap_device' or row[2] != 'label' or not row[3] or row[3] != row[6]
            or row[7:10] != ('1','1','1') or row[4] != layer or not layer
            or port[1] != row[3] or port[3] != '1' or port[4] != row[3] or not port[5]
            or not re.fullmatch(r'[0-9]+', port[2]) or int(port[2]) <= 0
            or v1._box(port[6]) != v1._box(row[5])):
            raise ValueError('executed MIM label or saved port differs')
    if (any(not re.fullmatch(r'[0-9]+', x) for x in ranges[0])
        or int(ranges[0][0]) != min(int(p[2]) for p in ports)
        or int(ranges[0][1]) != max(int(p[2]) for p in ports)):
        raise ValueError('MIM port range differs')
    if text.index('CLAY4_STYLE_BEFORE_END') >= text.index('CLAY4_STYLE_AFTER_BEGIN'):
        raise ValueError('MIM style observations are out of order')
    before, after = v1._style(text, 'BEFORE'), v1._style(text, 'AFTER')
    if before != after:
        raise ValueError('MIM trace changed Magic output style')
    return {'schema':'clay.pdk-mim-role-frames/v2', 'roles':by_role, 'ports':by_port,
            'children':children[0], 'port_range':ranges[0], 'style':before,
            'source_terminal_binding':'UNKNOWN', 'physical_admission':False}


def observe_mim_components(folder, lyp_bytes, stdout, checkpoint=lambda: None):
    import klayout.db as kdb
    from clay.layout.connectivity import extract_nodes
    from clay.layout.native_sky130 import _read_gds, native_connect_model, _METALS
    from pathlib import Path
    frames = verify_mim_frames(stdout)
    found = {}
    for entry in ET.fromstring(lyp_bytes).iter('properties'):
        name, source = entry.findtext('name'), entry.findtext('source')
        if name and source:
            short = name.split(' - ', 1)[0]
            if short in ('met3.label','met3.drawing','met4.label','met4.drawing'):
                found.setdefault(short, []).append(source.split('@', 1)[0])
    if len(found) != 4 or any(len(v) != 1 for v in found.values()):
        raise ValueError('PDK MIM layer purposes missing or ambiguous')
    def layer(name):
        raw = found[name][0].split('/')
        if len(raw) != 2 or not all(re.fullmatch(r'[0-9]+', x) and int(x) <= 65535 for x in raw):
            raise ValueError('invalid PDK layer purpose')
        return tuple(map(int,raw))
    folder = Path(folder)
    gdspath, scalepath = folder/'primitive.gds', folder/'scale_um.txt'
    gds, scale_bytes = v1._read(gdspath, checkpoint), v1._read(scalepath, checkpoint)
    v1.normalize_gds_timestamps(gds, checkpoint)
    layout = kdb.Layout(); layout.read_bytes(gds)
    if len(layout.top_cells()) != 1:
        raise ValueError('MIM GDS top cell differs')
    unit = Decimal(str(layout.dbu))*1000
    scale = Decimal(scale_bytes.decode('ascii').strip())*1000
    grid = int(scale.to_integral_value())
    if unit <= 0 or unit != unit.to_integral_value() or grid <= 0 or abs(scale-grid) >= Decimal('0.000001'):
        raise ValueError('MIM GDS unit or extraction grid differs')
    unit = int(unit)
    shapes, _ = _read_gds(gdspath, _bytes=gds)
    model = native_connect_model('mim'); nodes = extract_nodes(shapes, model)
    components = {}
    for role, row in sorted(frames['roles'].items()):
        checkpoint()
        short = {'m3':'met3', 'm4':'met4'}.get(row[4])
        if short is None:
            raise ValueError('MIM executed layer is unsupported')
        drawing = layer(short+'.drawing')
        if drawing != _METALS[{'met3':2, 'met4':3}[short]] or drawing not in model.metals:
            raise ValueError('PDK MIM drawing differs from native conductor')
        hits = []
        for purpose in ('met3.label', 'met4.label'):
            label = layer(purpose)
            index = layout.find_layer(*label)
            if index is None:
                continue
            it = layout.top_cell().begin_shapes_rec(index)
            while not it.at_end():
                checkpoint(); shape=it.shape()
                if shape.is_text() and shape.text.string == row[3]:
                    local = kdb.Point(shape.text.trans.disp.x,shape.text.trans.disp.y)
                    world = it.trans()*local
                    hits.append((local.x*unit,local.y*unit,world.x*unit,world.y*unit,label))
                it.next()
        if len(hits) != 1: raise ValueError('MIM GDS label missing or ambiguous')
        lx,ly,x,y,label = hits[0]; box=v1._box(row[5])
        if box[0]!=box[2] or box[1]!=box[3] or (lx,ly)!=(box[0]*grid,box[1]*grid):
            raise ValueError('MIM GDS label and frame point differ')
        probe=kdb.Region(kdb.Box(x,y,x+1,y+1))
        candidates=[i for i,n in enumerate(nodes) if drawing in n and not (n[drawing]&probe).is_empty()]
        if len(candidates)!=1: raise ValueError('MIM label does not meet one component')
        components[role]={'port_name':row[3], 'native_component_index':candidates[0],
                          'label_layer':list(label), 'drawing_layer':list(drawing),
                          'local_nm':[lx,ly], 'world_nm':[x,y]}
    if len({x['native_component_index'] for x in components.values()}) != 2:
        raise ValueError('MIM ports share a conductor')
    if v1._read(gdspath,checkpoint)!=gds or v1._read(scalepath,checkpoint)!=scale_bytes:
        raise ValueError('MIM evidence changed during observation')
    return {'schema':'clay.pdk-mim-label-components/v2', 'status':'OBSERVED', 'components':components,
            'lyp_observed_sha256':sha256(lyp_bytes).hexdigest(), 'lyp_observed_size_bytes':len(lyp_bytes),
            'lyp_authority':'source-image runtime observation',
            'trace_stdout_observed':{'sha256':sha256(stdout).hexdigest(),'size_bytes':len(stdout)},
            'gds_observed':{'sha256':sha256(gds).hexdigest(),'size_bytes':len(gds)},
            'scale_observed':{'sha256':sha256(scale_bytes).hexdigest(),'size_bytes':len(scale_bytes)},
            'source_terminal_binding':'UNKNOWN','ext_anchor_binding':'UNKNOWN',
            'dielectric_cut_coverage':'UNKNOWN','physical_admission':False}
