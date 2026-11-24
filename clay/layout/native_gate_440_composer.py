"""Rigid PDK-unit composition; independent draft, never imported by production."""
from hashlib import sha256
from pathlib import Path
import klayout.db as kdb

from clay.layout import native_gate_440_planner as P


def read(path):
    layout=kdb.Layout(); layout.read(str(path)); top=layout.top_cell()
    if layout.dbu!=.001 or top is None:
        raise P.UnsupportedComposition('unit geometry DBU or top differs')
    return layout,top


def regions(layout,top):
    return {(layout.get_info(i).layer,layout.get_info(i).datatype):
            kdb.Region(top.begin_shapes_rec(i)).merged() for i in layout.layer_indices()}


def check_unit_masks(labeled,noports):
    ll,lt=read(labeled); nl,nt=read(noports)
    a,b=regions(ll,lt),regions(nl,nt)
    for key in set(a)|set(b):
        if not (a.get(key,kdb.Region())^b.get(key,kdb.Region())).is_empty():
            raise P.UnsupportedComposition('doports changed PDK mask geometry')
    for i in nl.layer_indices():
        it=nt.begin_shapes_rec(i)
        while not it.at_end():
            if it.shape().is_text(): raise P.UnsupportedComposition('noport PCell still has labels')
            it.next()
    return nl,nt,b


def verify(labeled,noports,hierarchical,raw_gds,chain,pins, *, implementation=None):
    """Recompute every source-unit layer union and hierarchy/flat XOR."""
    unit=P.inspect_unit(labeled);plan=P.plan(chain,pins,unit,implementation=implementation);P.check_seams(plan,unit)
    _l,_t,unit_regions=check_unit_masks(labeled,noports)
    hl,ht=read(hierarchical);hierarchy=regions(hl,ht)
    rl,rt=read(raw_gds);actual=regions(rl,rt)
    if rt.child_cells()!=0:raise P.UnsupportedComposition('raw row was not flattened')
    all_keys=set(unit_regions)|set(hierarchy)|set(actual)
    for key in all_keys:
        expected=kdb.Region()
        for item in plan:expected+=unit_regions.get(key,kdb.Region()).transformed(item['transform'])
        if not (expected.merged()^actual.get(key,kdb.Region())).is_empty():
            raise P.UnsupportedComposition('raw differs from source unit union')
        if not (hierarchy.get(key,kdb.Region())^actual.get(key,kdb.Region())).is_empty():
            raise P.UnsupportedComposition('raw hierarchy to flat differs')
    active=actual[(65,20)];poly=actual[(66,20)]
    boxes=sorted((p.bbox().left,p.bbox().bottom,p.bbox().right,p.bbox().top) for p in (active&poly).merged().each())
    if len(boxes)!=10 or any((b[2]-b[0],b[3]-b[1])!=(150,2000) for b in boxes):
        raise P.UnsupportedComposition('source channel count or dimensions differ')
    if [boxes[i+1][0]-boxes[i][0] for i in range(9)]!=[440]*9:
        raise P.UnsupportedComposition('source channel pitch differs')
    for region in actual.values():
        for polygon in region.each():
            loops=[polygon.each_point_hull()]+[polygon.each_point_hole(i) for i in range(polygon.holes())]
            if any(p.x%5 or p.y%5 for points in loops for p in points):
                raise P.UnsupportedComposition('raw mask off manufacturing grid')
    return {'layer_keys':tuple(sorted(all_keys)),'channels':tuple(boxes),'occurrences':len(plan)}


def compose(labeled,noports,destination,chain,pins, *, implementation=None):
    destination=Path(destination)
    if destination.exists(): raise P.UnsupportedComposition('output already exists')
    hierarchical=destination.with_name(destination.stem+'-hierarchical.gds')
    if hierarchical.exists(): raise P.UnsupportedComposition('hierarchical output already exists')
    unit=P.inspect_unit(labeled); plan=P.plan(chain,pins,unit,implementation=implementation)
    P.check_seams(plan,unit)
    layout,top,raw=check_unit_masks(labeled,noports)
    top.name='native_unit'
    row=layout.create_cell('primitive')
    for item in plan:
        row.insert(kdb.CellInstArray(top.cell_index(),item['transform']))
    layout.write(str(hierarchical))
    hierarchy,hierarchy_top=read(hierarchical)
    before_flatten=regions(hierarchy,hierarchy_top)
    # Keep the source unit alive while the saved, read-only unit Regions are
    # still in use. Pruning its cell invalidates KLayout's native handles.
    row.flatten(-1,False)
    options=kdb.SaveLayoutOptions(); options.select_cell(row.cell_index())
    layout.write(str(destination),options)
    check,checktop=read(destination); actual=regions(check,checktop)
    if checktop.child_cells()!=0:
        raise P.UnsupportedComposition('raw row was not flattened')
    flat_xor={f'{l}/{d}':(before_flatten.get((l,d),kdb.Region())^actual.get((l,d),kdb.Region())).area()
              for l,d in sorted(set(before_flatten)|set(actual))}
    if any(flat_xor.values()):
        raise P.UnsupportedComposition('flattening changed PDK mask geometry')
    for key in set(raw)|set(actual):
        expected=kdb.Region()
        for item in plan:
            expected+=raw.get(key,kdb.Region()).transformed(item['transform'])
        if not (expected.merged()^actual.get(key,kdb.Region())).is_empty():
            raise P.UnsupportedComposition('composed mask differs from rigid source unit union')
    active=actual[(65,20)]; poly=actual[(66,20)]; channels=(active&poly).merged()
    boxes=sorted((p.bbox().left,p.bbox().bottom,p.bbox().right,p.bbox().top) for p in channels.each())
    if len(boxes)!=10 or any((b[2]-b[0],b[3]-b[1])!=(150,2000) for b in boxes):
        raise P.UnsupportedComposition('composed source channel count or dimensions differ')
    if [boxes[i+1][0]-boxes[i][0] for i in range(9)]!=[440]*9:
        raise P.UnsupportedComposition('composed pitch differs')
    for key,region in actual.items():
        for polygon in region.each():
            loops=[polygon.each_point_hull()]+[polygon.each_point_hole(i) for i in range(polygon.holes())]
            if any(p.x%5 or p.y%5 for points in loops for p in points):
                raise P.UnsupportedComposition('composed mask is off manufacturing grid')
    serial=[{**{k:v for k,v in item.items() if k!='transform'},
             'transform':str(item['transform'])} for item in plan]
    return {'source_identity':chain['source_identity'],'model':chain['model'],
            'unit_width_nm':chain['width_nm'],'length_nm':chain['length_nm'],
            'columns':chain['columns'],'occurrences':serial,'gate_boxes_nm':boxes,
            'labeled_unit_sha256':sha256(Path(labeled).read_bytes()).hexdigest(),
            'noports_unit_sha256':sha256(Path(noports).read_bytes()).hexdigest(),
            'hierarchical_gds_sha256':sha256(hierarchical.read_bytes()).hexdigest(),
            'flatten_layer_xor_area_nm2':flat_xor,
            'composed_gds_sha256':sha256(destination.read_bytes()).hexdigest(),
            'scope':'raw rigid PDK unit union; source terminal and extraction admission pending'}
