"""Isolated source-bound transform planning for a single native PCell row.

Planning only: does not emit artwork, alter production, or assert signoff.
"""
from pathlib import Path

import klayout.db as kdb

class UnsupportedComposition(ValueError): pass


def inspect_unit(path):
    path=Path(path)
    layout=kdb.Layout(); layout.read(str(path)); top=layout.top_cell()
    if top is None or layout.dbu!=.001:
        raise UnsupportedComposition('unit DBU or top differs')
    top.flatten(True)
    def region(layer):
        idx=layout.find_layer(*layer)
        return kdb.Region() if idx is None else kdb.Region(top.shapes(idx)).merged()
    active,poly,cuts=region((65,20)),region((66,20)),region((66,44))
    if not (poly & cuts).is_empty():
        raise UnsupportedComposition('no-gate-contact unit contains a gate licon')
    channels=(active&poly).merged()
    if channels.count()!=1: raise UnsupportedComposition('unit channel count differs')
    box=channels.bbox()
    if (box.width(),box.height())!=(150,2000):
        raise UnsupportedComposition('unit source W/L differs')
    cx,cy=(box.left+box.right)//2,(box.bottom+box.top)//2
    idx=layout.find_layer(67,5)
    labels=[] if idx is None else [shape.text for shape in top.shapes(idx).each() if shape.is_text()]
    positions={}
    for label in labels:
        if label.string in ('D','S'):
            if label.string in positions: raise UnsupportedComposition('duplicate unit port label')
            positions[label.string]=(label.trans.disp.x,label.trans.disp.y)
    if set(positions)!={'D','S'} or not positions['D'][0]<cx<positions['S'][0]:
        raise UnsupportedComposition('unit PDK D/S/G label positions differ')
    columns={}
    for p in (active&cuts).merged().each():
        b=p.bbox()
        if (b.width(),b.height())!=(170,170): raise UnsupportedComposition('unit active licon size differs')
        key='D' if b.center().x<cx else 'S'
        columns.setdefault(key,[]).append((b.left,b.bottom,b.right,b.top))
    if set(columns)!={'D','S'} or not all(columns.values()):
        raise UnsupportedComposition('unit contact columns differ')
    for key in columns: columns[key]=tuple(sorted(columns[key]))
    li=region((67,20))
    for x,y in positions.values():
        if (li & kdb.Region(kdb.Box(x,y,x+1,y+1))).is_empty():
            raise UnsupportedComposition('unit port label is outside PDK LI')
    return {'gate_center':(cx,cy),'columns':columns,'label_points':positions,'layout':layout,'top':top}


def plan(chain,pins,unit, pitch_nm=440, *, implementation=None):
    if pitch_nm!=440 or pitch_nm%5:
        raise UnsupportedComposition('unsupported pitch or grid')
    if (chain['model'],chain['width_nm'],chain['length_nm'],len(chain['fingers'])) != (
        'sky130_fd_pr__nfet_01v8_lvt',2000,150,10):
        raise UnsupportedComposition('source row model, dimensions or count differ')
    source_profile=chain.get('generation_profile')
    profile=source_profile if implementation is None else implementation
    if (implementation is not None and source_profile) or profile not in ('sky130_lvt_single_440_v1','sky130_lvt_single_440_output_m12_v1','sky130_lvt_single_440_output_input_m12_v1') or chain.get('coverage_profile') is not None or chain.get('access_profile'):
        raise UnsupportedComposition('source row generation profile differs')
    fingers=chain['fingers']; columns=chain['columns']
    if len(columns)!=len(fingers)+1: raise UnsupportedComposition('source diffusion column count differs')
    ids=[(finger['owner'],finger['occurrence']) for finger in fingers]
    if len(set(ids))!=len(ids): raise UnsupportedComposition('duplicate source occurrence')
    cx,cy=unit['gate_center']; out=[]
    for i,finger in enumerate(fingers):
        owner=finger['owner']; dev=pins.get(owner,{})
        if (finger['left'],finger['right']) != (columns[i],columns[i+1]) or (
            finger['gate'],finger['body']) != (dev.get('g'),dev.get('b')):
            raise UnsupportedComposition('source occurrence terminal mapping differs')
        pair=(finger['left'],finger['right']); direct=(dev.get('d'),dev.get('s'))
        if pair==direct: xflip=False
        elif pair==tuple(reversed(direct)): xflip=True
        else: raise UnsupportedComposition('source D/S mapping differs')
        # Existing contact producer expects even gates below and odd above.
        yflip=not bool(i%2)
        rot,mirror={(False,False):(0,False),(True,False):(2,True),
                    (False,True):(0,True),(True,True):(2,False)}[(xflip,yflip)]
        base=kdb.Trans(rot,mirror,0,0); transformed=base*kdb.Point(cx,cy)
        trans=kdb.Trans(rot,mirror,cx+i*pitch_nm-transformed.x,cy-transformed.y)
        if trans.disp.x%5 or trans.disp.y%5:
            raise UnsupportedComposition('unit transform is off manufacturing grid')
        out.append({'owner':owner,'occurrence':finger['occurrence'],'left':pair[0],
                    'right':pair[1],'gate':finger['gate'],'body':finger['body'],
                    'transform':trans,'xflip':xflip,'yflip':yflip})
    return tuple(out)


def check_seams(planned,unit):
    def transformed(items,trans):
        return tuple(sorted((b.left,b.bottom,b.right,b.top) for rect in items
                            for b in [(trans*kdb.Box(*rect))]))
    for a,b in zip(planned,planned[1:]):
        if a['right']!=b['left']:
            raise UnsupportedComposition('shared column net differs')
        right='D' if a['xflip'] else 'S'; left='S' if b['xflip'] else 'D'
        if transformed(unit['columns'][right],a['transform']) != transformed(unit['columns'][left],b['transform']):
            raise UnsupportedComposition('shared licon boxes are not identical')
    return True


def extraction_labels(planned,unit,gate_points):
    """Unique physical-site labels using actual unit PDK port locations.

    D/S names color the alternating extracted physical column graph; source
    net ownership remains the independently checked chain, not the names.
    """
    check_seams(planned,unit)
    if len(gate_points)!=len(planned):
        raise UnsupportedComposition('gate access point count differs')
    labels=[]
    for i,item in enumerate(planned):
        left='S' if item['xflip'] else 'D'
        point=item['transform']*kdb.Point(*unit['label_points'][left])
        labels.append((('D' if i%2==0 else 'S')+str(i),point.x,point.y,item['left']))
        gate=kdb.Point(*gate_points[i])
        labels.append(('G'+str(i),gate.x,gate.y,item['gate']))
    last=planned[-1]; right='D' if last['xflip'] else 'S'
    point=last['transform']*kdb.Point(*unit['label_points'][right]); i=len(planned)
    labels.append((('D' if i%2==0 else 'S')+str(i),point.x,point.y,last['right']))
    if len({row[0] for row in labels})!=len(labels) or len(labels)!=21:
        raise UnsupportedComposition('extraction label inventory differs')
    return tuple(labels)
