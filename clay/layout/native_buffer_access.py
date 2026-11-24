"""Source-owned SKY130 shared-buffer terminal access built from real PCell masks.

This is deliberately limited to the source-validated three-channel topology;
it neither creates route vias nor guesses a route or a placement coordinate.
"""
from math import ceil

import klayout.db as kdb

# Existing native PCell gate-contact operands, in nanometres, ordered as
# sky130::draw_contact {w h s o x}; the same values render the Tcl and prove
# its LI geometry. These are an explicitly supported fixed native template,
# not inferred from a different PDK's contact dimensions.
_POLY_LI_GATE_CONTACT_NM=(170,170,80,80,170)
_LI_M1_GATE_CONTACT_NM=(170,170,80,60,170)


def _native_gate_contact_templates(mcon_size, grid):
    poly=_POLY_LI_GATE_CONTACT_NM; metal=_LI_M1_GATE_CONTACT_NM
    if (len(poly)!=5 or len(metal)!=5 or
        any(type(n) is not int or n<=0 or n%grid for n in (*poly,*metal)) or
        metal[0]!=mcon_size or metal[1]!=mcon_size or metal[4]!=mcon_size):
        raise ValueError('native gate contact template differs from PDK mcon size or grid')
    # draw_contact expands w/h to at least x; reject rather than quietly rely
    # on that implicit change to the two source-bound gate contact templates.
    if poly[0]<poly[4] or poly[1]<poly[4]:
        raise ValueError('native poly gate contact template has an implicit size change')
    return poly,metal


def _draw_contact(template, layers):
    return 'sky130::draw_contact '+' '.join(f'{n/1000:.3f}' for n in template)+' '+layers


def _rule(value, identity):
    if value is None or value.rule_id != identity or type(value.value) is not int or value.value <= 0:
        raise ValueError('missing native access PDK rule '+identity)
    return value.value


def _box(region, label):
    pieces=list(region.merged().each())
    if len(pieces)!=1 or not pieces[0].is_box():
        raise ValueError('native access requires one rectangular '+label)
    return pieces[0].bbox()


def _nm_box(box, layer):
    return (f'box {box[0]/1000:.3f}um {box[1]/1000:.3f}um '
            f'{box[2]/1000:.3f}um {box[3]/1000:.3f}um\npaint {layer}')


def shared_source_rule_state(tech):
    """Current value, identity and provenance of every profile-consumed rule."""
    if tech.pdk_id!='sky130':raise ValueError('shared source access requires SKY130 PDK')
    li=tech.rules.rule_for('li1');poly=tech.rules.rule_for('poly');m1=tech.rules.rule_for('met1')
    via=tech.rules.via('mcon')
    values=(('li.1',li.min_width),('li.3',li.min_spacing),('li.6',li.min_area),
        ('poly.1a',poly.min_width),('poly.2',poly.min_spacing),
        ('met1.1',m1.min_width),('met1.2',m1.min_spacing),('met1.6',m1.min_area),
        ('mcon.1',via.size),('mcon.2',via.spacing),('met1.4',via.enclosure_above))
    rows=[]
    for expected,rule in values:
        value=_rule(rule,expected)
        rows.append((value,rule.rule_id,rule.source))
    grid=tech.manufacturing_grid_nm
    if type(grid) is not int or grid<=0:raise ValueError('missing manufacturing grid')
    templates=_native_gate_contact_templates(via.size.value,grid)
    return (grid,tuple(rows),templates)


def shared_source_access_tcl(*, layout, top, chain, tech, compact_source=False):
    """Build the gated profile from source column ownership and PDK rules.

    PCell source, LI and mcon contacts stay intact. The body tap remains a
    distinct M1 endpoint. The future routing engine owns any M1/M2 via.
    """
    if type(compact_source) is not bool:
        raise TypeError('compact source choice must be a bool')
    if tech.pdk_id!='sky130' or chain.access_profile not in ('shared_source_li_m1','shared_source_li_m1_far'):
        raise ValueError('shared source access requires its SKY130 source profile')
    if compact_source and chain.access_profile != 'shared_source_li_m1':
        raise ValueError('compact source candidate requires the declared near profile')
    shared_source_rule_state(tech)
    grid=tech.manufacturing_grid_nm
    if type(grid) is not int or grid<=0:raise ValueError('missing manufacturing grid')
    li=tech.rules.rule_for('li1');poly=tech.rules.rule_for('poly');m1=tech.rules.rule_for('met1')
    li_w=_rule(li.min_width,'li.1');li_gap=_rule(li.min_spacing,'li.3');li_area=_rule(li.min_area,'li.6')
    poly_w=_rule(poly.min_width,'poly.1a');poly_gap=_rule(poly.min_spacing,'poly.2')
    m1_w=_rule(m1.min_width,'met1.1');m1_gap=_rule(m1.min_spacing,'met1.2')
    m1_area=_rule(m1.min_area,'met1.6');via=tech.rules.via('mcon')
    cut=_rule(via.size,'mcon.1');cut_gap=_rule(via.spacing,'mcon.2');enclosure=_rule(via.enclosure_above,'met1.4')
    poly_contact,metal_contact=_native_gate_contact_templates(cut,grid)
    if li_w%2 or poly_w%2 or any(n%grid for n in (li_w,li_gap,poly_w,poly_gap,m1_w,m1_gap,cut,cut_gap,enclosure)):
        raise ValueError('native access rule phase differs from the PDK grid')
    active=kdb.Region(top.begin_shapes_rec(layout.layer(65,20))).merged()
    raw_poly=kdb.Region(top.begin_shapes_rec(layout.layer(66,20))).merged()
    raw_li=kdb.Region(top.begin_shapes_rec(layout.layer(67,20))).merged()
    raw_m1=kdb.Region(top.begin_shapes_rec(layout.layer(68,20))).merged()
    raw_mcon=kdb.Region(top.begin_shapes_rec(layout.layer(67,44))).merged()
    channels=sorted((active&raw_poly).merged().each(),key=lambda p:p.bbox().left)
    diffusions=sorted((active-raw_poly).merged().each(),key=lambda p:p.bbox().left)
    if len(channels)!=3 or len(diffusions)!=4:raise ValueError('native access channel inventory differs')
    source_indices=[i for i,net in enumerate(chain.columns) if net==chain.body]
    if source_indices!=[1,3]:raise ValueError('native access source column inventory differs')
    li_boxes=[];m1_boxes=[]
    for index in source_indices:
        diff=kdb.Region(diffusions[index])
        lands=[p for p in raw_li.each() if not (kdb.Region(p)&diff).is_empty()]
        if len(lands)!=1:raise ValueError('native source LI contact is ambiguous')
        land=kdb.Region(lands[0]);cuts=raw_mcon&land
        metals=[p for p in raw_m1.each() if not (kdb.Region(p)&cuts).is_empty()]
        if compact_source:
            if not cuts.is_empty() or metals:
                raise ValueError('compact native source must have no raw MCON/M1')
        elif cuts.is_empty() or len(metals)!=1:
            raise ValueError('native source mcon/M1 contact is ambiguous')
        li_boxes.append(_box(land,'LI source'))
        if not compact_source:
            m1_boxes.append(_box(kdb.Region(metals[0]),'M1 source'))
    if (li_boxes[0].right>=li_boxes[1].left or
            (not compact_source and m1_boxes[0].right>=m1_boxes[1].left)):
        raise ValueError('native source contacts are not distinct')
    # The bridge clears every raw LI/M1 terminal. Rectangles are constructed in
    # the actual PCell frame, on grid, using PDK minimum width and spacing.
    raw_li_top=max(p.bbox().top for p in raw_li.each())
    raw_m1_top=max(p.bbox().top for p in raw_m1.each())
    ceil_grid=lambda n:((n+grid-1)//grid)*grid
    rail_lo=ceil_grid(max(raw_li_top+li_gap,raw_m1_top+m1_gap));rail_hi=rail_lo+max(li_w,m1_w)
    if any(n%grid for b in (*li_boxes,*m1_boxes) for n in (b.left,b.bottom,b.right,b.top)):
        raise ValueError('native source contact lies off manufacturing grid')
    li_rects=[(b.left,b.top,b.right,rail_hi) for b in li_boxes]
    li_rects.append((li_boxes[0].left,rail_lo,li_boxes[1].right,rail_hi))
    m1_rects=[] if compact_source else [(b.left,b.top,b.right,rail_hi) for b in m1_boxes]
    if not compact_source:
        m1_rects.append((m1_boxes[0].left,rail_lo,m1_boxes[1].right,rail_hi))
    centres=[(p.bbox().left+p.bbox().right)//2 for p in channels]
    # PDK poly spacing plus LI width/spacing separates the low-side gate
    # accessory from the native active/contact row; no reference coordinates.
    gate_y=active.bbox().bottom-(li_gap+poly_gap+li_w)
    gate_x=(centres[0],(centres[1]+centres[2])//2)
    if any(n%grid for n in (*centres,*gate_x,gate_y)):
        raise ValueError('native gate contact does not land on the PDK grid')
    if gate_x[1]-gate_x[0]-cut<cut_gap:
        raise ValueError('native gate contacts fail PDK mcon spacing')
    # The actual native poly->LI call paints LI around the contact horizontally
    # by its own top-layer surround operand; evaluate that exact rectangle.
    li_contact_area=poly_contact[1]*(poly_contact[0]+2*poly_contact[3])
    if li_contact_area<li_area:
        raise ValueError('native gate contact fails PDK LI minimum area')
    half=poly_w//2
    poly_rects=[(c-half,gate_y-half,c+half,active.bbox().bottom) for c in centres]
    poly_rects.append((centres[1]-half,gate_y-half,centres[2]+half,gate_y+half))
    # Complete the same existing native poly/LI/mcon contact construction with
    # a metal rectangle satisfying native mcon enclosure and M1 minimum area.
    pad_h=max(m1_w,cut+2*enclosure)
    pad_half=ceil_grid(ceil(m1_area/(2*pad_h)))
    pad_half=max(pad_half,ceil_grid((cut+2*enclosure+1)//2))
    if pad_h%2:raise ValueError('native gate M1 pad phase differs')
    if chain.access_profile=='shared_source_li_m1_far':
        # The gate pads are at the low side of the raw source row. Derive an
        # M1 bar between those pads and raw active, retaining LI on the high
        # side and each original source M1 contact. Equality at the PDK rule
        # is intentional; the actual connected GDS and Magic deck arbitrate.
        far_lo=ceil_grid(gate_y+pad_h//2+m1_gap)
        far_hi=far_lo+m1_w
        if far_hi+m1_gap>min(b.bottom for b in m1_boxes):
            raise ValueError('no opposite-side source M1 bridge interval')
        m1_rects=[(b.left,far_lo,b.right,b.bottom) for b in m1_boxes]
        m1_rects.append((m1_boxes[0].left,far_lo,m1_boxes[1].right,far_hi))
    lines=[_nm_box(r,'poly') for r in poly_rects]
    for x in gate_x:
        lines += [f'box {x/1000:.3f}um {gate_y/1000:.3f}um {x/1000:.3f}um {gate_y/1000:.3f}um',
          _draw_contact(poly_contact,'poly pc li horz'),
          _draw_contact(metal_contact,'li mcon metal1 horz'),
          _nm_box((x-pad_half,gate_y-pad_h//2,x+pad_half,gate_y+pad_h//2),'metal1')]
    lines += [_nm_box(r,'li') for r in li_rects]
    compact_pads=[]
    if compact_source:
        pad_width=max(m1_w,cut+2*enclosure)
        pad_half_height=ceil_grid(ceil(m1_area/(2*pad_width)))
        pad_half_height=max(pad_half_height,ceil_grid((cut+2*enclosure+1)//2))
        if pad_width%2:
            raise ValueError('compact source pad phase differs')
        for box in li_boxes:
            x=(box.left+box.right)//2; y=(box.bottom+box.top)//2
            if x%grid or y%grid or box.height()<cut:
                raise ValueError('compact source contact is off grid or outside LI')
            pad=(x-pad_width//2,y-pad_half_height,x+pad_width//2,y+pad_half_height)
            compact_pads.append(pad)
            lines += [f'box {x/1000:.3f}um {y/1000:.3f}um {x/1000:.3f}um {y/1000:.3f}um',
                _draw_contact(metal_contact,'li mcon metal1 vert'),_nm_box(pad,'metal1')]
        # Keep the two M1 contacts as one routing island, but place the bridge
        # on the opposite side of active from the declared LI bridge. The gate
        # pad and active clearances are derived from the same PDK rules as the
        # existing far-side access profile.
        far_lo=ceil_grid(gate_y+pad_h//2+m1_gap)
        far_hi=far_lo+m1_w
        if far_hi+m1_gap>active.bbox().bottom or any(far_hi>pad[1] for pad in compact_pads):
            raise ValueError('no opposite-side compact source M1 bridge interval')
        m1_rects.extend((pad[0],far_hi,pad[2],pad[1]) for pad in compact_pads)
        m1_rects.append((compact_pads[0][0],far_lo,compact_pads[1][2],far_hi))
    lines += [_nm_box(r,'metal1') for r in m1_rects]
    evidence={'li_width_nm':li_w,'li_spacing_nm':li_gap,'li_min_area_nm2':li_area,
        'poly_width_nm':poly_w,'poly_spacing_nm':poly_gap,'m1_width_nm':m1_w,'m1_spacing_nm':m1_gap,
        'mcon_size_nm':cut,'mcon_spacing_nm':cut_gap,'mcon_m1_enclosure_nm':enclosure,'m1_min_area_nm2':m1_area,
        'manufacturing_grid_nm':grid,'poly_li_contact_nm':poly_contact,'li_m1_contact_nm':metal_contact,
        'gate_li_contact_area_nm2':li_contact_area,'gate_centres_nm':gate_x,'gate_y_nm':gate_y,
        'rail_y_nm':(rail_lo,rail_hi),'li_rectangles_nm':tuple(li_rects),'m1_rectangles_nm':tuple(m1_rects)}
    if compact_source:
        evidence['compact_source_pads_nm']=tuple(compact_pads)
    return '\n'.join(lines), evidence


def shared_source_contacts_tcl(*, layout, top, chain, tech, body_x, body_y, compact_source=False):
    """Complete deterministic Magic input, replayable at the source witness."""
    access,evidence=shared_source_access_tcl(layout=layout,top=top,chain=chain,tech=tech,
                                             compact_source=compact_source)
    return f'''crashbackups stop
drc off
gds read /work/raw.gds
load primitive
box {body_x/1000:.3f}um {body_y/1000:.3f}um {body_x/1000:.3f}um {body_y/1000:.3f}um
sky130::draw_contact 0.18 0.17 0.12 0.08 0.17 psd psc li horz
sky130::draw_contact 0.17 0.17 0.08 0.06 0.17 li mcon metal1 full
label B c psc
{access}
select top cell
gds write contacted.gds
set h [open scale_um.txt w]
puts $h [cif scale out]
close $h
flatten primitive_flat
load primitive_flat
select top cell
drc euclidean on
drc style drc(full)
drc check
drc catchup
set h [open drc.txt w]
puts $h [drc listall why]
close $h
extract path .
extract no capacitance
extract no coupling
extract no resistance
extract no length
extract all
ext2spice lvs
ext2spice merge none
ext2spice -p . -o native.spice
puts CONTACTED_AND_EXTRACTED
quit -noprompt
''', evidence
