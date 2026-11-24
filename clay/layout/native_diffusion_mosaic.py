"""Build source-owned joint row geometry from a checked native PCell template."""
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import klayout.db as kdb

from clay.frontend.unit_master_geometry import UnitMasterSource, UnitTerminalSite, normalize_unit_master_geometry
from clay.layout.common_centroid_array import _CCFinger
from clay.layout.connectivity import ConnectModel
from clay.layout.joint_occurrence_center_basis import capture_joint_channel_centers
from clay.layout.joint_row_geometry import (JointArtworkPart, JointRowOccurrence, JointRowGeometryBinding,
    _region, _shapes, joint_row_artwork, joint_row_polygons, joint_transform_shapes,
    _occurrence_channel_artwork, replay_joint_row_geometry)
from clay.layout.joint_terminal_inventory import (TerminalSiteRequirement, OccurrenceTerminalSite,
    JointTerminalLimits, compile_joint_terminal_inventory, _native_frame)
from clay.layout.mosaic_geometry import MosaicGeometry
from clay.layout.native_diffusion_pcell import _check_chain, _connected_precharge_profile
from clay.layout.native_sky130 import native_connect_model
from clay.layout.occurrence_channels import compile_row_channel_inventory
from clay.layout.place.row_program import RowMemberSpec
from clay.layout.route.resource_world import pdk_provenance_digest
from clay.pdk.sky130_native_geometry import native_geometry_spec


def _digest(path): return sha256(Path(path).read_bytes()).hexdigest()


def _regions(path):
    layout=kdb.Layout(); layout.read(str(path)); top=layout.top_cell()
    if layout.dbu != .001 or top is None or top.name != 'primitive':
        raise ValueError('native diffusion template top or DBU differs')
    result={}
    for index in layout.layer_indices():
        info=layout.get_info(index); observed=kdb.Region(top.begin_shapes_rec(index)).merged()
        region=kdb.Region()
        for polygon in observed.each(): region.insert(polygon.dup())
        region.merge()
        if not region.is_empty(): result[(info.layer,info.datatype)]=region
    return result


def _connected_gate_selectors(active, poly, channels, chain):
    """Partition shared gate poly across channels for precharge or parameterized merged gates."""
    is_param = getattr(chain, 'synthesis', '') in ('parameterized', 'cmos_gate')
    if not (_connected_precharge_profile(chain) or is_param):
        raise ValueError('shared native gate poly lacks the source-proven profile')
    components = tuple(poly.merged().each())
    valid_components = (1 <= len(components) <= len(chain.fingers)) if is_param else (len(components) == 1)
    if not valid_components or len(channels) != len(chain.fingers):
        raise ValueError('shared native gate component or channel denominator differs')
    boxes = [channel.bbox() for channel in channels]
    if any(left.right >= right.left for left, right in zip(boxes, boxes[1:])):
        raise ValueError('shared native gate channel separation is ambiguous')
    bounds = poly.bbox()
    cuts = [bounds.left] + [(left.right + right.left)//2
                           for left, right in zip(boxes, boxes[1:])] + [bounds.right]
    selectors = tuple((poly & kdb.Region(kdb.Box(left, bounds.bottom, right, bounds.top))).merged()
                      for left, right in zip(cuts, cuts[1:]))
    union = kdb.Region()
    for index, selector in enumerate(selectors):
        if selector.is_empty() or not (selector - poly).is_empty():
            raise ValueError('shared native gate selector differs from actual poly')
        if not ((selector & active) ^ kdb.Region(channels[index])).is_empty():
            raise ValueError('shared native gate selector does not own exactly one channel')
        if not (union & selector).is_empty():
            raise ValueError('shared native gate selectors overlap')
        union += selector
    if not (union.merged() ^ poly).is_empty():
        raise ValueError('shared native gate selectors do not cover actual poly')
    return selectors


def build_native_diffusion_mosaic(cell, chain, folder, tech):
    """Observe all source sites and raw/accessory masks for one declared row."""
    _check_chain(cell,chain)
    folder=Path(folder); raw_path=folder/'primitive.gds'; contact_path=folder/'contacted.gds'
    raw,contacts=_regions(raw_path),_regions(contact_path)
    if any(not (region-contacts.get(key,kdb.Region())).merged().is_empty() for key,region in raw.items()):
        raise ValueError('native contact producer removed original PCell masks')
    active,poly=raw[(65,20)],raw[(66,20)]
    channels=sorted((active&poly).merged().each(),key=lambda p:p.bbox().left)
    gates=sorted((p for p in poly.each() if not (kdb.Region(p)&active).is_empty()),
                 key=lambda p:p.bbox().left)
    diffusions=sorted((active-poly).merged().each(),key=lambda p:p.bbox().left)
    count=len(chain.fingers)
    if not (len(channels)==count and len(diffusions)==count+1):
        raise ValueError('native PCell channel or diffusion denominator differs')
    if len(gates)==count:
        gate_regions=tuple(kdb.Region(p) for p in gates)
    else:
        gate_regions=_connected_gate_selectors(active,poly,channels,chain)
    family='pmos' if '__pfet_' in chain.model else 'nmos'
    sites=[*(UnitTerminalSite(f'gate{i}','gate',(((66,20),_shapes(region)),))
                for i,region in enumerate(gate_regions)),
           *(UnitTerminalSite(f'sd{i}','sd',(((65,20),_shapes(kdb.Region(p))),))
                for i,p in enumerate(diffusions))]
    if family=='pmos' and count>1:
        sites.append(UnitTerminalSite('body','body',(((64,20),_shapes(raw[(64,20)])),)))
    sites=tuple(sorted(sites,key=lambda site:site.site_id))
    source_path=Path(cell.source_span['file']).resolve()
    provenance=sha256((_digest(source_path)+_digest(raw_path)+_digest(contact_path)).encode()).hexdigest()
    master=normalize_unit_master_geometry(pdk_id='sky130',
        parameter_class=f'{family}:{chain.width_nm}:{chain.length_nm}',
        sources=(UnitMasterSource(_digest(raw_path),'primitive',provenance),),
        geometry_by_layer_nm=tuple((key,_shapes(region)) for key,region in sorted(raw.items())),
        terminal_sites=sites,unresolved_access_obligations=('body_source_binding',))
    bounds=kdb.Region()
    for region in raw.values(): bounds+=region
    origin=bounds.bbox(); frame=(1,0,0,1,origin.left,origin.bottom)
    devices={device.instance_name:device for device in cell.devices}
    occurrences=[]; requirements=[]; positions=[]; fingers=[]; members={}
    for index,finger in enumerate(chain.fingers):
        owner=finger.owner; node=f'{owner}[{finger.occurrence:03d}]'; device=devices[owner]
        spec=native_geometry_spec(device)
        members[owner]=RowMemberSpec(device=owner,kind=family,width_nm=chain.width_nm,
            length_nm=chain.length_nm,fingers=spec['occurrences'],gate_net=device.pins['g'],
            drain_net=device.pins['d'],source_net=device.pins['s'],body_net=device.pins['b'])
        if count==1:
            occurrences.append(JointRowOccurrence(node,owner,'shared/raw',master,frame))
        else:
            names=tuple(sorted((f'gate{index}',f'sd{index}',f'sd{index+1}')+
                               (('body',) if family=='pmos' else ())))
            occurrences.append(JointRowOccurrence(node,owner,'shared/raw',master,frame,names,f'gate{index}'))
        site_nets=((f'gate{index}',finger.gate),(f'sd{index}',finger.left),(f'sd{index+1}',finger.right))
        if family=='pmos' and count>1: site_nets+=(('body',finger.body),)
        for name,net in site_nets:
            site=next(row for row in master.terminal_sites if row.site_id==name)
            layer,polygons=site.geometry_by_layer_nm[0]
            roles=(('g',) if site.role_group=='gate' else ('b',) if site.role_group=='body'
                   else tuple(sorted(role for role in ('d','s') if device.pins[role]==net)))
            if not roles: raise ValueError('native source terminal role differs')
            site_id=node+'/'+name
            requirements.append(TerminalSiteRequirement(site_id,owner,node,net,roles,layer,
                (66,20) if site.role_group=='gate' else (64,20) if site.role_group=='body'
                else (65000,1),polygons,master.variant_sha256,provenance))
            positions.append(OccurrenceTerminalSite(site_id,master.variant_sha256,frame,
                joint_transform_shapes(polygons,_native_frame(frame))))
        box=channels[index].bbox()
        fingers.append(_CCFinger(owner,finger.gate,box.left,box.right,(box.bottom+box.top)//2,
                                  'R0',(box.left,box.bottom,box.right,box.top)))
    occurrences=tuple(occurrences); associated=tuple(sorted(row.occurrence_id for row in occurrences))
    additions=tuple((key,_shapes(extra)) for key,region in sorted(contacts.items())
                    if not (extra:=(region-raw.get(key,kdb.Region())).merged()).is_empty())
    parts=(JointArtworkPart('shared/raw','raw_unit',provenance,associated,
             tuple((key,_shapes(region)) for key,region in sorted(raw.items()))),
           JointArtworkPart('shared/contacts','accessory',_digest(contact_path),associated,additions))
    substrate=None
    if family=='nmos' or (family=='pmos' and count==1):
        from clay.layout.native_substrate_binding import capture_native_substrate_binding
        scale=Decimal((folder/'scale_um.txt').read_text().strip())*1000
        grid=int(scale.to_integral_value())
        if grid<=0 or abs(scale-grid)>=Decimal('0.000001'):
            raise ValueError('native extraction grid is not integer nanometres')
        substrate,body_sites,body_positions=capture_native_substrate_binding(
            raw_gds_path=raw_path,contacted_gds_path=contact_path,spice_path=folder/'native.spice',
            ext_path=folder/'primitive_flat.ext',scale_nm=grid,occurrences=occurrences,members=members,
            tech=tech,source_cell=cell,source_path=source_path,receipt_path=folder/'template-receipt.json')
        requirements.extend(body_sites); positions.extend(body_positions)
    requirements=tuple(sorted(requirements,key=lambda row:row.site_id))
    positions=tuple(sorted(positions,key=lambda row:row.site_id))
    connect=native_connect_model('mos'); routing=ConnectModel.from_tech(tech); pdk=pdk_provenance_digest(tech)
    inventory=compile_joint_terminal_inventory(artwork=joint_row_artwork(parts),expected_sites=requirements,
        placed_sites=positions,connect_model=connect,routing_model=routing,pdk_provenance_sha256=pdk,
        checkpoint=lambda:None,limits=JointTerminalLimits())
    centers=capture_joint_channel_centers(occurrences,tech=tech,family=family)
    binding=JointRowGeometryBinding(parts,occurrences,requirements,positions,connect,routing,'sky130',pdk,inventory,
        quality_centers=centers,substrate_binding=substrate)
    polygons=joint_row_polygons(parts)
    channel_inventory=compile_row_channel_inventory(occurrence_artwork=tuple((occ.owner,
        _occurrence_channel_artwork(binding,occ,((65,20),),(66,20))) for occ in occurrences),
        row_polygons=polygons,od_layer_keys=((65,20),),poly_layer_key=(66,20))
    extent=kdb.Region()
    for part in parts:
        for _layer,shapes in part.geometry_by_layer_nm: extent+=_region(shapes)
    actual=extent.bbox(); comp=active.bbox()
    geometry=MosaicGeometry(polygons=polygons,kind=family,
        comp_bbox=(comp.left-actual.left,comp.bottom-actual.bottom,comp.right-actual.left,comp.top-actual.bottom),
        fingers=tuple(fingers),joint_row_binding=binding,channel_inventory=channel_inventory,tech=tech)
    replay_joint_row_geometry(geometry,tuple(sorted(members)),tech=tech,members=members)
    return geometry
