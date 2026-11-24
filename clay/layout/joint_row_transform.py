"""Transport complete raw/accessory/site evidence through one native D4 map."""
from dataclasses import replace

from clay.layout.joint_row_geometry import joint_transform_shapes, joint_row_artwork, replay_joint_row_geometry, joint_checkpoint
from clay.layout.joint_terminal_inventory import _native_frame, compile_joint_terminal_inventory


def transform_joint_row_geometry(original, transformed, *, matrix, translation_nm):
    checkpoint = joint_checkpoint()
    source = original.joint_row_binding
    members = tuple(sorted({occ.owner for occ in source.occurrences}))
    replay_joint_row_geometry(original, members)
    (aa,ab),(ac,ad) = matrix
    dx,dy = translation_nm
    native = _native_frame((aa,ab,ac,ad,dx,dy))

    def frame(value):
        a,b,c,d,x,y = value
        return (aa*a+ab*c, aa*b+ab*d, ac*a+ad*c, ac*b+ad*d,
                aa*x+ab*y+dx, ac*x+ad*y+dy)

    parts = tuple(replace(part, geometry_by_layer_nm=tuple((key, joint_transform_shapes(shapes, native))
        for key,shapes in part.geometry_by_layer_nm)) for part in source.parts)
    occurrences = tuple(replace(occ, master_to_local=frame(occ.master_to_local)) for occ in source.occurrences)
    placed_sites = tuple(replace(site, master_to_local=frame(site.master_to_local),
        geometry_nm=joint_transform_shapes(site.geometry_nm, native)) for site in source.placed_sites)
    inventory = compile_joint_terminal_inventory(artwork=joint_row_artwork(parts),
        expected_sites=source.expected_sites, placed_sites=placed_sites, connect_model=source.connect_model,
        routing_model=source.routing_model, pdk_provenance_sha256=source.pdk_provenance_sha256,
        checkpoint=checkpoint, limits=source.limits)
    binding = replace(source, parts=parts, occurrences=occurrences, placed_sites=placed_sites, joint_inventory=inventory)
    object.__setattr__(transformed, 'joint_row_binding', binding)
    replay_joint_row_geometry(transformed, members)
