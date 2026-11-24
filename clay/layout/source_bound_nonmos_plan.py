"""Placement plans whose internal geometry is owned by a native source assembly.

The rectangle tiler remains a separate contract. These plans retain the complete
unit/site/artwork binding and check actual member geometry at every translation.
"""
from dataclasses import dataclass
import hashlib
import json


def validate_native_assemblies(assemblies, *, tech_id, member_sizes):
    from clay.layout.bjt_local_assembly import NativeNonmosAssemblyBinding
    from clay.layout.source_resistor_pcell_assembly import CurrentResistorPCellAssembly
    if type(assemblies) is not tuple:
        raise TypeError('native assemblies require an exact tuple')
    owners = []
    keys = []
    sizes = {m.member: (m.width_nm, m.height_nm) for m in member_sizes}
    for assembly in assemblies:
        if type(assembly) not in (NativeNonmosAssemblyBinding, CurrentResistorPCellAssembly):
            raise TypeError('native assemblies require exact typed bindings')
        assembly.validate()
        pdk_id = assembly.pdk_id if type(assembly) is CurrentResistorPCellAssembly else assembly.source.pdk_id
        if pdk_id != tech_id:
            raise ValueError('native assembly PDK differs from compile inputs')
        key = tuple(m.owner for m in assembly.members)
        keys.append(key)
        owners.extend(key)
        for member in assembly.members:
            x0, y0, x1, y1 = member.bbox_nm
            if sizes.get(member.owner) != (x1-x0, y1-y0):
                raise ValueError('native assembly member size differs from compile inputs')
    if keys != sorted(set(keys)) or len(owners) != len(set(owners)):
        raise ValueError('native assemblies must be canonical and disjoint')


def _actual_member_artwork(views, *, translation):
    from clay.layout.occurrence_channels import _shape_keys
    from clay.layout.physical_terminal_inventory import PhysicalTerminalPolygonNM
    import klayout.db as kdb
    dx, dy = translation
    result = []
    for owner, view in sorted(views.items()):
        regions = {}
        for polygon in view.polys:
            if any(type(v) not in (int, float) or not float(v).is_integer()
                   for p in polygon.polygon.vertices for v in (p.x, p.y)):
                raise ValueError('native assembly actual artwork requires exact integer nm')
            key = (polygon.layer, polygon.datatype)
            region = regions.setdefault(key, kdb.Region())
            region.insert(kdb.Polygon([kdb.Point(int(p.x)-dx, int(p.y)-dy)
                                      for p in polygon.polygon.vertices]))
        result.append((owner, tuple((key, tuple(PhysicalTerminalPolygonNM(hull, holes)
                    for hull, holes in _shape_keys(region.merged())))
                    for key, region in sorted(regions.items()) if not region.is_empty())))
    return tuple(result)


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceBoundNonmosMacroPlan:
    macro_id: str
    assembly: object

    def __post_init__(self):
        from clay.layout.bjt_local_assembly import NativeNonmosAssemblyBinding
        from clay.layout.source_resistor_pcell_assembly import CurrentResistorPCellAssembly
        if type(self.macro_id) is not str or not self.macro_id:
            raise ValueError('native assembly plan requires a macro id')
        if type(self.assembly) not in (NativeNonmosAssemblyBinding, CurrentResistorPCellAssembly):
            raise TypeError('native assembly plan requires a materialized binding')
        self.assembly.validate()

    @property
    def kind(self):
        from clay.layout.source_resistor_pcell_assembly import CurrentResistorPCellAssembly
        return 'passive_bank' if type(self.assembly) is CurrentResistorPCellAssembly else 'bjt_cluster'

    @property
    def pdk_id(self):
        from clay.layout.source_resistor_pcell_assembly import CurrentResistorPCellAssembly
        return (self.assembly.pdk_id if type(self.assembly) is CurrentResistorPCellAssembly
                else self.assembly.source.pdk_id)

    @property
    def members(self):
        return tuple(m.owner for m in self.assembly.members)

    @property
    def local_bbox_nm(self):
        x0, y0, x1, y1 = self.assembly.bbox_nm
        return (0, 0, x1-x0, y1-y0)

    @property
    def member_plans(self):
        from clay.layout.nonrow_macro_plan import RigidNonmosMacroMemberPlan
        x0, y0, _x1, _y1 = self.assembly.bbox_nm
        return tuple(RigidNonmosMacroMemberPlan(member=m.owner,
            local_bbox_nm=(m.bbox_nm[0]-x0, m.bbox_nm[1]-y0,
                           m.bbox_nm[2]-x0, m.bbox_nm[3]-y0)) for m in self.assembly.members)

    @property
    def matching_pairs(self):
        return ()

    @property
    def local_relation_projections(self):
        return ()

    @property
    def local_alignment_projections(self):
        return ()

    def payload(self):
        # The public digest validates the current assembly before returning.
        return {'schema': 'clay.source-bound-nonmos-plan/v1', 'macro_id': self.macro_id,
                'kind': self.kind, 'assembly_sha256': self.assembly.sha256,
                'members': self.members, 'local_bbox_nm': self.local_bbox_nm,
                'member_plans': tuple(m.payload() for m in self.member_plans)}

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True,
                                         separators=(',', ':')).encode()).hexdigest()

    def translation_nm(self, devices):
        from clay.layout.source_resistor_pcell_assembly import CurrentResistorPCellAssembly
        if type(self.assembly) is CurrentResistorPCellAssembly:
            from clay.layout.source_resistor_placement import resistor_assembly_translation
            return resistor_assembly_translation(self.assembly, devices)
        self.assembly.validate()
        offsets = set()
        for member in self.member_plans:
            view = devices.get(member.member)
            if view is None:
                raise ValueError('native assembly placed member is missing')
            bbox = getattr(view, 'bbox', None)
            if (type(bbox) is not tuple or len(bbox) != 4
                    or any(type(v) is not int for v in bbox)):
                raise ValueError('native assembly placed member bbox is invalid')
            local = member.local_bbox_nm
            if (bbox[2]-bbox[0], bbox[3]-bbox[1]) != (local[2]-local[0], local[3]-local[1]):
                raise ValueError('native assembly placed member size changed')
            offsets.add((bbox[0]-local[0], bbox[1]-local[1]))
        if len(offsets) != 1:
            raise ValueError('native assembly placed member offsets changed')
        dx, dy = next(iter(offsets))
        ox, oy, _x1, _y1 = self.assembly.bbox_nm
        actual = _actual_member_artwork({m: devices[m] for m in self.members}, translation=(dx-ox, dy-oy))
        expected = tuple((m.owner, m.artwork) for m in self.assembly.members)
        if actual != expected:
            raise ValueError('native assembly actual member artwork changed')
        return dx, dy

    def placed_bbox(self, devices):
        dx, dy = self.translation_nm(devices)
        _x0, _y0, x1, y1 = self.local_bbox_nm
        return (dx, dy, dx+x1, dy+y1)


def compile_source_bound_plans(graph, inputs):
    """Join each complete assembly to exactly one declared physical aggregate."""
    assemblies = inputs.native_assemblies
    validate_native_assemblies(assemblies, tech_id=inputs.tech_id, member_sizes=inputs.member_sizes)
    result = {}
    for assembly in assemblies:
        members = tuple(m.owner for m in assembly.members)
        from clay.layout.source_resistor_pcell_assembly import CurrentResistorPCellAssembly
        kind = 'passive_bank' if type(assembly) is CurrentResistorPCellAssembly else 'bjt_cluster'
        macros = [m for m in graph.macros if m.kind == kind and m.members == members]
        if len(macros) != 1:
            raise ValueError('native assembly requires one complete physical macro owner')
        # A native assembly has fixed internal positions. A separate authored
        # packing/matching declaration must not silently replace those positions.
        if 'pack_rows' in dict(macros[0].attributes) or any(m.kind == 'matched_array' and len(m.members) > 1 and set(m.members) <= set(members)
                                      for m in graph.macros):
            raise ValueError('native assembly does not implement additional internal packing or matching')
        if any({r.left, r.right} <= set(members) for r in inputs.member_relations) or any(
                {r.first, r.second} <= set(members) for r in inputs.member_alignments):
            raise ValueError('native assembly does not implement additional internal member constraints')
        result[macros[0].macro_id] = SourceBoundNonmosMacroPlan(macro_id=macros[0].macro_id, assembly=assembly)
    return result
