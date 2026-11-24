"""Placement-only attribution from a freshly replayed complete joint row face.

The temporary model seeds attribute actual masks. Their cardinality is never
used as the electrical terminal denominator and they are not router endpoints.
"""
from collections import Counter
from dataclasses import dataclass

from clay.layout.joint_row_geometry import _require
from clay.layout.joint_row_terminals import joint_terminal_contracts


@dataclass(frozen=True, slots=True)
class JointPlacementTerminalFace:
    stamp_id: str
    binding_sha256: str
    consumed_devices: tuple[str, ...]
    delta_nm: tuple[int, int]
    joint_inventory: object
    accesses: tuple

    @property
    def demand_ids(self):
        return tuple((row.source_net, self.stamp_id + '/' + row.component_id)
                     for row in self.joint_inventory.joint_components)


@dataclass(frozen=True, slots=True)
class PlacementTerminalFace:
    legacy_pins: tuple
    joint_faces: tuple[JointPlacementTerminalFace, ...]
    consumed_devices: frozenset[str]

    def attribute_model_pins(self, pins_by_net):
        """Private model operand; callers must count demand_ids independently."""
        from clay.layout.route.engine import Pin
        result = {net: [pin for pin in pins if pin.device not in self.consumed_devices]
                  for net, pins in pins_by_net.items()}
        for pin in self.legacy_pins:
            result.setdefault(pin.net, []).append(pin)
        for face in self.joint_faces:
            seen = set()
            for terminal in face.accesses:
                for point in terminal.accesses_nm:
                    key = terminal.component_id, terminal.layer, point
                    if key in seen:
                        continue
                    seen.add(key)
                    absolute = tuple(point[i]+face.delta_nm[i] for i in range(2))
                    result.setdefault(terminal.net, []).append(Pin(net=terminal.net, device=terminal.member,
                        terminal=terminal.role, layer=terminal.layer, xy_nm=absolute,
                        xy_um=tuple(value/1000 for value in absolute)))
        return result

    def terminal_counts(self, attributed_pins):
        joint_members = {member for face in self.joint_faces for member in face.consumed_devices}
        result = Counter({net: sum(pin.device not in joint_members for pin in pins)
                          for net, pins in attributed_pins.items()})
        result.update(net for face in self.joint_faces for net, _identity in face.demand_ids)
        return result


def project_joint_placement_face(*, native_geometry, stamp, position, members, delta, tech,
                                 native_row_sha256, source_members):
    """Replay one native graph, then transport it through exact current artwork.

    The caller supplies the same realization and member map just checked by the
    complete materialization verifier. Its geometry SHA joins the stamp to that
    realization; a supplied digest is not independent source authority. This
    An active invocation may reuse that complete replay only while its detached
    source/PDK/native snapshots still match. Standalone calls replay independently.
    """
    from clay.layout.joint_row_geometry import joint_record_counter
    from clay.layout.joint_row_stamp import JointRowStampBinding
    from clay.layout.occurrence_artwork import polygon_records
    from clay.layout.owned_artwork import _stamp_records, _members
    from clay.layout.placed_geometry import placed_point_nm
    from clay.layout.place.row_program import RowMemberSpec
    from clay.layout.place.row_replay_session import current_row_terminal_inventory
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    _require(_members(members) == members and type(source_members) is dict
             and set(members) <= set(source_members)
             and all(type(source_members[name]) is RowMemberSpec and source_members[name].device == name
                     for name in members), 'placement source member denominator differs')
    _require(type(delta) is tuple and len(delta) == 2 and all(type(value) is int for value in delta),
             'placement frame requires an exact integer delta')
    binding = getattr(stamp, 'joint_row_stamp_binding', None)
    _require(type(binding) is JointRowStampBinding, 'placement face lacks its complete stamp binding')
    binding.__post_init__()
    _require(binding.source == getattr(native_geometry, 'joint_row_binding', None)
             and binding.consumed_devices == members
             and tuple(sorted(getattr(stamp, 'composite_members', ()))) == members,
             'placement source and actual stamp binding differ')
    _require(binding.native_row_sha256 == native_row_sha256, 'placement native row geometry binding differs')
    actual = _stamp_records(stamp, position)
    placed_origin = placed_point_nm(0, 0, position)
    _require(tuple(placed_origin[i]-binding.native_origin_nm[i] for i in range(2)) == delta,
             'placement stamp origin and current integer frame differ')
    local = tuple((layer, datatype, tuple((x-delta[0], y-delta[1]) for x, y in vertices))
                  for layer, datatype, vertices in actual)
    _require(joint_record_counter(local) == joint_record_counter(polygon_records(native_geometry.polygons)),
             'placement current stamp artwork differs from the complete native polygon multiset')
    # Equality above carries the complete all-layer multiset into this native
    # frame. One replay therefore checks both physical operands; expanding an
    # owner union here would duplicate shared accessories and graph extraction.
    inventory = current_row_terminal_inventory(native_geometry, members, source_members, tech)
    accesses = joint_terminal_contracts(inventory, grid=pdk_manufacturing_grid_nm(tech))
    return JointPlacementTerminalFace(stamp.name, binding.sha256, members, delta, inventory, accesses)


def build_placement_terminal_face(**operands):
    """Replay complete compiler context and current stamp geometry before use."""
    from clay.layout.row_terminal_probes import _replayed_row_pins
    joint_faces = []
    require_final = operands.get('row_final_certificate') is not None
    pins, members = _replayed_row_pins(**operands, _require_final_certificate=require_final,
                                     _joint_faces=joint_faces)
    generated_pins, generated_members = _generated_row_placement_pins(operands, members)
    return PlacementTerminalFace(pins + generated_pins, tuple(joint_faces), members | generated_members)


def _generated_row_placement_pins(operands, already_consumed):
    """Project native generated-row straps before solving the rail reservation.

    Routing already replaces standalone MOS probes with member_pins. Placement
    must seed exactly that interface too, including rows without a typed fixed
    terminal inventory. Every native polygon is replayed in the current frame.
    """
    from clay.layout.place.row_program import verify_row_program_materialization
    from clay.layout.joint_row_geometry import joint_record_counter
    from clay.layout.occurrence_artwork import polygon_records
    from clay.layout.owned_artwork import _stamp_records
    from clay.layout.route.engine import Pin

    program, materialization = operands.get('row_program'), operands.get('row_materialization')
    if program is None or materialization is None:
        return (), frozenset()
    entries = [entry for entry in program.entries
               if not set(entry.consumed_devices).intersection(already_consumed)]
    if not entries:
        return (), frozenset()
    tech = operands['tech']
    verify_row_program_materialization(materialization, program=program, tech_id=tech.pdk_id)
    realizations = {row.entry_id: row for row in materialization.realizations}
    source_members = {member.device: member for member in program.members}
    pins, consumed = [], set()
    role_fields = {'d': 'drain_net', 's': 'source_net', 'g': 'gate_net', 'b': 'body_net'}
    for entry in entries:
        realization = realizations[entry.entry_id]
        native = realization.geometry
        _require(callable(getattr(native, 'member_pins', None)), 'generated row has no native terminal interface')
        stamp = operands['geometry'][entry.group]
        actual = _stamp_records(stamp, operands['placement_um'][entry.group])
        points = [point for _layer, _datatype, vertices in actual for point in vertices]
        _require(bool(points), 'generated row artwork is empty')
        box = (min(x for x, y in points), min(y for x, y in points),
               max(x for x, y in points), max(y for x, y in points))
        local = realization.local_bbox_nm
        delta = box[0] - local[0], box[1] - local[1]
        _require((box[2] - local[2], box[3] - local[3]) == delta,
                 'generated row artwork is not a rigid translation')
        translated = tuple((layer, datatype, tuple((x-delta[0], y-delta[1]) for x, y in vertices))
                           for layer, datatype, vertices in actual)
        _require(joint_record_counter(translated) == joint_record_counter(polygon_records(native.polygons)),
                 'generated row artwork differs from the retained materialization')
        consumed.update(entry.consumed_devices)
        for member in entry.active_devices:
            source = source_members[member]
            for role, point in native.member_pins(member).items():
                _require(role in role_fields, 'generated row terminal role is unsupported')
                net = getattr(source, role_fields[role])
                _require(bool(net), 'generated row terminal has no source net')
                absolute = point[0] + delta[0], point[1] + delta[1]
                pins.append(Pin(net=net, device=member, terminal=role, layer=tuple(tech.metal_key(1)),
                                xy_nm=absolute, xy_um=tuple(value / 1000 for value in absolute)))
    return tuple(pins), frozenset(consumed)
