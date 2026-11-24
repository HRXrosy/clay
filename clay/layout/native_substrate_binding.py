"""Source-bound SKY130 native NMOS substrate access evidence.

Magic's substrate node is not a drawn polygon.  This record binds each raw
channel to Magic's body node, and separately binds that node's real P+ tap to
the ordinary physical conductor graph.  It makes no global substrate mask.
"""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import klayout.db as kdb

from clay.layout.native_sky130 import _raw_anchors, _raw_devices, _read_gds, native_connect_model
from clay.layout.connectivity import extract_nodes
from clay.layout.joint_terminal_inventory import (
    TerminalSiteRequirement, OccurrenceTerminalSite, _native_frame,
)


def _require(condition, message):
    if not condition:
        raise ValueError('native substrate ' + message)


def _artwork(shapes):
    from clay.layout.joint_row_geometry import _shapes
    return tuple((key, _shapes(kdb.Region(list(polygons)).merged()))
                 for key, polygons in sorted(shapes.items()))


def _region(shapes):
    from clay.layout.joint_row_geometry import _region as region
    return region(shapes)


def _digest(data):
    return sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class NativeSubstrateBinding:
    raw_gds_sha256: str
    contacted_gds_sha256: str
    spice_text: str
    ext_text: str
    scale_nm: int
    raw_artwork: tuple
    contacted_artwork: tuple
    pdk_provenance_sha256: str
    source_bodies: tuple
    source_sha256: str
    source_identity: str
    source_substrate_net: str
    receipt_text: str
    extraction_frame: tuple


def capture_native_substrate_binding(*, raw_gds_path, contacted_gds_path, spice_path,
                                     ext_path, scale_nm, occurrences, members, tech,
                                     source_cell, source_path, receipt_path):
    """Observe physical source material and return the body site ledger.

    All paths refer to one produced PCell and its matching Magic extraction.
    The binding stores the actual geometry and extraction for independent replay.
    """
    from clay.layout.route.resource_world import pdk_provenance_digest
    from clay.layout.native_sky130 import native_source_identity
    from clay.pdk.sky130_native_geometry import native_geometry_spec
    from clay.port_roles import resolve_substrate_net
    from clay.parser import parse_clay_file
    source_bytes = Path(source_path).read_bytes()
    source_cells = parse_clay_file(source_path).cell_defs
    _require(len(source_cells) == 1 and
             native_source_identity(source_cells[0]) == native_source_identity(source_cell),
             'source cell differs from the extraction source')
    devices = {device.instance_name: device for device in source_cell.devices}
    receipt = json.loads(Path(receipt_path).read_text())
    # The original three-finger evidence predates explicit model metadata.
    model = receipt.get('model', 'sky130_fd_pr__nfet_01v8' if len(occurrences) == 3 else None)
    parameterized = receipt.get('synthesis') in ('parameterized', 'cmos_gate')
    is_single_pmos = (
        parameterized
        and len(occurrences) == 1
        and model in ('sky130_fd_pr__pfet_01v8', 'sky130_fd_pr__pfet_01v8_lvt')
    )
    if is_single_pmos:
        declared_bodies = {
            member.body_net
            for name, member in members.items()
            if name in devices and member.body_net and devices[name].pins.get('b') == member.body_net
        }
        _require(
            len(declared_bodies) == 1 and len(members) == 1 and all(
                name in devices and devices[name].pins.get('b') == member.body_net
                for name, member in members.items()
            ),
            'source PMOS body is not uniquely declared',
        )
        substrate_net, = declared_bodies
    else:
        substrate_net = resolve_substrate_net(source_cell, fallback='')
        _require(model in ('sky130_fd_pr__nfet_01v8', 'sky130_fd_pr__nfet_01v8_lvt'),
                 'native substrate model is unsupported')
        _require(substrate_net and all(name in devices and devices[name].pins.get('b') == substrate_net
                 and member.body_net == substrate_net for name, member in members.items()),
                 'source body is not the uniquely declared substrate')
    for name, member in members.items():
        device = devices[name]
        spec = native_geometry_spec(device)
        _require(spec['model'] == model and spec['l_nm'] == member.length_nm
                 and spec['w_nm'] == member.width_nm * spec['nf']
                 and spec['occurrences'] == member.fingers and
                 tuple(device.pins[role] for role in ('g', 'd', 's', 'b')) ==
                 (member.gate_net, member.drain_net, member.source_net, member.body_net),
                 'current native source device differs from the row member')
    raw, _ = _read_gds(raw_gds_path)
    contacted, _ = _read_gds(contacted_gds_path)
    evidence = NativeSubstrateBinding(
        _digest(Path(raw_gds_path).read_bytes()), _digest(Path(contacted_gds_path).read_bytes()),
        Path(spice_path).read_text(), Path(ext_path).read_text(), scale_nm,
        _artwork(raw), _artwork(contacted), pdk_provenance_digest(tech),
        tuple(sorted((name, member.body_net) for name, member in members.items())),
        _digest(source_bytes), native_source_identity(source_cell), substrate_net,
        Path(receipt_path).read_text(), occurrences[0].master_to_local)
    sites, positions = substrate_sites(evidence, occurrences=occurrences, members=members, tech=tech)
    return evidence, sites, positions


def _validate_parameterized_substrate_profile(receipt, *, occurrences):
    count = len(occurrences)
    width_nm = receipt.get('unit_width_nm', receipt.get('width_nm'))
    length_nm = receipt.get('length_nm')
    model = receipt.get('model')
    occ_ids = [
        o.occurrence_id if hasattr(o, 'occurrence_id') else str(o)
        for o in occurrences
    ]
    valid_nmos = (
        model in ('sky130_fd_pr__nfet_01v8', 'sky130_fd_pr__nfet_01v8_lvt')
        and type(count) is int
        and 1 <= count <= 4096
    )
    valid_single_pmos = (
        model in ('sky130_fd_pr__pfet_01v8', 'sky130_fd_pr__pfet_01v8_lvt')
        and type(count) is int
        and count == 1
    )
    _require((valid_nmos or valid_single_pmos) and
             type(width_nm) is int and width_nm >= 420 and width_nm % 5 == 0 and
             type(length_nm) is int and length_nm >= 150 and length_nm % 5 == 0 and
             len(set(occ_ids)) == len(occ_ids),
             'supported distinct parameterized MOS occurrences are required')
    return model, (count, width_nm, length_nm)


def substrate_sites(evidence, *, occurrences, members, tech):
    """Recompute body sites from current source channels and actual masks."""
    from clay.layout.route.resource_world import pdk_provenance_digest
    from clay.layout.joint_row_geometry import _shapes
    _require(type(evidence) is NativeSubstrateBinding and tech.pdk_id == 'sky130', 'binding or PDK differs')
    _require(pdk_provenance_digest(tech) == evidence.pdk_provenance_sha256, 'PDK provenance differs')
    receipt = json.loads(evidence.receipt_text)
    outputs = receipt.get('outputs', {})
    _require(receipt.get('source_sha256') == evidence.source_sha256 and
             outputs.get('primitive.gds') == evidence.raw_gds_sha256 and
             outputs.get('contacted.gds') == evidence.contacted_gds_sha256 and
             outputs.get('native.spice') == _digest(evidence.spice_text.encode()) and
             outputs.get('primitive_flat.ext') == _digest(evidence.ext_text.encode()),
             'extraction receipt does not bind source and actual artifacts')
    _require(type(evidence.scale_nm) is int and evidence.scale_nm == 5, 'extraction scale differs')
    # Historical three-finger evidence predates explicit size metadata.  Its
    # original dimensions remain independently checked against the source and
    # every extracted channel below; newer two-finger evidence must carry them.
    legacy = len(occurrences) == 3
    profile=(len(occurrences), receipt.get('unit_width_nm', 500 if legacy else None),
             receipt.get('length_nm', 150 if legacy else None))
    model=receipt.get('model', 'sky130_fd_pr__nfet_01v8' if legacy else None)
    count,width_nm,length_nm=profile
    parameterized = receipt.get('synthesis') in ('parameterized', 'cmos_gate')
    if parameterized:
        model, (count, width_nm, length_nm) = _validate_parameterized_substrate_profile(
            receipt, occurrences=occurrences
        )
    else:
        _require((model,profile) in (
                 ('sky130_fd_pr__nfet_01v8',(3,500,150)),
                 ('sky130_fd_pr__nfet_01v8',(2,420,150)),
                 ('sky130_fd_pr__nfet_01v8',(6,2000,150)),
                 ('sky130_fd_pr__nfet_01v8',(10,500,150)),
                 ('sky130_fd_pr__nfet_01v8_lvt',(10,2000,150))) and
                 len({o.occurrence_id for o in occurrences}) == len(occurrences),
                 'supported distinct source occurrences are required')
    expected_kind = 'pmos' if model in ('sky130_fd_pr__pfet_01v8', 'sky130_fd_pr__pfet_01v8_lvt') else 'nmos'
    _require(len({o.raw_part_id for o in occurrences}) == 1 and
             len({o.master_to_local for o in occurrences}) == 1, 'source raw frame differs')
    expected_owners = {o.owner for o in occurrences}
    _require(type(evidence.source_bodies) is tuple and evidence.source_bodies == tuple(sorted(evidence.source_bodies))
             and len(dict(evidence.source_bodies)) == len(evidence.source_bodies)
             and set(dict(evidence.source_bodies)) == expected_owners, 'source body denominator differs')
    if members is None:
        members = {name: SimpleNamespace(kind=expected_kind, width_nm=width_nm, length_nm=length_nm, body_net=net)
                   for name, net in evidence.source_bodies}
    else:
        _require(all(name in members and members[name].body_net == net for name, net in evidence.source_bodies),
                 'current source body differs')
    source_members = [members[o.owner] for o in occurrences]
    _require(all(m.kind == expected_kind and m.width_nm == width_nm and m.length_nm == length_nm and m.body_net
                 for m in source_members), 'source MOS channel or body differs')
    _require(len({m.body_net for m in source_members}) == 1, 'source bodies differ')
    _require(all(m.body_net == evidence.source_substrate_net for m in source_members),
             'source body differs from the declared substrate')
    raw = {key: _region(shapes) for key, shapes in evidence.raw_artwork}
    scene = {key: _region(shapes) for key, shapes in evidence.contacted_artwork}
    _require(all(key in scene and (region - scene[key]).is_empty() for key, region in raw.items()),
             'contact producer changed raw PCell geometry')
    frame = _native_frame(evidence.extraction_frame)
    current_frame = _native_frame(occurrences[0].master_to_local)
    for occurrence in occurrences:
        master = {key: _region(shapes) for key, shapes in occurrence.master.geometry_by_layer_nm}
        _require(any(source.auxiliary_gds_sha256 == evidence.raw_gds_sha256
                     for source in occurrence.master.sources), 'raw PCell producer differs')
        _require(set(master) == set(raw) and
                 all((master[key].transformed(frame) ^ raw[key]).is_empty() for key in raw),
                 'raw PCell differs from occurrence master')
    rows = _raw_devices(evidence.spice_text, model, 4)
    _require(len(rows) == count and all(float(p['w']) * 1000 == width_nm and float(p['l']) * 1000 == length_nm for _, p in rows),
             'native extraction channel dimensions differ')
    bodies = {nodes[3] for nodes, _ in rows}
    _require(len(bodies) == 1, 'native extracted body nodes differ')
    body, = bodies
    anchors = _raw_anchors(evidence.ext_text, evidence.scale_nm)
    if expected_kind == 'pmos':
        _require(body in anchors and anchors[body][2] in ('nw', 'nsc', 'nnd'),
                 'body lacks a physical N-well tap anchor')
        x, y, _ = anchors[body]
        probe = kdb.Region(kdb.Box(x, y, x+1, y+1))
        nplus = scene.get(tuple(tech.build_layer('nplus')), kdb.Region())
        pplus = scene.get(tuple(tech.build_layer('pplus')), kdb.Region())
        nwell = scene.get(tuple(tech.build_layer('nwell')), kdb.Region())
        _require(not (probe & nwell).is_empty(), 'body anchor is not in the actual N-well')
        nodes = extract_nodes({key: list(region.each()) for key, region in scene.items()}, native_connect_model('mos'))
        hits = [node for node in nodes if (64, 20) in node and not (node[(64, 20)] & probe).is_empty()]
        _require(len(hits) == 1 and (65, 44) in hits[0] and not hits[0][(65, 44)].is_empty()
                 and (68, 20) in hits[0] and not hits[0][(68, 20)].is_empty(),
                 'N-well tap has no actual contact to routing metal')
        physical_tap = hits[0][(65, 44)] & nplus
        _require(not physical_tap.is_empty() and (hits[0][(65, 44)] - nplus).is_empty()
                 and (physical_tap & pplus).is_empty(),
                 'tap is not entirely N+ implanted')
        _require((physical_tap - hits[0][(64, 20)]).is_empty() and (raw[(65, 20)] - hits[0][(64, 20)]).is_empty(),
                 'PMOS channel or N+ tap is outside connected N-well')
        for kind in ('dnwell', 'pwell'):
            well = scene.get(tuple(tech.build_layer(kind)), kdb.Region())
            _require((well & physical_tap).is_empty() and (well & raw[(65, 20)]).is_empty(),
                     'isolated or P-well substrate context is unsupported')
    else:
        _require(body in anchors and anchors[body][2] == 'ppd', 'body lacks a physical P+ tap anchor')
        x, y, _ = anchors[body]
        probe = kdb.Region(kdb.Box(x, y, x+1, y+1))
        tap = scene.get((65, 44), kdb.Region())
        pplus = scene.get(tuple(tech.build_layer('pplus')), kdb.Region())
        _require(not (probe & tap & pplus).is_empty(), 'body anchor is not in the actual P+ tap')
        nodes = extract_nodes({key: list(region.each()) for key, region in scene.items()}, native_connect_model('mos'))
        hits = [node for node in nodes if (65, 44) in node and not (node[(65, 44)] & probe).is_empty()]
        _require(len(hits) == 1 and (68, 20) in hits[0] and not hits[0][(68, 20)].is_empty(),
                 'P+ tap has no actual contact to routing metal')
        physical_tap = hits[0][(65, 44)] & pplus
        _require(not physical_tap.is_empty() and (hits[0][(65, 44)] - pplus).is_empty(),
                 'tap is not entirely P+ implanted')
        nwell = scene.get(tuple(tech.build_layer('nwell')), kdb.Region())
        _require((nwell & physical_tap).is_empty() and (nwell & raw[(65, 20)]).is_empty(),
                 'isolated or N-well substrate context is unsupported')
        for kind in ('dnwell', 'pwell'):
            well = scene.get(tuple(tech.build_layer(kind)), kdb.Region())
            _require((well & physical_tap).is_empty() and (well & raw[(65, 20)]).is_empty(),
                     'isolated or P-well substrate context is unsupported')
    # Each device record carries a channel coordinate and its actual gate/body
    # extraction nodes.  Match that coordinate to the selected raw gate crossing.
    extracted = []
    for line in evidence.ext_text.splitlines():
        fields = shlex.split(line)
        if len(fields) >= 11 and fields[:3] == ['device', 'msubckt', model]:
            extracted.append((int(fields[3])*evidence.scale_nm, int(fields[4])*evidence.scale_nm,
                              fields[9].casefold(), fields[10].casefold()))
    _require(len(extracted) == count and Counter((b, g) for _, _, b, g in extracted) ==
             Counter((nodes[3], nodes[1]) for nodes, _ in rows),
             'spice and extraction channel nodes differ')
    unmatched = list(extracted)
    for occurrence in occurrences:
        sites = {site.site_id: site for site in occurrence.master.terminal_sites}
        if occurrence.channel_site_id:
            _require(occurrence.channel_site_id in sites, 'source channel gate is missing')
            gate = sites[occurrence.channel_site_id]
        else:
            gates = [site for site in occurrence.master.terminal_sites if site.role_group == 'gate']
            _require(len(gates) == 1, 'source channel gate is missing')
            gate = gates[0]
        _require(gate.role_group == 'gate' and len(gate.geometry_by_layer_nm) == 1,
                 'source channel gate is malformed')
        channel = raw[(65, 20)] & _region(gate.geometry_by_layer_nm[0][1]).transformed(frame)
        matches = [r for r in unmatched if r[2] == body and
                   not (channel & kdb.Region(kdb.Box(r[0], r[1], r[0]+1, r[1]+1))).is_empty()]
        _require(len(matches) == 1, 'source occurrence does not identify one extracted channel')
        unmatched.remove(matches[0])
    _require(not unmatched, 'extracted channel denominator differs')
    tap_shapes = _shapes(physical_tap.merged().transformed(frame.inverted()))
    placed_tap_shapes = _shapes(physical_tap.merged().transformed(frame.inverted()).transformed(current_frame))
    provenance = _digest((evidence.raw_gds_sha256 + evidence.contacted_gds_sha256 +
                          _digest(evidence.spice_text.encode()) + _digest(evidence.ext_text.encode())).encode())
    sites, positions = [], []
    for occurrence in occurrences:
        site_id = occurrence.occurrence_id + '/substrate-tap'
        sites.append(TerminalSiteRequirement(site_id, occurrence.owner, occurrence.occurrence_id,
            members[occurrence.owner].body_net, ('b',), (65, 44), (65, 44), tap_shapes,
            occurrence.master.variant_sha256, provenance))
        positions.append(OccurrenceTerminalSite(site_id, occurrence.master.variant_sha256,
            occurrence.master_to_local, placed_tap_shapes))
    return tuple(sorted(sites, key=lambda s:s.site_id)), tuple(sorted(positions, key=lambda s:s.site_id))
