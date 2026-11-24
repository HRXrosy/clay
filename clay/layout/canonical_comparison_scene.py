"""Serialize a live, completely consumed canonical scene without rebuilding it.

This is a representation boundary for shared external inputs. Placement-time
structure evidence is retained in its original scope; device LVS and full DRC
must still qualify each final routed artifact before any comparison score.
"""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path


def capture_current_comparison_scene(*, cell, tech, compact, structure, observation, intent=None):
    from clay.layout.place.current_reference_acceptance import CurrentReferenceStructureEvidence
    from clay.layout.place.final_quality_observation import observe_materialized_source
    from clay.layout.route.exact_artifact_evidence import _emitted_device_snapshot_sha256
    from clay.layout.route.route_cell import (
        _BridgedIntent, bridge_devices_from_cell_def, bridge_bjts_from_cell_def,
        _declared_ground_ports, _relocate_array_member_pins,
        _guard_ring_taps_from_stamps, _append_guard_ring_tap_pins)
    from clay.layout.flow_placement import project_final_route_placement
    from clay.layout.route.lvs_gate import route_cell_reference_text
    from clay.layout.route.pins import extract_pins
    from clay.layout.owned_artwork import _stamp_records
    from clay.port_roles import resolve_substrate_net, resolve_positive_supply_net
    from clay.well_isolation import declared_well_isolated_devices, deep_well_lvs_model_overrides

    if type(structure) is not CurrentReferenceStructureEvidence:
        raise TypeError('canonical scene requires actual complete current structure consumption')
    if intent is None:
        raise ValueError('canonical scene requires the consumed source intent')
    actual = observe_materialized_source(cell=cell, compact=compact)
    if actual.canonical_data() != observation.canonical_data():
        raise ValueError('canonical scene final observation differs from the live candidate')
    proof = structure.canonical_data()
    positions, geometry = compact.ring_inclusive
    if set(geometry) - set(positions):
        raise ValueError('canonical scene has artwork without a certified position')
    artwork_sha = _emitted_device_snapshot_sha256(geometry, positions)
    if (proof['final_artwork_sha256'] != artwork_sha
            or proof['observation_sha256'] != actual.sha256
            or proof['required_structure_status'] != 'pass'
            or proof['current_structure_status'] != 'pass'):
        raise ValueError('canonical scene structure belongs to a different or incomplete candidate')
    devices, passives = bridge_devices_from_cell_def(cell)
    substrate = resolve_substrate_net(cell, fallback=tech.substrate_net())
    positive = resolve_positive_supply_net(cell, fallback='')
    bridged = _BridgedIntent(devices, passives, bridge_bjts_from_cell_def(cell),
        well_isolated_devices=declared_well_isolated_devices(cell),
        substrate_net=substrate, deep_nwell_net=positive)
    report = extract_pins(bridged, positions, tech=tech, geometry=geometry,
                          return_report=True, include_power=True, per_device_fold=True,
                          extra_power_nets=_declared_ground_ports(cell))
    final_placement = project_final_route_placement(compact.logical_placement_basis,
        positions, geometry, final_placement_certificate=compact.final_placement_certificate,
        intent=intent)
    _relocate_array_member_pins(report.pins_by_net, bridged, cell, intent, tech, positions,
                               placement_obj=final_placement, geometry=geometry)
    # Pin seeds remain placement diagnostics. Retain the independent complete
    # AND/OR face alongside them; its masks must not be flattened into endpoints.
    from clay.layout.primitive_joint_access import compile_current_joint_access_ir
    from clay.layout.route.canonical_input_capture import _json_value
    joint_access_ir = compile_current_joint_access_ir(tech=tech, geometry=geometry,
        placement_um=positions, row_program=final_placement.row_program,
        row_materialization=final_placement.row_materialization,
        row_final_certificate=final_placement.row_final_certificate,
        intent=intent, source_cell=cell)
    _append_guard_ring_tap_pins(report.pins_by_net,
                               _guard_ring_taps_from_stamps(geometry, positions))
    pins = [asdict(pin) for net in sorted(report.pins_by_net) for pin in report.pins_by_net[net]]
    # Serialize all stamps, including source-independent rail/guard artwork.
    # No shape filtering, PCell reconstruction or parameter reinterpretation.
    stamps = {name: dict(position_um=positions[name],
        polygons_nm=_stamp_records(stamp, positions[name]),
        composite_members=list(getattr(stamp, 'composite_members', ())))
        for name, stamp in sorted(geometry.items())}
    if _emitted_device_snapshot_sha256(geometry, positions) != artwork_sha:
        raise ValueError('canonical scene pin extraction changed actual geometry')
    return dict(schema='clay.canonical-comparison-scene/v1',
        scope='complete_current_placement_scene_pending_outer_binding_completion_and_physical_signoff',
        physical_admission=False, source_sha256=proof['source_sha256'],
        source_inventory_sha256=proof['source_inventory_sha256'], pdk=tech.pdk_id,
        substrate=substrate, positive_supply=positive,
        isolated_lvs_models=deep_well_lvs_model_overrides(cell, tech.pdk_id),
        logical_devices=len(cell.devices), source_population=proof['source_population'],
        final_artwork_sha256=artwork_sha, observation_sha256=actual.sha256,
        final_placement_state_sha256=actual.final_placement_state_sha256,
        current_structure_sha256=structure.sha256,
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stamps=stamps, pins=pins, pin_counts_by_net=dict(Counter(pin['net'] for pin in pins)),
        joint_access_ir=_json_value(joint_access_ir), joint_access_ir_sha256=joint_access_ir.sha256,
        joint_access_implementation_sha256=hashlib.sha256(
            Path(__file__).with_name('primitive_joint_access.py').read_bytes()).hexdigest(),
        pin_gaps=list(report.uncovered),
        lvs_reference_text=route_cell_reference_text(
            cell, tech_id=tech.pdk_id, top_cell='CANONICAL_SCENE', full_extraction=True))


def emit_comparison_scene_gds(scene, path):
    """Emit every captured polygon verbatim; JSON status cannot grant admission."""
    import klayout.db as db
    experimental = scene.get('experimental_routing')
    if (scene.get('schema') != 'clay.canonical-prepared-routing-input/v5'
            and type(experimental) is dict and 'fixed_index_seal' in experimental):
        raise ValueError('fixed index seal requires prepared input v5')
    if (scene.get('schema') != 'clay.canonical-prepared-routing-input/v6'
            and type(experimental) is dict and 'group_full_feedback' in experimental):
        raise ValueError('group full feedback requires prepared input v6')
    if scene.get('schema') not in ('clay.canonical-comparison-scene/v1',
                                  'clay.canonical-prepared-routing-input/v1',
                                  'clay.canonical-prepared-routing-input/v2',
                                  'clay.canonical-prepared-routing-input/v3',
                                  'clay.canonical-prepared-routing-input/v4',
                                  'clay.canonical-prepared-routing-input/v5',
                                  'clay.canonical-prepared-routing-input/v6'):
        raise ValueError('unknown canonical scene schema')
    if (scene.get('schema') in ('clay.canonical-prepared-routing-input/v2',
                                'clay.canonical-prepared-routing-input/v3',
                                'clay.canonical-prepared-routing-input/v4',
                                'clay.canonical-prepared-routing-input/v5',
                                'clay.canonical-prepared-routing-input/v6')
            and (scene.get('schema') == 'clay.canonical-prepared-routing-input/v2'
                 or scene.get('pdk') == 'sky130')
            and (scene.get('pdk') != 'sky130'
                 or scene.get('lvs_reference_kind') != 'native-spice-source/v1')):
        raise ValueError('native prepared scene lacks its source reference identity')
    if (scene.get('schema') in ('clay.canonical-prepared-routing-input/v3',
                                'clay.canonical-prepared-routing-input/v4',
                                'clay.canonical-prepared-routing-input/v5',
                                'clay.canonical-prepared-routing-input/v6')
            and scene.get('pdk') != 'sky130' and 'lvs_reference_kind' in scene):
        raise ValueError('synthesized prepared scene has a foreign source reference')
    if scene.get('schema') in ('clay.canonical-prepared-routing-input/v4', 'clay.canonical-prepared-routing-input/v5',
                               'clay.canonical-prepared-routing-input/v6'):
        from clay.layout.route.canonical_input_capture import validate_experimental_prepared
        validate_experimental_prepared(scene)
    layout = db.Layout()
    layout.dbu = 0.001
    top = layout.create_cell('CANONICAL_SCENE')
    for name, stamp in sorted(scene['stamps'].items()):
        child = layout.create_cell(name)
        for layer, datatype, points in stamp['polygons_nm']:
            child.shapes(layout.layer(layer, datatype)).insert(db.Polygon([db.Point(*point) for point in points]))
        top.insert(db.CellInstArray(child.cell_index(), db.Trans()))
    options = db.SaveLayoutOptions()
    options.gds2_write_timestamps = False
    layout.write(str(path), options)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
