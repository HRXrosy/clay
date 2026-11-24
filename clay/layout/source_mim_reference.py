"""Reopen retained occurrence-capture operands for MIM source correspondence.

This adapter consumes the public capture schema and explicit installed evidence
paths. It does not import a development circuit, infer electrical capacitance,
or promote the retained capture to a fresh reference extraction/signoff result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from clay.frontend.reference_occurrence_binding import (
    LogicalOwner, PhysicalOccurrence, solve_reference_occurrence_binding,
)
from clay.frontend.unit_master_geometry import UnitMasterGeometry, UnitMasterSource, UnitTerminalSite
from clay.layout.intent import ArrayIntent, replay_array_intent
from clay.layout.physical_terminal_inventory import PhysicalTerminalPolygonNM
from clay.layout.source_mim_materialization import (
    MimSourceRecipe, _files, _member, _reopen, _require, _source, _technology,
    capture_mim_source_association,
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(type(value) is dict, "capture artifact must be an object")
    return value


def _path(root, relative):
    path = (root / relative).resolve(strict=True)
    _require(path.is_relative_to(root) and path.is_file(), "capture artifact path escapes its declared root")
    return path


def _geometry(rows):
    return tuple((tuple(layer), tuple(PhysicalTerminalPolygonNM(
        tuple(map(tuple, shape["hull_nm"])), tuple(tuple(map(tuple, ring)) for ring in shape["holes_nm"]))
        for shape in polygons)) for layer, polygons in rows)


def _master(row):
    result = UnitMasterGeometry(row["pdk_id"], row["parameter_class"],
        tuple(UnitMasterSource(**source) for source in row["sources"]), _geometry(row["geometry_by_layer_nm"]),
        tuple(UnitTerminalSite(site["site_id"], site["role_group"], _geometry(site["geometry_by_layer_nm"]))
              for site in row["terminal_sites"]), tuple(row["unresolved_access_obligations"]))
    _require(result.variant_sha256 == row["variant_sha256"] and result.artwork_sha256 == row["artwork_sha256"],
             "captured master artwork or variant digest differs")
    return result


def _roles(rows):
    return tuple((role, tuple(nodes)) for role, nodes in rows)


def _bound_capture(directory, root, pdk_evidence_paths, tech):
    header_path = directory / "capture.json"
    header_sha = _digest(header_path)
    header = _read(header_path)
    _require(header.get("schema") == "clay.reference-occurrence-capture/v1"
             and header.get("status") == "UNIQUE_TERMINAL_GRAPH_OWNERSHIP"
             and header.get("scope") == "reference_targets_only_not_acceptance", "capture scope/status differs")
    expected = {str(header_path): header_sha}
    _require(set(header["output_sha256"]) == {"binding.json", "physical_graph.json", "quality_contract.json"},
             "capture output denominator differs")
    for name, digest in header["output_sha256"].items():
        path = _path(directory, name)
        _require(_digest(path) == digest, "capture artifact digest changed")
        expected[str(path)] = digest
    for relative, digest in header["source_inputs"]:
        path = _path(root, relative)
        _require(_digest(path) == digest, "retained source/input artifact digest changed")
        _require(str(path) not in expected or expected[str(path)] == digest, "capture binds conflicting artifact digests")
        expected[str(path)] = digest
    mapping = header["mapping"]
    _require(hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode()).hexdigest() == header["mapping_sha256"], "capture mapping digest differs")
    _require(type(pdk_evidence_paths) is dict and set(pdk_evidence_paths) == {name for name, _ in mapping["pdk_files"]},
             "explicit current PDK evidence path denominator differs")
    for name, digest in mapping["pdk_files"]:
        path = Path(pdk_evidence_paths[name]).resolve(strict=True)
        _require(_digest(path) == digest, "current PDK parameter evidence digest changed")
        _require(str(path) not in expected or expected[str(path)] == digest, "capture binds conflicting PDK digests")
        expected[str(path)] = digest
    quality = _read(directory / "quality_contract.json")
    authority = quality["source_authority"]
    _require(_technology(tech) == (authority["pdk_id"], authority["pdk_snapshot_sha256"],
                authority["pdk_installed_authority_sha256"]), "capture and current PDK identity differs")
    source_path = _path(root, header["source_definition"]["source_artifact"])
    _require(_digest(source_path) == mapping["source_sha256"], "captured source bytes differ")
    from clay.parser import parse_clay_file
    cells = [cell for cell in parse_clay_file(source_path).cell_defs
             if cell.local_name == header["source_definition"]["top_cell"]]
    _require(len(cells) == 1, "captured source cell is not unique")
    bound = tuple(sorted(expected.items()))
    _reopen(bound)
    return header, cells[0], bound


def _binding_graph(data):
    physical = tuple(PhysicalOccurrence(row["occurrence_id"], row["parameter_class"], _roles(row["terminals"]))
                     for row in data["physical"])
    logical = tuple(LogicalOwner(row["owner_id"], row["parameter_class"], _roles(row["terminals"]), row["multiplicity"])
                    for row in data["logical"])
    anchors = tuple(map(tuple, data["anchors"]))
    fresh = solve_reference_occurrence_binding(physical, logical, port_anchors=anchors, prove_net_mapping=True)
    _require(fresh.status == "UNIQUE" and fresh.net_mapping_proof.status == "UNIQUE",
             "retained owner/net correspondence does not replay uniquely")
    saved = data["binding"]
    _require(fresh.bindings == tuple(map(tuple, saved["bindings"]))
             and fresh.witness.input_sha256 == saved["witness"]["input_sha256"]
             and fresh.net_mapping_proof.net_to_node == tuple(map(tuple, saved["net_mapping_proof"]["net_to_node"])),
             "capture owner/net binding differs from replay")
    return physical, logical, fresh


def _site_roles(master, observations, physical):
    by_id, result = {row.occurrence_id: row for row in physical}, None
    for observation in observations:
        local = observation["local_unit_observation"]
        _require(local["variant_sha256"] == master.variant_sha256
                 and local["artwork_sha256"] == master.artwork_sha256, "observed master identity differs")
        terminals = dict(by_id[observation["occurrence_id"]].terminals)
        rows = []
        _require(tuple(site["site_id"] for site in local["terminal_sites"]) == tuple(site.site_id for site in master.terminal_sites),
                 "reference unit site denominator differs")
        for site in local["terminal_sites"]:
            _require(site["global_uncovered_area_nm2"] == 0 and site["local_uncovered_area_nm2"] == 0,
                     "reference site has unresolved artwork/node coverage")
            hits = tuple(map(str, site["global_node_hits"]))
            roles = [role for role, nodes in terminals.items() if hits == nodes]
            _require(len(roles) == 1 and roles[0] in {"p", "n"}, "reference site has no unique bound source role")
            rows.append((site["site_id"], roles[0]))
        rows = tuple(rows)
        _require(result is None or rows == result, "reference units disagree on site/source role binding")
        result = rows
    return result


def mim_source_recipes_from_capture(source_cell, *, capture_directory, project_root, tech, choices, pdk_evidence_paths):
    """Bind explicit rectangular choices to reopened retained source/graph facts.

The selected source can be a freshly authored subset with identical electrical
declarations. Every capacitor in that source must be requested and retained.
No source orientation or observed rank is converted to an automatic grid.
"""
    directory, root = Path(capture_directory).resolve(strict=True), Path(project_root).resolve(strict=True)
    source_sha, _inventory, canonical = _source(source_cell)
    names = {device.instance_name for device in canonical.devices if device.device_class == "cap"}
    _require(type(choices) is dict and set(choices) == names and bool(names), "complete source capacitor choice denominator differs")
    header, original, files = _bound_capture(directory, root, pdk_evidence_paths, tech)
    graph, data = _read(directory / "physical_graph.json"), _read(directory / "binding.json")
    _require(not graph["port_unknowns"] and not any(graph["coverage"].values())
             and all(row["equivalent"] and row["xor_area_nm2"] == 0 for row in graph["selected_layer_equivalence"]),
             "retained reference geometry association is unresolved")
    physical, logical, binding = _binding_graph(data)
    owned = dict(binding.bindings)
    observations = {row["occurrence_id"]: row for row in graph["occurrences"]}
    masters = {row["variant_sha256"]: row for row in graph["local_unit_library"]}
    parameters = {row["instance"]: row for row in data["logical_parameters"]}
    cap_logical = tuple(row for row in logical if row.owner_id in names)
    cap_physical = tuple(row for row in physical if owned[row.occurrence_id] in names)
    used_nets = {net for row in cap_logical for _role, nets in row.terminals for net in nets}
    anchors = tuple((net, node) for net, node in binding.net_mapping_proof.net_to_node if net in used_nets)
    from clay.layout.generic_placement import _resistor_orient_map
    orientations = _resistor_orient_map(canonical)
    result = {}
    for name in sorted(names):
        choice = replay_array_intent(choices[name])
        _require(type(choice) is ArrayIntent and choice.group == name and choice.style == "discrete" and choice.order == "ratio",
                 "MIM construction choice must be an explicit discrete ratio array")
        _require(getattr(choice, "unit_pattern", None) is None, "MIM sparse unit_pattern construction is unsupported")
        _require(not choice.pin_facing, "MIM pin_facing construction is unsupported")
        member, value, count, _pins = _member(canonical, name)
        _require(member == _member(original, name)[0], "current source electrical declaration differs from retained source")
        parameter = parameters[name]
        _require(parameter["device_class"] == "cap" and parameter["source_model"] == member[3]
                 and parameter["multiplicity"] == count and parameter["source_value_f"] == value
                 and tuple(sorted(parameter["sizing"].items())) == member[5],
                 "captured source parameter correspondence differs")
        ids = tuple(sorted(row.occurrence_id for row in cap_physical if owned[row.occurrence_id] == name))
        actual = tuple(observations[key] for key in ids)
        variants = {row["local_unit_observation"]["variant_sha256"] for row in actual}
        _require(len(actual) == count and len(variants) == 1, "complete source multiplicity requires one proven unit variant")
        master = _master(masters[next(iter(variants))])
        association = capture_mim_source_association(canonical, logical_id=name, master=master, source_member=member,
            physical_occurrences=cap_physical, logical_owners=cap_logical, port_anchors=anchors,
            observed_variants=tuple((key, master.variant_sha256) for key in ids),
            evidence_files=tuple(path for path, _sha in files), tech=tech)
        result[name] = MimSourceRecipe(association, master, _site_roles(master, actual, cap_physical),
            rows=choice.rows, unit_transforms=choice.transforms or ("R0",) * count,
            orient=orientations.get(name, "R0"), bank_align=choice.align or "bottom", bank_interconnect=choice.bank_interconnect)
    _reopen(files)
    _require(_source(source_cell)[0] == source_sha, "current source changed during association replay")
    return result


__all__ = ["mim_source_recipes_from_capture"]
