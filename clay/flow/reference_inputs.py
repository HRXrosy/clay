"""Normalize reference input identity; physical evidence remains a separate API.

Portable packages bind bytes without claiming conversion provenance or physical
PASS. Repository manifests retain their existing, stricter audit contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

from clay.flow.circuit_definition import DesignInputError, InputBinding, artifact_path, file_sha256


@dataclass(frozen=True)
class ReferenceInputs:
    pdk_id: str
    topology_id: str
    topology: InputBinding
    geometry: InputBinding
    bindings: tuple[InputBinding, ...]
    observation_topologies: tuple[InputBinding, ...]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DesignInputError("REFERENCE_INPUT_INVALID", f"duplicate reference field: {key}")
        result[key] = value
    return result


def _portable_binding(root, value):
    if type(value) is not dict or set(value) != {"artifact", "sha256"}:
        raise DesignInputError("REFERENCE_INPUT_INVALID", "reference input requires artifact and sha256")
    expected = value["sha256"]
    if type(expected) is not str or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise DesignInputError("REFERENCE_INPUT_INVALID", "reference hash must be lowercase SHA-256")
    path = artifact_path(root, value["artifact"])
    if file_sha256(path) != expected:
        raise DesignInputError("REFERENCE_INPUT_CHANGED", f"reference bytes differ: {value['artifact']}")
    return InputBinding(value["artifact"], expected)


def load_reference_inputs(prepared):
    root = prepared.project_root
    ref = prepared.definition.reference
    path = artifact_path(root, ref.manifest_artifact)
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if type(data) is not dict:
        raise DesignInputError("REFERENCE_INPUT_INVALID", "reference manifest must be an object")
    if data.get("schema") == "clay.reference-inputs/v1":
        required = {"schema", "pdk_id", "topology_id", "topology", "geometry"}
        if not required <= set(data) or set(data) - required - {"auxiliary"}:
            raise DesignInputError("REFERENCE_INPUT_INVALID", "unsupported or missing portable reference fields")
        for key in ("pdk_id", "topology_id"):
            if type(data[key]) is not str or not data[key] or data[key].strip() != data[key]:
                raise DesignInputError("REFERENCE_INPUT_INVALID", f"reference {key} must be a non-empty string")
        auxiliary = data.get("auxiliary", [])
        if type(auxiliary) is not list:
            raise DesignInputError("REFERENCE_INPUT_INVALID", "reference auxiliary must be a list")
        topology = _portable_binding(root, data["topology"])
        geometry = _portable_binding(root, data["geometry"])
        bindings = (topology, geometry, *(_portable_binding(root, row) for row in auxiliary))
        if len({b.artifact for b in bindings}) != len(bindings):
            raise DesignInputError("REFERENCE_ROLE_CONFLICT", "reference input roles overlap")
        pdk_id, topology_id = data["pdk_id"], data["topology_id"]
        observation_topologies = (topology,)
    elif data.get("schema") == "clay.reference-manifest/v1":
        from clay.flow.reference_evidence import audit_reference_manifest
        manifest = audit_reference_manifest(path, repo_root=root).manifest
        candidates = [a for a in manifest.reference_artifacts if a.repository_path == ref.geometry_artifact
                      and a.role in ("layout_gds", "oa_spatial_anchor_gds")]
        if len(candidates) != 1:
            raise DesignInputError("REFERENCE_GEOMETRY_UNBOUND", "primary geometry must be one manifest spatial artifact")
        canonical = next(a for a in manifest.reference_artifacts if a.role == "canonical_netlist")
        topology = InputBinding(canonical.repository_path, canonical.sha256)
        geometry = InputBinding(candidates[0].repository_path, candidates[0].sha256)
        artifacts = (*manifest.source_artifacts, *manifest.reference_artifacts, manifest.conversion_receipt)
        if manifest.reference_characterization is not None:
            artifacts += (manifest.reference_characterization,)
        bindings = tuple(InputBinding(a.repository_path, a.sha256) for a in artifacts
                         if a.availability == "required_evidence")
        pdk_id, topology_id = manifest.reference_pdk_id, manifest.topology_id
        # The audited converter proves raw geometry-netlist -> canonical
        # connectivity (including replica collapse and terminal permutations).
        # An arbitrary manifest dependency is not such a proof.
        from clay.flow.reference_evidence import load_topology_conversion_receipt
        conversion = load_topology_conversion_receipt(artifact_path(root, manifest.conversion_receipt.repository_path), repo_root=root)
        conversion_sources = {(a.repository_path, a.sha256) for a in conversion.source_artifacts}
        observation_topologies = (topology, *(InputBinding(a.repository_path, a.sha256)
            for a in manifest.source_artifacts if a.role == "geometry_netlist"
            and (a.repository_path, a.sha256) in conversion_sources))
    else:
        raise DesignInputError("UNSUPPORTED_REFERENCE_SCHEMA", "unsupported reference manifest schema")
    if geometry.artifact != ref.geometry_artifact:
        raise DesignInputError("REFERENCE_GEOMETRY_UNBOUND", "selected primary geometry differs from the reference manifest")
    declared = {b.artifact for b in bindings}
    if not set(ref.auxiliary_artifacts) <= declared:
        raise DesignInputError("REFERENCE_AUXILIARY_UNBOUND", "auxiliary inputs must also be bound in the reference manifest")
    result = ReferenceInputs(pdk_id, topology_id, topology, geometry, bindings, observation_topologies)
    require_reference_unchanged(prepared, bindings)
    return result


def require_reference_unchanged(prepared, bindings):
    for binding in bindings:
        try:
            if file_sha256(artifact_path(prepared.project_root, binding.artifact)) != binding.sha256:
                raise DesignInputError("INPUT_CHANGED", f"reference input changed during check: {binding.artifact}", location={"artifact": binding.artifact})
        except OSError as exc:
            raise DesignInputError("INPUT_UNREADABLE", "bound reference input can no longer be read", location={"artifact": binding.artifact}) from exc
