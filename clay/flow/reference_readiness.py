"""Reference identity and formal receipt replay for common preflight."""
from __future__ import annotations

import json
from pathlib import Path

from clay.flow.circuit_definition import DesignInputError, InputBinding, PreparedDesign, artifact_path, file_sha256
from clay.flow.reference_inputs import load_reference_inputs, require_reference_unchanged
from clay.flow.readiness_result import ReadinessCheck
from clay.flow.reference_topology_projection import verify_primitive_reference_topology


def _failure(name, exc, *, fallback="REFERENCE_INVALID", artifact=None):
    code = getattr(exc, "code", fallback)
    status = "unsupported" if code.endswith("UNAVAILABLE") or code.endswith("UNSUPPORTED") else "fail"
    return ReadinessCheck(name, status, code, str(exc), ({"artifact": artifact},) if artifact else ())


def pdk_snapshot(pdk_id):
    from clay.pdk import get_pdk
    try:
        pdk = get_pdk(pdk_id)
        snapshot = pdk.source_snapshot(refresh=True)
    except Exception as exc:
        raise DesignInputError("PDK_UNAVAILABLE", f"cannot resolve PDK {pdk_id}: {exc}") from exc
    if snapshot is None or snapshot.installed_authority_sha256 is None:
        raise DesignInputError("PDK_AUTHORITY_UNAVAILABLE", f"installed PDK authority unavailable: {pdk_id}")
    return snapshot


def _receipt_check(prepared, *, kind, path, reference_identity, observation=None):
    """Join the selected authority and reject drift before formal replay."""
    root = prepared.project_root
    data = json.loads(artifact_path(root, path).read_text(encoding="utf-8"))
    if kind == "observation":
        from clay.frontend.reference_observation import reference_observation_receipt_from_data, replay_reference_observation_receipt
        receipt = reference_observation_receipt_from_data(data)
        replay = replay_reference_observation_receipt
        changed_code = "REFERENCE_OBSERVATION_IMPLEMENTATION_CHANGED"
        pdk_id = receipt.pdk_id
        ref = prepared.definition.reference
        allowed = (ref.geometry_artifact,) if ref.observation_role == "primary" else ref.auxiliary_artifacts
        if receipt.gds_repository_path not in allowed:
            raise DesignInputError("REFERENCE_OBSERVATION_ROLE_MISMATCH", "receipt geometry is not bound in its declared reference role")
        topologies = {(b["artifact"], b["sha256"]) for b in reference_identity.get("reference_observation_topologies", ())}
        reference_hashes = {b["artifact"]: b["sha256"] for b in reference_identity.get("reference_inputs", ())}
        if ((receipt.topology_repository_path, receipt.topology_sha256) not in topologies
                or receipt.pdk_id != reference_identity.get("reference_pdk_id")
                or receipt.gds_sha256 != reference_hashes.get(receipt.gds_repository_path)):
            raise DesignInputError("REFERENCE_OBSERVATION_IDENTITY_MISMATCH", "observation does not bind the selected reference topology, geometry and PDK")
        receipt_bindings = (
            InputBinding(receipt.gds_repository_path, receipt.gds_sha256),
            InputBinding(receipt.topology_repository_path, receipt.topology_sha256),
            InputBinding(receipt.source_repository_path, receipt.source_sha256),
        )
    else:
        from clay.frontend.reference_macro_grammar import reference_macro_grammar_receipt_from_data, replay_reference_macro_grammar_receipt
        receipt = reference_macro_grammar_receipt_from_data(data)
        replay = replay_reference_macro_grammar_receipt
        changed_code = "REFERENCE_MACRO_IMPLEMENTATION_CHANGED"
        if receipt.observation_receipt_repository_path != prepared.definition.reference.observation_artifact:
            raise DesignInputError("REFERENCE_MACRO_OBSERVATION_MISMATCH", "macro grammar binds a different observation")
        pdk_id = receipt.pdk_id
        if (pdk_id != reference_identity.get("reference_pdk_id")
                or not observation or pdk_id != observation.get("pdk_id")
                or receipt.placement_source_repository_path != prepared.definition.source_artifact
                or receipt.placement_source_sha256 != prepared.source_sha256
                or receipt.top_cell != prepared.definition.top_cell):
            raise DesignInputError("REFERENCE_MACRO_IDENTITY_MISMATCH", "macro grammar does not bind the selected source, top and reference PDK")
        reference_hashes = {b["artifact"]: b["sha256"] for b in reference_identity.get("reference_inputs", ())}
        if reference_hashes.get(receipt.def_repository_path) != receipt.def_sha256:
            raise DesignInputError("REFERENCE_MACRO_DEF_UNBOUND", "macro DEF must bind one declared reference input")
        receipt_bindings = (
            InputBinding(receipt.observation_receipt_repository_path, receipt.observation_receipt_file_sha256),
            InputBinding(receipt.def_repository_path, receipt.def_sha256),
            InputBinding(receipt.placement_source_repository_path, receipt.placement_source_sha256),
        )
    require_reference_unchanged(prepared, receipt_bindings)
    reference_identity.setdefault("receipt_inputs", []).extend(
        {"artifact": b.artifact, "sha256": b.sha256, "domain": "file_bytes"} for b in receipt_bindings)
    # Implementation belongs to the installed package, not a user project that
    # could supply replacement clay/*.py files. Formal replay is still mandatory.
    implementation_root = Path(__file__).resolve().parents[2]
    changed = [p for p, sha in receipt.implementation_sha256.items()
               if file_sha256(artifact_path(implementation_root, p)) != sha]
    if changed:
        raise DesignInputError(changed_code, f"receipt implementation changed: {', '.join(changed)}")
    snapshot = pdk_snapshot(pdk_id)
    options = {"implementation_root": implementation_root} if kind == "macro" else {}
    replay(root, receipt, pdk_source_snapshot_sha256=snapshot.snapshot_sha256,
           pdk_installed_authority_sha256=snapshot.installed_authority_sha256,
           pdk_layermap=snapshot.layermap(), **options)
    return ReadinessCheck(f"reference.{kind}", "pass", "REFERENCE_REPLAY_VERIFIED",
                          f"formal {kind} replay passed; role remains explicit", detail={"receipt_sha256": receipt.receipt_sha256})


def check_reference(prepared: PreparedDesign, cell, *, replay: bool):
    from clay.flow.reference_evidence import canonical_topology_from_spice
    ref = prepared.definition.reference
    root = prepared.project_root
    checks, identity = [], {}
    if ref is None:
        if prepared.definition.mode != "generate":
            checks.append(ReadinessCheck("reference", "fail", "REFERENCE_REQUIRED", "selected mode requires an explicit reference"))
        return checks, identity
    if ref.manifest_artifact is None:
        return [ReadinessCheck("reference", "unsupported", "REFERENCE_MANIFEST_REQUIRED", "a versioned reference manifest is required")], identity
    try:
        inputs = load_reference_inputs(prepared)
        identity.update({"reference_pdk_id": inputs.pdk_id, "reference_topology_id": inputs.topology_id,
                         "reference_observation_topologies": [{"artifact": b.artifact, "sha256": b.sha256, "domain": "file_bytes"} for b in inputs.observation_topologies],
                         "reference_inputs": [{"artifact": b.artifact, "sha256": b.sha256, "domain": "file_bytes"} for b in inputs.bindings],
                         "reference_topology_file": {"domain": "file_bytes", "sha256": inputs.topology.sha256,
                                                     "artifact": inputs.topology.artifact},
                         "reference_geometry_file": {"domain": "file_bytes", "sha256": inputs.geometry.sha256,
                                                     "artifact": ref.geometry_artifact}})
        canonical = canonical_topology_from_spice(artifact_path(root, inputs.topology.artifact).read_text(encoding="utf-8"))
        checks.append(ReadinessCheck("reference.identity", "pass", "REFERENCE_INPUTS_VERIFIED", "manifest and selected geometry roles verified"))
        digest = verify_primitive_reference_topology(cell, canonical)
        identity["reference_connectivity"] = {"domain": "canonical_named_connectivity/v1", "sha256": digest}
        checks.append(ReadinessCheck("reference.topology", "pass", "REFERENCE_TOPOLOGY_VERIFIED", "named primitive connectivity agrees; sizing and geometry are separate"))
    except Exception as exc:
        checks.append(_failure("reference.inputs", exc, artifact=ref.manifest_artifact))
    if replay:
        observation = None
        for kind, path in (("observation", ref.observation_artifact), ("macro", ref.macro_grammar_artifact)):
            if path is None:
                if kind == "observation" and ref.representation == "hierarchical":
                    checks.append(ReadinessCheck("reference.observation", "unsupported", "REFERENCE_OBSERVATION_UNAVAILABLE", "hierarchical observation receipt is required"))
                continue
            try:
                if kind == "observation":
                    observation = json.loads(artifact_path(root, path).read_text(encoding="utf-8"))
                checks.append(_receipt_check(prepared, kind=kind, path=path, reference_identity=identity, observation=observation))
            except Exception as exc:
                checks.append(_failure(f"reference.{kind}", exc, artifact=path))
        if ref.representation == "flat":
            checks.append(ReadinessCheck("reference.instance_binding", "unsupported", "FLAT_INSTANCE_BINDING_UNAVAILABLE",
                                         "flat geometry needs a logical-instance binding producer; auxiliary hierarchy cannot supply its verdict"))
        elif ref.observation_role != "primary":
            checks.append(ReadinessCheck("reference.observation.primary", "unsupported", "PRIMARY_REFERENCE_OBSERVATION_UNAVAILABLE",
                                         "auxiliary hierarchy does not observe primary reference geometry"))
    return checks, identity


def require_reference_inputs(prepared: PreparedDesign):
    """Validate reference input identity for a build spec, without admission."""
    from clay.parser import parse_clay_source
    parsed = parse_clay_source(prepared.source_bytes.decode("utf-8-sig"))
    cell = next(c for c in parsed.cell_defs if c.local_name == prepared.definition.top_cell)
    checks, identity = check_reference(prepared, cell, replay=False)
    failed = [check for check in checks if check.status != "pass"]
    if failed:
        raise DesignInputError(failed[0].code, failed[0].message)
    if "reference_geometry_file" not in identity:
        raise DesignInputError("REFERENCE_REQUIRED", "physical reference build spec requires bound geometry")
    prepared.require_unchanged()
    require_reference_identity_unchanged(prepared, identity)
    return identity


def require_reference_identity_unchanged(prepared, identity):
    require_reference_unchanged(prepared, tuple(InputBinding(b["artifact"], b["sha256"])
                                              for key in ("reference_inputs", "receipt_inputs")
                                              for b in identity.get(key, ())))
