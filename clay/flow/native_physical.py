"""Adapt the existing complete native verification to canonical flow steps.

The tool runner owns DRC/LVS/RC extraction and source-net proof. This boundary
only binds that evidence to the current candidate and publishes step artifacts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid

from lark.exceptions import LarkError

from clay.flow.spec import StepResult
from clay.frontend.native_spice_import import emit_native_spice
from clay.parser import parse_clay_file, parse_clay_source


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_cell(spec, ctx):
    path = Path(ctx.extra.get("effective_design_clay") or spec.case_dir / "source/design.clay")
    compiled = ctx.extra.get("cell_def")
    native_required = (
        getattr(spec, "native_runtime_manifest", None) is not None
        or ctx.extra.get("native_design_binding") is not None
        or any(device.device_class == "native" for device in getattr(compiled, "devices", ()))
    )
    if not path.is_file():
        if native_required:
            raise ValueError("native physical flow is missing its effective Clay source")
        return None
    bound_source = ctx.extra.get('_primitive_stage_source_bytes')
    if bound_source is None:
        bound_source = path.read_bytes()
        parsed = parse_clay_file(path, _source_bytes=bound_source)
        source_digest = hashlib.sha256(bound_source).hexdigest()
    else:
        from clay.flow.run_admission import current_admission
        if not getattr(current_admission(), 'is_primitive_stage', False):
            raise ValueError('bound primitive source bytes require stage admission')
        if type(bound_source) is not bytes:
            raise ValueError('primitive stage source must be bound bytes')
        parsed = parse_clay_source(bound_source.decode('utf-8-sig'))
        source_digest = hashlib.sha256(bound_source).hexdigest()
    cells = [item for item in parsed.cell_defs if item.local_name == spec.top_cell]
    if len(cells) != 1:
        raise ValueError("native physical flow requires exactly one source cell")
    cell = cells[0]
    if not any(device.device_class == "native" for device in cell.devices):
        if native_required:
            raise ValueError("native physical flow source no longer contains native devices")
        return None
    if not all(device.device_class == "native" for device in cell.devices):
        raise ValueError("native physical flow requires a complete native cell")
    if parsed.pdk_id != spec.pdk:
        raise ValueError("native physical source PDK differs from the case")
    if compiled is not None:
        from clay.flow.candidate_bindings import native_semantic_envelope

        compiled_file = getattr(compiled, "_clay_file", None)
        before = compiled_file if compiled_file is not None else compiled
        after = parsed if compiled_file is not None else cell
        if native_semantic_envelope(before) != native_semantic_envelope(after):
            raise ValueError("compiled native cell differs from the current effective Clay source")
    binding = {"path": str(path.resolve()), "sha256": source_digest}
    previous = ctx.extra.get("native_design_binding")
    if previous is not None and binding != previous:
        raise ValueError("native effective Clay changed after physical materialization")
    ctx.extra["native_design_binding"] = binding
    return cell


def native_runtime(spec):
    """Read the exact runtime DATA bound when the source case was compiled."""
    path = spec.native_runtime_manifest
    if path is None or not spec.native_runtime_sha256:
        raise ValueError("native physical flow requires a bound native_runtime_manifest")
    runtime_bytes = Path(path).read_bytes()
    if hashlib.sha256(runtime_bytes).hexdigest() != spec.native_runtime_sha256:
        raise ValueError("native runtime manifest changed after case compilation")
    data = json.loads(runtime_bytes.decode('utf-8'))
    if data.get("schema") != "clay.native-physical-runtime/v1":
        raise ValueError("unsupported native physical runtime schema")
    required = {"image", "pdk_root", "files", "timeout_seconds", "max_workers"}
    if not required <= data.keys():
        raise ValueError("native physical runtime lacks explicit tool settings")
    if not isinstance(data["files"], dict) or not data["files"]:
        raise ValueError("native physical runtime lacks its actual file inventory")
    return data


def native_input_closure(spec, ctx, *paths):
    """Expose the source, runtime and native producers in canonical step inputs."""
    result = [Path(path) for path in paths]
    if getattr(spec, "native_runtime_manifest", None) is not None:
        result.extend((Path(ctx.extra.get("effective_design_clay") or spec.case_dir / "source/design.clay"),
                       Path(spec.native_runtime_manifest), Path(__file__)))
        from clay.verification.native_sky130 import _runner_sources

        result.extend(Path(row["path"]) for row in _runner_sources().values())
    return list(dict.fromkeys(result))


def ensure_native_verification(spec, ctx, *, gds, reference, cell, stage):
    from clay.verification.native_sky130 import run_native_verification, validate_native_evidence

    if spec.pdk != "sky130":
        raise ValueError(f"no complete native verification provider for {spec.pdk!r}")
    runtime = native_runtime(spec)
    source_binding = dict(ctx.extra["native_design_binding"])
    reference = Path(reference).resolve()
    gds = Path(gds).resolve()
    expected = emit_native_spice(cell, pdk_id=spec.pdk, top_name=spec.gds_cell_name)
    if reference.read_text(encoding="utf-8").strip() != expected.strip():
        raise ValueError("native physical reference differs from the current candidate Clay")
    result = ctx.extra.get("native_verification")
    if result is None:
        output = Path(ctx.work_dir) / ("native-" + uuid.uuid4().hex[:12])
        result = run_native_verification(
            gds, reference, spec.gds_cell_name, output,
            image=runtime["image"], pdk_root=runtime["pdk_root"],
            expected_runtime_files=runtime["files"],
            timeout_seconds=runtime["timeout_seconds"], max_workers=runtime["max_workers"],
        )
        ctx.extra["native_verification"] = result
        ctx.extra["native_verification_design"] = source_binding
    if ctx.extra.get("native_verification_design") != source_binding:
        raise ValueError("native verification is bound to a different effective Clay source")
    result = validate_native_evidence(result, required_stages=(stage,))
    for name, path in (("gds", gds), ("reference_spice", reference)):
        row = result["inputs"][name]
        if Path(row["path"]).resolve() != path or row["sha256"] != digest(path):
            raise ValueError(f"native {name} changed between canonical flow steps")
    if (result["image"] != runtime["image"] or result["pdk_root"] != runtime["pdk_root"]
            or result["expected_runtime_files"] != runtime["files"]):
        raise ValueError("native verification runtime differs from the source-bound runtime")
    native_cell(spec, ctx)
    native_runtime(spec)
    return result


def native_verification_step(stage, spec, ctx, *, gds, reference):
    """Return None for portable sources; native callers never fall back."""
    try:
        cell = native_cell(spec, ctx)
        if cell is None:
            return None
        result = ensure_native_verification(spec, ctx, gds=gds, reference=reference, cell=cell, stage=stage)
        output = Path(result["output"])
        artifacts = {"native_summary": str(output / "summary.json"),
                     "native_manifest": str(output / "manifest.json")}
        stable = bool(result["inputs_unchanged"])
        if stage == "drc":
            passed = stable and bool(result["drc"]["passed"])
            reports = result["drc"].get("reports", {})
            counts = {key: int(row["item_count"]) for key, row in reports.items()}
            return StepResult(stage, "pass" if passed else "fail",
                              "Complete native Magic and KLayout DRC",
                              data={"is_clean": passed, "n_violations": sum(counts.values()),
                                    "by_deck": counts, "errors": result["errors"]},
                              artifacts=artifacts, blocking=not passed)
        if stage == "lvs":
            passed = stable and bool(result["lvs"]["passed"])
            artifacts["lvs_reference"] = str(reference)
            if result["lvs"].get("report"):
                artifacts["lvs_report"] = result["lvs"]["report"]["path"]
            return StepResult(stage, "pass" if passed else "fail", "Complete native LVS",
                              data={**result["lvs"], "errors": result["errors"]},
                              artifacts=artifacts, blocking=not passed)
        if stage != "pex":
            raise ValueError(f"unknown native verification step {stage!r}")
        from clay.verification.native_sky130 import source_bound_pex_report, validate_native_evidence

        report = source_bound_pex_report(result)
        sealed_identity = {"path": str(output.resolve()),
                           "summary_sha256": digest(output / "summary.json"),
                           "manifest_sha256": digest(output / "manifest.json")}
        raw = Path(result["pex"]["native_spice"]["path"])
        expected_sha = result["pex"]["native_spice"]["sha256"]
        payload = raw.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise ValueError("native PEX changed after evidence replay")
        destination = Path(ctx.work_dir) / f"{spec.top_cell}.pex.spice"
        destination.write_bytes(payload)
        if digest(destination) != expected_sha or digest(raw) != expected_sha:
            raise ValueError("native PEX changed during canonical artifact publication")
        native_cell(spec, ctx)
        native_runtime(spec)
        sidecar = Path(str(destination) + ".native-verification.json")
        proof = {
            "schema": "clay.native-pex-reference/v1",
            "native_output": sealed_identity,
            "candidate_source": {"path": str(Path(reference).resolve()),
                                 "sha256": result["inputs"]["reference_spice"]["sha256"]},
            "pex": {"path": str(destination.resolve()), "sha256": expected_sha},
        }
        sidecar_payload = (json.dumps(proof, indent=2) + "\n").encode("utf-8")
        sidecar.write_bytes(sidecar_payload)
        # Publication can race source, report, runtime or producer changes.
        # Replay the original seal rather than hashing changed bytes into a new proof.
        validate_native_evidence(result, required_stages=())
        if (digest(output / "summary.json") != sealed_identity["summary_sha256"]
                or digest(output / "manifest.json") != sealed_identity["manifest_sha256"]
                or digest(destination) != expected_sha or digest(raw) != expected_sha
                or digest(sidecar) != hashlib.sha256(sidecar_payload).hexdigest()):
            raise ValueError("native PEX evidence changed during canonical artifact publication")
        native_cell(spec, ctx)
        native_runtime(spec)
        ctx.extra["pex_report"] = report
        artifacts.update(pex_spice=str(destination), raw_pex_spice=str(raw), native_reference=str(sidecar))
        return StepResult(stage, "pass", "Complete unchanged native RC with replayed source binding",
                          data={"backend": "magic", "calibrated": True,
                                "n_caps": result["pex"]["capacitor_count"],
                                "n_resistors": result["pex"]["resistor_count"],
                                "total_cap_fF": report.total_cap_fF}, artifacts=artifacts)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, LarkError) as exc:
        return StepResult(stage, "fail", f"native {stage} failed: {type(exc).__name__}: {exc}",
                          data={"error": str(exc), "error_type": type(exc).__name__}, blocking=True)
