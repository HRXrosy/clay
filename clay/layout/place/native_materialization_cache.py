"""Content-addressed disk cache for Sky130 native geometry and diffusion rows.

Every cache hit verifies a complete file-level SHA-256 manifest before copying
regular files into the current invocation's ``work_dir`` and running the exact
same native toolchain, witness, receipt, GDS/SPICE, and row-program verifiers
as a cold Docker/Magic materialization. Any tampered or stale cache entry is
discarded and falls back cleanly to live producer materialization.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def default_native_cache_root() -> Path:
    override = os.environ.get("CLAY_NATIVE_CACHE_DIR", "").strip()
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[3] / ".clay" / "cache" / "native-primitives"


def _directory_manifest(root: Path) -> dict[str, str]:
    resolved = root.resolve(strict=True)
    manifest: dict[str, str] = {}
    for item in sorted(resolved.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"native cache directory contains a symlink: {item}")
        if item.is_file():
            rel = item.relative_to(resolved).as_posix()
            manifest[rel] = _sha256_file(item)
    if not manifest:
        raise ValueError("native cache directory is empty")
    return manifest


def _verify_directory_manifest(root: Path, expected: dict[str, str]) -> None:
    if type(expected) is not dict or not expected:
        raise ValueError("native cache manifest file table is missing or empty")
    actual = _directory_manifest(root)
    if actual != expected:
        raise ValueError("native cache artifact SHA-256 mismatch")


def _copy_tree_regular(src: Path, dst: Path) -> None:
    src_resolved = src.resolve(strict=True)
    dst_resolved = dst.resolve()
    dst_resolved.mkdir(parents=True, exist_ok=False)
    for item in sorted(src_resolved.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"refusing to copy symlink from native cache: {item}")
        rel = item.relative_to(src_resolved)
        target = dst_resolved / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.read_bytes())


def _atomic_populate_cache_entry(
    category_dir: Path,
    cache_key: str,
    source_dir: Path,
    extra_metadata: dict[str, Any],
) -> None:
    category_dir.mkdir(parents=True, exist_ok=True)
    target_dir = category_dir / cache_key
    if target_dir.exists():
        return
    staging_dir = category_dir / f".staging-{cache_key[:12]}-{uuid.uuid4().hex[:10]}"
    try:
        files_dir = staging_dir / "files"
        _copy_tree_regular(source_dir, files_dir)
        files_manifest = _directory_manifest(files_dir)
        manifest = {
            "schema": "clay.native-primitive-cache-entry/v1",
            "cache_key": cache_key,
            "files": files_manifest,
            **extra_metadata,
        }
        (staging_dir / "cache-manifest.json").write_bytes(
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        try:
            staging_dir.replace(target_dir)
        except OSError:
            if not target_dir.exists():
                raise
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def _rehydrate_cached_geometry_bundle(
    cell_def: Any,
    *,
    output_dir: Path,
    pdk_root: str,
    expected_runtime_files: Any,
    origin_output_dir: str,
    preverified_geometries: Mapping[str, Any] | None = None,
) -> Any:
    from clay.layout import native_sky130 as ns
    from clay.verification.native_sky130 import (
        _runtime_expectations,
        _validate_toolchain,
    )

    output = output_dir.resolve(strict=True)
    source_identity = ns.native_source_identity(cell_def)
    source_specs = ns._cell_native_specs(cell_def)
    specs: dict[str, Any] = {}
    members: dict[str, str] = {}
    for device in cell_def.devices:
        spec = source_specs[device.instance_name]
        key = "pcell_" + ns._digest(spec)[:20]
        specs[key] = spec
        members[device.instance_name] = key

    source_contracts = {
        "models": {spec["model"]: spec["contract"] for spec in specs.values()}
    }
    expected = _runtime_expectations(expected_runtime_files, pdk_root)
    runtime = _validate_toolchain(output, source_contracts, pdk_root, expected)
    copied = (
        ("sky130A.tech", "libs.tech/magic/sky130A.tech"),
        ("sky130A_mr.drc", "libs.tech/klayout/drc/sky130A_mr.drc"),
    )
    if any((output / local).exists() for local, _ in copied):
        inventory = {row["path"]: row for row in runtime["files"]}
        for local, relative in copied:
            target = output / local
            remote = str(Path(pdk_root) / "sky130A" / relative).replace("\\", "/")
            row = inventory.get(remote)
            if (
                not target.is_file()
                or row is None
                or target.stat().st_size != row["bytes"]
                or _sha256_file(target) != row["sha256"]
            ):
                raise ValueError(
                    "cached native PCell copied PDK rule bytes differ from validated toolchain"
                )

    geometry_payload = json.loads((output / "geometry.json").read_text(encoding="utf-8"))
    bundle = object.__new__(ns.NativeGeometryBundle)
    bundle.source_identity = source_identity
    bundle.producer_sha256 = ns._sha(ns.__file__)
    if preverified_geometries is not None:
        if set(preverified_geometries) != set(members):
            raise ValueError("preverified native geometries denominator differs from cell")
        bundle.geometries = {name: preverified_geometries[name] for name in members}
    else:
        observed = {
            key: ns.observe_native_primitive(output / key, spec)
            for key, spec in specs.items()
        }
        bundle.geometries = {name: observed[key] for name, key in members.items()}
    bundle.folders = {name: output / key for name, key in members.items()}
    bundle.geometry_identities = {
        name: ns._geometry_identity(geom) for name, geom in bundle.geometries.items()
    }
    bundle.output_dir = output
    bundle.common_artifacts = {
        p.name: _sha256_file(p) for p in output.iterdir() if p.is_file()
    }
    bundle.dependencies = ns._producer_dependencies()
    bundle._live_session_seal = True
    bundle._cache_origin_output_dir = origin_output_dir
    if geometry_payload.get("geometry_identities") != bundle.geometry_identities:
        raise ValueError("cached native geometry identities differ from observed primitives")
    return bundle.require_current(cell_def)


def materialize_or_reuse_native_geometry(
    provider: Any,
    cell_def: Any,
    *,
    output_dir: Path,
    image: str,
    pdk_root: str = "/foss/pdks",
    expected_runtime_files: Any = None,
    timeout_seconds: int = 600,
    cache_root: Path | None = None,
) -> Any:
    """Reuse verified cached native geometry or run the live Docker producer."""
    from clay.layout import native_sky130 as ns

    output = Path(output_dir).resolve()
    if getattr(provider, "materialize", None) is not ns.materialize_native_sky130:
        return provider.materialize(
            cell_def,
            output_dir=output,
            image=image,
            pdk_root=pdk_root,
            expected_runtime_files=expected_runtime_files,
            timeout_seconds=timeout_seconds,
        )
    root = (cache_root or default_native_cache_root()).resolve()
    category_dir = root / "geometry"

    cache_key: str | None = None
    try:
        cache_key = _digest_json(
            {
                "schema": "clay.native-geometry-cache-key/v1",
                "source_identity": ns.native_source_identity(cell_def),
                "image": str(image),
                "pdk_root": str(pdk_root),
                "expected_runtime_files": expected_runtime_files,
                "producer_sha256": ns._sha(ns.__file__),
                "dependencies": ns._producer_dependencies(),
            }
        )
        entry_dir = category_dir / cache_key
        manifest_path = entry_dir / "cache-manifest.json"
        files_dir = entry_dir / "files"
        if manifest_path.is_file() and files_dir.is_dir():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("cache_key") != cache_key:
                raise ValueError("native geometry cache key mismatch")
            origin_dir = manifest.get("origin_output_dir")
            if type(origin_dir) is not str or not origin_dir:
                raise ValueError("native geometry cache origin path missing")
            _verify_directory_manifest(files_dir, manifest.get("files"))
            if output.exists():
                shutil.rmtree(output)
            _copy_tree_regular(files_dir, output)
            return _rehydrate_cached_geometry_bundle(
                cell_def,
                output_dir=output,
                pdk_root=pdk_root,
                expected_runtime_files=expected_runtime_files,
                origin_output_dir=origin_dir,
            )
    except Exception:
        if output.exists():
            shutil.rmtree(output, ignore_errors=True)

    bundle = provider.materialize(
        cell_def,
        output_dir=output,
        image=image,
        pdk_root=pdk_root,
        expected_runtime_files=expected_runtime_files,
        timeout_seconds=timeout_seconds,
    )
    if cache_key is not None:
        try:
            _atomic_populate_cache_entry(
                category_dir,
                cache_key,
                output,
                {"origin_output_dir": output.as_posix()},
            )
        except Exception:
            pass
    return bundle


def _candidate_key_rows(row_candidates: tuple) -> list[Any]:
    rows = []
    for candidate in row_candidates:
        if is_dataclass(candidate) and not isinstance(candidate, type):
            rows.append(asdict(candidate))
        else:
            rows.append(repr(candidate))
    return rows


def materialize_or_reuse_native_diffusion(
    cell_def: Any,
    intent: Any,
    tech: Any,
    bundle: Any,
    *,
    output_dir: Path,
    source_cir: Path | str | None = None,
    timeout_seconds: int = 60,
    row_candidates: tuple = (),
    cache_root: Path | None = None,
) -> tuple[Any, Any]:
    """Reuse verified cached native diffusion templates or run live generation."""
    from clay.layout import native_diffusion_pcell as ndp
    from clay.layout import native_diffusion_producer as producer_mod
    from clay.layout.native_diffusion_source import _require_canonical_native_arrays
    from clay.layout.route.resource_world import pdk_provenance_digest

    output = Path(output_dir).resolve()
    if (
        getattr(
            producer_mod.materialize_declared_native_diffusion, "__name__", ""
        )
        != "materialize_declared_native_diffusion"
        or getattr(
            producer_mod.materialize_declared_native_diffusion, "__module__", ""
        )
        != "clay.layout.native_diffusion_producer"
    ):
        candidate_kwargs = {"row_candidates": row_candidates} if row_candidates else {}
        return producer_mod.materialize_declared_native_diffusion(
            cell_def,
            intent,
            tech,
            bundle,
            output_dir=output,
            source_cir=source_cir,
            timeout_seconds=timeout_seconds,
            **candidate_kwargs,
        )
    root = (cache_root or default_native_cache_root()).resolve()
    category_dir = root / "diffusion"

    producer_mod._require_canonical_native_arrays(cell_def, intent)
    chains = producer_mod.declared_native_chains(cell_def, intent)
    candidates = producer_mod._checked_candidates(cell_def, chains, row_candidates)
    if not chains:
        return intent, None

    cache_key: str | None = None
    try:
        source_sha, _canonical = _require_canonical_native_arrays(cell_def)
        cir_sha = (
            None
            if source_cir is None
            else _sha256_file(Path(source_cir).resolve(strict=True))
        )
        bundle_geom_sha = _sha256_file(Path(bundle.output_dir) / "geometry.json")
        template_entries = []
        for group, chain in chains:
            candidate = candidates.get(group)
            template_entries.append(
                (
                    group,
                    producer_mod._native_template_key(chain, candidate),
                    producer_mod._native_template_folder(chain, candidate),
                )
            )
        cache_payload: dict[str, Any] = {
            "schema": "clay.native-diffusion-cache-key/v1",
            "source_sha256": source_sha,
            "topology_sha256": cir_sha,
            "pdk_id": tech.pdk_id,
            "pdk_provenance_sha256": pdk_provenance_digest(tech),
            "bundle_geometry_json_sha256": bundle_geom_sha,
            "templates": template_entries,
            "row_candidates": _candidate_key_rows(row_candidates),
            "pcell_producer_sha256": _sha256_file(Path(ndp.__file__).resolve()),
        }
        if any(getattr(chain, "synthesis", "") for _, chain in chains):
            from clay.layout import native_cmos_gate_composer, native_parameterized_row
            from clay.pdk import sky130_native_geometry

            cache_payload["parameterized_producer_sha256"] = _digest_json(
                [
                    _sha256_file(Path(native_parameterized_row.__file__).resolve()),
                    _sha256_file(Path(native_cmos_gate_composer.__file__).resolve()),
                    _sha256_file(Path(sky130_native_geometry.__file__).resolve()),
                ]
            )
        cache_key = _digest_json(cache_payload)
        entry_dir = category_dir / cache_key
        manifest_path = entry_dir / "cache-manifest.json"
        files_dir = entry_dir / "files"
        if manifest_path.is_file() and files_dir.is_dir():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("cache_key") != cache_key:
                raise ValueError("native diffusion cache key mismatch")
            _verify_directory_manifest(files_dir, manifest.get("files"))
            if output.exists():
                shutil.rmtree(output)
            _copy_tree_regular(files_dir, output)
            templates: dict[str, Path] = {}
            for group, chain in chains:
                candidate = candidates.get(group)
                key = producer_mod._native_template_key(chain, candidate)
                if key not in templates:
                    templates[key] = output / producer_mod._native_template_folder(
                        chain, candidate
                    )
            extra = {"row_candidates": row_candidates} if row_candidates else {}
            return producer_mod._materialize_native_diffusion_templates(
                cell_def,
                intent,
                tech,
                bundle,
                chains,
                templates,
                source_cir=source_cir,
                **extra,
            )
    except Exception:
        if output.exists():
            shutil.rmtree(output, ignore_errors=True)

    candidate_kwargs = {"row_candidates": row_candidates} if row_candidates else {}
    projected_intent, composite_row = producer_mod.materialize_declared_native_diffusion(
        cell_def,
        intent,
        tech,
        bundle,
        output_dir=output,
        source_cir=source_cir,
        timeout_seconds=timeout_seconds,
        **candidate_kwargs,
    )
    if cache_key is not None and composite_row is not None:
        try:
            _atomic_populate_cache_entry(
                category_dir,
                cache_key,
                output,
                {"origin_output_dir": output.as_posix()},
            )
        except Exception:
            pass
    return projected_intent, composite_row


__all__ = [
    "default_native_cache_root",
    "materialize_or_reuse_native_diffusion",
    "materialize_or_reuse_native_geometry",
]
