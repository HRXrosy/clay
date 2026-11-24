"""Authenticated export and load of a prepared compact scene for fast routing.

``export_prepared_scene`` serializes a verified ``DeclaredCompactOverride`` and
its exact logical ``Placement`` / row-projection / native-primitive basis into a
directory sealed by SHA-256 digests over:
  - the live ``.clay`` source bytes and canonical source inventory,
  - the live ``.cir`` topology bytes (when present),
  - the live ``LayoutTech`` PDK id, provenance digest, and installed snapshot,
  - the serialized payload bytes,
  - the ``override`` and ``ring_inclusive`` placement/geometry artwork,
  - the ``FinalPlacementCertificate`` state digest, and
  - any ``RowProgram`` / ``RowProgramMaterialization`` / ``RowSourceProjection``
    and Sky130 native geometry / diffusion template artifacts.

``load_prepared_scene`` verifies every manifest and live-input digest BEFORE
deserializing with a restricted ``clay.*``-only unpickler, re-runs full compiler
and row-materialization replay, and registers the authenticated
``RowSourceProjection`` / ``DeclaredCompactOverride`` so standalone deterministic
routing replay via ``route_with_declared_compaction(..., prepared_compact=...)``
completes in seconds while rejecting any tampered or stale scene fail-closed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import io
import json
from pathlib import Path
import pickle
import shutil
from typing import Any, Dict, Mapping, Optional
import uuid

from clay.layout.flow_placement import (
    DeclaredCompactOverride,
    _require_prepared_declared_compact,
    register_authenticated_prepared_compact,
)

PREPARED_SCENE_SCHEMA = "clay.prepared-scene/v1"

_SEAL_MODULES: tuple[str, ...] = (
    "clay.layout.bjt_local_assembly",
    "clay.layout.layered_placement_assembly",
    "clay.layout.nonmos_local_assembly",
    "clay.layout.nonmos_placement_geometry",
    "clay.layout.passive_terminal_binding",
    "clay.layout.source_mim_materialization",
    "clay.layout.source_resistor_pcell_assembly",
    "clay.layout.well_component_access",
)

_ALLOWED_NON_CLAY_MODULES: frozenset[str] = frozenset(
    {
        "builtins",
        "collections",
        "copyreg",
        "_codecs",
        "decimal",
        "fractions",
        "pathlib",
        "types",
    }
)

_DISALLOWED_BUILTINS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "__import__",
        "breakpoint",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
    }
)


class PreparedSceneAuthenticationError(ValueError):
    """Raised when a serialized PreparedScene fails cryptographic or replay checks."""


def _require_auth(condition: bool, message: str) -> None:
    if not condition:
        raise PreparedSceneAuthenticationError(message)


@dataclass(frozen=True)
class PreparedScene:
    """Authenticated pre-route scene ready for ``route_with_declared_compaction``."""

    prepared_compact: DeclaredCompactOverride
    placement: Any
    intent: Any
    effective_policy: Any
    device_records: Optional[Dict[str, Dict[str, Any]]]
    intent_fact_set: Any = None
    source_inventory: Any = None
    native_geometry: Any = None
    native_composite_row: Any = None
    manifest: Optional[Dict[str, Any]] = None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _manifest_seal(manifest_without_seal: Mapping[str, Any]) -> str:
    body = {k: v for k, v in manifest_without_seal.items() if k != "manifest_seal_sha256"}
    return _digest_json(body)


def _canonical_artwork_class_tag(kind: type) -> str:
    mod = kind.__module__
    if mod.startswith("clay.layout.route.rc_"):
        from clay.layout.route import route_cell as _rc_facade

        if getattr(_rc_facade, kind.__qualname__, None) is kind:
            mod = "clay.layout.route.route_cell"
    return f"{mod}.{kind.__qualname__}"


def _artwork_value_digest(value: Any) -> str:
    """Canonical Merkle SHA-256 digest without pointer ``id(...)`` values or DAG blowup."""
    from dataclasses import fields, is_dataclass
    from decimal import Decimal
    from enum import Enum
    from fractions import Fraction
    import math
    import struct

    memo: dict[int, tuple[Any, bytes]] = {}
    pack_u32 = struct.Struct(">I").pack
    pack_u64 = struct.Struct(">Q").pack
    class_tag_memo: dict[type, str] = {}

    def class_tag_of(kind: type) -> str:
        tag = class_tag_memo.get(kind)
        if tag is None:
            tag = _canonical_artwork_class_tag(kind)
            class_tag_memo[kind] = tag
        return tag

    def convert(item: Any, depth: int = 0) -> bytes:
        _require_auth(depth <= 64, "prepared scene artwork exceeds bounded depth")
        kind = type(item)
        if item is None:
            return b"N"
        if kind is bool:
            return b"B1" if item else b"B0"
        if kind is int:
            return b"I" + str(item).encode("ascii")
        if kind is str:
            raw = item.encode("utf-8")
            return b"S" + pack_u32(len(raw)) + raw
        if kind is float:
            _require_auth(math.isfinite(item), "non-finite float in prepared scene artwork")
            return b"F" + item.hex().encode("ascii")
        if kind is bytes:
            return b"Y" + hashlib.sha256(item).digest()
        if kind is Decimal:
            _require_auth(item.is_finite(), "non-finite Decimal in prepared scene artwork")
            return b"D" + str(item).encode("ascii")
        if kind is Fraction:
            return f"R{item.numerator}/{item.denominator}".encode("ascii")
        # Fast-path 2D numeric coordinate tuples/lists before ABC isinstance checks.
        if (kind is tuple or kind is list) and len(item) == 2:
            a, b = item[0], item[1]
            ta, tb = type(a), type(b)
            if ta is int and tb is int:
                _require_auth(depth + 1 <= 64, "prepared scene artwork exceeds bounded depth")
                prefix = b"t2i:" if kind is tuple else b"l2i:"
                return prefix + f"{a},{b}".encode("ascii")
            if ta is float and tb is float:
                _require_auth(depth + 1 <= 64, "prepared scene artwork exceeds bounded depth")
                _require_auth(
                    math.isfinite(a) and math.isfinite(b),
                    "non-finite float in prepared scene artwork",
                )
                prefix = b"t2f:" if kind is tuple else b"l2f:"
                return prefix + f"{a.hex()},{b.hex()}".encode("ascii")
        if isinstance(item, Path):
            raw = item.name.encode("utf-8")
            return b"P" + pack_u32(len(raw)) + raw
        if isinstance(item, Enum):
            val_tok = convert(item.value, depth + 1)
            cls_tag = class_tag_of(kind).encode("utf-8")
            return b"E" + pack_u32(len(cls_tag)) + cls_tag + pack_u32(len(val_tok)) + val_tok

        obj_id = id(item)
        cached = memo.get(obj_id)
        if cached is not None:
            return cached[1]

        if kind in (tuple, list):
            h = hashlib.sha256()
            h.update(b"tuple:" if kind is tuple else b"list:")
            h.update(pack_u64(len(item)))
            for part in item:
                tok = convert(part, depth + 1)
                h.update(pack_u32(len(tok)))
                h.update(tok)
            res = b"H" + h.digest()
            memo[obj_id] = (item, res)
            return res
        if kind is dict:
            pairs = sorted(
                (convert(k, depth + 1), convert(v, depth + 1)) for k, v in item.items()
            )
            h = hashlib.sha256()
            h.update(b"dict:")
            h.update(pack_u64(len(pairs)))
            for k_tok, v_tok in pairs:
                h.update(pack_u32(len(k_tok)))
                h.update(k_tok)
                h.update(pack_u32(len(v_tok)))
                h.update(v_tok)
            res = b"H" + h.digest()
            memo[obj_id] = (item, res)
            return res
        if kind in (set, frozenset):
            elems = sorted(convert(v, depth + 1) for v in item)
            h = hashlib.sha256()
            h.update(b"set:" if kind is set else b"frozenset:")
            h.update(pack_u64(len(elems)))
            for tok in elems:
                h.update(pack_u32(len(tok)))
                h.update(tok)
            res = b"H" + h.digest()
            memo[obj_id] = (item, res)
            return res
        if is_dataclass(item) and not isinstance(item, type):
            names = [f.name for f in fields(kind)]
            extra = sorted(set(getattr(item, "__dict__", ())) - set(names))
            h = hashlib.sha256()
            h.update(f"dataclass:{class_tag_of(kind)}:".encode("utf-8"))
            for name in (*names, *extra):
                if name in ("_seal", "tech"):
                    continue
                name_b = name.encode("utf-8")
                tok = convert(getattr(item, name), depth + 1)
                h.update(pack_u32(len(name_b)))
                h.update(name_b)
                h.update(pack_u32(len(tok)))
                h.update(tok)
            res = b"H" + h.digest()
            memo[obj_id] = (item, res)
            return res
        # Slotted stamp classes (_PlacedStamp, _PlacedBjtStamp, _PlacedPassiveStamp)
        slots = [
            slot
            for cls in kind.__mro__
            for slot in cls.__dict__.get("__slots__", ())
            if slot != "__weakref__" and hasattr(item, slot)
        ]
        if slots:
            h = hashlib.sha256()
            h.update(f"slotted:{class_tag_of(kind)}:".encode("utf-8"))
            for slot in sorted(set(slots)):
                if slot in ("_seal", "tech"):
                    continue
                slot_b = slot.encode("utf-8")
                tok = convert(getattr(item, slot), depth + 1)
                h.update(pack_u32(len(slot_b)))
                h.update(slot_b)
                h.update(pack_u32(len(tok)))
                h.update(tok)
            res = b"H" + h.digest()
            memo[obj_id] = (item, res)
            return res
        if hasattr(item, "__dict__"):
            h = hashlib.sha256()
            h.update(f"object:{class_tag_of(kind)}:".encode("utf-8"))
            for k, v in sorted(vars(item).items()):
                if k in ("_seal", "tech"):
                    continue
                k_b = k.encode("utf-8")
                tok = convert(v, depth + 1)
                h.update(pack_u32(len(k_b)))
                h.update(k_b)
                h.update(pack_u32(len(tok)))
                h.update(tok)
            res = b"H" + h.digest()
            memo[obj_id] = (item, res)
            return res
        raise PreparedSceneAuthenticationError(
            f"unsupported artwork value type in prepared scene: {kind.__module__}.{kind.__qualname__}"
        )

    root_tok = convert(value)
    return hashlib.sha256(root_tok).hexdigest()


def _artwork_scene_digest(scene_pair: tuple[Dict[str, Any], Dict[str, Any]]) -> str:
    positions, geometry = scene_pair
    return _artwork_value_digest((positions, geometry))


def prepared_compact_snapshot_digest(
    compact: DeclaredCompactOverride,
    *,
    override_digest: Optional[str] = None,
    ring_inclusive_digest: Optional[str] = None,
) -> str:
    """Canonical SHA-256 digest of all mutable and structural compact fields."""
    ov_sha = (
        override_digest
        if override_digest is not None
        else _artwork_scene_digest(compact.override)
    )
    ring_sha = (
        ring_inclusive_digest
        if ring_inclusive_digest is not None
        else _artwork_scene_digest(compact.ring_inclusive)
    )
    return _artwork_value_digest(
        (
            ov_sha,
            ring_sha,
            compact.width_um,
            compact.height_um,
            compact.loose_width_um,
            compact.loose_height_um,
            compact.n_rings,
            compact.n_declared_rings,
            compact.n_source_guard_ring_declarations,
            compact.member_ring_pad_top_nm,
            compact.member_ring_pad_bottom_nm,
            compact.shared_well_guard_envelope_plans,
            compact.shared_well_guard_ring_inventory,
            compact.shared_well_guard_child_plans,
            compact.shared_well_flexible_content_bboxes_nm,
            compact.source_mos_names,
            compact.placement_state_digest,
        )
    )


def _seal_object_to_module() -> dict[int, str]:
    mapping: dict[int, str] = {}
    for mod_name in _SEAL_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        seal = getattr(mod, "_SEAL", None)
        if seal is not None:
            mapping[id(seal)] = mod_name
    return mapping


class _PreparedScenePickler(pickle.Pickler):
    def __init__(
        self,
        stream: io.BytesIO,
        *,
        canonical_cell: Any,
        canonical_source_sha256: str,
        tech: Any,
        native_bundle: Any,
    ) -> None:
        super().__init__(stream, protocol=5)
        self._canonical_cell = canonical_cell
        self._canonical_source_sha256 = canonical_source_sha256
        self._tech = tech
        self._native_bundle = native_bundle
        self._seal_map = _seal_object_to_module()

    def persistent_id(self, obj: Any) -> Any:
        from clay.ir import CellDef
        from clay.layout.generic_placement import _PhysicalMacroScopeToken
        from clay.layout.tech import LayoutTech

        obj_id = id(obj)
        if obj_id in self._seal_map:
            return ("module_seal", self._seal_map[obj_id])
        if type(obj) is _PhysicalMacroScopeToken:
            return ("physical_macro_scope_token", "mint_on_load")
        if type(obj) is LayoutTech:
            return ("live_layout_tech", obj.pdk_id)
        if type(obj) is CellDef:
            if obj is self._canonical_cell:
                return ("live_cell_def", "canonical")
            span = getattr(obj, "source_span", None)
            if isinstance(span, dict) and span.get("file"):
                from clay.frontend.source_revision import reopen_complete_source

                try:
                    info = reopen_complete_source(obj)
                    if (
                        info["source_sha256"] == self._canonical_source_sha256
                        and obj.local_name == self._canonical_cell.local_name
                    ):
                        return ("live_cell_def", "canonical")
                except Exception:
                    pass
        if self._native_bundle is not None and obj is self._native_bundle:
            return ("native_geometry_bundle", "primary")
        mod_name = getattr(type(obj), "__module__", "")
        cls_name = getattr(type(obj), "__qualname__", "")
        if (mod_name, cls_name) == (
            "clay.layout.native_diffusion_source",
            "NativePCellWitness",
        ):
            row_specs = tuple((group, geometry, group) for group, geometry, _folder in obj.rows)
            return ("native_pcell_witness", row_specs, obj.row_candidates, obj.identity, obj.overlay)
        if mod_name in (
            "clay.layout.native_row_matching_evidence",
            "clay.layout.native_cross_row_matching_evidence",
        ) and cls_name in (
            "NativeRowMatchingEvidence",
            "NativeRowMatchingInventory",
            "NativeCrossRowMatchingInventory",
        ):
            return ("native_matching_placeholder", cls_name)
        return None


class _SafeClayUnpickler(pickle.Unpickler):
    def __init__(
        self,
        stream: io.BytesIO,
        *,
        canonical_cell: Any,
        tech: Any,
        native_bundle: Any,
        native_diffusion_root: Optional[Path],
        source_cir: Optional[str],
    ) -> None:
        super().__init__(stream)
        self._canonical_cell = canonical_cell
        self._tech = tech
        self._native_bundle = native_bundle
        self._native_diffusion_root = native_diffusion_root
        self._source_cir = source_cir
        self.rehydrated_witness: Any = None

    def find_class(self, module: str, name: str) -> Any:
        if module == "builtins":
            _require_auth(
                name not in _DISALLOWED_BUILTINS,
                f"disallowed builtin in prepared scene stream: {name}",
            )
            mod = importlib.import_module("builtins")
            _require_auth(hasattr(mod, name), f"unknown builtin in prepared scene: {name}")
            return getattr(mod, name)
        allowed = module in _ALLOWED_NON_CLAY_MODULES or (
            module == "clay" or module.startswith("clay.")
        )
        _require_auth(
            allowed,
            f"disallowed module in prepared scene stream: {module}.{name}",
        )
        if module == "pathlib":
            _require_auth(
                name in ("Path", "PosixPath", "WindowsPath", "PurePosixPath", "PureWindowsPath"),
                f"disallowed pathlib symbol in prepared scene: {name}",
            )
        mod = importlib.import_module(module)
        target: Any = mod
        for part in name.split("."):
            _require_auth(
                hasattr(target, part),
                f"missing symbol in prepared scene stream: {module}.{name}",
            )
            target = getattr(target, part)
        return target

    def persistent_load(self, pid: Any) -> Any:
        _require_auth(type(pid) is tuple and len(pid) >= 2, "malformed persistent id")
        tag = pid[0]
        if tag == "module_seal":
            mod_name = pid[1]
            _require_auth(mod_name in _SEAL_MODULES, f"unauthorized seal module: {mod_name}")
            mod = importlib.import_module(mod_name)
            return getattr(mod, "_SEAL")
        if tag == "physical_macro_scope_token":
            return None
        if tag == "live_layout_tech":
            pdk_id = pid[1]
            _require_auth(pdk_id == self._tech.pdk_id, "LayoutTech PDK id mismatch in payload")
            return self._tech
        if tag == "live_cell_def":
            _require_auth(pid[1] == "canonical", "unknown CellDef persistent reference")
            return self._canonical_cell
        if tag == "native_geometry_bundle":
            _require_auth(
                self._native_bundle is not None,
                "prepared scene payload references missing NativeGeometryBundle",
            )
            return self._native_bundle
        if tag == "native_pcell_witness":
            _require_auth(
                self._native_bundle is not None and self._native_diffusion_root is not None,
                "prepared scene payload references missing native diffusion templates",
            )
            if len(pid) == 5:
                from clay.layout.native_diffusion_source import (
                    _rehydrate_sealed_native_pcell_witness,
                )

                _, row_specs, row_candidates, expected_identity, sealed_overlay = pid
                rows = tuple(
                    (group, geometry, self._native_diffusion_root / rel_group)
                    for group, geometry, rel_group in row_specs
                )
                witness = _rehydrate_sealed_native_pcell_witness(
                    cell=self._canonical_cell,
                    bundle=self._native_bundle,
                    tech=self._tech,
                    rows=rows,
                    source_cir=self._source_cir,
                    row_candidates=row_candidates,
                    expected_identity=expected_identity,
                    overlay=sealed_overlay,
                )
            else:
                from clay.layout.native_diffusion_source import _capture_native_pcell_witness

                _, row_specs, row_candidates, expected_identity = pid
                rows = tuple(
                    (group, geometry, self._native_diffusion_root / rel_group)
                    for group, geometry, rel_group in row_specs
                )
                witness = _capture_native_pcell_witness(
                    cell=self._canonical_cell,
                    bundle=self._native_bundle,
                    tech=self._tech,
                    rows=rows,
                    source_cir=self._source_cir,
                    row_candidates=row_candidates,
                )
            _require_auth(
                witness.sha256 == expected_identity,
                "rehydrated NativePCellWitness identity differs from captured scene",
            )
            self.rehydrated_witness = witness
            return witness
        if tag == "native_matching_placeholder":
            return None
        raise PreparedSceneAuthenticationError(f"unknown persistent tag: {tag!r}")


def export_prepared_scene(
    scene_dir: str | Path,
    *,
    prepared_compact: DeclaredCompactOverride,
    cell_def: Any,
    intent: Any,
    pdk: Any,
    effective_policy: Any,
    source_cir: str | Path | None = None,
    device_records: Optional[Mapping[str, Mapping[str, Any]]] = None,
    intent_fact_set: Any = None,
    source_inventory: Any = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Export an authenticated ``PreparedScene`` bundle to ``scene_dir``."""
    from clay.frontend.source_revision import reopen_complete_source
    from clay.layout.place.native_materialization_cache import (
        _copy_tree_regular,
        _directory_manifest,
    )
    from clay.layout.place.row_source_projection import require_row_source_projection
    from clay.layout.route.resource_world import pdk_provenance_digest
    from clay.layout.tech import LayoutTech

    tech = pdk if isinstance(pdk, LayoutTech) else LayoutTech.resolve(str(pdk))
    _require_prepared_declared_compact(prepared_compact, cell_def=cell_def, pdk=tech)

    source_info = reopen_complete_source(cell_def)
    source_clay_sha = source_info["source_sha256"]
    source_inventory_sha = source_info["source_inventory_sha256"]

    placement = prepared_compact.logical_placement_basis
    program = getattr(placement, "row_program", None)
    materialization = getattr(placement, "row_materialization", None)
    projection = getattr(materialization, "source_projection", None)

    resolved_cir_path: Optional[Path] = None
    if source_cir is not None:
        resolved_cir_path = Path(source_cir).resolve(strict=True)
    elif projection is not None and projection.source_cir is not None:
        resolved_cir_path = Path(projection.source_cir).resolve(strict=True)
    topology_sha = _sha256_file(resolved_cir_path) if resolved_cir_path is not None else None

    if projection is not None:
        require_row_source_projection(projection, program=program)
        _require_auth(
            projection.source_sha256 == source_clay_sha
            and projection.source_inventory_sha256 == source_inventory_sha,
            "prepared_compact row projection does not match live .clay source",
        )
        if topology_sha is not None:
            _require_auth(
                projection.topology_sha256 == topology_sha,
                "prepared_compact row projection does not match live .cir topology",
            )

    pdk_snapshot = tech.adapter.source_snapshot()
    pdk_prov_sha = pdk_provenance_digest(tech)

    out_dir = Path(scene_dir).resolve()
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"prepared scene directory already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    native_bundle = getattr(placement, "native_geometry", None)
    native_geometry_files: Optional[Dict[str, str]] = None
    native_geometry_origin_dir: Optional[str] = None
    native_expected_runtime_files: Any = None
    native_pdk_root: str = "/foss/pdks"
    native_geometries_sha256: Optional[str] = None
    if native_bundle is not None:
        native_bundle.require_current(cell_def)
        geom_dst = out_dir / "native-geometry"
        _copy_tree_regular(Path(native_bundle.output_dir), geom_dst)
        native_geometry_files = _directory_manifest(geom_dst)
        bundle_geometries = getattr(native_bundle, "geometries", None)
        if isinstance(bundle_geometries, Mapping):
            geom_buf = io.BytesIO()
            geom_pickler = _PreparedScenePickler(
                geom_buf,
                canonical_cell=cell_def,
                canonical_source_sha256=source_clay_sha,
                tech=tech,
                native_bundle=None,
            )
            geom_pickler.dump(dict(bundle_geometries))
            geom_bin_bytes = geom_buf.getvalue()
            (out_dir / "native-geometries.bin").write_bytes(geom_bin_bytes)
            native_geometries_sha256 = _sha256_bytes(geom_bin_bytes)
        origin = getattr(native_bundle, "_cache_origin_output_dir", None)
        native_geometry_origin_dir = (
            origin
            if (type(origin) is str and origin)
            else Path(native_bundle.output_dir).resolve().as_posix()
        )
        command_path = geom_dst / "command.json"
        if command_path.is_file():
            cmd = json.loads(command_path.read_text(encoding="utf-8"))
            for arg in cmd:
                if type(arg) is str and arg.startswith("PDK_ROOT="):
                    native_pdk_root = arg.split("=", 1)[1]
                    break
        toolchain_path = geom_dst / "toolchain.json"
        if toolchain_path.is_file():
            tc = json.loads(toolchain_path.read_text(encoding="utf-8"))
            native_expected_runtime_files = {
                row["path"]: row["sha256"]
                for row in tc.get("files", ())
                if isinstance(row, dict) and "path" in row and "sha256" in row
            }

    witness = getattr(projection, "native_witness", None)
    native_diffusion_files: Optional[Dict[str, str]] = None
    if witness is not None:
        diff_dst = out_dir / "native-diffusion"
        diff_dst.mkdir(parents=True, exist_ok=False)
        for group, _geom, folder in witness.rows:
            _copy_tree_regular(Path(folder), diff_dst / group)
        native_diffusion_files = _directory_manifest(diff_dst)

    frame_proof = getattr(placement, "native_row_cp_frame_proof", None)
    group_authority_sha = (
        frame_proof.group_authority.sha256
        if frame_proof is not None and getattr(frame_proof, "group_authority", None) is not None
        else None
    )

    payload_obj = {
        "prepared_compact": prepared_compact,
        "intent": intent,
        "effective_policy": effective_policy,
        "device_records": dict(device_records) if device_records is not None else None,
        "intent_fact_set": intent_fact_set,
        "source_inventory": source_inventory,
    }
    buf = io.BytesIO()
    pickler = _PreparedScenePickler(
        buf,
        canonical_cell=cell_def,
        canonical_source_sha256=source_clay_sha,
        tech=tech,
        native_bundle=native_bundle,
    )
    pickler.dump(payload_obj)
    scene_bytes = buf.getvalue()
    (out_dir / "scene.bin").write_bytes(scene_bytes)

    manifest: Dict[str, Any] = {
        "schema": PREPARED_SCENE_SCHEMA,
        "source_cell_name": str(cell_def.local_name),
        "source_clay_sha256": source_clay_sha,
        "source_inventory_sha256": source_inventory_sha,
        "topology_sha256": topology_sha,
        "pdk_id": tech.pdk_id,
        "pdk_provenance_sha256": pdk_prov_sha,
        "pdk_snapshot_sha256": (
            None if pdk_snapshot is None else pdk_snapshot.snapshot_sha256
        ),
        "pdk_installed_authority_sha256": (
            None if pdk_snapshot is None else pdk_snapshot.installed_authority_sha256
        ),
        "final_placement_state_sha256": prepared_compact.placement_state_digest,
        "override_artwork_sha256": _artwork_scene_digest(prepared_compact.override),
        "ring_inclusive_artwork_sha256": _artwork_scene_digest(
            prepared_compact.ring_inclusive
        ),
        "row_program_sha256": (
            None
            if program is None
            else _sha256_bytes(program.to_canonical_bytes())
        ),
        "row_materialization_sha256": (
            None
            if materialization is None
            else materialization.certificate.sha256
        ),
        "row_source_projection_sha256": (
            None if projection is None else projection.sha256
        ),
        "native_row_group_authority_sha256": group_authority_sha,
        "has_native_geometry": native_bundle is not None,
        "native_geometry_files": native_geometry_files,
        "native_geometries_sha256": native_geometries_sha256,
        "native_geometry_origin_dir": native_geometry_origin_dir,
        "native_pdk_root": native_pdk_root,
        "native_expected_runtime_files": native_expected_runtime_files,
        "has_native_diffusion": witness is not None,
        "native_diffusion_files": native_diffusion_files,
        "payload_sha256": _sha256_bytes(scene_bytes),
    }
    manifest["manifest_seal_sha256"] = _manifest_seal(manifest)
    (out_dir / "manifest.json").write_bytes(
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return manifest


def load_prepared_scene(
    scene_dir: str | Path,
    *,
    cell_def: Any,
    pdk: Any,
    source_cir: str | Path | None = None,
    work_dir: str | Path | None = None,
) -> PreparedScene:
    """Authenticate and load a serialized ``PreparedScene`` in seconds."""
    from clay.frontend.source_revision import reopen_complete_source
    from clay.layout.generic_placement import (
        _mint_physical_macro_scope,
        _physical_macro_compiler_binding,
        _seal_physical_macro_scope,
        verify_row_source_compiler_authority,
    )
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.place.native_materialization_cache import (
        _copy_tree_regular,
        _rehydrate_cached_geometry_bundle,
        _verify_directory_manifest,
    )
    from clay.layout.place.native_row_group_authority import (
        authenticate_serialized_native_row_group_authority,
    )
    from clay.layout.place.row_replay_session import (
        require_current_row_materialization,
        shared_row_replay_session,
    )
    from clay.layout.place.row_source_projection import (
        authenticate_serialized_row_source_projection,
    )
    from clay.layout.route.resource_world import pdk_provenance_digest
    from clay.layout.tech import LayoutTech

    in_dir = Path(scene_dir).resolve(strict=True)
    manifest_path = in_dir / "manifest.json"
    payload_path = in_dir / "scene.bin"
    _require_auth(
        manifest_path.is_file() and payload_path.is_file(),
        "prepared scene directory is missing manifest.json or scene.bin",
    )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PreparedSceneAuthenticationError(f"malformed manifest.json: {exc}") from exc
    _require_auth(
        isinstance(manifest, dict) and manifest.get("schema") == PREPARED_SCENE_SCHEMA,
        "prepared scene manifest schema mismatch",
    )
    _require_auth(
        manifest.get("manifest_seal_sha256") == _manifest_seal(manifest),
        "prepared scene manifest seal SHA-256 verification failed (tampered manifest)",
    )

    scene_bytes = payload_path.read_bytes()
    _require_auth(
        _sha256_bytes(scene_bytes) == manifest.get("payload_sha256"),
        "prepared scene payload SHA-256 verification failed (tampered scene.bin)",
    )

    try:
        source_info = reopen_complete_source(cell_def)
    except Exception as exc:
        raise PreparedSceneAuthenticationError(
            f"live .clay source cannot be verified: {exc}"
        ) from exc
    _require_auth(
        str(getattr(cell_def, "local_name", "")) == manifest.get("source_cell_name"),
        "live CellDef name differs from prepared scene",
    )
    _require_auth(
        source_info["source_sha256"] == manifest.get("source_clay_sha256")
        and source_info["source_inventory_sha256"]
        == manifest.get("source_inventory_sha256"),
        "live .clay source SHA-256 differs from prepared scene",
    )

    expected_topo_sha = manifest.get("topology_sha256")
    resolved_cir_str: Optional[str] = None
    if expected_topo_sha is not None:
        _require_auth(
            source_cir is not None,
            "prepared scene requires live source_cir to verify topology SHA-256",
        )
        cir_path = Path(source_cir).resolve(strict=True)
        actual_topo_sha = _sha256_file(cir_path)
        _require_auth(
            actual_topo_sha == expected_topo_sha,
            "live .cir topology SHA-256 differs from prepared scene",
        )
        resolved_cir_str = str(cir_path)
    elif source_cir is not None:
        raise PreparedSceneAuthenticationError(
            "source_cir supplied for a prepared scene captured without topology"
        )

    tech = pdk if isinstance(pdk, LayoutTech) else LayoutTech.resolve(str(pdk))
    pdk_snapshot = tech.adapter.source_snapshot()
    _require_auth(
        tech.pdk_id == manifest.get("pdk_id")
        and pdk_provenance_digest(tech) == manifest.get("pdk_provenance_sha256")
        and (None if pdk_snapshot is None else pdk_snapshot.snapshot_sha256)
        == manifest.get("pdk_snapshot_sha256")
        and (
            None
            if pdk_snapshot is None
            else pdk_snapshot.installed_authority_sha256
        )
        == manifest.get("pdk_installed_authority_sha256"),
        "live PDK rules or provenance digest differs from prepared scene",
    )

    native_bundle: Any = None
    if manifest.get("has_native_geometry"):
        geom_archive = in_dir / "native-geometry"
        try:
            _verify_directory_manifest(geom_archive, manifest.get("native_geometry_files"))
        except Exception as exc:
            raise PreparedSceneAuthenticationError(
                f"native geometry artifact verification failed: {exc}"
            ) from exc
        if work_dir is not None:
            geom_runtime_dir = Path(work_dir).resolve() / (
                "native-geometry-" + uuid.uuid4().hex[:12]
            )
            _copy_tree_regular(geom_archive, geom_runtime_dir)
        else:
            geom_runtime_dir = geom_archive
        preverified_geometries: Optional[Dict[str, Any]] = None
        geom_bin_sha = manifest.get("native_geometries_sha256")
        if geom_bin_sha is not None:
            geom_bin_path = in_dir / "native-geometries.bin"
            _require_auth(
                geom_bin_path.is_file(),
                "prepared scene missing native-geometries.bin",
            )
            geom_bin_bytes = geom_bin_path.read_bytes()
            _require_auth(
                _sha256_bytes(geom_bin_bytes) == geom_bin_sha,
                "native-geometries.bin SHA-256 verification failed",
            )
            geom_unpickler = _SafeClayUnpickler(
                io.BytesIO(geom_bin_bytes),
                canonical_cell=cell_def,
                tech=tech,
                native_bundle=None,
                native_diffusion_root=None,
                source_cir=resolved_cir_str,
            )
            loaded_geoms = geom_unpickler.load()
            _require_auth(
                isinstance(loaded_geoms, dict),
                "native-geometries.bin payload must be a dict",
            )
            preverified_geometries = loaded_geoms
        try:
            native_bundle = _rehydrate_cached_geometry_bundle(
                cell_def,
                output_dir=geom_runtime_dir,
                pdk_root=str(manifest.get("native_pdk_root") or "/foss/pdks"),
                expected_runtime_files=manifest.get("native_expected_runtime_files"),
                origin_output_dir=str(manifest.get("native_geometry_origin_dir") or ""),
                preverified_geometries=preverified_geometries,
            )
        except Exception as exc:
            raise PreparedSceneAuthenticationError(
                f"native geometry bundle replay failed: {exc}"
            ) from exc

    diff_runtime_dir: Optional[Path] = None
    if manifest.get("has_native_diffusion"):
        diff_archive = in_dir / "native-diffusion"
        try:
            _verify_directory_manifest(diff_archive, manifest.get("native_diffusion_files"))
        except Exception as exc:
            raise PreparedSceneAuthenticationError(
                f"native diffusion template verification failed: {exc}"
            ) from exc
        if work_dir is not None:
            diff_runtime_dir = Path(work_dir).resolve() / (
                "native-diffusion-" + uuid.uuid4().hex[:12]
            )
            _copy_tree_regular(diff_archive, diff_runtime_dir)
        else:
            diff_runtime_dir = diff_archive

    try:
        unpickler = _SafeClayUnpickler(
            io.BytesIO(scene_bytes),
            canonical_cell=cell_def,
            tech=tech,
            native_bundle=native_bundle,
            native_diffusion_root=diff_runtime_dir,
            source_cir=resolved_cir_str,
        )
        loaded = unpickler.load()
    except PreparedSceneAuthenticationError:
        raise
    except Exception as exc:
        raise PreparedSceneAuthenticationError(
            f"prepared scene payload deserialization failed: {exc}"
        ) from exc

    _require_auth(isinstance(loaded, dict), "prepared scene payload root must be a dict")
    prepared_compact = loaded.get("prepared_compact")
    _require_auth(
        type(prepared_compact) is DeclaredCompactOverride,
        "prepared scene payload lacks DeclaredCompactOverride",
    )
    placement = prepared_compact.logical_placement_basis
    program = getattr(placement, "row_program", None)
    materialization = getattr(placement, "row_materialization", None)
    projection = getattr(materialization, "source_projection", None)

    native_composite_row = None
    try:
        if projection is not None:
            authenticate_serialized_row_source_projection(
                projection,
                canonical_source_cell=cell_def,
                tech=tech,
                expected_source_sha256=source_info["source_sha256"],
                expected_topology_sha256=(
                    expected_topo_sha if expected_topo_sha is not None else projection.topology_sha256
                ),
                expected_pdk_provenance_sha256=manifest["pdk_provenance_sha256"],
                expected_payload_sha256=manifest["row_source_projection_sha256"],
                source_cir=resolved_cir_str,
                native_witness=unpickler.rehydrated_witness,
            )
        if program is not None and materialization is not None:
            _require_auth(
                _sha256_bytes(program.to_canonical_bytes())
                == manifest.get("row_program_sha256")
                and materialization.certificate.sha256
                == manifest.get("row_materialization_sha256"),
                "row program or materialization canonical digest mismatch",
            )
        with shared_row_replay_session(
            program=program,
            materialization=materialization,
            tech=tech,
            source_cell=cell_def,
            trust_sealed_scene=True,
        ) as session_proof:
            if session_proof is None and program is not None and materialization is not None:
                require_current_row_materialization(
                    materialization, program=program, tech_id=tech.pdk_id
                )
            graph = getattr(placement, "physical_macro_graph", None)
            if graph is not None:
                token = _mint_physical_macro_scope(graph.sha256)
                _seal_physical_macro_scope(
                    token,
                    graph.sha256,
                    _physical_macro_compiler_binding(placement),
                    row_program=program,
                    row_materialization=materialization,
                )
                object.__setattr__(placement, "physical_macro_scope_token", token)
                if projection is not None:
                    verify_row_source_compiler_authority(placement, cell_def, tech)
            frame_proof = getattr(placement, "native_row_cp_frame_proof", None)
            if (
                frame_proof is not None
                and getattr(frame_proof, "group_authority", None) is not None
            ):
                from clay.layout.place.native_row_cp_frames import (
                    replay_native_row_cp_frame_proof,
                )

                _auth, _frames_and_groups = authenticate_serialized_native_row_group_authority(
                    frame_proof.group_authority,
                    source_placement=placement,
                    source_cell=cell_def,
                    source_intent=layout_intent_from_cell_def(cell_def),
                    program=program,
                    materialization=materialization,
                    tech=tech,
                    expected_authority_sha256=str(
                        manifest.get("native_row_group_authority_sha256") or ""
                    ),
                    _return_replay_tuple=True,
                )
                replay_native_row_cp_frame_proof(
                    frame_proof,
                    program,
                    materialization,
                    tech,
                    _precomputed_frames_and_groups=_frames_and_groups,
                )
            if (
                unpickler.rehydrated_witness is not None
                and program is not None
                and materialization is not None
            ):
                try:
                    from clay.layout.native_composite_row import (
                        _capture_native_composite_row,
                    )

                    first_folder = Path(unpickler.rehydrated_witness.rows[0][2])
                    receipt_path = first_folder / "template-receipt.json"
                    deps = tuple(unpickler.rehydrated_witness.dependencies)
                    native_composite_row = _capture_native_composite_row(
                        cell=cell_def,
                        native_bundle=native_bundle,
                        tech=tech,
                        projection=projection,
                        program=program,
                        materialization=materialization,
                        geometry=materialization.realizations[0].geometry,
                        receipt_path=receipt_path,
                        receipt_sha256=_sha256_file(receipt_path),
                        dependencies=deps,
                    )
                except Exception as exc:
                    raise PreparedSceneAuthenticationError(
                        f"native composite row rehydration failed: {exc}"
                    ) from exc
    except PreparedSceneAuthenticationError:
        raise
    except Exception as exc:
        raise PreparedSceneAuthenticationError(
            f"prepared scene authority replay failed: {exc}"
        ) from exc

    _require_auth(
        prepared_compact.placement_state_digest
        == manifest.get("final_placement_state_sha256"),
        "final placement state digest mismatch",
    )
    ov_sha = _artwork_scene_digest(prepared_compact.override)
    ring_sha = _artwork_scene_digest(prepared_compact.ring_inclusive)
    _require_auth(
        ov_sha == manifest.get("override_artwork_sha256")
        and ring_sha == manifest.get("ring_inclusive_artwork_sha256"),
        "prepared_compact artwork digest mismatch",
    )

    snap_value = prepared_compact_snapshot_digest(
        prepared_compact,
        override_digest=ov_sha,
        ring_inclusive_digest=ring_sha,
    )
    register_authenticated_prepared_compact(
        prepared_compact,
        source_sha256=source_info["source_sha256"],
        topology_sha256=expected_topo_sha,
        pdk_id=tech.pdk_id,
        pdk_provenance_sha256=manifest["pdk_provenance_sha256"],
        snapshot_value=snap_value,
    )
    _require_prepared_declared_compact(prepared_compact, cell_def=cell_def, pdk=tech)

    return PreparedScene(
        prepared_compact=prepared_compact,
        placement=placement,
        intent=loaded["intent"],
        effective_policy=loaded["effective_policy"],
        device_records=loaded["device_records"],
        intent_fact_set=loaded.get("intent_fact_set"),
        source_inventory=loaded.get("source_inventory"),
        native_geometry=native_bundle,
        native_composite_row=native_composite_row,
        manifest=manifest,
    )


__all__ = [
    "PREPARED_SCENE_SCHEMA",
    "PreparedScene",
    "PreparedSceneAuthenticationError",
    "export_prepared_scene",
    "load_prepared_scene",
]
