"""Adapter-owned BJT model/geometry capability with five-state evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATES = frozenset({"pass", "fail", "unmeasurable", "unsupported", "advisory"})


@dataclass(frozen=True)
class BjtTapBinding:
    """Adapter-owned realization of a declared device-level ``@tap``.

    ``intrinsic_terminal`` means the fixed cell already exposes the substrate
    contact through ``terminal_role``. ``external_companion`` asks the generic
    BJT materializer to add a rule-sized diffusion/contact/M1 island outside the
    named well. Layer/rule tokens are PDK data; the circuit supplies only the net.
    """

    mode: str
    where: tuple[str, ...] = ("substrate",)
    terminal_role: str = ""
    diffusion_layer: str = ""
    diffusion_rule_layer: str = ""
    implant_layer: str = ""
    excluded_well_layer: str = ""
    contact_via: str = ""
    metal_layer: str = "metal1"
    implant_enclosure_nm: int = 0
    fixed_cell_clearance_nm: int = 0
    max_distance_nm: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"intrinsic_terminal", "external_companion"}:
            raise ValueError("BJT tap mode must be intrinsic_terminal or external_companion")
        if not self.where or any(not str(item) for item in self.where):
            raise ValueError("BJT tap where vocabulary must be non-empty")
        if self.mode == "intrinsic_terminal":
            if self.terminal_role not in {"c", "b", "e"}:
                raise ValueError("intrinsic BJT tap must name c, b, or e")
        else:
            required = (
                self.diffusion_layer,
                self.diffusion_rule_layer,
                self.implant_layer,
                self.excluded_well_layer,
                self.contact_via,
                self.metal_layer,
            )
            if any(not value for value in required):
                raise ValueError("external BJT companion tap requires exact layer/rule tokens")
            if self.implant_enclosure_nm < 0 or self.fixed_cell_clearance_nm < 0:
                raise ValueError("BJT companion tap dimensions must be non-negative")
            if self.max_distance_nm <= 0:
                raise ValueError("BJT companion tap max distance must be positive")


@dataclass(frozen=True)
class BjtBinding:
    """One semantic BJT family bound to foundry DATA, never circuit identity."""

    semantic_kind: str
    simulator_model: str
    lvs_model: str
    nominal_voltage_v: float
    emitter_area_um2: float
    geometry_provider: str
    geometry_path: str
    provenance: str
    geometry_sha256: str = ""
    primitive_evidence_state: str = "unmeasurable"
    terminal_order: tuple[str, str, str] = ("c", "b", "e")
    full_extraction_gds: str = ""
    full_extraction_cdl: str = ""
    proof_gds_sha256: str = ""
    proof_cdl_sha256: str = ""
    canonical_materializer: bool = False
    terminal_label_layer: tuple[int, int] | None = None
    terminal_labels: Mapping[str, str] = field(default_factory=dict)
    terminal_access_points_nm: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    terminal_access_layers: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    array_spacing_nm: int = 0
    array_landing_offsets_nm: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    lvs_parallel_parameter: str = "m"
    tap_binding: BjtTapBinding | None = None

    def __post_init__(self) -> None:
        if self.semantic_kind not in ("pnp", "npn"):
            raise ValueError("BJT semantic_kind must be pnp or npn")
        if self.terminal_order != ("c", "b", "e"):
            raise ValueError("BJT terminal order must be exact c/b/e")
        if self.primitive_evidence_state not in _STATES:
            raise ValueError("invalid five-state BJT evidence")
        if self.nominal_voltage_v <= 0 or self.emitter_area_um2 <= 0:
            raise ValueError("BJT voltage and emitter area must be positive")
        if type(self.array_spacing_nm) is not int or self.array_spacing_nm < 0:
            raise ValueError("BJT array_spacing_nm must be a non-negative integer")
        if self.array_landing_offsets_nm:
            if set(self.array_landing_offsets_nm) != {"c", "b", "e"}:
                raise ValueError("BJT array landing offsets require exact c/b/e roles")
            for role, offset in self.array_landing_offsets_nm.items():
                if (
                    not isinstance(offset, tuple)
                    or len(offset) != 2
                    or any(type(value) is not int or value < 0 for value in offset)
                ):
                    raise ValueError(
                        f"BJT array landing offset for {role} must be two non-negative integers"
                    )
        if not self.lvs_parallel_parameter or not self.lvs_parallel_parameter.replace("_", "").isalnum():
            raise ValueError("BJT LVS parallel parameter must be an identifier")
        if self.geometry_provider == "fixed_gds":
            if len(self.geometry_sha256) != 64:
                raise ValueError("fixed_gds BJT requires a pinned geometry sha256")
            label_access = (
                self.terminal_label_layer is not None
                and set(self.terminal_labels) == {"c", "b", "e"}
            )
            point_access = (
                set(self.terminal_access_points_nm) == {"c", "b", "e"}
                and set(self.terminal_access_layers) == {"c", "b", "e"}
            )
            if not label_access and not point_access:
                raise ValueError(
                    "fixed_gds BJT requires exact c/b/e labels or authoritative access points"
                )
            if self.terminal_access_points_nm or self.terminal_access_layers:
                if not point_access:
                    raise ValueError(
                        "fixed_gds BJT access points/layers require exact c/b/e roles"
                    )
                for role in ("c", "b", "e"):
                    point = self.terminal_access_points_nm[role]
                    layer = self.terminal_access_layers[role]
                    if (
                        not isinstance(point, tuple)
                        or len(point) != 2
                        or any(type(value) is not int or value < 0 for value in point)
                    ):
                        raise ValueError(
                            f"BJT terminal access point for {role} must be two non-negative integers"
                        )
                    if (
                        not isinstance(layer, tuple)
                        or len(layer) != 2
                        or any(type(value) is not int or value < 0 for value in layer)
                    ):
                        raise ValueError(
                            f"BJT terminal access layer for {role} must be a non-negative GDS key"
                        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_path(value: str) -> Path | None:
    return (_REPO_ROOT / Path(value)).resolve() if value else None


def assess_bjt_binding(adapter: Any, semantic_kind: str) -> Mapping[str, Any]:
    """Return a fail-closed capability row without running a physical tool."""
    binding = adapter.bjt_bindings.get(str(semantic_kind).lower())
    if binding is None:
        return {
            "pdk": adapter.pdk_id,
            "semantic_kind": str(semantic_kind).lower(),
            "verdict": "unsupported",
            "primitive_evidence": "unsupported",
            "geometry_artifact": "unsupported",
            "full_extraction_fixture": "unsupported",
            "canonical_materializer": "unsupported",
            "reason": "adapter declares no BJT binding",
        }

    geometry = _bound_path(binding.geometry_path)
    geometry_state = "pass" if geometry is not None and geometry.is_file() else "fail"
    if (
        geometry_state == "pass"
        and binding.geometry_sha256
        and _sha256(geometry) != binding.geometry_sha256
    ):
        geometry_state = "fail"
    fixture_state = "unmeasurable"
    proof_errors: list[str] = []
    if binding.full_extraction_gds and binding.full_extraction_cdl:
        proof_gds = _bound_path(binding.full_extraction_gds)
        proof_cdl = _bound_path(binding.full_extraction_cdl)
        for label, path, expected in (
            ("gds", proof_gds, binding.proof_gds_sha256),
            ("cdl", proof_cdl, binding.proof_cdl_sha256),
        ):
            if path is None or not path.is_file():
                proof_errors.append(f"missing_{label}")
            elif expected and _sha256(path) != expected:
                proof_errors.append(f"digest_{label}")
        fixture_state = "pass" if not proof_errors else "fail"

    materializer_state = "pass" if binding.canonical_materializer else "unsupported"
    if geometry_state == "fail" or fixture_state == "fail":
        verdict = "fail"
    elif materializer_state != "pass":
        verdict = "unsupported"
    elif binding.primitive_evidence_state == "pass" and fixture_state == "pass":
        verdict = "pass"
    else:
        verdict = binding.primitive_evidence_state

    return {
        "pdk": adapter.pdk_id,
        "semantic_kind": binding.semantic_kind,
        "simulator_model": binding.simulator_model,
        "lvs_model": binding.lvs_model,
        "primitive_evidence": binding.primitive_evidence_state,
        "geometry_artifact": geometry_state,
        "full_extraction_fixture": fixture_state,
        "canonical_materializer": materializer_state,
        "verdict": verdict,
        "proof_errors": proof_errors,
        "provenance": binding.provenance,
    }


__all__ = ["BjtBinding", "BjtTapBinding", "assess_bjt_binding"]
