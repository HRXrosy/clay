"""Digest-bound geometric quality over canonical physical evidence.

This module is a measurement adapter.  Aggregate authority remains the
``DimensionVerdict`` type in :mod:`clay.flow.reference_convergence`.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
from clay.core_contract import VerificationStatus
from clay.flow.portfolio_signoff import ArtifactLocator
from clay.flow.reference_convergence import DimensionVerdict, ReferenceConvergenceError


CANONICAL_FIDELITY_METRICS = (
    "enclosure_efficiency",
    "ring_granularity",
    "fill_density",
    "aspect_band",
    "dead_region",
    "contact_density",
    "bbox_fill",
    "row_balance",
    "over_cell_fraction",
)


class ReferenceQualityError(ReferenceConvergenceError):
    """Canonical geometric evidence is missing, ambiguous, or cross-bound."""


def _file_sha256(path: Path, *, label: str) -> str:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReferenceQualityError(f"{label} is missing: {path}") from exc
    if not payload:
        raise ReferenceQualityError(f"{label} is empty: {path}")
    return hashlib.sha256(payload).hexdigest()


def _require_manifest_binding(
    path: Path, *, reference_manifest: object
) -> None:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ReferenceQualityError(
                    f"reference manifest JSON contains duplicate key {key!r}"
                )
            output[key] = value
        return output

    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=reject_duplicates
        )
    except ReferenceQualityError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReferenceQualityError(
            f"reference manifest locator is invalid JSON: {path}"
        ) from exc
    if type(payload) is not dict:
        raise ReferenceQualityError("reference manifest locator root is not an object")
    expected = reference_manifest.canonical_data()
    if payload != expected:
        raise ReferenceQualityError(
            "reference manifest locator bytes differ from the typed manifest"
        )


def _repository_path(root: Path, relative: str, *, label: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise ReferenceQualityError(
            f"{label} must be a canonical repository-relative path"
        )
    resolved = root.joinpath(*path.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ReferenceQualityError(f"{label} escapes the repository") from exc
    return resolved


def _inventory_for_cell(cell: object) -> dict[str, int]:
    counts = {name: 0 for name in ("pmos", "nmos", "pnp", "resistor", "capacitor")}
    for device in tuple(getattr(cell, "devices", ()) or ()):
        kind = str(getattr(device, "device_class", "")).casefold()
        if kind.startswith("pmos"):
            counts["pmos"] += 1
        elif kind.startswith("nmos"):
            counts["nmos"] += 1
        elif kind.startswith("pnp"):
            counts["pnp"] += 1
        elif kind in {"res", "resistor"} or kind.startswith("res"):
            counts["resistor"] += 1
        elif kind in {"cap", "capacitor", "mimcap"} or "cap" in kind:
            counts["capacitor"] += 1
    return counts


def _source_intent(source: Path, *, reference_manifest: object) -> object:
    from clay.layout.envelope_place import _extracted_intent_from_cell_def
    from clay.parser import parse_clay_file

    expected_inventory = reference_manifest.inventory.canonical_data()
    expected_ports = tuple(reference_manifest.ports)
    try:
        parsed = parse_clay_file(source)
    except (OSError, TypeError, ValueError) as exc:
        raise ReferenceQualityError(
            f"physical row exact source cannot be parsed: {exc}"
        ) from exc
    candidates = tuple(
        cell
        for cell in parsed.cell_defs
        if tuple(str(port) for port in tuple(getattr(cell, "ports", ()) or ()))
        == expected_ports
        and _inventory_for_cell(cell) == expected_inventory
    )
    if len(candidates) != 1:
        inventories = tuple(_inventory_for_cell(cell) for cell in parsed.cell_defs)
        raise ReferenceQualityError(
            "physical row exact source inventory/ports do not identify the unique "
            f"registered topology: expected={expected_inventory}/{expected_ports}, "
            f"observed_inventories={inventories}"
        )
    try:
        return _extracted_intent_from_cell_def(candidates[0])
    except (TypeError, ValueError) as exc:
        raise ReferenceQualityError(
            f"physical row exact source cannot produce fidelity intent: {exc}"
        ) from exc


def _require_unique_top(path: Path, *, label: str) -> None:
    try:
        import klayout.db as kdb

        layout = kdb.Layout()
        layout.read(str(path))
        tops = tuple(layout.top_cells())
    except Exception as exc:
        raise ReferenceQualityError(f"{label} cannot be opened: {exc}") from exc
    if len(tops) != 1:
        raise ReferenceQualityError(
            f"{label} requires exactly one top cell; found {len(tops)}"
        )


def _require_canonical_metrics(metrics: tuple[dict[str, object], ...]) -> None:
    names = tuple(str(item["name"]) for item in metrics)
    if len(metrics) != len(CANONICAL_FIDELITY_METRICS) or names != CANONICAL_FIDELITY_METRICS:
        raise ReferenceQualityError(
            "geometric fidelity metric inventory must equal the exact canonical "
            f"nine-name order: {CANONICAL_FIDELITY_METRICS}; observed={names}"
        )
    for item in metrics:
        status = item["status"]
        observable = item["observable"]
        value = item["value"]
        if status not in {"pass", "fail", "unmeasurable"}:
            raise ReferenceQualityError("geometric fidelity metric status is not closed")
        if status in {"pass", "fail"} and (observable is not True or value is None):
            raise ReferenceQualityError(
                "measured geometric fidelity metric status contradicts observability"
            )
        if status == "unmeasurable" and (observable is not False or value is not None):
            raise ReferenceQualityError(
                "unmeasurable geometric fidelity metric contradicts observability"
            )


def _metric_data(metric: object) -> dict[str, object]:
    satisfied = getattr(metric, "satisfied", None)
    status = (
        VerificationStatus.PASS
        if satisfied is True
        else VerificationStatus.FAIL
        if satisfied is False
        else VerificationStatus.UNMEASURABLE
    )
    return {
        "name": str(getattr(metric, "name")),
        "status": status.value,
        "observable": bool(getattr(metric, "observable", False)),
        "value": getattr(metric, "value", None),
        "expected": getattr(metric, "expected", None),
        "threshold": str(getattr(metric, "threshold", "")),
        "detail": str(getattr(metric, "detail", "")),
    }


def _unmeasurable_verdict(
    *,
    mode: str,
    exact_gds: ArtifactLocator,
    reference_manifest_locator: ArtifactLocator,
    blocker: str,
) -> DimensionVerdict:
    return DimensionVerdict(
        dimension="geometric_quality",
        status=VerificationStatus.UNMEASURABLE,
        evidence=(reference_manifest_locator, exact_gds),
        measurements={
            "mode": mode,
            "exact_gds_sha256": exact_gds.sha256,
            "metrics": [],
            "metric_status_counts": {},
        },
        thresholds={"digest_bound_exact_gds_measurement_required": True},
        blocker_reason=blocker,
    )


def geometric_quality_verdict(
    *,
    physical_row: object,
    reference_manifest: object,
    exact_gds: ArtifactLocator,
    reference_manifest_locator: ArtifactLocator,
    repo_root: str | Path,
) -> DimensionVerdict:
    """Measure canonical exact GDS bytes selected only by their physical row."""

    from clay.flow.portfolio_signoff import PortfolioSignoffRow
    from clay.flow.reference_evidence import ReferenceManifest
    from clay.flow.signoff_artifacts import SignoffArtifactError, verify_artifact_locator

    if not isinstance(physical_row, PortfolioSignoffRow):
        raise ReferenceQualityError("geometric quality requires a typed physical row")
    if not isinstance(reference_manifest, ReferenceManifest):
        raise ReferenceQualityError(
            "geometric quality requires a typed reference manifest"
        )
    if not isinstance(exact_gds, ArtifactLocator) or exact_gds.logical_id != "gds":
        raise ReferenceQualityError("geometric quality exact GDS locator is invalid")
    if (
        not isinstance(reference_manifest_locator, ArtifactLocator)
        or reference_manifest_locator.logical_id != "reference_manifest"
    ):
        raise ReferenceQualityError(
            "geometric quality reference manifest locator is invalid"
        )
    if (
        physical_row.circuit_id != reference_manifest.circuit_id
        or physical_row.topology_id != reference_manifest.topology_id
        or physical_row.normalized_connectivity_sha256
        != reference_manifest.normalized_connectivity_sha256
    ):
        raise ReferenceQualityError(
            "physical row identity differs from the registered reference manifest"
        )

    physical_gds = tuple(
        item for item in physical_row.artifacts if item.logical_id == "gds"
    )
    if len(physical_gds) != 1 or physical_gds[0].canonical_data() != exact_gds.canonical_data():
        raise ReferenceQualityError(
            "geometric quality exact GDS differs from the unique physical row GDS"
        )
    sources = tuple(
        item for item in physical_row.artifacts if item.logical_id == "source"
    )
    if len(sources) != 1:
        raise ReferenceQualityError(
            "geometric quality requires one unique physical row exact source"
        )
    source = sources[0]
    root = Path(repo_root).resolve()
    try:
        for locator in (source, exact_gds, reference_manifest_locator):
            verify_artifact_locator(root, locator)
    except (SignoffArtifactError, OSError, ValueError) as exc:
        raise ReferenceQualityError(
            f"geometric quality bound artifact is invalid: {exc}"
        ) from exc
    manifest_path = _repository_path(
        root,
        reference_manifest_locator.path,
        label="reference manifest locator",
    )
    _require_manifest_binding(
        manifest_path, reference_manifest=reference_manifest
    )

    reference_gds = tuple(
        item
        for item in reference_manifest.reference_artifacts
        if item.role == "layout_gds"
    )
    if not reference_gds:
        return _unmeasurable_verdict(
            mode="unmeasurable",
            exact_gds=exact_gds,
            reference_manifest_locator=reference_manifest_locator,
            blocker="registered reference manifest has no exact layout GDS",
        )
    if len(reference_gds) != 1 or reference_gds[0].repository_path is None:
        raise ReferenceQualityError(
            "registered reference manifest must bind one repository layout GDS"
        )
    if physical_row.pdk_id != reference_manifest.reference_pdk_id:
        return _unmeasurable_verdict(
            mode="spatial_advisory",
            exact_gds=exact_gds,
            reference_manifest_locator=reference_manifest_locator,
            blocker=(
                "reference layout is cross-PDK spatial evidence and cannot issue "
                "a direct geometric verdict"
            ),
        )
    required_scope = (
        reference_manifest.scope.floorplan,
        reference_manifest.scope.device_geometry,
        reference_manifest.scope.routing_geometry,
    )
    if any(status is not VerificationStatus.PASS for status in required_scope):
        return _unmeasurable_verdict(
            mode="spatial_advisory",
            exact_gds=exact_gds,
            reference_manifest_locator=reference_manifest_locator,
            blocker="registered same-PDK reference geometry scope is incomplete",
        )

    registered_gds = reference_gds[0]
    reference_path = _repository_path(
        root, registered_gds.repository_path, label="registered reference GDS"
    )
    if _file_sha256(reference_path, label="registered reference GDS") != registered_gds.sha256:
        raise ReferenceQualityError("registered reference GDS digest differs")
    source_path = _repository_path(root, source.path, label="physical row source")
    exact_path = _repository_path(root, exact_gds.path, label="physical row exact GDS")
    _require_unique_top(exact_path, label="physical row exact GDS top-cell binding")
    _require_unique_top(reference_path, label="registered reference GDS top-cell binding")
    intent = _source_intent(source_path, reference_manifest=reference_manifest)
    try:
        from clay.layout.spec.geometric_fidelity import (
            measure_fidelity,
            measure_principal_physical_body_area_um2,
        )

        report = measure_fidelity(str(exact_path), intent)
        area_um2 = measure_principal_physical_body_area_um2(str(exact_path))
        reference_area_um2 = measure_principal_physical_body_area_um2(
            str(reference_path)
        )
    except Exception as exc:
        raise ReferenceQualityError(
            f"canonical exact-GDS fidelity measurement failed: {exc}"
        ) from exc
    if reference_area_um2 <= 0.0:
        raise ReferenceQualityError(
            "registered reference GDS has non-positive bbox area"
        )
    area_ratio = area_um2 / reference_area_um2
    max_area_ratio = 1.25
    area_ratio_failed = area_ratio > max_area_ratio
    metrics = tuple(_metric_data(metric) for metric in report.metrics)
    _require_canonical_metrics(metrics)
    status_counts = Counter(str(metric["status"]) for metric in metrics)
    if area_ratio_failed or status_counts[VerificationStatus.FAIL.value]:
        status = VerificationStatus.FAIL
    elif status_counts[VerificationStatus.UNMEASURABLE.value]:
        status = VerificationStatus.UNMEASURABLE
    else:
        status = VerificationStatus.PASS
    counts = {
        name: status_counts[name]
        for name in (
            VerificationStatus.FAIL.value,
            VerificationStatus.PASS.value,
            VerificationStatus.UNMEASURABLE.value,
        )
        if status_counts[name]
    }
    reference_locator = ArtifactLocator(
        logical_id="reference_gds",
        path=registered_gds.repository_path,
        sha256=registered_gds.sha256,
        required=False,
    )
    blocker_parts: list[str] = []
    if area_ratio_failed:
        blocker_parts.append(
            f"area_ratio={area_ratio:.6g} exceeds {max_area_ratio}"
        )
    if (
        status_counts[VerificationStatus.FAIL.value]
        or status_counts[VerificationStatus.UNMEASURABLE.value]
    ):
        blocker_parts.append(
            "canonical exact-GDS fidelity metrics are not all pass: "
            + ", ".join(f"{name}={count}" for name, count in counts.items())
        )
    blocker = "; ".join(blocker_parts) or None
    return DimensionVerdict(
        dimension="geometric_quality",
        status=status,
        evidence=(exact_gds, reference_locator, reference_manifest_locator, source),
        measurements={
            "mode": "direct_reference",
            "source_sha256": source.sha256,
            "exact_gds_sha256": exact_gds.sha256,
            "reference_gds_sha256": registered_gds.sha256,
            "area_um2": area_um2,
            "reference_bbox_area_um2": reference_area_um2,
            "area_ratio": area_ratio,
            "metric_status_counts": counts,
            "metrics": list(metrics),
        },
        thresholds={
            "all_fidelity_metrics_required": True,
            "unmeasurable_blocks": True,
            "area_ratio_comparator": "<=",
            "max_area_ratio": max_area_ratio,
            "metric_thresholds": {
                str(metric["name"]): str(metric["threshold"])
                for metric in metrics
            },
        },
        blocker_reason=blocker,
    )


def human_gap_markdown(vector: object) -> str:
    """Render the typed vector as deterministic text without image authority."""

    from clay.flow.reference_convergence import HumanGapVector

    if not isinstance(vector, HumanGapVector):
        raise ReferenceQualityError("human-gap Markdown requires a typed vector")
    lines = [
        "# Human gap vector",
        "",
        f"- PDK: `{vector.pdk_id}`",
        f"- Topology: `{vector.binding.topology_id}`",
        f"- Source SHA-256: `{vector.binding.source_sha256}`",
        (
            "- Reference scope SHA-256: "
            f"`{vector.binding.reference_scope_sha256}`"
        ),
        (
            "- Generated GDS SHA-256: "
            f"`{vector.binding.generated_gds_sha256}`"
        ),
        "",
        "| Dimension | Observation | Authority | Level | Status |",
        "|---|---|---|---|---|",
    ]
    for dimension in vector.dimensions:
        for observation in dimension.observations:
            lines.append(
                "| "
                + " | ".join(
                    (
                        dimension.dimension,
                        observation.name,
                        observation.authority.value,
                        observation.level.value,
                        observation.status.value,
                    )
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


__all__ = [
    "ReferenceQualityError",
    "geometric_quality_verdict",
    "human_gap_markdown",
]
