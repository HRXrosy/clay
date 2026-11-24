"""Calibrated, exact-GDS-bound electrical budget obligations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable

from clay.core_contract import VerificationStatus
from clay.layout.intent import (
    LayoutIntent,
    LayoutObligation,
    PowerBudgetIntent,
    SensitiveNetBudgetIntent,
    replay_power_budget_inventory,
    replay_sensitive_net_budget_inventory,
)
from clay.layout.intent_obligations import resolve_distributed_net_members
from .base import PexReport
from .magic_parser import parse_ext2spice


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SCHEMA = "clay.calibrated-pex-receipt/v1"


class ElectricalBudgetEvidenceError(ValueError):
    """The PEX bytes do not replay the claimed exact electrical evidence."""


def _sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ElectricalBudgetEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _stable_bytes(path: str | Path) -> bytes:
    source = Path(path).resolve(strict=True)
    first = source.read_bytes()
    second = source.read_bytes()
    if first != second:
        raise ElectricalBudgetEvidenceError(f"artifact changed while reading: {source}")
    if not first:
        raise ElectricalBudgetEvidenceError(f"artifact is empty: {source}")
    return first


def _report_data(report: PexReport) -> dict[str, object]:
    if type(report) is not PexReport:
        raise TypeError("electrical evidence requires an exact PexReport")
    caps = []
    for item in report.caps:
        value = item.cap_aF
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise ElectricalBudgetEvidenceError("PEX capacitance is not finite")
        caps.append((item.a, item.b, value))
    resistors = []
    for item in report.resistors:
        value = item.res_ohm
        if type(value) is not float or not math.isfinite(value) or value <= 0.0:
            raise ElectricalBudgetEvidenceError("PEX resistance is not positive finite")
        resistors.append((item.a, item.b, value))
    return {
        "backend": report.backend,
        "calibrated": report.calibrated,
        "caps": sorted(caps),
        "cell_name": report.cell_name,
        "connectivity_source": report.connectivity_source,
        "constants_source": report.constants_source,
        "field_solver": report.field_solver,
        "resistors": sorted(resistors),
    }


def pex_report_sha256(report: PexReport) -> str:
    return hashlib.sha256(
        json.dumps(
            _report_data(report),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CalibratedPexReceipt:
    """Content binding for one Magic RC artifact and its exact input GDS."""

    pdk_id: str
    pdk_source_snapshot_sha256: str
    cell_name: str
    exact_gds_sha256: str
    pex_sha256: str
    pex_report_sha256: str
    calibrated: bool
    resistance_extracted: bool
    backend: str = "magic"
    schema: str = _SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA or self.backend != "magic":
            raise ElectricalBudgetEvidenceError("PEX receipt scope is unsupported")
        if type(self.pdk_id) is not str or not self.pdk_id:
            raise TypeError("PEX receipt pdk_id must be a non-empty string")
        if type(self.cell_name) is not str or not self.cell_name:
            raise TypeError("PEX receipt cell_name must be a non-empty string")
        _sha256(self.exact_gds_sha256, "PEX receipt GDS digest")
        _sha256(
            self.pdk_source_snapshot_sha256, "PEX receipt PDK source snapshot"
        )
        _sha256(self.pex_sha256, "PEX receipt artifact digest")
        _sha256(self.pex_report_sha256, "PEX receipt report digest")
        if type(self.calibrated) is not bool or type(self.resistance_extracted) is not bool:
            raise TypeError("PEX receipt calibration flags must be exact booleans")

    def canonical_data(self) -> dict[str, object]:
        receipt = replay_calibrated_pex_receipt(self)
        return {
            "schema": receipt.schema,
            "backend": receipt.backend,
            "pdk_id": receipt.pdk_id,
            "pdk_source_snapshot_sha256": receipt.pdk_source_snapshot_sha256,
            "cell_name": receipt.cell_name,
            "exact_gds_sha256": receipt.exact_gds_sha256,
            "pex_sha256": receipt.pex_sha256,
            "pex_report_sha256": receipt.pex_report_sha256,
            "calibrated": receipt.calibrated,
            "resistance_extracted": receipt.resistance_extracted,
        }


def replay_calibrated_pex_receipt(value: object) -> CalibratedPexReceipt:
    if type(value) is not CalibratedPexReceipt:
        raise TypeError("electrical budget receipt must be CalibratedPexReceipt")
    return CalibratedPexReceipt(
        pdk_id=value.pdk_id,
        pdk_source_snapshot_sha256=value.pdk_source_snapshot_sha256,
        cell_name=value.cell_name,
        exact_gds_sha256=value.exact_gds_sha256,
        pex_sha256=value.pex_sha256,
        pex_report_sha256=value.pex_report_sha256,
        calibrated=value.calibrated,
        resistance_extracted=value.resistance_extracted,
        backend=value.backend,
        schema=value.schema,
    )


def write_calibrated_pex_receipt(
    path: str | Path, receipt: CalibratedPexReceipt
) -> Path:
    """Atomically persist the canonical receipt emitted by active extraction."""

    payload = json.dumps(
        replay_calibrated_pex_receipt(receipt).canonical_data(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(payload, encoding="ascii", newline="\n")
    temporary.replace(target)
    return target


def electrical_budget_pex_path(exact_artifact: object) -> Path:
    """Return the isolated raw-PEX locator for one exact artifact."""

    top_cell = str(exact_artifact.scope.top_cell)
    exact_sha256 = str(exact_artifact.truth_manifest.exact_gds_sha256)
    return (
        Path(exact_artifact.exact_gds_path).parent
        / "electrical_budget_pex"
        / exact_sha256
        / f"{top_cell}.pex.spice"
    )


def _capture_calibrated_pex_receipt(
    *,
    exact_gds_path: str | Path,
    expected_exact_gds_sha256: str,
    pex_path: str | Path,
    cell_name: str,
    pdk_id: str,
    pdk_source_snapshot_sha256: str,
    calibrated: bool,
) -> tuple[CalibratedPexReceipt, PexReport]:
    """Reparse one extractor output; this helper never grants promotion alone."""

    expected = _sha256(expected_exact_gds_sha256, "expected exact GDS digest")
    measured_gds = hashlib.sha256(_stable_bytes(exact_gds_path)).hexdigest()
    if measured_gds != expected:
        raise ElectricalBudgetEvidenceError("exact GDS digest changed")
    pex_bytes = _stable_bytes(pex_path)
    try:
        text = pex_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ElectricalBudgetEvidenceError("PEX artifact is not UTF-8 SPICE") from exc
    if type(calibrated) is not bool:
        raise TypeError("extractor calibration state must be an exact boolean")
    report = parse_ext2spice(
        text, cell_name=cell_name, pdk=pdk_id, calibrated=calibrated
    )
    report_digest = pex_report_sha256(report)
    return (
        CalibratedPexReceipt(
            pdk_id=pdk_id,
            pdk_source_snapshot_sha256=pdk_source_snapshot_sha256,
            cell_name=cell_name,
            exact_gds_sha256=expected,
            pex_sha256=hashlib.sha256(pex_bytes).hexdigest(),
            pex_report_sha256=report_digest,
            calibrated=report.calibrated,
            resistance_extracted=bool(report.resistors),
        ),
        report,
    )


def _blocked(
    kind: str, net: str, status: VerificationStatus, reason: str
) -> LayoutObligation:
    return LayoutObligation(
        kind=kind,
        pair=(net, net),
        measured_imbalance=None,
        rel_tol=1.0,
        missing_nets=(reason,),
        verification_status=status,
    )


def _declared_rows(
    power: Iterable[PowerBudgetIntent],
    sensitive: Iterable[SensitiveNetBudgetIntent],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        [("power_ir_drop", item.net) for item in power]
        + [
            pair
            for item in sensitive
            for pair in (
                ("sensitive_net_resistance", item.net),
                ("sensitive_net_capacitance", item.net),
            )
        ]
    )


def _net_measurements(
    report: PexReport, net: str
) -> tuple[float | None, float | None]:
    members = frozenset(resolve_distributed_net_members(report, net))
    if not members:
        return None, None
    resistors = tuple(
        item for item in report.resistors if item.a in members and item.b in members
    )
    if any(
        (item.a in members) != (item.b in members) for item in report.resistors
    ):
        return None, None
    resistance = None if not resistors else sum(item.res_ohm for item in resistors)
    caps = tuple(
        item
        for item in report.caps
        if (item.a in members) != (item.b in members)
    )
    capacitance_f = None if not caps else sum(item.cap_aF for item in caps) * 1e-18
    return resistance, capacitance_f


def _intent_inventory(
    intent: LayoutIntent,
) -> tuple[
    tuple[PowerBudgetIntent, ...], tuple[SensitiveNetBudgetIntent, ...]
]:
    if type(intent) is not LayoutIntent:
        raise TypeError("electrical budget checker requires exact LayoutIntent")
    stored = vars(intent)
    if not {"power_budgets", "sensitive_net_budgets"}.issubset(stored):
        raise ElectricalBudgetEvidenceError("electrical budget inventory is missing")
    power = replay_power_budget_inventory(stored["power_budgets"])
    sensitive = replay_sensitive_net_budget_inventory(
        stored["sensitive_net_budgets"]
    )
    return power, sensitive


def declared_electrical_budget_rows(
    intent: LayoutIntent,
) -> tuple[tuple[str, str], ...]:
    """Replay the exact obligation denominator declared by one intent."""

    power, sensitive = _intent_inventory(intent)
    return _declared_rows(power, sensitive)


def replay_electrical_budget_obligations(
    intent: LayoutIntent, obligations: object
) -> tuple[LayoutObligation, ...]:
    """Reject partial, extra, duplicate, or mismatched verifier output rows."""

    if type(obligations) is not tuple or any(
        type(item) is not LayoutObligation for item in obligations
    ):
        raise TypeError("electrical budget obligations must be an exact typed tuple")
    expected = declared_electrical_budget_rows(intent)
    observed = tuple(
        (item.kind, item.pair[0])
        for item in obligations
        if type(item.pair) is tuple
        and len(item.pair) == 2
        and item.pair[0] == item.pair[1]
    )
    if observed != expected or len(observed) != len(obligations):
        raise ElectricalBudgetEvidenceError(
            "electrical budget obligation denominator does not replay"
        )
    return obligations


def _measure_electrical_budgets(
    power: tuple[PowerBudgetIntent, ...],
    sensitive: tuple[SensitiveNetBudgetIntent, ...],
    report: PexReport,
) -> tuple[LayoutObligation, ...]:

    obligations: list[LayoutObligation] = []
    measurements = {
        net: _net_measurements(report, net)
        for net in {item.net for item in power + sensitive}
    }
    for item in power:
        resistance, _capacitance = measurements[item.net]
        if resistance is None:
            obligations.append(
                _blocked(
                    "power_ir_drop",
                    item.net,
                    VerificationStatus.UNMEASURABLE,
                    "distributed_resistance_missing",
                )
            )
            continue
        observed_drop = item.current_a * resistance
        obligations.append(
            LayoutObligation(
                "power_ir_drop",
                (item.net, item.net),
                observed_drop / item.max_drop_v,
                1.0,
                measurements={
                    "observed_drop_v": observed_drop,
                    "observed_resistance_ohm": resistance,
                },
            )
        )
    for item in sensitive:
        resistance, capacitance = measurements[item.net]
        obligations.append(
            _blocked(
                "sensitive_net_resistance",
                item.net,
                VerificationStatus.UNMEASURABLE,
                "distributed_resistance_missing",
            )
            if resistance is None
            else LayoutObligation(
                "sensitive_net_resistance",
                (item.net, item.net),
                resistance / item.max_resistance_ohm,
                1.0,
                measurements={"observed_resistance_ohm": resistance},
            )
        )
        obligations.append(
            _blocked(
                "sensitive_net_capacitance",
                item.net,
                VerificationStatus.UNMEASURABLE,
                "distributed_capacitance_missing",
            )
            if capacitance is None
            else LayoutObligation(
                "sensitive_net_capacitance",
                (item.net, item.net),
                capacitance / item.max_capacitance_f,
                1.0,
                measurements={"observed_capacitance_f": capacitance},
            )
        )
    return tuple(obligations)


def verify_electrical_budgets_on_exact_artifact(
    intent: LayoutIntent,
    *,
    exact_artifact: object,
    pdk_id: str,
) -> tuple[tuple[LayoutObligation, ...], CalibratedPexReceipt | None]:
    """Run Magic on the exact emitted GDS and verify all electrical budgets.

    This is the promotion boundary.  It does not accept a caller-minted receipt
    or a pre-existing PEX file: the checker validates the emitter authority,
    launches the calibrated extractor itself, reparses the emitted SPICE bytes,
    and compares that replay with the backend report before measuring anything.
    """

    power, sensitive = _intent_inventory(intent)
    rows = declared_electrical_budget_rows(intent)
    if not rows:
        return (), None
    from clay.layout.route.exact_artifact_evidence import ExactEmittedArtifact

    if type(exact_artifact) is not ExactEmittedArtifact:
        raise TypeError("electrical budget verification requires ExactEmittedArtifact")
    if type(pdk_id) is not str or not pdk_id:
        raise TypeError("electrical budget verification requires a PDK identifier")
    try:
        exact_artifact.validate()
        from clay.pdk import exact_source_identity

        pdk_identity = exact_source_identity(pdk_id)
        exact_artifact.truth_manifest.bind_pdk_identity(pdk_identity)
        snapshot_sha256 = pdk_identity["source_snapshot_sha256"]
        if snapshot_sha256 is None:
            return (
                tuple(
                    _blocked(
                        kind,
                        net,
                        VerificationStatus.UNSUPPORTED,
                        "pdk_source_snapshot_unavailable",
                    )
                    for kind, net in rows
                ),
                None,
            )
        top_cell = exact_artifact.scope.top_cell
        exact_gds_path = exact_artifact.exact_gds_path
        expected_gds_sha256 = exact_artifact.truth_manifest.exact_gds_sha256
        exact_gds_bytes = _stable_bytes(exact_gds_path)
        before = hashlib.sha256(exact_gds_bytes).hexdigest()
        if before != expected_gds_sha256:
            raise ElectricalBudgetEvidenceError("exact artifact GDS digest changed")

        from .magic_backend import MagicPexBackend

        backend = MagicPexBackend()
        if not backend.available() or not backend.supports_pdk(pdk_id):
            return (
                tuple(
                    _blocked(
                        kind,
                        net,
                        VerificationStatus.UNSUPPORTED,
                        "calibrated_extractor_unavailable",
                    )
                    for kind, net in rows
                ),
                None,
            )
        pex_path = electrical_budget_pex_path(exact_artifact)
        extraction_gds = pex_path.parent / Path(exact_gds_path).name
        pex_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_gds = extraction_gds.with_name(extraction_gds.name + ".tmp")
        temporary_gds.write_bytes(exact_gds_bytes)
        temporary_gds.replace(extraction_gds)
        backend_report = backend.extract(
            str(extraction_gds), top_cell, pdk_id, with_resistance=True
        )
        receipt, parsed_report = _capture_calibrated_pex_receipt(
            exact_gds_path=extraction_gds,
            expected_exact_gds_sha256=expected_gds_sha256,
            pex_path=pex_path,
            cell_name=top_cell,
            pdk_id=pdk_id,
            pdk_source_snapshot_sha256=snapshot_sha256,
            calibrated=backend_report.calibrated,
        )
        if pex_report_sha256(backend_report) != pex_report_sha256(parsed_report):
            raise ElectricalBudgetEvidenceError(
                "extractor report differs from emitted PEX bytes"
            )
        confirmed_pdk_identity = exact_source_identity(pdk_id)
        if confirmed_pdk_identity != pdk_identity:
            raise ElectricalBudgetEvidenceError(
                "PDK source snapshot changed during extraction"
            )
        exact_artifact.truth_manifest.bind_pdk_identity(confirmed_pdk_identity)
        exact_artifact.validate()
        if not receipt.calibrated or not backend_report.calibrated:
            return (
                tuple(
                    _blocked(
                        kind,
                        net,
                        VerificationStatus.UNSUPPORTED,
                        "extractor_uncalibrated",
                    )
                    for kind, net in rows
                ),
                receipt,
            )
        return _measure_electrical_budgets(power, sensitive, parsed_report), receipt
    except Exception as exc:  # noqa: BLE001 - named fail-closed evidence boundary
        return (
            tuple(
                _blocked(
                    kind,
                    net,
                    VerificationStatus.UNMEASURABLE,
                    f"pex_evidence_mismatch:{type(exc).__name__}",
                )
                for kind, net in rows
            ),
            None,
        )


__all__ = [
    "CalibratedPexReceipt",
    "ElectricalBudgetEvidenceError",
    "declared_electrical_budget_rows",
    "electrical_budget_pex_path",
    "pex_report_sha256",
    "replay_calibrated_pex_receipt",
    "replay_electrical_budget_obligations",
    "verify_electrical_budgets_on_exact_artifact",
    "write_calibrated_pex_receipt",
]
