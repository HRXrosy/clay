from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from clay.core_contract import (
    ContractCertificate,
    ContractCertificateBundle,
    DesignDomain,
    VerificationStatus,
    valid_certificate,
    valid_certificate_bundle,
)
from clay.core_multisetup_result_bridge import certificate_from_multisetup_result
from clay.core_observation_extract import ObservationMetadata
from clay.core_verify_bridge import certificate_from_verify_result
from clay.core_verify_metadata import metadata_from_verify_result


@dataclass
class CoreCertificatePreview:
    status: str
    reason: str
    certificate: ContractCertificate | ContractCertificateBundle | None
    source_obligation_ids: tuple[str, ...]
    row_count: int


def preview_from_verify_result(
    verify_result,
    *,
    metadata: Iterable[ObservationMetadata] | None = None,
    units: dict[str, str] | None = None,
    kinds: dict[str, str] | None = None,
) -> CoreCertificatePreview:
    try:
        if metadata is None:
            metadata = metadata_from_verify_result(
                verify_result,
                units=units,
                kinds=kinds,
            )
        certificate = certificate_from_verify_result(verify_result, tuple(metadata))
    except ValueError as exc:
        return _rejected_without_certificate(str(exc))

    return _preview_from_certificate(certificate)


def preview_from_verify_results(
    verify_results: Iterable,
    *,
    metadata: Iterable[ObservationMetadata] | None = None,
    units: dict[str, str] | None = None,
    kinds: dict[str, str] | None = None,
) -> CoreCertificatePreview:
    results = tuple(verify_results)
    if not results:
        return _rejected_without_certificate("no VerifyResult objects to preview")

    try:
        metadata_items = (
            _merged_metadata_from_verify_results(
                results,
                units=units,
                kinds=kinds,
            )
            if metadata is None
            else _dedup_metadata(tuple(metadata))
        )
        certificates = tuple(
            certificate_from_verify_result(result, metadata_items)
            for result in results
        )
        certificate = _merge_sweep_certificates(certificates)
    except ValueError as exc:
        return _rejected_without_certificate(str(exc))

    return _preview_from_certificate(certificate)


def preview_from_multisetup_result(
    result,
    *,
    metadata: Iterable[ObservationMetadata] | None = None,
    units: dict[str, str] | None = None,
    kinds: dict[str, str] | None = None,
) -> CoreCertificatePreview:
    try:
        if metadata is None:
            metadata = _merged_metadata_from_verify_results(
                tuple(_iter_setup_verify_results(result)),
                units=units,
                kinds=kinds,
            )
        certificate = certificate_from_multisetup_result(result, tuple(metadata))
    except ValueError as exc:
        return _rejected_without_certificate(str(exc))

    return _preview_from_certificate(certificate)


def preview_from_certificate_previews(
    previews: Iterable[CoreCertificatePreview],
) -> CoreCertificatePreview:
    preview_items = tuple(previews)
    if not preview_items:
        return _rejected_without_certificate("no Core previews to bundle")

    certificates = []
    for preview in preview_items:
        if preview.status != "accepted":
            return _rejected_without_certificate(
                f"member preview rejected: {preview.reason}"
            )
        if preview.certificate is None:
            return _rejected_without_certificate(
                "member preview accepted without certificate"
            )
        if isinstance(preview.certificate, ContractCertificateBundle):
            certificates.extend(preview.certificate.certificates)
        else:
            certificates.append(preview.certificate)

    bundle = ContractCertificateBundle(certificates=tuple(certificates))
    is_valid = valid_certificate_bundle(bundle)
    return CoreCertificatePreview(
        status="accepted" if is_valid else "rejected",
        reason=(
            "Core certificate bundle valid"
            if is_valid
            else "Core certificate bundle invalid"
        ),
        certificate=bundle,
        source_obligation_ids=tuple(
            obligation.id
            for certificate in bundle.certificates
            for obligation in certificate.source_obligations
        ),
        row_count=sum(
            len(certificate.observations) for certificate in bundle.certificates
        ),
    )


def _preview_from_certificate(
    certificate: ContractCertificate,
) -> CoreCertificatePreview:
    is_valid = valid_certificate(certificate) and _all_required_rows_pass(certificate)
    return CoreCertificatePreview(
        status="accepted" if is_valid else "rejected",
        reason="Core certificate valid" if is_valid else "Core certificate invalid",
        certificate=certificate,
        source_obligation_ids=tuple(
            obligation.id for obligation in certificate.source_obligations
        ),
        row_count=len(certificate.observations),
    )


def _merged_metadata_from_verify_results(
    verify_results: tuple,
    *,
    units: dict[str, str] | None,
    kinds: dict[str, str] | None,
) -> tuple[ObservationMetadata, ...]:
    return _dedup_metadata(
        item
        for result in verify_results
        for item in metadata_from_verify_result(result, units=units, kinds=kinds)
    )


def _iter_setup_verify_results(result):
    if hasattr(result, "per_setup"):
        yield from tuple(result.per_setup.values())
        return

    for setup_result in tuple(result.setup_results):
        yield setup_result.verify_result


def _dedup_metadata(
    metadata: Iterable[ObservationMetadata],
) -> tuple[ObservationMetadata, ...]:
    by_source_id: dict[str, ObservationMetadata] = {}
    for item in metadata:
        existing = by_source_id.get(item.source_obligation_id)
        if existing is None:
            by_source_id[item.source_obligation_id] = item
            continue
        if existing.observation_kind != item.observation_kind or existing.unit != item.unit:
            raise ValueError(
                f"conflicting metadata for {item.source_obligation_id}: "
                f"{existing.observation_kind}/{existing.unit} != "
                f"{item.observation_kind}/{item.unit}"
            )
    return tuple(by_source_id.values())


def _merge_sweep_certificates(
    certificates: tuple[ContractCertificate, ...],
) -> ContractCertificate:
    if not certificates:
        raise ValueError("no certificates to merge")

    source_obligations = []
    observation_basis = []
    observations = []
    seen_sources = {}
    seen_basis = set()

    for certificate in certificates:
        for source in certificate.source_obligations:
            existing = seen_sources.get(source.id)
            if existing is None:
                seen_sources[source.id] = source
                source_obligations.append(source)
            elif existing != source:
                raise ValueError(f"conflicting source obligation id: {source.id}")

        for signature in certificate.observation_basis:
            key = (
                signature.observation_id,
                signature.observation_kind,
                signature.unit,
            )
            if key not in seen_basis:
                seen_basis.add(key)
                observation_basis.append(signature)

        observations.extend(certificate.observations)

    first = certificates[0]
    return ContractCertificate(
        intent_id=first.intent_id,
        domain=DesignDomain(id=first.domain.id, description=first.domain.description),
        source_obligations=tuple(source_obligations),
        observation_basis=tuple(observation_basis),
        observations=tuple(observations),
    )


def _all_required_rows_pass(certificate: ContractCertificate) -> bool:
    return all(
        row.status == VerificationStatus.PASS
        for row in certificate.observations
        if row.required
    )


def _rejected_without_certificate(reason: str) -> CoreCertificatePreview:
    return CoreCertificatePreview(
        status="rejected",
        reason=reason,
        certificate=None,
        source_obligation_ids=(),
        row_count=0,
    )
