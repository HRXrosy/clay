from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class VerificationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    PENDING = "pending"
    UNSUPPORTED = "unsupported"
    NO_CONVERGENCE = "no_convergence"
    ERROR = "error"
    UNKNOWN = "unknown"
    # The architecture's explicit-incompleteness states: a capability that cannot be
    # measured on this artifact/toolchain (distinct from UNSUPPORTED = not implemented),
    # and an informational advisory that never blocks. Added so a checker NEVER silently
    # skips -- it returns one of pass/fail/unmeasurable/unsupported/advisory.
    UNMEASURABLE = "unmeasurable"
    ADVISORY = "advisory"
    # The priority axis' explicit-incompleteness state: the obligation SET is
    # self-contradictory -- two obligations cannot both be satisfied and there is
    # no declared superiority + equal specificity to break the tie. Distinct from
    # FAIL (no geometry is violated), UNMEASURABLE (it was measured), and
    # UNSUPPORTED (the capability is present). Emitted only by the priority
    # resolver; it ALWAYS blocks signoff (no-silent-priority, the dual of
    # no-silent-pass). A pre-existing checker never returns it, so it is purely
    # additive: no existing design's verdict changes.
    CONFLICT = "conflict"


@dataclass(frozen=True)
class DesignDomain:
    id: str
    description: str = ""


@dataclass(frozen=True)
class SourceObligation:
    id: str
    observation_id: str
    observation_kind: str
    unit: str
    required: bool = True


@dataclass(frozen=True)
class ObservationSignature:
    observation_id: str
    observation_kind: str
    unit: str


@dataclass(frozen=True)
class IntentDecl:
    id: str
    domain: DesignDomain
    source_obligations: tuple[SourceObligation, ...] = field(default_factory=tuple)
    observation_basis: tuple[ObservationSignature, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ObservationRow:
    source_obligation_id: str
    observation_id: str
    observation_kind: str
    unit: str
    status: VerificationStatus
    required: bool = True


@dataclass(frozen=True)
class RepairStepEvidence:
    id: str
    source_obligation_id: str = ""
    description: str = ""


@dataclass(frozen=True)
class ContractCertificate:
    intent_id: str
    domain: DesignDomain
    source_obligations: tuple[SourceObligation, ...] = field(default_factory=tuple)
    observation_basis: tuple[ObservationSignature, ...] = field(default_factory=tuple)
    observations: tuple[ObservationRow, ...] = field(default_factory=tuple)
    repair_steps: tuple[RepairStepEvidence, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ContractCertificateBundle:
    certificates: tuple[ContractCertificate, ...] = field(default_factory=tuple)


def lower_intent(intent: IntentDecl) -> ContractCertificate:
    return ContractCertificate(
        intent_id=intent.id,
        domain=intent.domain,
        source_obligations=tuple(intent.source_obligations),
        observation_basis=tuple(intent.observation_basis),
    )


def row_certifies_obligation(
    row: ObservationRow,
    obligation: SourceObligation,
    basis: tuple[ObservationSignature, ...],
) -> bool:
    if row.status != VerificationStatus.PASS:
        return False
    if not row.required:
        return False
    if row.source_obligation_id != obligation.id:
        return False
    if row.observation_id != obligation.observation_id:
        return False
    if row.observation_kind != obligation.observation_kind:
        return False
    if row.unit != obligation.unit:
        return False

    return any(
        signature.observation_id == obligation.observation_id
        and signature.observation_kind == obligation.observation_kind
        and signature.unit == obligation.unit
        for signature in basis
    )


def required_obligation_covered(
    cert: ContractCertificate, obligation: SourceObligation
) -> bool:
    return any(
        row_certifies_obligation(row, obligation, cert.observation_basis)
        for row in cert.observations
    )


def valid_certificate(cert: ContractCertificate) -> bool:
    return all(
        required_obligation_covered(cert, obligation)
        for obligation in cert.source_obligations
        if obligation.required
    )


def valid_certificate_bundle(bundle: ContractCertificateBundle) -> bool:
    if not all(valid_certificate(cert) for cert in bundle.certificates):
        return False

    seen: set[tuple[str, str, str]] = set()
    for cert in bundle.certificates:
        for obligation in cert.source_obligations:
            identity = (cert.intent_id, cert.domain.id, obligation.id)
            if identity in seen:
                return False
            seen.add(identity)
    return True
