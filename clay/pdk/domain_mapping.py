"""Fail-honest semantic MOS voltage-domain mapping through PDK adapters.

Clay device classes carry a process-independent thin/thick obligation.  Concrete
model names and the domains they implement are PDK facts.  This module joins the
two without circuit, instance-name, or PDK-name branches and returns auditable
evidence rather than silently treating a model alias as semantic preservation.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

from clay.pdk import PdkAdapter, get_pdk


_SEMANTIC_DOMAINS = {
    "nmos": "thin",
    "pmos": "thin",
    "nmos3v": "thick",
    "pmos3v": "thick",
}


@dataclass(frozen=True)
class DeviceDomainMapping:
    semantic_class: str
    source_domain: str
    target_model: str
    target_domain: str
    status: str
    reason: str
    count: int = 1

    def canonical_data(self) -> dict[str, object]:
        return {
            "count": self.count,
            "reason": self.reason,
            "semantic_class": self.semantic_class,
            "source_domain": self.source_domain,
            "status": self.status,
            "target_domain": self.target_domain,
            "target_model": self.target_model,
        }


def map_device_domains(
    device_classes: Iterable[str],
    adapter_or_id: PdkAdapter | str,
) -> tuple[DeviceDomainMapping, ...]:
    """Map semantic MOS classes to one adapter with explicit domain evidence.

    Adapter ``extra['device_semantic_domains']`` maps each semantic class to the
    actual domain implemented by its target model.  Missing model or metadata is
    ``unsupported``.  A deliberate domain change is named ``retarget``; callers
    decide whether that status is allowed and must never promote it as ``pass``.
    """

    adapter = (
        adapter_or_id
        if isinstance(adapter_or_id, PdkAdapter)
        else get_pdk(str(adapter_or_id))
    )
    counts = Counter(str(item).strip().casefold() for item in device_classes)
    raw_domains = adapter.extra.get("device_semantic_domains", {})
    target_domains: Mapping[str, object] = (
        raw_domains if isinstance(raw_domains, Mapping) else {}
    )
    rows: list[DeviceDomainMapping] = []
    for semantic_class in sorted(counts):
        source_domain = _SEMANTIC_DOMAINS.get(semantic_class, "")
        target_model = str(adapter.device_bindings.get(semantic_class, "") or "")
        target_domain = str(target_domains.get(semantic_class, "") or "").casefold()
        if not source_domain:
            status = "unsupported"
            reason = f"{semantic_class!r} is not a semantic MOS class"
        elif not target_model:
            status = "unsupported"
            reason = f"target model is missing for {semantic_class}"
        elif target_domain not in {"thin", "thick"}:
            status = "unsupported"
            reason = f"target-domain metadata is missing for {semantic_class}"
        elif target_domain == source_domain:
            status = "pass"
            reason = "semantic voltage domain preserved"
        else:
            status = "retarget"
            reason = (
                "voltage_domain_retarget: "
                f"{source_domain} -> {target_domain} through adapter capability"
            )
        rows.append(
            DeviceDomainMapping(
                semantic_class=semantic_class,
                source_domain=source_domain,
                target_model=target_model,
                target_domain=target_domain,
                status=status,
                reason=reason,
                count=counts[semantic_class],
            )
        )
    return tuple(rows)


__all__ = ["DeviceDomainMapping", "map_device_domains"]
