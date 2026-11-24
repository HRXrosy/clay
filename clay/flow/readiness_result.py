"""Public preflight result vocabulary; never a physical admission token."""
from dataclasses import asdict, dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: str
    code: str
    message: str
    locations: tuple[dict, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntentReadiness:
    requested_until: str
    checks: tuple[ReadinessCheck, ...]
    design: dict[str, Any] | None = None
    identity: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return bool(self.checks) and all(row.status == "pass" for row in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps({
            "schema": "clay.intent-readiness/v1", "status": "ready" if self.ready else "blocked",
            "requested_until": self.requested_until, "physical_admission": False,
            "scope": "preflight_only", "design": self.design, "identity": self.identity,
            "checks": [asdict(row) for row in self.checks],
        }))
