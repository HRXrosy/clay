"""Intent Implementation REGISTRY -- keyed on intent KIND, circuit names HARD-banned.

The single lookup surface for the registered :class:`~clay.layout.intent_impl.
IntentImplementation` set (spec §3.2). Three enforcement layers keep the
architecture's central law -- the engine never learns a circuit -- true at the
registration boundary (enforcement > discipline):

1. :func:`register` REJECTS a key that is not a declarative layout-intent kind, and a
   key that LOOKS LIKE a circuit / device / cell name ('ota' / 'comparator' / 'ldo' /
   a '5t'-style token). A per-circuit implementation cannot even be registered.
2. a ``check-known-errors`` hook (Wave A6) statically blocks a source write that pins
   an impl ``kind`` to a circuit name.
3. :mod:`tests.test_intent_registry` asserts every declarative craft field has exactly
   one registered impl whose declared status matches the ELV台账 (a stronger gate than
   "has an E/L/V row").

Lookup is :func:`implementation_for`: an unregistered / non-applicable kind raises the
TYPED :class:`UnsupportedIntentError` (a 5-state UNSUPPORTED stop), NEVER a silent
fallback to some default placer -- the caller renders one honest verdict.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from clay.core_contract import VerificationStatus
from clay.layout.intent_impl import IntentImplementation

__all__ = [
    "UnsupportedIntentError",
    "register",
    "implementation_for",
    "get",
    "registered_kinds",
    "all_implementations",
]

_IMPLS: Dict[str, IntentImplementation] = {}

# The built-in craft implementations register themselves on import of
# ``intent_impls_craft``; nothing else imports that module, so a caller that reaches a
# READER (get/implementation_for/registered_kinds/all_implementations) without having
# imported it would otherwise see an EMPTY registry (a lookup-ordering fragility). We
# import it lazily on first read -- a module-level import would cycle, since
# ``intent_impls_craft`` imports :func:`register` from here. ``register`` (the writer)
# never bootstraps, so the import's own ``register`` calls do not recurse.
_BOOTSTRAPPED = False


def _ensure_bootstrapped() -> None:
    """Import the built-in craft implementations once, so the registered set is present
    for any reader regardless of import order. Idempotent; failure is swallowed so a
    partial environment still returns whatever is registered."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _BOOTSTRAPPED = True  # set BEFORE the import so re-entrant reads do not recurse
    try:
        import clay.layout.intent_impls_craft  # noqa: F401  (registers on import)
    except Exception:
        pass

# Tokens that mark a key as a CIRCUIT / device / cell name rather than a generic
# intent KIND. A registry key naming a circuit is the per-circuit-placer anti-pattern
# the architecture bans (memory ``feedback_no_per_circuit_placers_ever``).
_CIRCUIT_NAME_TOKENS = frozenset(
    {
        "ota", "comparator", "ldo", "amp", "amplifier", "opamp", "pll", "vco",
        "bandgap", "bgr", "adc", "dac", "regulator", "buffer", "strongarm",
        "two_stage", "twostage", "fc_ota", "telescopic", "cascode_ota",
    }
)
# a '5t' / '5T' / 'nmos_pair'-style circuit token
_CIRCUIT_NAME_RE = re.compile(r"(^|_)\d+t($|_)", re.IGNORECASE)


class UnsupportedIntentError(Exception):
    """Typed 5-state stop: no registered/applicable implementation for an intent kind.

    Carries ``status == VerificationStatus.UNSUPPORTED`` so the caller/flow renders a
    5-state verdict instead of crashing or silently substituting a default. Raised by
    :func:`implementation_for` -- the registry NEVER falls back."""

    def __init__(self, kind: str, reason: str = "") -> None:
        self.kind = kind
        self.status = VerificationStatus.UNSUPPORTED
        self.reason = reason or f"no registered intent implementation for kind {kind!r}"
        super().__init__(self.reason)


def _looks_like_circuit_name(k: str) -> bool:
    """Does key ``k`` name a circuit / device / cell rather than a generic intent
    KIND? Used by :func:`register` to reject a per-circuit implementation key."""
    low = k.lower()
    if low in _CIRCUIT_NAME_TOKENS:
        return True
    if _CIRCUIT_NAME_RE.search(low):
        return True
    return any(tok in low.split("_") for tok in _CIRCUIT_NAME_TOKENS)


def _intent_kind_universe() -> frozenset:
    """The set of legal registry keys = the declarative LayoutIntent field names
    (derived, never hand-listed), so a typo'd or non-field kind cannot register."""
    from clay.layout.intent_elv import declarative_intent_fields

    return frozenset(declarative_intent_fields())


def register(impl: IntentImplementation) -> IntentImplementation:
    """Register ``impl`` under ``impl.kind``. Rejects a non-vocabulary key and a
    circuit-name-shaped key (enforcement layer 1). Returns ``impl`` so it can be used
    as a decorator. Idempotent re-registration of the SAME kind replaces the entry
    (module reload safe)."""
    k = impl.kind
    if not k:
        raise ValueError("intent implementation has no kind")
    if _looks_like_circuit_name(k):
        raise ValueError(
            f"intent impl key {k!r} looks like a CIRCUIT/device/cell name -- the "
            "registry is keyed on generic intent KIND only (per-circuit placers are "
            "banned, memory feedback_no_per_circuit_placers_ever)."
        )
    if k not in _intent_kind_universe():
        raise ValueError(
            f"intent impl kind {k!r} is not a declarative layout-intent field "
            f"(universe = {sorted(_intent_kind_universe())})."
        )
    _IMPLS[k] = impl
    return impl


def get(kind: str) -> Optional[IntentImplementation]:
    """The registered impl for ``kind`` (or ``None``). A plain lookup -- no
    applicability probe, no raise (use :func:`implementation_for` for the gate)."""
    _ensure_bootstrapped()
    return _IMPLS.get(kind)


def implementation_for(decl: Any, pdk: Any) -> IntentImplementation:
    """The applicable implementation for declaration ``decl`` on ``pdk``. Raises the
    typed :class:`UnsupportedIntentError` (5-state UNSUPPORTED) when no impl is
    registered for ``decl.kind`` OR the impl is not applicable -- NEVER a fallback."""
    _ensure_bootstrapped()
    kind = str(getattr(decl, "kind", decl))
    impl = _IMPLS.get(kind)
    if impl is None:
        raise UnsupportedIntentError(kind)
    if not impl.applicable(decl, pdk):
        raise UnsupportedIntentError(
            kind, f"impl for {kind!r} is not applicable to this declaration/PDK"
        )
    return impl


def registered_kinds() -> frozenset:
    """The set of kinds with a registered implementation."""
    _ensure_bootstrapped()
    return frozenset(_IMPLS)


def all_implementations() -> List[IntentImplementation]:
    """Every registered implementation (kind order)."""
    _ensure_bootstrapped()
    return [_IMPLS[k] for k in sorted(_IMPLS)]
