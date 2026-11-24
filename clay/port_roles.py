"""Shared semantic port-role helpers.

Physical materialization and signoff must choose the same deck-global
p-substrate net.  Keeping the choice here prevents layout generation, routing,
and LVS reference lowering from independently interpreting port spellings.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def declared_substrate_ports(
    port_roles: Mapping[object, object] | None,
) -> tuple[str, ...]:
    """Return the semantic substrate port, if declared.

    An explicit ``ground`` role is the substrate domain.  If none exists, the
    negative supply (``power_n``) is the substrate domain.  This distinguishes a
    split-domain cell (``GNDA: power_n``, ``SUB: ground``) from a conventional
    cell whose only low rail is ``GND: power_n``.

    The supported decks expose one global p-substrate.  More than one port at
    the selected semantic tier is therefore ambiguous and fails closed.
    """

    roles = port_roles or {}
    for semantic_role in ("ground", "power_n"):
        candidates = tuple(
            str(net)
            for net, role in roles.items()
            if str(net).strip()
            and str(role or "").strip().casefold() == semantic_role
        )
        if not candidates:
            continue
        if len(candidates) != 1:
            raise ValueError(
                "single-substrate extraction requires exactly one declared "
                f"{semantic_role} port; found {candidates}"
            )
        return candidates
    return ()


def resolve_substrate_net(cell_def: Any, *, fallback: str) -> str:
    """Return the declared substrate port or the PDK-owned fallback spelling."""

    declared = declared_substrate_ports(
        getattr(cell_def, "port_roles", {}) if cell_def is not None else {}
    )
    return declared[0] if declared else str(fallback)


def declared_positive_supply_ports(
    port_roles: Mapping[object, object] | None,
) -> tuple[str, ...]:
    """Return the single declared positive-supply port.

    A deep N-well is a real routed conductor, not an implicit spelling such as
    ``vdd``.  Its bias therefore comes from the source's semantic ``power_p``
    role.  Multiple positive domains are ambiguous until the source declares
    which one owns the well, so this helper fails closed rather than guessing.
    """

    roles = port_roles or {}
    candidates = tuple(
        str(net)
        for net, role in roles.items()
        if str(net).strip()
        and str(role or "").strip().casefold() == "power_p"
    )
    if len(candidates) > 1:
        raise ValueError(
            "deep-N-well materialization requires exactly one declared power_p "
            f"port; found {candidates}"
        )
    return candidates


def resolve_positive_supply_net(cell_def: Any, *, fallback: str) -> str:
    """Return the declared positive-supply port or a PDK-owned fallback."""

    declared = declared_positive_supply_ports(
        getattr(cell_def, "port_roles", {}) if cell_def is not None else {}
    )
    return declared[0] if declared else str(fallback)


__all__ = [
    "declared_positive_supply_ports",
    "declared_substrate_ports",
    "resolve_positive_supply_net",
    "resolve_substrate_net",
]
