"""Frozen data model shared by the front-end paradigm mapper (round
front-end-paradigm-mapper).

The role-inference agent (A) PRODUCES a :class:`RoleDecomposition` from a parsed
SPICE ``.cir``; the design.clay emitter (B) CONSUMES it. Keeping the contract here
lets both build in parallel without depending on each other's module landing. The
role tokens are open free tokens (the Clay functional_role catalog reads them as
strings), so the inferencer may emit any descriptive role name; the canonical
analog set is suggested below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

# Suggested canonical role tokens (open set; descriptive names are fine).
CANONICAL_ROLES = (
    "input_pair",
    "tail_current",
    "current_mirror",
    "load_mirror",
    "cascode",
    "bias_tree",
    "pass_device",
    "compensation",
    "output_stage",
)


@dataclass(frozen=True)
class DeviceCard:
    """One parsed device. ``terminals`` maps a pin name to its net: a MOS uses
    ``{"d","g","s","b"}``; a 2-terminal device uses ``{"p","n"}``; a BJT uses
    ``{"c","b","e"}``. ``w``/``l``/``m`` are the sizing tokens as written in the
    .cir -- symbolic (``"W_diff"``) or literal (``"4.4u"``)."""

    name: str
    kind: str  # nmos | pmos | res | cap | bjt | diode | other
    terminals: Dict[str, str] = field(default_factory=dict)
    w: str = ""
    l: str = ""
    m: str = ""
    source_model: str = ""
    area: str = ""
    # Native source adapters retain every original parameter token separately
    # from the analysis-only W/L view. Legacy cards keep this empty.
    native_parameters: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class FunctionalRoleHit:
    """One inferred functional role over a set of devices, with its evidence."""

    role: str
    devices: Tuple[str, ...]
    evidence: str
    confidence: float = 1.0


@dataclass(frozen=True)
class RoleDecomposition:
    """The typed functional-role decomposition of a netlist (the Clay paradigm
    front-end's intermediate form)."""

    cell: str
    devices: Tuple[DeviceCard, ...] = ()
    roles: Tuple[FunctionalRoleHit, ...] = ()
    matched_pairs: Tuple[Tuple[str, str], ...] = ()
    mirror_axis: Optional[str] = None
    source_ports: Tuple[str, ...] = ()

    def device(self, name: str) -> Optional[DeviceCard]:
        for d in self.devices:
            if d.name == name:
                return d
        return None


def disjoint_role_partition(
    decomposition: RoleDecomposition,
    *,
    matched_pairs: Iterable[Tuple[str, str]] = (),
) -> Tuple[FunctionalRoleHit, ...]:
    """Normalize inferred roles into a complete, disjoint source partition.

    Role inference is intentionally best-effort and a device may satisfy more
    than one structural signature.  Physical intent cannot use overlapping
    ``functional_group`` membership, and an explicit matching pair must never
    be split between groups.  This adapter therefore treats declared/inferred
    matching components as indivisible atoms, lets earlier (higher-priority)
    role hits claim still-free atoms, and emits deterministic ``unclassified``
    groups for the residual inventory.

    This is topology-only normalization: it has no circuit, PDK, coordinate,
    or geometry branches.
    """

    ordered_names = tuple(card.name for card in decomposition.devices)
    if len(ordered_names) != len(set(ordered_names)):
        raise ValueError("role decomposition contains duplicate device names")
    known = set(ordered_names)
    parent = {name: name for name in ordered_names}

    def find(name: str) -> str:
        root = name
        while parent[root] != root:
            root = parent[root]
        while parent[name] != name:
            next_name = parent[name]
            parent[name] = root
            name = next_name
        return root

    def union(left: str, right: str) -> None:
        if left not in known or right not in known or left == right:
            return
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        # Device order, rather than lexical spelling, is the stable source
        # order carried by RoleDecomposition.
        order = {name: index for index, name in enumerate(ordered_names)}
        low, high = sorted((left_root, right_root), key=order.__getitem__)
        parent[high] = low

    for left, right in tuple(decomposition.matched_pairs) + tuple(matched_pairs):
        union(str(left), str(right))

    roots_by_name = {name: find(name) for name in ordered_names}
    assigned_roots: set[str] = set()
    out: list[FunctionalRoleHit] = []
    for hit in decomposition.roles:
        roots = {
            roots_by_name[name]
            for name in hit.devices
            if name in roots_by_name and roots_by_name[name] not in assigned_roots
        }
        if not roots:
            continue
        members = tuple(
            name for name in ordered_names if roots_by_name[name] in roots
        )
        out.append(
            FunctionalRoleHit(
                role=hit.role,
                devices=members,
                evidence=hit.evidence,
                confidence=hit.confidence,
            )
        )
        assigned_roots.update(roots)

    for name in ordered_names:
        root = roots_by_name[name]
        if root in assigned_roots:
            continue
        members = tuple(
            member for member in ordered_names if roots_by_name[member] == root
        )
        out.append(
            FunctionalRoleHit(
                role="unclassified",
                devices=members,
                evidence="no structural role claimed this matching-safe device atom",
                confidence=0.0,
            )
        )
        assigned_roots.add(root)
    return tuple(out)


__all__ = [
    "CANONICAL_ROLES",
    "DeviceCard",
    "FunctionalRoleHit",
    "RoleDecomposition",
    "disjoint_role_partition",
]
