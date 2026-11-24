"""Deterministic structural-intent candidates for compact analog layout.

This module contains no placer, router, or circuit-specific policy.  It defines
the normalized DATA searched by the compact-intent flow and the exact
lexicographic ordering used after physical signoff.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Iterable, Mapping, Sequence


_ORIENTATIONS = frozenset({"R0", "R90", "R180", "R270"})
_PASSIVE_SIDES = frozenset({"center", "left", "right"})
_SHA256 = re.compile(r"[0-9a-f]{64}")


class CompactIntentTruth(str, Enum):
    """Evidence maturity carried by one materialized compact candidate."""

    STAGING = "staging"
    PHYSICAL_EXACT_ARTIFACT = "physical_exact_artifact"
    POST_PEX = "post_pex"


def _normalized_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a 64-hex SHA-256 digest")
    digest = value.strip().lower()
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a 64-hex SHA-256 digest")
    return digest


def load_compact_intent_provenance_json(
    encoded: str,
    *,
    label: str = "compact_intent provenance",
) -> Mapping[str, object]:
    """Parse one provenance object while rejecting duplicate keys at any depth."""

    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"{label} JSON contains duplicate key {key!r}")
            output[key] = value
        return output

    try:
        payload = json.loads(encoded, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


@dataclass(frozen=True)
class PhysicalPexArtifact:
    """A topology-matched PEX artifact, without an electrical claim."""

    sha256: str
    topology_match: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sha256",
            _normalized_sha256(
                self.sha256,
                label="physical_pex_artifact.sha256",
            ),
        )
        if type(self.topology_match) is not bool:
            raise ValueError(
                "physical_pex_artifact.topology_match must be a Boolean"
            )


@dataclass(frozen=True)
class ElectricalResultReceipt:
    """Content identity of electrical results and their exact PEX input."""

    sha256: str
    physical_pex_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sha256",
            _normalized_sha256(
                self.sha256,
                label="electrical_result_receipt.sha256",
            ),
        )
        object.__setattr__(
            self,
            "physical_pex_sha256",
            _normalized_sha256(
                self.physical_pex_sha256,
                label="electrical_result_receipt.physical_pex_sha256",
            ),
        )


@dataclass(frozen=True)
class CompactIntentEvidence:
    """Typed, fail-closed truth boundary for compact-intent provenance."""

    truth: CompactIntentTruth
    physical_pex_artifact: PhysicalPexArtifact | None = None
    electrical_result_receipt: ElectricalResultReceipt | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.truth, CompactIntentTruth):
            raise TypeError("compact intent truth must be CompactIntentTruth")
        physical = self.physical_pex_artifact
        electrical = self.electrical_result_receipt
        if self.truth is CompactIntentTruth.STAGING:
            if physical is not None or electrical is not None:
                raise ValueError("staging compact intent cannot carry signed evidence")
            return
        if physical is None or not physical.topology_match:
            raise ValueError(
                f"{self.truth.value} requires a topology-matched "
                "physical_pex_artifact"
            )
        if self.truth is CompactIntentTruth.PHYSICAL_EXACT_ARTIFACT:
            if electrical is not None:
                raise ValueError(
                    "physical_exact_artifact requires "
                    "electrical_result_receipt=null"
                )
            return
        if electrical is None:
            raise ValueError(
                "post_pex requires a digest-bound electrical_result_receipt"
            )
        if electrical.physical_pex_sha256 != physical.sha256:
            raise ValueError(
                "electrical_result_receipt physical PEX digest does not match "
                "physical_pex_artifact"
            )

    def canonical_data(self) -> dict[str, object]:
        physical = self.physical_pex_artifact
        electrical = self.electrical_result_receipt
        return {
            "truth": self.truth.value,
            "physical_pex_artifact": (
                None
                if physical is None
                else {
                    "sha256": physical.sha256,
                    "topology_match": physical.topology_match,
                }
            ),
            "electrical_result_receipt": (
                None
                if electrical is None
                else {
                    "sha256": electrical.sha256,
                    "physical_pex_sha256": electrical.physical_pex_sha256,
                }
            ),
        }


def _closed_receipt(
    provenance: Mapping[str, object],
    *,
    field: str,
    expected: frozenset[str],
) -> Mapping[str, object] | None:
    raw = provenance.get(field)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field} must be an object or null")
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown or missing:
        raise ValueError(
            f"{field} fields are not closed: missing={missing}, unknown={unknown}"
        )
    return raw


def compact_intent_evidence_from_provenance(
    provenance: Mapping[str, object],
) -> CompactIntentEvidence:
    """Parse only typed truth fields from a full compact provenance row."""

    try:
        truth = CompactIntentTruth(str(provenance.get("truth", "")))
    except ValueError as exc:
        raise ValueError("compact_intent truth is unknown") from exc
    if "verified" in provenance and not (
        truth is CompactIntentTruth.STAGING
        and provenance["verified"] is False
    ):
        raise ValueError(
            "legacy verified is ambiguous for typed compact intent truth"
        )
    physical_data = _closed_receipt(
        provenance,
        field="physical_pex_artifact",
        expected=frozenset({"sha256", "topology_match"}),
    )
    electrical_data = _closed_receipt(
        provenance,
        field="electrical_result_receipt",
        expected=frozenset({"sha256", "physical_pex_sha256"}),
    )
    if truth is CompactIntentTruth.PHYSICAL_EXACT_ARTIFACT and (
        "physical_pex_artifact" not in provenance
        or "electrical_result_receipt" not in provenance
    ):
        raise ValueError(
            "physical_exact_artifact requires explicit physical_pex_artifact "
            "and electrical_result_receipt=null"
        )
    physical = (
        None
        if physical_data is None
        else PhysicalPexArtifact(
            sha256=physical_data["sha256"],  # type: ignore[arg-type]
            topology_match=physical_data["topology_match"],
        )
    )
    electrical = (
        None
        if electrical_data is None
        else ElectricalResultReceipt(
            sha256=electrical_data["sha256"],  # type: ignore[arg-type]
            physical_pex_sha256=electrical_data[  # type: ignore[arg-type]
                "physical_pex_sha256"
            ],
        )
    )
    return CompactIntentEvidence(
        truth=truth,
        physical_pex_artifact=physical,
        electrical_result_receipt=electrical,
    )


@dataclass(frozen=True, order=True)
class PassiveSplitChoice:
    """One topology-preserving physical split of a two-terminal passive."""

    passive: str
    unit_count: int
    connection: str
    orientation: str = "R0"


@dataclass(frozen=True, order=True)
class PassiveArrayChoice:
    """One rectangular lattice shape for a parallel passive-unit bank."""

    passive: str
    rows: int

    def __post_init__(self) -> None:
        if type(self.passive) is not str:
            raise TypeError("passive array name must be an exact string")
        if (
            not self.passive
            or self.passive != self.passive.strip()
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.passive) is None
        ):
            raise ValueError("passive array name must be a canonical Clay identifier")
        if type(self.rows) is not int:
            raise TypeError("passive array rows must be an exact integer")
        if self.rows < 1:
            raise ValueError("passive array rows must be positive")


@dataclass(frozen=True, order=True)
class NonmosMacroRowsChoice:
    """Selected internal row count for one graph-owned rigid non-MOS macro."""

    macro_id: str
    rows: int

    def __post_init__(self) -> None:
        if type(self.macro_id) is not str:
            raise TypeError("non-MOS macro row choice id must be an exact string")
        if not self.macro_id or self.macro_id != self.macro_id.strip():
            raise ValueError("non-MOS macro row choice id must be canonical")
        if type(self.rows) is not int:
            raise TypeError("non-MOS macro rows must be an exact integer")
        if self.rows < 1:
            raise ValueError("non-MOS macro rows must be positive")


@dataclass(frozen=True, order=True)
class GuardRingChoice:
    """One explicit guard-ring island in a structural candidate."""

    members: tuple[str, ...]
    well: str
    tie_net: str

    def __post_init__(self) -> None:
        normalized = tuple(sorted({str(member) for member in self.members}))
        if not normalized:
            raise ValueError("guard ring members must be non-empty")
        if len(normalized) != len(self.members):
            raise ValueError("guard ring members must be unique")
        well = str(self.well).lower()
        if well not in {"p", "n"}:
            raise ValueError(f"guard ring well must be 'p' or 'n', got {self.well!r}")
        tie_net = str(self.tie_net).strip()
        if not tie_net:
            raise ValueError("guard ring tie_net must be non-empty")
        object.__setattr__(self, "members", normalized)
        object.__setattr__(self, "well", well)
        object.__setattr__(self, "tie_net", tie_net)


@dataclass(frozen=True, order=True)
class MacroPartitionChoice:
    """One canonical partition of a typed physical macro's full membership."""

    macro_id: str
    member_groups: tuple[tuple[str, ...], ...]
    declaration_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        macro_id = str(self.macro_id).strip()
        if not macro_id:
            raise ValueError("macro partition macro_id must be non-empty")
        groups: list[tuple[str, ...]] = []
        owned: set[str] = set()
        for raw_group in self.member_groups:
            group = tuple(sorted(str(member).strip() for member in raw_group))
            if not group or any(not member for member in group):
                raise ValueError("macro partition member groups must be non-empty")
            if len(group) != len(set(group)):
                raise ValueError("macro partition member group contains duplicates")
            overlap = owned.intersection(group)
            if overlap:
                raise ValueError(
                    "macro partition member groups overlap at "
                    f"{sorted(overlap)!r}"
                )
            owned.update(group)
            groups.append(group)
        if not groups:
            raise ValueError("macro partition member groups must be non-empty")
        refs = tuple(sorted(str(ref).strip() for ref in self.declaration_refs))
        if any(not ref for ref in refs) or len(refs) != len(set(refs)):
            raise ValueError(
                "macro partition declaration_refs must be non-empty and unique"
            )
        object.__setattr__(self, "macro_id", macro_id)
        object.__setattr__(self, "member_groups", tuple(sorted(groups)))
        object.__setattr__(self, "declaration_refs", refs)


@dataclass(frozen=True, order=True)
class GuardRingAtom:
    """Smallest function/matching cluster the ring search may not split."""

    members: tuple[str, ...]
    well: str
    tie_net: str
    functional_group: str
    isolated: bool = False
    sensitive: bool = False
    output: bool = False
    passive_only: bool = False

    def __post_init__(self) -> None:
        if type(self.passive_only) is not bool:
            raise TypeError("guard ring atom passive_only must be bool")
        normalized = tuple(sorted({str(member) for member in self.members}))
        if not normalized or len(normalized) != len(self.members):
            raise ValueError("guard ring atom members must be non-empty and unique")
        well = str(self.well).lower()
        if well not in {"p", "n"}:
            raise ValueError("guard ring atom well must be 'p' or 'n'")
        tie_net = str(self.tie_net).strip()
        if not tie_net:
            raise ValueError("guard ring atom tie_net must be non-empty")
        group = str(self.functional_group).strip()
        if not group:
            raise ValueError("guard ring atom functional_group must be non-empty")
        object.__setattr__(self, "members", normalized)
        object.__setattr__(self, "well", well)
        object.__setattr__(self, "tie_net", tie_net)
        object.__setattr__(self, "functional_group", group)


@dataclass(frozen=True)
class GuardAtomCompatibility:
    compatible: bool
    reason: str = ""


def guard_atom_compatibility(
    left: GuardRingAtom,
    right: GuardRingAtom,
) -> GuardAtomCompatibility:
    """Return the stable legality verdict for merging two ring atoms."""

    if left.well != right.well:
        return GuardAtomCompatibility(False, "WELL_POLARITY")
    if left.tie_net != right.tie_net:
        return GuardAtomCompatibility(False, "BODY_NET")
    shared_passive_isolation = (
        left.isolated
        and right.isolated
        and left.passive_only
        and right.passive_only
    )
    if (left.isolated or right.isolated) and not shared_passive_isolation:
        return GuardAtomCompatibility(False, "ISOLATION")
    if (left.sensitive and right.output) or (right.sensitive and left.output):
        return GuardAtomCompatibility(False, "SENSITIVE_OUTPUT")
    return GuardAtomCompatibility(True)


def _mos_well_and_body(device: object) -> tuple[str, str] | None:
    kind = str(getattr(device, "device_class", "")).lower()
    if "pmos" in kind or kind.startswith("pfet"):
        well = "n"
    elif "nmos" in kind or kind.startswith("nfet"):
        well = "p"
    else:
        return None
    pins = getattr(device, "pins", {}) or {}
    body = str(pins.get("b", pins.get("body", ""))).strip()
    if not body:
        name = str(getattr(device, "instance_name", ""))
        raise ValueError(f"guard ring atom requires declared body net for {name!r}")
    return well, body


def _mos_geometry_signature(
    device: object,
) -> tuple[str, str, float, float]:
    """Return the canonical per-occurrence MOS geometry class.

    Functional role and body ownership constrain which devices may share a
    ring, but they do not prove that unlike PCell geometries form one
    indivisible placement atom.  W/L select the physical primitive shape;
    parallel multiplicity M controls how many occurrences belong to the atom
    and therefore must not prevent otherwise identical devices from sharing.
    """

    sizing = getattr(device, "sizing", None)
    if type(sizing) is not dict:
        raise TypeError("guard ring MOS sizing must be an exact dict")
    normalized: dict[str, float] = {}
    for key in ("W", "L"):
        value = sizing.get(key)
        if (
            type(value) not in {int, float}
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            name = str(getattr(device, "instance_name", ""))
            raise ValueError(
                f"guard ring atom requires positive finite {key} for {name!r}"
            )
        normalized[key] = float(value)
    device_class = str(getattr(device, "device_class", "")).strip().casefold()
    model = str(
        getattr(device, "pdk_variant", None)
        or getattr(device, "source_model", None)
        or ""
    ).strip().casefold()
    if not device_class:
        raise ValueError("guard ring MOS device class must be non-empty")
    return (device_class, model, normalized["W"], normalized["L"])


def _is_passive_device(device: object) -> bool:
    kind = str(getattr(device, "device_class", "")).strip().lower()
    return kind.startswith("res") or "cap" in kind


def _declares_isolation(attributes: dict[str, object]) -> bool:
    value = str(attributes.get("well_isolation", "") or "").strip().lower()
    return value not in {"", "none", "shared", "common"}


def derive_guard_ring_atoms(cell_def: object) -> tuple[GuardRingAtom, ...]:
    """Derive function/matching atoms from parsed Clay DATA, never cell names."""

    all_devices = {
        str(getattr(device, "instance_name", "")): device
        for device in (getattr(cell_def, "devices", ()) or ())
    }
    devices = {
        str(getattr(device, "instance_name", "")): device
        for device in (getattr(cell_def, "devices", ()) or ())
        if _mos_well_and_body(device) is not None
    }
    roles = tuple(getattr(cell_def, "functional_roles", ()) or ())
    traits: dict[str, dict[str, bool]] = {}
    role_memberships: dict[str, list[str]] = {}
    dummy_members: set[str] = set()
    for role in roles:
        attributes = dict(getattr(role, "attributes", {}) or {})
        members = tuple(str(name) for name in attributes.get("devices", ()) or ())
        if str(attributes.get("role", "") or "").strip().lower() == "dummy":
            dummy_members.update(members)
        for name in members:
            role_memberships.setdefault(name, []).append(
                str(getattr(role, "name", "functional"))
            )
            state = traits.setdefault(
                name,
                {"isolated": False, "sensitive": False, "output": False},
            )
            state["isolated"] |= _declares_isolation(attributes)
            state["sensitive"] |= bool(attributes.get("input_nets"))
            state["output"] |= bool(attributes.get("output_nets"))

    from clay.layout.intent import layout_intent_from_cell_def

    declared_rings = tuple(
        sorted(
            layout_intent_from_cell_def(cell_def).guard_rings,
            key=lambda ring: (
                tuple(sorted(str(member) for member in ring.members)),
                str(ring.well),
                str(ring.tie_net),
            ),
        )
    )
    declared_owner: dict[str, int] = {}
    for index, ring in enumerate(declared_rings):
        members = tuple(sorted(str(member) for member in ring.members))
        unknown = sorted(set(members) - set(all_devices))
        overlap = sorted(set(members) & set(declared_owner))
        if unknown or overlap:
            detail = f"unknown={unknown!r}, overlap={overlap!r}"
            raise ValueError(f"declared guard-ring ownership is invalid: {detail}")
        for name in members:
            well_body = _mos_well_and_body(devices[name]) if name in devices else None
            if well_body is not None and well_body != (ring.well, ring.tie_net):
                raise ValueError(
                    f"declared guard ring disagrees with {name!r} well/body"
                )
        for name in members:
            declared_owner[name] = index

    assigned: set[str] = set()
    atoms: list[GuardRingAtom] = []
    for index, ring in enumerate(declared_rings):
        members = tuple(
            sorted(name for name, owner in declared_owner.items() if owner == index)
        )
        if not set(members) & dummy_members:
            continue
        atoms.append(
            GuardRingAtom(
                members=members,
                well=ring.well,
                tie_net=ring.tie_net,
                functional_group=f"dummy_owned_row_ring:{index}",
                isolated=(
                    str(getattr(ring, "role", "") or "").lower() == "isolation"
                    or any(
                        traits.get(name, {}).get("isolated", False)
                        for name in members
                    )
                ),
                sensitive=any(
                    traits.get(name, {}).get("sensitive", False) for name in members
                ),
                output=any(
                    traits.get(name, {}).get("output", False) for name in members
                ),
            )
        )
        assigned.update(members)

    # Functional roles are semantic labels and may overlap.  Physical ring
    # atoms must not: group MOS devices by the complete role-membership
    # signature, and retain a declared-ring boundary when one exists.  This
    # keeps the result independent of declaration order without freezing a
    # whole existing ring into one indivisible search atom.
    functional_partitions: dict[
        tuple[
            tuple[str, ...],
            int | None,
            str,
            str,
            tuple[str, str, float, float],
        ],
        list[str],
    ] = {}
    for name in sorted((set(role_memberships) & set(devices)) - assigned):
        well_body = _mos_well_and_body(devices[name])
        assert well_body is not None
        role_signature = tuple(sorted(set(role_memberships[name])))
        functional_partitions.setdefault(
            (
                role_signature,
                declared_owner.get(name),
                *well_body,
                _mos_geometry_signature(devices[name]),
            ),
            [],
        ).append(name)
    for (role_signature, _owner, well, body, _geometry), members in sorted(
        functional_partitions.items()
    ):
        atoms.append(
            GuardRingAtom(
                members=tuple(members),
                well=well,
                tie_net=body,
                functional_group="+".join(role_signature),
                isolated=any(
                    traits.get(name, {}).get("isolated", False) for name in members
                ),
                sensitive=any(
                    traits.get(name, {}).get("sensitive", False) for name in members
                ),
                output=any(
                    traits.get(name, {}).get("output", False) for name in members
                ),
            )
        )
        assigned.update(members)

    # Non-MOS members do not participate in functional-role atom derivation.
    # Preserve any explicit ring ownership so passive isolation declarations
    # remain in the structural search instead of disappearing silently.
    for index, ring in enumerate(declared_rings):
        members = tuple(
            sorted(
                name
                for name, owner in declared_owner.items()
                if owner == index and name not in devices and name not in assigned
            )
        )
        if not members:
            continue
        atoms.append(
            GuardRingAtom(
                members=members,
                well=ring.well,
                tie_net=ring.tie_net,
                functional_group=f"declared_guard_ring:{index}",
                isolated=str(getattr(ring, "role", "") or "").lower()
                == "isolation",
                passive_only=all(
                    _is_passive_device(all_devices[name]) for name in members
                ),
                sensitive=any(
                    traits.get(name, {}).get("sensitive", False) for name in members
                ),
                output=any(
                    traits.get(name, {}).get("output", False) for name in members
                ),
            )
        )
        assigned.update(members)

    # Matching declarations are physical intent too.  A matched MOS pair that
    # has no broader functional_group must not be split into independently
    # searchable ring atoms: doing so lets a random ring partition give the pair
    # different well environments even though the source explicitly asks them to
    # match.  Build annotation-connected components, then partition only where a
    # different well/body net makes a shared ring physically impossible.
    from clay.layout.annotations import Matching

    remaining = set(devices).difference(assigned)
    parent = {name: name for name in remaining}

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
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    annotations = dict(
        getattr(cell_def, "_layout_annotations", {}) or {}
    )
    for name, declarations in sorted(annotations.items()):
        if name not in remaining:
            continue
        for declaration in declarations or ():
            if not isinstance(declaration, Matching):
                continue
            other = str(getattr(declaration, "other", ""))
            if other in remaining:
                union(name, other)

    components: dict[str, list[str]] = {}
    for name in sorted(remaining):
        components.setdefault(find(name), []).append(name)
    for component_index, members in enumerate(
        sorted(tuple(sorted(value)) for value in components.values())
    ):
        partitions: dict[tuple[str, str], list[str]] = {}
        for name in members:
            well_body = _mos_well_and_body(devices[name])
            assert well_body is not None
            partitions.setdefault(well_body, []).append(name)
        for partition_index, ((well, body), partition_members) in enumerate(
            sorted(partitions.items())
        ):
            group_name = (
                f"matching:{':'.join(members)}"
                if len(members) > 1 and len(partitions) == 1
                else (
                    f"matching:{component_index}:{partition_index}"
                    if len(members) > 1
                    else f"device:{partition_members[0]}"
                )
            )
            atoms.append(
                GuardRingAtom(
                    members=tuple(partition_members),
                    well=well,
                    tie_net=body,
                    functional_group=group_name,
                )
            )
    return tuple(sorted(atoms))


def _group_accepts(group: Sequence[GuardRingAtom], atom: GuardRingAtom) -> bool:
    return all(guard_atom_compatibility(member, atom).compatible for member in group)


def enumerate_guard_ring_partitions(
    atoms: Iterable[GuardRingAtom],
) -> tuple[tuple[GuardRingChoice, ...], ...]:
    """Enumerate every deterministic legal set partition of ring atoms."""

    ordered = tuple(sorted(atoms))
    all_members = [member for atom in ordered for member in atom.members]
    if len(all_members) != len(set(all_members)):
        raise ValueError("guard ring atoms must not share device members")

    partitions: list[tuple[tuple[GuardRingAtom, ...], ...]] = [()]
    for atom in ordered:
        expanded: set[tuple[tuple[GuardRingAtom, ...], ...]] = set()
        for partition in partitions:
            expanded.add(tuple(sorted(partition + ((atom,),))))
            for index, group in enumerate(partition):
                if not _group_accepts(group, atom):
                    continue
                merged = tuple(sorted(group + (atom,)))
                replacement = partition[:index] + (merged,) + partition[index + 1 :]
                expanded.add(tuple(sorted(replacement)))
        partitions = sorted(expanded)

    choices: set[tuple[GuardRingChoice, ...]] = set()
    for partition in partitions:
        rings: list[GuardRingChoice] = []
        for group in partition:
            first = group[0]
            rings.append(
                GuardRingChoice(
                    members=tuple(
                        sorted(member for atom in group for member in atom.members)
                    ),
                    well=first.well,
                    tie_net=first.tie_net,
                )
            )
        choices.add(tuple(sorted(rings)))
    return tuple(sorted(choices))


def _validate_aspect_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode == "natural":
        return mode
    prefix = "aspect:"
    if not mode.startswith(prefix):
        raise ValueError(
            "aspect_mode must be 'natural' or 'aspect:<positive finite ratio>'"
        )
    try:
        ratio = float(mode[len(prefix) :])
    except ValueError as exc:
        raise ValueError(f"invalid aspect_mode {value!r}") from exc
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError(f"invalid aspect_mode {value!r}")
    return prefix + format(ratio, ".12g")


def _validate_pack_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in {"rows", "dense_2d"}:
        raise ValueError(
            "pack_mode must be one of ['dense_2d', 'rows'], "
            f"got {value!r}"
        )
    return mode


@dataclass(frozen=True)
class CompactIntentCandidate:
    """One normalized point in the bounded structural-intent domain."""

    materialization_mode: str = "derived"
    passive_splits: tuple[PassiveSplitChoice, ...] = ()
    passive_arrays: tuple[PassiveArrayChoice, ...] = ()
    nonmos_macro_rows: tuple[NonmosMacroRowsChoice, ...] = ()
    guard_ring_groups: tuple[GuardRingChoice, ...] = ()
    group_order: tuple[str, ...] = ()
    group_floorplan: tuple[tuple[str, int, int], ...] = ()
    passive_orientations: tuple[tuple[str, str], ...] = ()
    passive_side: str = "center"
    aspect_mode: str = "natural"
    pack_mode: str = "rows"
    edge_rail_width_um: float | None = None
    mos_fold_height_um: float | None = None
    block_gap_um: float | None = None
    physical_macro_graph_sha256: str | None = None
    passive_bank_partitions: tuple[MacroPartitionChoice, ...] = ()
    well_island_partitions: tuple[MacroPartitionChoice, ...] = ()
    left_column_group: str = ""
    common_centroid_rows: tuple[tuple[str, int], ...] = ()
    channel_strategy: str = "declared"

    def __post_init__(self) -> None:
        materialization_mode = str(self.materialization_mode).strip().lower()
        if materialization_mode not in {"derived", "authoring"}:
            raise ValueError(
                "materialization_mode must be one of ['authoring', 'derived']"
            )
        object.__setattr__(self, "materialization_mode", materialization_mode)
        splits: list[PassiveSplitChoice] = []
        seen_passives: set[str] = set()
        for choice in self.passive_splits:
            passive = str(choice.passive).strip()
            if not passive:
                raise ValueError("passive split name must be non-empty")
            if passive in seen_passives:
                raise ValueError(f"duplicate passive split for {passive!r}")
            seen_passives.add(passive)
            if type(choice.unit_count) is not int or choice.unit_count < 1:
                raise ValueError("passive split unit_count must be a positive integer")
            connection = str(choice.connection).lower()
            if connection != "parallel":
                raise ValueError(
                    "phase-1 passive split connection must be 'parallel'"
                )
            orientation = str(choice.orientation).upper()
            if orientation not in _ORIENTATIONS:
                raise ValueError(
                    f"passive split orientation must be one of "
                    f"{sorted(_ORIENTATIONS)}, got {choice.orientation!r}"
                )
            splits.append(
                PassiveSplitChoice(
                    passive=passive,
                    unit_count=choice.unit_count,
                    connection=connection,
                    orientation=orientation,
                )
            )

        if type(self.passive_arrays) is not tuple:
            raise TypeError("passive_arrays must be an exact tuple")
        passive_arrays: list[PassiveArrayChoice] = []
        seen_array_passives: set[str] = set()
        for choice in self.passive_arrays:
            if type(choice) is not PassiveArrayChoice:
                raise TypeError(
                    "passive_arrays entries must be exact PassiveArrayChoice values"
                )
            replayed = PassiveArrayChoice(
                passive=choice.passive,
                rows=choice.rows,
            )
            if replayed.passive in seen_array_passives:
                raise ValueError(
                    f"duplicate passive array for {replayed.passive!r}"
                )
            seen_array_passives.add(replayed.passive)
            passive_arrays.append(replayed)

        rings = tuple(sorted(self.guard_ring_groups))
        ring_members = [member for ring in rings for member in ring.members]
        if len(ring_members) != len(set(ring_members)):
            raise ValueError("guard ring groups must not share members")

        if type(self.group_order) is not tuple:
            raise TypeError("group_order must be an exact tuple")
        if any(type(group) is not str for group in self.group_order):
            raise TypeError("group_order entries must be exact strings")
        order = self.group_order
        if (
            any(not group or group != group.strip() for group in order)
            or len(order) != len(set(order))
        ):
            raise ValueError("group_order entries must be non-empty and unique")

        if type(self.group_floorplan) is not tuple:
            raise TypeError("group_floorplan must be an exact tuple")
        floorplan: list[tuple[str, int, int]] = []
        for item in self.group_floorplan:
            if type(item) is not tuple or len(item) != 3:
                raise TypeError(
                    "group_floorplan entries must be exact (group, tier, column) tuples"
                )
            group, tier, column = item
            if (
                type(group) is not str
                or not group
                or group != group.strip()
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", group) is None
            ):
                raise ValueError(
                    "group_floorplan group must be an exact Clay identifier"
                )
            if (
                type(tier) is not int
                or type(column) is not int
                or tier < 0
                or column < 0
            ):
                raise TypeError(
                    "group_floorplan requires non-negative exact integer ordinals"
                )
            floorplan.append((group, tier, column))
        if len({group for group, _tier, _column in floorplan}) != len(floorplan):
            raise ValueError("group_floorplan groups must be unique")
        passive_orientations = tuple(sorted(
            (str(passive).strip(), str(orientation).upper())
            for passive, orientation in self.passive_orientations
        ))
        orientation_names = [passive for passive, _orientation in passive_orientations]
        if (
            any(not passive for passive in orientation_names)
            or len(orientation_names) != len(set(orientation_names))
            or any(orientation not in _ORIENTATIONS for _passive, orientation in passive_orientations)
            or set(orientation_names) & seen_passives
        ):
            raise ValueError(
                "passive_orientations must contain unique non-split passives and "
                f"orientations from {sorted(_ORIENTATIONS)}"
            )

        side = str(self.passive_side).lower()
        if side not in _PASSIVE_SIDES:
            raise ValueError(
                f"passive_side must be one of {sorted(_PASSIVE_SIDES)}, got "
                f"{self.passive_side!r}"
            )

        object.__setattr__(self, "passive_splits", tuple(sorted(splits)))
        object.__setattr__(self, "passive_arrays", tuple(sorted(passive_arrays)))

        if type(self.nonmos_macro_rows) is not tuple:
            raise TypeError("nonmos_macro_rows must be an exact tuple")
        nonmos_macro_rows: list[NonmosMacroRowsChoice] = []
        for choice in self.nonmos_macro_rows:
            if type(choice) is not NonmosMacroRowsChoice:
                raise TypeError(
                    "nonmos_macro_rows entries must be exact "
                    "NonmosMacroRowsChoice values"
                )
            nonmos_macro_rows.append(
                NonmosMacroRowsChoice(macro_id=choice.macro_id, rows=choice.rows)
            )
        if len({choice.macro_id for choice in nonmos_macro_rows}) != len(
            nonmos_macro_rows
        ):
            raise ValueError("duplicate non-MOS macro row choice")
        object.__setattr__(
            self, "nonmos_macro_rows", tuple(sorted(nonmos_macro_rows))
        )
        object.__setattr__(self, "guard_ring_groups", rings)
        object.__setattr__(self, "group_order", order)
        object.__setattr__(self, "group_floorplan", tuple(floorplan))
        object.__setattr__(self, "passive_orientations", passive_orientations)
        object.__setattr__(self, "passive_side", side)
        object.__setattr__(self, "aspect_mode", _validate_aspect_mode(self.aspect_mode))
        object.__setattr__(self, "pack_mode", _validate_pack_mode(self.pack_mode))
        rail_width = self.edge_rail_width_um
        if rail_width is not None:
            rail_width = float(rail_width)
            if not math.isfinite(rail_width) or rail_width <= 0.0:
                raise ValueError(
                    "edge_rail_width_um must be a positive finite value or None"
                )
        object.__setattr__(self, "edge_rail_width_um", rail_width)
        fold_height = self.mos_fold_height_um
        if fold_height is not None:
            fold_height = float(fold_height)
            if not math.isfinite(fold_height) or fold_height <= 0.0:
                raise ValueError(
                    "mos_fold_height_um must be a positive finite value or None"
                )
        object.__setattr__(self, "mos_fold_height_um", fold_height)
        block_gap = self.block_gap_um
        if block_gap is not None:
            block_gap = float(block_gap)
            if not math.isfinite(block_gap) or block_gap < 0.0:
                raise ValueError(
                    "block_gap_um must be a non-negative finite value or None"
                )
        object.__setattr__(self, "block_gap_um", block_gap)

        left_column_group = self.left_column_group
        if type(left_column_group) is not str:
            raise TypeError("left_column_group must be an exact string")
        if left_column_group and (
            left_column_group != left_column_group.strip()
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", left_column_group) is None
        ):
            raise ValueError("left_column_group must be an exact Clay identifier")

        if type(self.common_centroid_rows) is not tuple:
            raise TypeError("common_centroid_rows must be an exact tuple")
        centroid_rows: list[tuple[str, int]] = []
        for item in self.common_centroid_rows:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError(
                    "common_centroid_rows entries must be exact (group, rows) tuples"
                )
            group, rows = item
            if (
                type(group) is not str
                or not group
                or group != group.strip()
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", group) is None
            ):
                raise ValueError(
                    "common_centroid_rows group must be an exact Clay identifier"
                )
            if type(rows) is not int or rows not in (1, 2):
                raise ValueError("common_centroid_rows row count must be 1 or 2")
            centroid_rows.append((group, rows))
        if len({group for group, _rows in centroid_rows}) != len(centroid_rows):
            raise ValueError("common_centroid_rows groups must be unique")
        object.__setattr__(self, "common_centroid_rows", tuple(sorted(centroid_rows)))

        channel_strategy = self.channel_strategy
        if type(channel_strategy) is not str:
            raise TypeError("channel_strategy must be an exact string")
        if channel_strategy not in {"declared", "block_gap_only"}:
            raise ValueError(
                "channel_strategy must be one of ['block_gap_only', 'declared']"
            )

        passive_partitions = tuple(sorted(self.passive_bank_partitions))
        well_partitions = tuple(sorted(self.well_island_partitions))
        if any(
            not isinstance(choice, MacroPartitionChoice)
            for choice in passive_partitions + well_partitions
        ):
            raise TypeError("macro partitions must be MacroPartitionChoice values")
        macro_ids = [
            choice.macro_id for choice in passive_partitions + well_partitions
        ]
        if len(macro_ids) != len(set(macro_ids)):
            raise ValueError("duplicate macro partition ownership")

        graph_sha = self.physical_macro_graph_sha256
        if graph_sha is not None:
            graph_sha = str(graph_sha).strip().lower()
            if re.fullmatch(r"[0-9a-f]{64}", graph_sha) is None:
                raise ValueError(
                    "physical_macro_graph_sha256 must be a 64-hex digest"
                )
        if (
            passive_partitions
            or well_partitions
            or nonmos_macro_rows
        ) and graph_sha is None:
            raise ValueError(
                "physical_macro_graph_sha256 is required for graph-owned transforms"
            )

        if well_partitions and rings:
            partition_groups = {
                frozenset(group)
                for choice in well_partitions
                for group in choice.member_groups
            }
            legacy_groups = {frozenset(ring.members) for ring in rings}
            if partition_groups != legacy_groups:
                raise ValueError(
                    "legacy guard-ring membership disagrees with graph well partition"
                )

        object.__setattr__(self, "physical_macro_graph_sha256", graph_sha)
        object.__setattr__(self, "passive_bank_partitions", passive_partitions)
        object.__setattr__(self, "well_island_partitions", well_partitions)
        if materialization_mode == "authoring" and any(
            (
                self.passive_splits,
                self.passive_arrays,
                self.nonmos_macro_rows,
                self.guard_ring_groups,
                self.group_order,
                self.group_floorplan,
                self.passive_orientations,
                self.passive_side != "center",
                self.aspect_mode != "natural",
                self.pack_mode != "rows",
                self.edge_rail_width_um is not None,
                self.mos_fold_height_um is not None,
                self.block_gap_um is not None,
                self.physical_macro_graph_sha256 is not None,
                self.passive_bank_partitions,
                self.well_island_partitions,
                self.left_column_group,
                self.common_centroid_rows,
                self.channel_strategy != "declared",
            )
        ):
            raise ValueError(
                "authoring materialization_mode cannot carry derived transforms"
            )

    def canonical_data(self) -> dict[str, object]:
        data = {
            "aspect_mode": self.aspect_mode,
            "block_gap_um": self.block_gap_um,
            "edge_rail_width_um": self.edge_rail_width_um,
            "group_order": list(self.group_order),
            "guard_ring_groups": [asdict(item) for item in self.guard_ring_groups],
            "mos_fold_height_um": self.mos_fold_height_um,
            "pack_mode": self.pack_mode,
            "passive_side": self.passive_side,
            "passive_splits": [asdict(item) for item in self.passive_splits],
        }
        if self.materialization_mode != "derived":
            data["materialization_mode"] = self.materialization_mode
        # Keep the already-published v1 candidate bytes/digests stable when this
        # later optional intent axis is unused.
        if self.passive_orientations:
            data["passive_orientations"] = [
                {"passive": passive, "orientation": orientation}
                for passive, orientation in self.passive_orientations
            ]
        if self.passive_arrays:
            data["passive_arrays"] = [asdict(item) for item in self.passive_arrays]
        if self.nonmos_macro_rows:
            data["nonmos_macro_rows"] = [
                asdict(item) for item in self.nonmos_macro_rows
            ]
        if self.group_floorplan:
            data["group_floorplan"] = [list(item) for item in self.group_floorplan]
        if self.physical_macro_graph_sha256 is not None:
            data["physical_macro_graph_sha256"] = self.physical_macro_graph_sha256
        if self.passive_bank_partitions:
            data["passive_bank_partitions"] = [
                _macro_partition_data(choice)
                for choice in self.passive_bank_partitions
            ]
        if self.well_island_partitions:
            data["well_island_partitions"] = [
                _macro_partition_data(choice)
                for choice in self.well_island_partitions
            ]
        if self.left_column_group:
            data["left_column_group"] = self.left_column_group
        if self.common_centroid_rows:
            data["common_centroid_rows"] = [
                [group, rows] for group, rows in self.common_centroid_rows
            ]
        if self.channel_strategy != "declared":
            data["channel_strategy"] = self.channel_strategy
        return data

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


def compact_intent_candidate_from_data(
    data: Mapping[str, object],
) -> CompactIntentCandidate:
    """Rehydrate one normalized candidate from auditable Clay provenance."""

    expected = {
        "aspect_mode",
        "block_gap_um",
        "edge_rail_width_um",
        "group_order",
        "guard_ring_groups",
        "mos_fold_height_um",
        "pack_mode",
        "passive_side",
        "passive_splits",
    }
    optional = {
        "materialization_mode",
        "passive_orientations",
        "passive_arrays",
        "nonmos_macro_rows",
        "physical_macro_graph_sha256",
        "passive_bank_partitions",
        "well_island_partitions",
        "left_column_group",
        "common_centroid_rows",
        "channel_strategy",
        "group_floorplan",
    }
    unknown = sorted(set(data) - expected - optional)
    missing = sorted(expected - set(data))
    if unknown or missing:
        raise ValueError(
            "compact candidate provenance fields disagree with the canonical "
            f"schema: missing={missing}, unknown={unknown}"
        )
    raw_splits = data["passive_splits"]
    raw_rings = data["guard_ring_groups"]
    raw_order = data["group_order"]
    raw_orientations = data.get("passive_orientations", [])
    raw_passive_arrays = data.get("passive_arrays", [])
    raw_nonmos_macro_rows = data.get("nonmos_macro_rows", [])
    raw_passive_partitions = data.get("passive_bank_partitions", [])
    raw_well_partitions = data.get("well_island_partitions", [])
    raw_centroid_rows = data.get("common_centroid_rows", [])
    raw_group_floorplan = data.get("group_floorplan", [])
    if not isinstance(raw_splits, list) or not isinstance(raw_rings, list):
        raise ValueError("compact candidate provenance lists are malformed")
    if not isinstance(raw_order, list) or any(
        type(item) is not str for item in raw_order
    ):
        raise ValueError("compact candidate group_order provenance is malformed")
    if not isinstance(raw_orientations, list):
        raise ValueError("compact candidate passive orientations are malformed")
    if not isinstance(raw_passive_arrays, list):
        raise ValueError("compact candidate passive arrays are malformed")
    if not isinstance(raw_nonmos_macro_rows, list):
        raise ValueError("compact candidate non-MOS macro rows are malformed")
    if not isinstance(raw_passive_partitions, list) or not isinstance(
        raw_well_partitions, list
    ):
        raise ValueError("compact candidate macro partitions are malformed")
    if not isinstance(raw_centroid_rows, list):
        raise ValueError("compact candidate common-centroid rows are malformed")
    if not isinstance(raw_group_floorplan, list) or any(
        type(item) is not list
        or len(item) != 3
        or type(item[0]) is not str
        or type(item[1]) is not int
        or type(item[2]) is not int
        for item in raw_group_floorplan
    ):
        raise ValueError("compact candidate group_floorplan provenance is malformed")
    try:
        splits = tuple(
            PassiveSplitChoice(
                passive=str(item["passive"]),
                unit_count=int(item["unit_count"]),
                connection=str(item["connection"]),
                orientation=str(item["orientation"]),
            )
            for item in raw_splits
            if isinstance(item, Mapping)
        )
        rings = tuple(
            GuardRingChoice(
                members=tuple(str(member) for member in item["members"]),
                well=str(item["well"]),
                tie_net=str(item["tie_net"]),
            )
            for item in raw_rings
            if isinstance(item, Mapping)
        )
        passive_orientations = tuple(
            (str(item["passive"]), str(item["orientation"]))
            for item in raw_orientations
            if isinstance(item, Mapping)
        )
        passive_arrays = tuple(
            PassiveArrayChoice(
                passive=item["passive"],
                rows=item["rows"],
            )
            for item in raw_passive_arrays
            if isinstance(item, Mapping)
        )
        nonmos_macro_rows = tuple(
            NonmosMacroRowsChoice(
                macro_id=item["macro_id"],
                rows=item["rows"],
            )
            for item in raw_nonmos_macro_rows
            if isinstance(item, Mapping)
        )
        passive_partitions = _macro_partitions_from_data(raw_passive_partitions)
        well_partitions = _macro_partitions_from_data(raw_well_partitions)
        centroid_rows = tuple(
            (item[0], item[1])
            for item in raw_centroid_rows
            if type(item) is list
            and len(item) == 2
            and type(item[0]) is str
            and type(item[1]) is int
        )
        group_floorplan = tuple(
            (item[0], item[1], item[2]) for item in raw_group_floorplan
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("compact candidate provenance is malformed") from exc
    if (
        len(splits) != len(raw_splits)
        or len(rings) != len(raw_rings)
        or len(passive_orientations) != len(raw_orientations)
        or len(passive_arrays) != len(raw_passive_arrays)
        or len(nonmos_macro_rows) != len(raw_nonmos_macro_rows)
        or len(passive_partitions) != len(raw_passive_partitions)
        or len(well_partitions) != len(raw_well_partitions)
        or len(centroid_rows) != len(raw_centroid_rows)
    ):
        raise ValueError("compact candidate provenance entries must be objects")
    return CompactIntentCandidate(
        materialization_mode=str(data.get("materialization_mode", "derived")),
        passive_splits=splits,
        passive_arrays=passive_arrays,
        nonmos_macro_rows=nonmos_macro_rows,
        guard_ring_groups=rings,
        group_order=tuple(raw_order),
        passive_orientations=passive_orientations,
        passive_side=str(data["passive_side"]),
        aspect_mode=str(data["aspect_mode"]),
        pack_mode=str(data["pack_mode"]),
        edge_rail_width_um=(
            None
            if data["edge_rail_width_um"] is None
            else float(data["edge_rail_width_um"])
        ),
        mos_fold_height_um=(
            None
            if data["mos_fold_height_um"] is None
            else float(data["mos_fold_height_um"])
        ),
        block_gap_um=(
            None if data["block_gap_um"] is None else float(data["block_gap_um"])
        ),
        physical_macro_graph_sha256=(
            None
            if data.get("physical_macro_graph_sha256") is None
            else str(data["physical_macro_graph_sha256"])
        ),
        passive_bank_partitions=passive_partitions,
        well_island_partitions=well_partitions,
        left_column_group=str(data.get("left_column_group", "")),
        common_centroid_rows=centroid_rows,
        channel_strategy=data.get("channel_strategy", "declared"),
        group_floorplan=group_floorplan,
    )


def replay_compact_intent_candidate(value: object) -> CompactIntentCandidate:
    """Re-run the exact candidate schema at every physical authority boundary."""

    if type(value) is not CompactIntentCandidate:
        raise TypeError(
            "compact intent candidate must be an exact CompactIntentCandidate"
        )
    return replace(value)


def _macro_partition_data(choice: MacroPartitionChoice) -> dict[str, object]:
    return {
        "macro_id": choice.macro_id,
        "member_groups": [list(group) for group in choice.member_groups],
        "declaration_refs": list(choice.declaration_refs),
    }


def _macro_partitions_from_data(
    raw: list[object],
) -> tuple[MacroPartitionChoice, ...]:
    result: list[MacroPartitionChoice] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        groups = item.get("member_groups")
        refs = item.get("declaration_refs", [])
        if not isinstance(groups, list) or not isinstance(refs, list):
            raise ValueError("compact candidate macro partition is malformed")
        if any(not isinstance(group, list) for group in groups):
            raise ValueError("compact candidate macro partition groups are malformed")
        result.append(
            MacroPartitionChoice(
                macro_id=str(item["macro_id"]),
                member_groups=tuple(
                    tuple(str(member) for member in group) for group in groups
                ),
                declaration_refs=tuple(str(ref) for ref in refs),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class CompactIntentMetrics:
    """Measured truth used to rank candidates that passed every hard gate."""

    bbox_area_um2: float
    whitespace_um2: float
    wire_length_um: float
    via_count: int
    ring_perimeter_um: float
    post_pex_fom: float

    def __post_init__(self) -> None:
        continuous = (
            self.bbox_area_um2,
            self.whitespace_um2,
            self.wire_length_um,
            self.ring_perimeter_um,
            self.post_pex_fom,
        )
        if not all(math.isfinite(float(value)) for value in continuous):
            raise ValueError("compact-intent metrics must be finite")
        if any(float(value) < 0.0 for value in continuous):
            raise ValueError("compact-intent metrics must be non-negative")
        if type(self.via_count) is not int or self.via_count < 0:
            raise ValueError("via_count must be a non-negative integer")

    def ranking_key(self, digest: str) -> tuple[float | int | str, ...]:
        return (
            self.bbox_area_um2,
            self.whitespace_um2,
            self.wire_length_um,
            self.via_count,
            self.ring_perimeter_um,
            -self.post_pex_fom,
            str(digest),
        )


SignedCandidate = tuple[CompactIntentCandidate, CompactIntentMetrics]


def rank_signed_candidates(
    items: Iterable[SignedCandidate],
) -> list[SignedCandidate]:
    """Return signed candidates in the design-specified lexicographic order."""

    materialized: Sequence[SignedCandidate] = tuple(items)
    return sorted(
        materialized,
        key=lambda item: item[1].ranking_key(item[0].digest),
    )


__all__ = [
    "CompactIntentCandidate",
    "CompactIntentEvidence",
    "CompactIntentMetrics",
    "CompactIntentTruth",
    "ElectricalResultReceipt",
    "GuardAtomCompatibility",
    "GuardRingAtom",
    "GuardRingChoice",
    "MacroPartitionChoice",
    "NonmosMacroRowsChoice",
    "PassiveArrayChoice",
    "PassiveSplitChoice",
    "PhysicalPexArtifact",
    "SignedCandidate",
    "compact_intent_candidate_from_data",
    "compact_intent_evidence_from_provenance",
    "load_compact_intent_provenance_json",
    "derive_guard_ring_atoms",
    "enumerate_guard_ring_partitions",
    "guard_atom_compatibility",
    "rank_signed_candidates",
    "replay_compact_intent_candidate",
]
