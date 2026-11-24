"""Immutable bindings from logical devices to extracted physical occurrences."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re


_SCHEMA = "clay.semantic-spatial-overlay/v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ORIENTATIONS = frozenset(
    {"R0", "R90", "R180", "R270", "MR0", "MR90", "MR180", "MR270"}
)
_DEVICE_CLASSES = frozenset({"nmos", "pmos", "capacitor", "resistor", "pnp", "npn"})
_SOURCE_KINDS = frozenset({"reference_gds", "current_placement", "native_pcell"})


class SemanticOverlayError(ValueError):
    """A semantic occurrence cover is ambiguous or incomplete."""


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SemanticOverlayError(f"{label} must be one lowercase SHA-256")
    return value


def _require_canonical_strings(value: object, *, label: str) -> tuple[str, ...]:
    if (
        type(value) is not tuple
        or any(type(item) is not str or not item or item != item.strip() for item in value)
        or tuple(sorted(set(value))) != value
    ):
        raise SemanticOverlayError(f"{label} must be sorted unique strings")
    return value


@dataclass(frozen=True, slots=True, order=True)
class SemanticParameterKey:
    family: str
    width_nm: int
    length_nm: int
    units: int

    def __post_init__(self) -> None:
        if (
            type(self.family) is not str
            or not self.family
            or self.family != self.family.strip()
            or any(
                type(value) is not int or value <= 0
                for value in (self.width_nm, self.length_nm, self.units)
            )
        ):
            raise SemanticOverlayError("semantic parameter key is malformed")

    def canonical_data(self) -> list[object]:
        return [self.family, self.width_nm, self.length_nm, self.units]


@dataclass(frozen=True, slots=True, order=True)
class SemanticPhysicalBinding:
    anchor_nm: tuple[int, int]
    envelope_nm: tuple[int, int, int, int]
    orientation: str

    def __post_init__(self) -> None:
        if (
            type(self.anchor_nm) is not tuple
            or len(self.anchor_nm) != 2
            or any(type(value) is not int for value in self.anchor_nm)
            or type(self.envelope_nm) is not tuple
            or len(self.envelope_nm) != 4
            or any(type(value) is not int for value in self.envelope_nm)
            or self.envelope_nm[2] <= self.envelope_nm[0]
            or self.envelope_nm[3] <= self.envelope_nm[1]
            or not self.envelope_nm[0] <= self.anchor_nm[0] <= self.envelope_nm[2]
            or not self.envelope_nm[1] <= self.anchor_nm[1] <= self.envelope_nm[3]
            or self.orientation not in _ORIENTATIONS
        ):
            raise SemanticOverlayError("semantic physical binding is malformed")

    def canonical_data(self) -> dict[str, object]:
        return {
            "anchor_nm": list(self.anchor_nm),
            "envelope_nm": list(self.envelope_nm),
            "orientation": self.orientation,
        }

    def translated(self, *, dx_nm: int, dy_nm: int) -> "SemanticPhysicalBinding":
        if type(dx_nm) is not int or type(dy_nm) is not int:
            raise TypeError("semantic translation must use exact integer nm")
        x0, y0, x1, y1 = self.envelope_nm
        return SemanticPhysicalBinding(
            anchor_nm=(self.anchor_nm[0] + dx_nm, self.anchor_nm[1] + dy_nm),
            envelope_nm=(x0 + dx_nm, y0 + dy_nm, x1 + dx_nm, y1 + dy_nm),
            orientation=self.orientation,
        )


@dataclass(frozen=True, slots=True, order=True)
class SemanticOccurrence:
    node_id: str
    logical_owner: str
    occurrence_ordinal: int
    device_class: str
    parameter_key: SemanticParameterKey
    master_content_sha256: str
    master_artwork_sha256: str
    physical: SemanticPhysicalBinding
    terminal_ids: tuple[str, ...]
    body_well_id: str | None
    binding_status: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        expected_node = f"{self.logical_owner}[{self.occurrence_ordinal:03d}]"
        if (
            type(self.logical_owner) is not str
            or not self.logical_owner
            or type(self.occurrence_ordinal) is not int
            or self.occurrence_ordinal < 0
            or self.node_id != expected_node
            or self.device_class not in _DEVICE_CLASSES
            or type(self.parameter_key) is not SemanticParameterKey
            or type(self.physical) is not SemanticPhysicalBinding
            or type(self.binding_status) is not str
            or not self.binding_status
        ):
            raise SemanticOverlayError("semantic occurrence is malformed")
        _require_sha256(self.master_content_sha256, label="master content identity")
        _require_sha256(self.master_artwork_sha256, label="master artwork identity")
        _require_canonical_strings(self.terminal_ids, label="occurrence terminal ids")
        _require_canonical_strings(self.evidence_ids, label="occurrence evidence ids")
        is_mos = self.device_class in {"nmos", "pmos"}
        if is_mos != (type(self.body_well_id) is str and bool(self.body_well_id)):
            raise SemanticOverlayError("MOS body/well binding coverage differs")

    def canonical_data(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "logical_owner": self.logical_owner,
            "occurrence_ordinal": self.occurrence_ordinal,
            "device_class": self.device_class,
            "parameter_key": self.parameter_key.canonical_data(),
            "master_content_sha256": self.master_content_sha256,
            "master_artwork_sha256": self.master_artwork_sha256,
            "physical": self.physical.canonical_data(),
            "terminal_ids": list(self.terminal_ids),
            "body_well_id": self.body_well_id,
            "binding_status": self.binding_status,
            "evidence_ids": list(self.evidence_ids),
        }

    @property
    def semantic_signature(self) -> tuple[object, ...]:
        return (
            self.node_id,
            self.logical_owner,
            self.occurrence_ordinal,
            self.device_class,
            self.parameter_key,
            self.master_content_sha256,
            self.master_artwork_sha256,
            self.physical.orientation,
        )


@dataclass(frozen=True, slots=True, order=True)
class SemanticTerminal:
    terminal_id: str
    occurrence_id: str
    role: str
    net: str
    verification_status: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.occurrence_id) is not str
            or not self.occurrence_id
            or type(self.role) is not str
            or not self.role
            or self.terminal_id != f"{self.occurrence_id}:{self.role}"
            or type(self.net) is not str
            or not self.net
            or type(self.verification_status) is not str
            or not self.verification_status
        ):
            raise SemanticOverlayError("semantic terminal is malformed")
        _require_canonical_strings(self.evidence_ids, label="terminal evidence ids")

    def canonical_data(self) -> dict[str, object]:
        return {
            "terminal_id": self.terminal_id,
            "occurrence_id": self.occurrence_id,
            "role": self.role,
            "net": self.net,
            "verification_status": self.verification_status,
            "evidence_ids": list(self.evidence_ids),
        }

    @property
    def semantic_signature(self) -> tuple[str, str, str]:
        return (self.occurrence_id, self.role, self.net)


@dataclass(frozen=True, slots=True, order=True)
class SemanticNet:
    name: str
    terminal_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name or self.name != self.name.strip():
            raise SemanticOverlayError("semantic net name is malformed")
        _require_canonical_strings(self.terminal_ids, label="net terminal ids")
        if not self.terminal_ids:
            raise SemanticOverlayError("semantic net must bind at least one terminal")

    def canonical_data(self) -> dict[str, object]:
        return {"name": self.name, "terminal_ids": list(self.terminal_ids)}


@dataclass(frozen=True, slots=True, order=True)
class SemanticBodyWellBinding:
    binding_id: str
    occurrence_id: str
    body_net: str
    domain_class: str
    verification_status: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.occurrence_id) is not str
            or not self.occurrence_id
            or self.binding_id != f"{self.occurrence_id}:body-well"
            or any(
                type(value) is not str or not value or value != value.strip()
                for value in (
                    self.body_net,
                    self.domain_class,
                    self.verification_status,
                )
            )
        ):
            raise SemanticOverlayError("semantic body/well binding is malformed")
        _require_canonical_strings(self.evidence_ids, label="body/well evidence ids")

    def canonical_data(self) -> dict[str, object]:
        return {
            "binding_id": self.binding_id,
            "occurrence_id": self.occurrence_id,
            "body_net": self.body_net,
            "domain_class": self.domain_class,
            "verification_status": self.verification_status,
            "evidence_ids": list(self.evidence_ids),
        }

    @property
    def semantic_signature(self) -> tuple[str, str, str]:
        return (self.occurrence_id, self.body_net, self.domain_class)


@dataclass(frozen=True, slots=True)
class SemanticOverlay:
    source_kind: str
    source_sha256: str
    topology_sha256: str
    occurrences: tuple[SemanticOccurrence, ...]
    terminals: tuple[SemanticTerminal, ...]
    nets: tuple[SemanticNet, ...]
    body_wells: tuple[SemanticBodyWellBinding, ...]

    def __post_init__(self) -> None:
        if self.source_kind not in _SOURCE_KINDS:
            raise SemanticOverlayError("semantic overlay source kind is unsupported")
        _require_sha256(self.source_sha256, label="semantic source identity")
        _require_sha256(self.topology_sha256, label="semantic topology identity")
        for values, expected_type, label in (
            (self.occurrences, SemanticOccurrence, "occurrences"),
            (self.terminals, SemanticTerminal, "terminals"),
            (self.nets, SemanticNet, "nets"),
            (self.body_wells, SemanticBodyWellBinding, "body/well bindings"),
        ):
            if type(values) is not tuple or any(type(row) is not expected_type for row in values):
                raise TypeError(f"semantic {label} contain foreign values")

        origin_x = min(row.physical.envelope_nm[0] for row in self.occurrences)
        origin_y = min(row.physical.envelope_nm[1] for row in self.occurrences)
        occurrences = tuple(
            sorted(
                (
                    replace(
                        row,
                        physical=row.physical.translated(
                            dx_nm=-origin_x,
                            dy_nm=-origin_y,
                        ),
                    )
                    for row in self.occurrences
                ),
                key=lambda row: row.node_id,
            )
        )
        terminals = tuple(sorted(self.terminals, key=lambda row: row.terminal_id))
        nets = tuple(sorted(self.nets, key=lambda row: row.name))
        body_wells = tuple(sorted(self.body_wells, key=lambda row: row.binding_id))
        if len({row.node_id for row in occurrences}) != len(occurrences):
            raise SemanticOverlayError("occurrence node ids must be unique")
        if len({row.terminal_id for row in terminals}) != len(terminals):
            raise SemanticOverlayError("terminal ids must be unique")
        if len({row.name for row in nets}) != len(nets):
            raise SemanticOverlayError("net names must be unique")
        if len({row.binding_id for row in body_wells}) != len(body_wells):
            raise SemanticOverlayError("body/well binding ids must be unique")

        occurrence_by_id = {row.node_id: row for row in occurrences}
        terminal_by_id = {row.terminal_id: row for row in terminals}
        body_by_id = {row.binding_id: row for row in body_wells}
        for occurrence in occurrences:
            owned_terminals = tuple(
                row.terminal_id
                for row in terminals
                if row.occurrence_id == occurrence.node_id
            )
            if owned_terminals != occurrence.terminal_ids:
                raise SemanticOverlayError(
                    f"terminal coverage differs for {occurrence.node_id}"
                )
            if occurrence.body_well_id is not None:
                body = body_by_id.get(occurrence.body_well_id)
                if body is None or body.occurrence_id != occurrence.node_id:
                    raise SemanticOverlayError(
                        f"body/well coverage differs for {occurrence.node_id}"
                    )
        if any(row.occurrence_id not in occurrence_by_id for row in terminals):
            raise SemanticOverlayError("terminal references an unknown occurrence")
        if any(row.occurrence_id not in occurrence_by_id for row in body_wells):
            raise SemanticOverlayError("body/well references an unknown occurrence")

        net_terminal_ids = tuple(
            terminal_id for net in nets for terminal_id in net.terminal_ids
        )
        if len(net_terminal_ids) != len(set(net_terminal_ids)):
            raise SemanticOverlayError("a terminal is multiply bound to semantic nets")
        if set(net_terminal_ids) != set(terminal_by_id):
            raise SemanticOverlayError("semantic net terminal cover differs")
        if any(
            terminal_by_id[terminal_id].net != net.name
            for net in nets
            for terminal_id in net.terminal_ids
        ):
            raise SemanticOverlayError("semantic net membership differs")

        by_logical: dict[str, list[int]] = {}
        for occurrence in occurrences:
            by_logical.setdefault(occurrence.logical_owner, []).append(
                occurrence.occurrence_ordinal
            )
        if any(sorted(ordinals) != list(range(len(ordinals))) for ordinals in by_logical.values()):
            raise SemanticOverlayError("occurrence ordinals must be contiguous per owner")

        object.__setattr__(self, "occurrences", occurrences)
        object.__setattr__(self, "terminals", terminals)
        object.__setattr__(self, "nets", nets)
        object.__setattr__(self, "body_wells", body_wells)

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(row.node_id for row in self.occurrences)

    @property
    def logical_owners(self) -> tuple[str, ...]:
        return tuple(sorted({row.logical_owner for row in self.occurrences}))

    @property
    def logical_multiplicities(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (owner, sum(row.logical_owner == owner for row in self.occurrences))
            for owner in self.logical_owners
        )

    @property
    def unbound_node_ids(self) -> tuple[str, ...]:
        terminal_owners = {row.occurrence_id for row in self.terminals}
        body_owners = {row.occurrence_id for row in self.body_wells}
        return tuple(
            row.node_id
            for row in self.occurrences
            if row.node_id not in terminal_owners
            or (
                row.device_class in {"nmos", "pmos"}
                and row.node_id not in body_owners
            )
        )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "source_kind": self.source_kind,
            "source_sha256": self.source_sha256,
            "topology_sha256": self.topology_sha256,
            "occurrences": [row.canonical_data() for row in self.occurrences],
            "terminals": [row.canonical_data() for row in self.terminals],
            "nets": [row.canonical_data() for row in self.nets],
            "body_wells": [row.canonical_data() for row in self.body_wells],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


def semantic_nets_from_terminals(
    terminals: tuple[SemanticTerminal, ...],
) -> tuple[SemanticNet, ...]:
    """Build the exact deterministic net cover for terminal records."""

    if type(terminals) is not tuple or any(type(row) is not SemanticTerminal for row in terminals):
        raise TypeError("semantic terminals contain foreign values")
    names = tuple(sorted({row.net for row in terminals}))
    return tuple(
        SemanticNet(
            name=name,
            terminal_ids=tuple(
                sorted(row.terminal_id for row in terminals if row.net == name)
            ),
        )
        for name in names
    )


__all__ = [
    "SemanticBodyWellBinding",
    "SemanticNet",
    "SemanticOccurrence",
    "SemanticOverlay",
    "SemanticOverlayError",
    "SemanticParameterKey",
    "SemanticPhysicalBinding",
    "SemanticTerminal",
    "semantic_nets_from_terminals",
]
