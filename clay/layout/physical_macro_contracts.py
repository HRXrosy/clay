"""Exact pre-solver interfaces and placed bindings for physical macro nodes.

This module starts with declared singleton power arrays.  A singleton is already
one indivisible solver footprint, which makes it the smallest honest non-row
macro boundary: its exact PDK geometry and routing accesses are closed before the
solver runs, then replayed as a translation-only view after placement.  Larger
matched/passive/BJT aggregates are intentionally not claimed by this schema yet.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping


_HEX = frozenset("0123456789abcdef")
_MAX_ITEMS = 4096
_MAX_ARTWORK_POLYGONS = 262_144
_MAX_COORDINATE_NM = 10**12
_SUPPORTED_SINGLETON_KINDS = frozenset({"power_array"})


class PhysicalMacroContractError(ValueError):
    """A declared macro cannot be lowered to one exact pre-solver interface."""


class PhysicalMacroProjectionError(RuntimeError):
    """A placed macro no longer replays its exact pre-solver interface."""

    def __init__(self, code: str) -> None:
        self.code = str(code)
        super().__init__(self.code)


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise PhysicalMacroContractError(f"{field} must be a non-empty exact string")
    return value


def _digest(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in _HEX for char in value)
    ):
        raise PhysicalMacroContractError(f"{field} must be one canonical sha256")
    return value


def _bounded_tuple(value: object, field: str, *, allow_empty: bool = True) -> tuple:
    if type(value) is not tuple:
        raise PhysicalMacroContractError(f"{field} must be an exact tuple")
    if len(value) > _MAX_ITEMS:
        raise PhysicalMacroContractError(f"{field} exceeds the physical item cap")
    if not allow_empty and not value:
        raise PhysicalMacroContractError(f"{field} must not be empty")
    return value


def _coordinate(value: object, field: str) -> int:
    if (
        type(value) is not int
        or value < -_MAX_COORDINATE_NM
        or value > _MAX_COORDINATE_NM
    ):
        raise PhysicalMacroContractError(f"{field} must be one bounded exact integer")
    return value


def _point(value: object, field: str) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        raise PhysicalMacroContractError(f"{field} must be one exact integer pair")
    return (_coordinate(value[0], f"{field} x"), _coordinate(value[1], f"{field} y"))


def _bbox(value: object, field: str) -> tuple[int, int, int, int]:
    if type(value) is not tuple or len(value) != 4:
        raise PhysicalMacroContractError(f"{field} must be one exact bbox tuple")
    result = tuple(_coordinate(item, field) for item in value)
    if result[2] <= result[0] or result[3] <= result[1]:
        raise PhysicalMacroContractError(f"{field} must have positive area")
    return result  # type: ignore[return-value]


def _layer(value: object) -> tuple[int, int]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(item) is not int or item < 0 or item > 65_535 for item in value)
    ):
        raise PhysicalMacroContractError(
            "terminal layer must be an exact non-negative integer pair"
        )
    return value


def _sha(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _canonical_ring(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        raise PhysicalMacroContractError("macro artwork polygon is degenerate")
    variants = []
    for sequence in (points, tuple(reversed(points))):
        variants.extend(
            sequence[index:] + sequence[:index]
            for index in range(len(sequence))
        )
    return min(variants)


def _artwork_sha256(polygons: object, *, origin_nm: tuple[int, int]) -> str:
    try:
        values = tuple(polygons)
    except TypeError as exc:
        raise PhysicalMacroContractError("macro artwork must be iterable") from exc
    if len(values) > _MAX_ARTWORK_POLYGONS:
        raise PhysicalMacroContractError("macro artwork exceeds the physical item cap")
    if not values:
        raise PhysicalMacroContractError("macro artwork must not be empty")
    ox, oy = origin_nm
    rows = []
    for polygon in values:
        layer = getattr(polygon, "layer", None)
        datatype = getattr(polygon, "datatype", None)
        vertices = tuple(getattr(getattr(polygon, "polygon", None), "vertices", ()) or ())
        if (
            type(layer) is not int
            or type(datatype) is not int
            or not 0 <= layer <= 65_535
            or not 0 <= datatype <= 65_535
        ):
            raise PhysicalMacroContractError("macro artwork lacks an exact layer key")
        points = tuple(
            (
                _coordinate(getattr(point, "x", None), "artwork x") - ox,
                _coordinate(getattr(point, "y", None), "artwork y") - oy,
            )
            for point in vertices
        )
        rows.append((layer, datatype, _canonical_ring(points)))
    return _sha(tuple(sorted(rows)))


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class PhysicalMacroTerminalAccess:
    member: str
    role: str
    net: str
    layer: tuple[int, int]
    accesses_nm: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "member", _text(self.member, "terminal member"))
        object.__setattr__(self, "role", _text(self.role, "terminal role"))
        object.__setattr__(self, "net", _text(self.net, "terminal net"))
        object.__setattr__(self, "layer", _layer(self.layer))
        accesses = _bounded_tuple(
            self.accesses_nm, "terminal accesses", allow_empty=False
        )
        rebuilt = tuple(sorted(_point(item, "terminal access") for item in accesses))
        if len(rebuilt) != len(set(rebuilt)):
            raise PhysicalMacroContractError("terminal accesses contain duplicates")
        object.__setattr__(self, "accesses_nm", rebuilt)

    def payload(self) -> tuple[object, ...]:
        return (self.member, self.role, self.net, self.layer, self.accesses_nm)


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class PhysicalMacroRouteDemand:
    net: str
    members: tuple[str, ...]
    terminal_count: int
    access_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "net", _text(self.net, "route demand net"))
        members = _bounded_tuple(self.members, "route demand members", allow_empty=False)
        if any(type(item) is not str or not item for item in members):
            raise PhysicalMacroContractError(
                "route demand members must be non-empty exact strings"
            )
        canonical = tuple(sorted(members))
        if members != canonical or len(members) != len(set(members)):
            raise PhysicalMacroContractError("route demand members must be canonical")
        for field in ("terminal_count", "access_count"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0 or value > _MAX_ITEMS:
                raise PhysicalMacroContractError(
                    f"route demand {field} must be a bounded positive exact integer"
                )

    def payload(self) -> tuple[object, ...]:
        return (self.net, self.members, self.terminal_count, self.access_count)


@dataclass(frozen=True, slots=True, kw_only=True)
class PhysicalMacroContract:
    macro_id: str
    kind: str
    members: tuple[str, ...]
    local_bbox_nm: tuple[int, int, int, int]
    terminal_accesses: tuple[PhysicalMacroTerminalAccess, ...]
    local_symmetry_axes_2nm: tuple[tuple[str, int], ...]
    body_tie_domain: tuple[str, ...]
    legal_orientations: tuple[str, ...]
    route_demand: tuple[PhysicalMacroRouteDemand, ...]
    artwork_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "macro_id", _text(self.macro_id, "macro_id"))
        if self.kind not in _SUPPORTED_SINGLETON_KINDS:
            raise PhysicalMacroContractError(
                f"unsupported singleton macro kind {self.kind!r}"
            )
        members = _bounded_tuple(self.members, "macro members", allow_empty=False)
        if (
            len(members) != 1
            or any(type(item) is not str or not item for item in members)
        ):
            raise PhysicalMacroContractError(
                "singleton macro contract must own exactly one member"
            )
        object.__setattr__(self, "local_bbox_nm", _bbox(
            self.local_bbox_nm, "macro local bbox"
        ))
        if self.local_bbox_nm[0:2] != (0, 0):
            raise PhysicalMacroContractError("macro local bbox must start at the origin")
        terminals = _bounded_tuple(
            self.terminal_accesses, "macro terminal accesses", allow_empty=False
        )
        if any(type(item) is not PhysicalMacroTerminalAccess for item in terminals):
            raise PhysicalMacroContractError(
                "macro terminal accesses must contain exact typed values"
            )
        canonical_terminals = tuple(sorted(terminals))
        if terminals != canonical_terminals:
            raise PhysicalMacroContractError("macro terminals must use canonical order")
        if any(item.member != members[0] for item in terminals):
            raise PhysicalMacroContractError("macro terminal names a foreign member")
        keys = tuple((item.member, item.role) for item in terminals)
        if len(keys) != len(set(keys)):
            raise PhysicalMacroContractError("macro terminal roles must be unique")
        axes = _bounded_tuple(self.local_symmetry_axes_2nm, "macro symmetry axes")
        if axes:
            raise PhysicalMacroContractError(
                "singleton power macro cannot claim an inferred symmetry axis"
            )
        body = _bounded_tuple(self.body_tie_domain, "macro body tie domain")
        if any(type(item) is not str or not item for item in body):
            raise PhysicalMacroContractError("body tie domain must contain exact strings")
        if body != tuple(sorted(set(body))):
            raise PhysicalMacroContractError("body tie domain must be canonical")
        orientations = _bounded_tuple(
            self.legal_orientations, "macro legal orientations", allow_empty=False
        )
        if (
            orientations != tuple(sorted(set(orientations)))
            or any(item not in {"R0", "R90", "R180", "R270"} for item in orientations)
        ):
            raise PhysicalMacroContractError("macro legal orientations are invalid")
        demand = _bounded_tuple(self.route_demand, "macro route demand", allow_empty=False)
        if any(type(item) is not PhysicalMacroRouteDemand for item in demand):
            raise PhysicalMacroContractError(
                "macro route demand must contain exact typed values"
            )
        if demand != tuple(sorted(demand)):
            raise PhysicalMacroContractError("macro route demand must be canonical")
        derived = []
        for net in sorted({item.net for item in terminals}):
            selected = tuple(item for item in terminals if item.net == net)
            derived.append(PhysicalMacroRouteDemand(
                net=net,
                members=tuple(sorted({item.member for item in selected})),
                terminal_count=len(selected),
                access_count=sum(len(item.accesses_nm) for item in selected),
            ))
        if demand != tuple(derived):
            raise PhysicalMacroContractError(
                "macro route demand differs from its terminal interface"
            )
        object.__setattr__(self, "artwork_sha256", _digest(
            self.artwork_sha256, "macro artwork_sha256"
        ))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "clay.physical-macro-contract/v1",
            "macro_id": self.macro_id,
            "kind": self.kind,
            "members": self.members,
            "local_bbox_nm": self.local_bbox_nm,
            "terminal_accesses": tuple(item.payload() for item in self.terminal_accesses),
            "local_symmetry_axes_2nm": self.local_symmetry_axes_2nm,
            "body_tie_domain": self.body_tie_domain,
            "legal_orientations": self.legal_orientations,
            "route_demand": tuple(item.payload() for item in self.route_demand),
            "artwork_sha256": self.artwork_sha256,
        }

    @property
    def structure_fingerprint(self) -> str:
        return _sha(self.payload())


@dataclass(frozen=True, slots=True, kw_only=True)
class PhysicalMacroContractSet:
    graph_sha256: str
    tech_id: str
    macros: tuple[PhysicalMacroContract, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "graph_sha256", _digest(
            self.graph_sha256, "graph_sha256"
        ))
        object.__setattr__(self, "tech_id", _text(self.tech_id, "contract tech_id"))
        macros = _bounded_tuple(self.macros, "physical macro contracts")
        if any(type(item) is not PhysicalMacroContract for item in macros):
            raise PhysicalMacroContractError(
                "physical macro contract set requires exact typed contracts"
            )
        if macros != tuple(sorted(macros, key=lambda item: item.macro_id)):
            raise PhysicalMacroContractError("physical macro contracts must be canonical")
        if len({item.macro_id for item in macros}) != len(macros):
            raise PhysicalMacroContractError("physical macro contract ids must be unique")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "clay.physical-macro-contract-set/v1",
            "graph_sha256": self.graph_sha256,
            "tech_id": self.tech_id,
            "macros": tuple(item.payload() for item in self.macros),
        }

    @property
    def sha256(self) -> str:
        return _sha(self.payload())


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class PhysicalMacroMemberSnapshot:
    """Frozen exact geometry/pin input captured before placement."""

    member: str
    local_bbox_nm: tuple[int, int, int, int]
    terminal_accesses: tuple[PhysicalMacroTerminalAccess, ...]
    body_tie_domain: tuple[str, ...]
    legal_orientations: tuple[str, ...]
    artwork_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "member", _text(self.member, "snapshot member"))
        object.__setattr__(self, "local_bbox_nm", _bbox(
            self.local_bbox_nm, "snapshot local bbox"
        ))
        if self.local_bbox_nm[0:2] != (0, 0):
            raise PhysicalMacroContractError(
                "snapshot local bbox must start at the origin"
            )
        terminals = _bounded_tuple(
            self.terminal_accesses, "snapshot terminal accesses", allow_empty=False
        )
        if any(type(item) is not PhysicalMacroTerminalAccess for item in terminals):
            raise PhysicalMacroContractError(
                "snapshot terminals must contain exact typed accesses"
            )
        if terminals != tuple(sorted(terminals)) or any(
            item.member != self.member for item in terminals
        ):
            raise PhysicalMacroContractError(
                "snapshot terminals must be canonical and owned"
            )
        body = _bounded_tuple(self.body_tie_domain, "snapshot body tie domain")
        if body != tuple(sorted(set(body))) or any(
            type(item) is not str or not item for item in body
        ):
            raise PhysicalMacroContractError(
                "snapshot body tie domain must be canonical exact strings"
            )
        orientations = _bounded_tuple(
            self.legal_orientations,
            "snapshot legal orientations",
            allow_empty=False,
        )
        if orientations != tuple(sorted(set(orientations))) or any(
            item not in {"R0", "R90", "R180", "R270"}
            for item in orientations
        ):
            raise PhysicalMacroContractError(
                "snapshot legal orientations are invalid"
            )
        object.__setattr__(self, "artwork_sha256", _digest(
            self.artwork_sha256, "snapshot artwork_sha256"
        ))

    def payload(self) -> dict[str, object]:
        return {
            "member": self.member,
            "local_bbox_nm": self.local_bbox_nm,
            "terminal_accesses": tuple(
                item.payload() for item in self.terminal_accesses
            ),
            "body_tie_domain": self.body_tie_domain,
            "legal_orientations": self.legal_orientations,
            "artwork_sha256": self.artwork_sha256,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class PhysicalMacroCompileInputs:
    """Canonical graph/PDK/member facts from which contracts must replay."""

    graph_sha256: str
    tech_id: str
    members: tuple[PhysicalMacroMemberSnapshot, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "graph_sha256", _digest(
            self.graph_sha256, "compile inputs graph_sha256"
        ))
        object.__setattr__(self, "tech_id", _text(
            self.tech_id, "compile inputs tech_id"
        ))
        members = _bounded_tuple(self.members, "compile input members")
        if any(type(item) is not PhysicalMacroMemberSnapshot for item in members):
            raise PhysicalMacroContractError(
                "compile inputs require exact typed member snapshots"
            )
        if members != tuple(sorted(members)) or len({item.member for item in members}) != len(members):
            raise PhysicalMacroContractError("compile input members must be canonical")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "clay.physical-macro-compile-inputs/v1",
            "graph_sha256": self.graph_sha256,
            "tech_id": self.tech_id,
            "members": tuple(item.payload() for item in self.members),
        }

    @property
    def sha256(self) -> str:
        return _sha(self.payload())


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class PhysicalMacroPlacementBinding:
    macro_id: str
    contract_sha256: str
    members: tuple[str, ...]
    translation_nm: tuple[int, int]
    bbox_nm: tuple[int, int, int, int]
    artwork_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "macro_id", _text(self.macro_id, "binding macro_id"))
        object.__setattr__(self, "contract_sha256", _digest(
            self.contract_sha256, "binding contract_sha256"
        ))
        members = _bounded_tuple(self.members, "binding members", allow_empty=False)
        if (
            len(members) != 1
            or any(type(item) is not str or not item for item in members)
        ):
            raise PhysicalMacroContractError("singleton binding must own one member")
        object.__setattr__(self, "translation_nm", _point(
            self.translation_nm, "binding translation"
        ))
        object.__setattr__(self, "bbox_nm", _bbox(self.bbox_nm, "binding bbox"))
        if self.translation_nm != self.bbox_nm[0:2]:
            raise PhysicalMacroContractError(
                "singleton binding translation must equal its bbox origin"
            )
        object.__setattr__(self, "artwork_sha256", _digest(
            self.artwork_sha256, "binding artwork_sha256"
        ))

    def payload(self) -> tuple[object, ...]:
        return (
            self.macro_id, self.contract_sha256, self.members,
            self.translation_nm, self.bbox_nm, self.artwork_sha256,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PhysicalMacroPlacementCertificate:
    graph_sha256: str
    contracts_sha256: str
    compile_inputs_sha256: str
    bindings: tuple[PhysicalMacroPlacementBinding, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "graph_sha256", _digest(
            self.graph_sha256, "certificate graph_sha256"
        ))
        object.__setattr__(self, "contracts_sha256", _digest(
            self.contracts_sha256, "certificate contracts_sha256"
        ))
        object.__setattr__(self, "compile_inputs_sha256", _digest(
            self.compile_inputs_sha256, "certificate compile_inputs_sha256"
        ))
        bindings = _bounded_tuple(self.bindings, "macro placement bindings")
        if any(type(item) is not PhysicalMacroPlacementBinding for item in bindings):
            raise PhysicalMacroContractError(
                "macro placement certificate requires exact typed bindings"
            )
        if bindings != tuple(sorted(bindings)):
            raise PhysicalMacroContractError("macro placement bindings must be canonical")
        if len({item.macro_id for item in bindings}) != len(bindings):
            raise PhysicalMacroContractError("macro placement binding ids must be unique")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "clay.physical-macro-placement-certificate/v1",
            "graph_sha256": self.graph_sha256,
            "contracts_sha256": self.contracts_sha256,
            "compile_inputs_sha256": self.compile_inputs_sha256,
            "bindings": tuple(item.payload() for item in self.bindings),
        }

    @property
    def sha256(self) -> str:
        return _sha(self.payload())


def _terminal_center(terminal: object, field: str) -> tuple[int, int]:
    bbox = _bbox(getattr(terminal, "bbox", None), field)
    return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)


def compile_physical_macro_contracts(
    graph: object,
    *,
    geometries: Mapping[str, object],
    device_pins: Mapping[str, Mapping[str, str]],
    tech: object,
) -> PhysicalMacroContractSet:
    """Compile exact interfaces for supported non-row graph nodes before solve."""

    from clay.layout.physical_macro_graph import PhysicalMacroGraph

    if type(graph) is not PhysicalMacroGraph:
        raise PhysicalMacroContractError("graph must be an exact PhysicalMacroGraph")
    if not isinstance(geometries, Mapping) or not isinstance(device_pins, Mapping):
        raise PhysicalMacroContractError("macro geometry and pin inputs must be mappings")
    tech_id = _text(getattr(tech, "pdk_id", ""), "contract tech identity")
    try:
        default_layer = _layer(tuple(tech.metal_key(1)))
    except Exception as exc:
        raise PhysicalMacroContractError(
            "contract technology exposes no exact routable metal1 layer"
        ) from exc
    contracts = []
    for macro in graph.macros:
        if macro.kind not in _SUPPORTED_SINGLETON_KINDS or len(macro.members) != 1:
            continue
        member = macro.members[0]
        geometry = geometries.get(member)
        if geometry is None:
            raise PhysicalMacroContractError(
                f"declared singleton macro lacks geometry: {macro.macro_id}"
            )
        raw_bbox = _bbox(geometry.extent(), "macro geometry bbox")
        ox, oy = raw_bbox[0], raw_bbox[1]
        local_bbox = (0, 0, raw_bbox[2] - ox, raw_bbox[3] - oy)
        pins = device_pins.get(member)
        if not isinstance(pins, Mapping):
            raise PhysicalMacroContractError(
                f"declared singleton macro lacks pin binding: {macro.macro_id}"
            )
        raw_terminals = getattr(geometry, "terminals", None)
        if not isinstance(raw_terminals, Mapping) or not raw_terminals:
            raise PhysicalMacroContractError(
                f"declared singleton macro lacks terminal geometry: {macro.macro_id}"
            )
        raw_accesses = getattr(geometry, "terminal_accesses", {}) or {}
        raw_layers = getattr(geometry, "terminal_layers", {}) or {}
        terminals = []
        for raw_role, terminal in sorted(raw_terminals.items()):
            role = _text(raw_role, "geometry terminal role")
            # The canonical MOS placement suppresses a separately routable bulk
            # access when the device already drew a source-tied tap pad: that tie
            # is discharged inside the cell.  Close the same routing interface
            # here, before solve, instead of advertising a terminal placement will
            # deliberately remove.
            if role == "b" and getattr(terminal, "bbox", None) is not None:
                continue
            net = _text(pins.get(role), f"terminal net {member}.{role}")
            access_terms = tuple(raw_accesses.get(role, ()) or (terminal,))
            accesses = tuple(
                (
                    point[0] - ox,
                    point[1] - oy,
                )
                for point in (
                    _terminal_center(item, f"terminal {member}.{role}")
                    for item in access_terms
                )
            )
            raw_layer = raw_layers.get(role, default_layer)
            terminals.append(PhysicalMacroTerminalAccess(
                member=member,
                role=role,
                net=net,
                layer=tuple(raw_layer),
                accesses_nm=accesses,
            ))
        terminals_tuple = tuple(sorted(terminals))
        demands = tuple(
            PhysicalMacroRouteDemand(
                net=net,
                members=(member,),
                terminal_count=len(selected),
                access_count=sum(len(item.accesses_nm) for item in selected),
            )
            for net in sorted({item.net for item in terminals_tuple})
            if (selected := tuple(item for item in terminals_tuple if item.net == net))
        )
        orientation = str(getattr(geometry, "orient", "R0") or "R0").upper()
        contracts.append(PhysicalMacroContract(
            macro_id=macro.macro_id,
            kind=macro.kind,
            members=macro.members,
            local_bbox_nm=local_bbox,
            terminal_accesses=terminals_tuple,
            local_symmetry_axes_2nm=(),
            body_tie_domain=tuple(sorted({
                str(pins.get("b")) for _ in (0,) if pins.get("b")
            })),
            legal_orientations=(orientation,),
            route_demand=demands,
            artwork_sha256=_artwork_sha256(
                getattr(geometry, "polygons", ()), origin_nm=(ox, oy)
            ),
        ))
    return PhysicalMacroContractSet(
        graph_sha256=graph.sha256,
        tech_id=tech_id,
        macros=tuple(sorted(contracts, key=lambda item: item.macro_id)),
    )


def compile_physical_macro_contract_snapshot(
    graph: object,
    *,
    geometries: Mapping[str, object],
    device_pins: Mapping[str, Mapping[str, str]],
    tech: object,
) -> tuple[PhysicalMacroContractSet, PhysicalMacroCompileInputs]:
    """Compile contracts and freeze their exact pre-solver geometry/pin inputs."""

    contracts = compile_physical_macro_contracts(
        graph,
        geometries=geometries,
        device_pins=device_pins,
        tech=tech,
    )
    inputs = PhysicalMacroCompileInputs(
        graph_sha256=contracts.graph_sha256,
        tech_id=contracts.tech_id,
        members=tuple(sorted(
            PhysicalMacroMemberSnapshot(
                member=contract.members[0],
                local_bbox_nm=contract.local_bbox_nm,
                terminal_accesses=contract.terminal_accesses,
                body_tie_domain=contract.body_tie_domain,
                legal_orientations=contract.legal_orientations,
                artwork_sha256=contract.artwork_sha256,
            )
            for contract in contracts.macros
        )),
    )
    replayed = replay_physical_macro_contracts(graph, inputs)
    if replayed.sha256 != contracts.sha256 or replayed.payload() != contracts.payload():
        raise PhysicalMacroContractError(
            "pre-solver physical macro snapshot does not replay contracts"
        )
    return contracts, inputs


def replay_physical_macro_contracts(
    graph: object,
    inputs: PhysicalMacroCompileInputs,
) -> PhysicalMacroContractSet:
    """Recompile downstream contracts solely from canonical pre-solver inputs."""

    from clay.layout.physical_macro_graph import PhysicalMacroGraph

    if type(graph) is not PhysicalMacroGraph:
        raise PhysicalMacroContractError("graph must be an exact PhysicalMacroGraph")
    if type(inputs) is not PhysicalMacroCompileInputs:
        raise PhysicalMacroContractError(
            "compile inputs must be exact PhysicalMacroCompileInputs"
        )
    if inputs.graph_sha256 != graph.sha256:
        raise PhysicalMacroContractError(
            "compile inputs do not bind the current physical macro graph"
        )
    snapshots = {item.member: item for item in inputs.members}
    supported = tuple(
        macro for macro in graph.macros
        if macro.kind in _SUPPORTED_SINGLETON_KINDS and len(macro.members) == 1
    )
    expected_members = {macro.members[0] for macro in supported}
    if set(snapshots) != expected_members:
        raise PhysicalMacroContractError(
            "compile input members differ from supported graph macros"
        )
    contracts = []
    for macro in supported:
        snapshot = snapshots[macro.members[0]]
        terminals = snapshot.terminal_accesses
        demands = tuple(
            PhysicalMacroRouteDemand(
                net=net,
                members=(snapshot.member,),
                terminal_count=len(selected),
                access_count=sum(len(item.accesses_nm) for item in selected),
            )
            for net in sorted({item.net for item in terminals})
            if (selected := tuple(item for item in terminals if item.net == net))
        )
        contracts.append(PhysicalMacroContract(
            macro_id=macro.macro_id,
            kind=macro.kind,
            members=macro.members,
            local_bbox_nm=snapshot.local_bbox_nm,
            terminal_accesses=terminals,
            local_symmetry_axes_2nm=(),
            body_tie_domain=snapshot.body_tie_domain,
            legal_orientations=snapshot.legal_orientations,
            route_demand=demands,
            artwork_sha256=snapshot.artwork_sha256,
        ))
    return PhysicalMacroContractSet(
        graph_sha256=inputs.graph_sha256,
        tech_id=inputs.tech_id,
        macros=tuple(sorted(contracts, key=lambda item: item.macro_id)),
    )


def bind_physical_macro_placement(
    graph: object,
    contracts: PhysicalMacroContractSet,
    devices: Mapping[str, object],
    *,
    compile_inputs: PhysicalMacroCompileInputs,
) -> PhysicalMacroPlacementCertificate:
    """Bind every compiled singleton to an exact translation-only placed view."""

    if type(contracts) is not PhysicalMacroContractSet:
        raise PhysicalMacroProjectionError("MACRO_CONTRACT_SET_TYPE_MISMATCH")
    if type(compile_inputs) is not PhysicalMacroCompileInputs:
        raise PhysicalMacroProjectionError("MACRO_COMPILE_INPUTS_MISMATCH")
    try:
        replayed = replay_physical_macro_contracts(graph, compile_inputs)
    except PhysicalMacroContractError as exc:
        raise PhysicalMacroProjectionError("MACRO_COMPILE_INPUTS_MISMATCH") from exc
    if replayed.sha256 != contracts.sha256 or replayed.payload() != contracts.payload():
        raise PhysicalMacroProjectionError("MACRO_COMPILE_INPUTS_MISMATCH")
    if not isinstance(devices, Mapping):
        raise PhysicalMacroProjectionError("MACRO_PLACEMENT_DEVICE_MAP_INVALID")
    bindings = []
    for contract in contracts.macros:
        member = contract.members[0]
        view = devices.get(member)
        if view is None:
            raise PhysicalMacroProjectionError(f"MACRO_MEMBER_MISSING:{member}")
        try:
            placed_bbox = _bbox(getattr(view, "bbox", None), "placed macro bbox")
            translation = (placed_bbox[0], placed_bbox[1])
            expected_bbox = (
                translation[0] + contract.local_bbox_nm[0],
                translation[1] + contract.local_bbox_nm[1],
                translation[0] + contract.local_bbox_nm[2],
                translation[1] + contract.local_bbox_nm[3],
            )
            orientation = getattr(view, "orientation", "")
            if type(orientation) is not str:
                raise PhysicalMacroProjectionError(
                    f"MACRO_MEMBER_VIEW_MISMATCH:{member}"
                )
            placed_terms = getattr(view, "terminals", {}) or {}
            placed_accesses = getattr(view, "terminal_accesses", {}) or {}
            placed_layers = getattr(view, "term_layers", {}) or {}
            expected_roles = {item.role for item in contract.terminal_accesses}
            if (
                set(placed_terms) != expected_roles
                or not set(placed_accesses) <= expected_roles
                or not set(placed_layers) <= expected_roles
            ):
                raise PhysicalMacroProjectionError(
                    f"MACRO_MEMBER_VIEW_MISMATCH:{member}"
                )
            observed = []
            for terminal in contract.terminal_accesses:
                role = terminal.role
                if role not in placed_terms:
                    raise PhysicalMacroProjectionError(
                        f"MACRO_MEMBER_VIEW_MISMATCH:{member}"
                    )
                if role in placed_accesses:
                    raw_points = placed_accesses[role]
                    if type(raw_points) is not tuple:
                        raise PhysicalMacroProjectionError(
                            f"MACRO_MEMBER_VIEW_MISMATCH:{member}"
                        )
                    if not raw_points:
                        raise PhysicalMacroProjectionError(
                            f"MACRO_MEMBER_VIEW_MISMATCH:{member}"
                        )
                    points = raw_points
                else:
                    points = (placed_terms[role],)
                if placed_terms[role] not in points:
                    raise PhysicalMacroProjectionError(
                        f"MACRO_MEMBER_VIEW_MISMATCH:{member}"
                    )
                local_points = tuple(sorted(
                    (
                        _point(point, f"placed terminal {member}.{role}")[0]
                        - translation[0],
                        _point(point, f"placed terminal {member}.{role}")[1]
                        - translation[1],
                    )
                    for point in points
                ))
                effective_layer = _layer(placed_layers.get(role, terminal.layer))
                observed.append((role, effective_layer, local_points))
            expected = tuple(
                (item.role, item.layer, item.accesses_nm)
                for item in contract.terminal_accesses
            )
            artwork_sha = _artwork_sha256(
                getattr(view, "polys", ()), origin_nm=translation
            )
        except PhysicalMacroProjectionError:
            raise
        except Exception as exc:
            raise PhysicalMacroProjectionError(
                f"MACRO_MEMBER_VIEW_MISMATCH:{member}"
            ) from exc
        if (
            placed_bbox != expected_bbox
            or orientation not in contract.legal_orientations
            or tuple(observed) != expected
            or artwork_sha != contract.artwork_sha256
        ):
            raise PhysicalMacroProjectionError(
                f"MACRO_MEMBER_VIEW_MISMATCH:{member}"
            )
        bindings.append(PhysicalMacroPlacementBinding(
            macro_id=contract.macro_id,
            contract_sha256=contract.structure_fingerprint,
            members=contract.members,
            translation_nm=translation,
            bbox_nm=placed_bbox,
            artwork_sha256=artwork_sha,
        ))
    return PhysicalMacroPlacementCertificate(
        graph_sha256=contracts.graph_sha256,
        contracts_sha256=contracts.sha256,
        compile_inputs_sha256=compile_inputs.sha256,
        bindings=tuple(sorted(bindings)),
    )


__all__ = [
    "PhysicalMacroContract",
    "PhysicalMacroContractError",
    "PhysicalMacroContractSet",
    "PhysicalMacroCompileInputs",
    "PhysicalMacroMemberSnapshot",
    "PhysicalMacroPlacementBinding",
    "PhysicalMacroPlacementCertificate",
    "PhysicalMacroProjectionError",
    "PhysicalMacroRouteDemand",
    "PhysicalMacroTerminalAccess",
    "bind_physical_macro_placement",
    "compile_physical_macro_contract_snapshot",
    "compile_physical_macro_contracts",
    "replay_physical_macro_contracts",
]
