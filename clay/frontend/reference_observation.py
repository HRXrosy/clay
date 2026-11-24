"""Independent, coordinate-free completeness observations for reference GDS.

The spatial-intent extractor intentionally ignores physical artwork.  This module
closes the complementary authority boundary: a reference cannot be promoted from
``oa_spatial_only`` merely because its SREF placement names are readable.  The
recognized device masters must contain non-text geometry on installed-PDK layers,
the declared device families must be covered, and routing geometry must be present.

Only dimensions, counts, family identities, layer tokens, and digests leave this
module.  Absolute reference coordinates never do.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping

__all__ = [
    "ReferenceDeviceMasterOccurrenceObservation",
    "ReferenceDeviceMasterOccurrenceRow",
    "ReferenceGdsGeometryObservation",
    "ReferenceObservationReceipt",
    "ReferenceObservationError",
    "build_reference_observation_receipt",
    "observe_reference_device_master_occurrences",
    "observe_reference_gds_geometry",
    "reference_device_family_from_cell_name",
    "reference_observation_receipt_from_data",
    "replay_reference_observation_receipt",
]


_DEVICE_FAMILIES = frozenset(
    {"capacitor", "nmos", "npn", "pmos", "pnp", "resistor"}
)
_DEVICE_NAME_PATTERNS = (
    (re.compile(r"^(?:p33|p18|pmos)(?:_|\d|$)"), "pmos"),
    (re.compile(r"^(?:n33|n18|nmos)(?:_|\d|$)"), "nmos"),
    (
        re.compile(r"^(?:ppolyf_u|npolyf_u|poly_res|resistor)(?:_|\d|$)"),
        "resistor",
    ),
    (re.compile(r"^(?:mim|mimcap|capacitor)(?:_|\d|$)"), "capacitor"),
    (re.compile(r"^pnp(?:_|\d|$)"), "pnp"),
    (re.compile(r"^npn(?:_|\d|$)"), "npn"),
)
_METAL_TOKEN_RE = re.compile(r"metal[1-9][0-9]*\Z")
_MAX_CELLS = 16_384
_MAX_SHAPES = 100_000_000
_MAX_INSTANCE_OCCURRENCES = 100_000_000
_SEMANTIC_VERSION = "clay.reference-obligation-semantics/v1"
_ID_RE = re.compile(r"[a-z][a-z0-9_.]*\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IMPLEMENTATION_PATHS = (
    "clay/frontend/reference_observation.py",
    "clay/frontend/reference_spatial_intent.py",
)


class ReferenceObservationError(ValueError):
    """One closed typed failure at the reference-observation boundary."""

    def __init__(self, code: str, detail: str) -> None:
        if type(code) is not str or not code or not code.startswith("REFERENCE_"):
            raise TypeError("reference observation error requires a typed code")
        if type(detail) is not str or not detail:
            raise TypeError("reference observation error requires a detail")
        self.code = code
        super().__init__(f"{code}: {detail}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exact_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_RECEIPT_MALFORMED",
            f"{field} must be one lowercase SHA-256",
        )
    return value


def _exact_layermap(value: object) -> dict[str, tuple[int, int]]:
    if type(value) is not dict or not value:
        raise ReferenceObservationError(
            "REFERENCE_PDK_LAYER_AUTHORITY_MALFORMED",
            "PDK layermap must be a non-empty exact dict",
        )
    result: dict[str, tuple[int, int]] = {}
    for token, key in value.items():
        if (
            type(token) is not str
            or not token
            or token != token.strip()
            or type(key) is not tuple
            or len(key) != 2
            or any(type(item) is not int or item < 0 for item in key)
        ):
            raise ReferenceObservationError(
                "REFERENCE_PDK_LAYER_AUTHORITY_MALFORMED",
                "every PDK layer must bind one exact nonnegative (layer, datatype) tuple",
            )
        result[token] = key
    if not any(_METAL_TOKEN_RE.fullmatch(token) for token in result):
        raise ReferenceObservationError(
            "REFERENCE_PDK_LAYER_AUTHORITY_MALFORMED",
            "PDK layermap exposes no canonical routing metal",
        )
    return result


def _exact_inventory(value: object) -> dict[str, int]:
    if type(value) is not dict or not value:
        raise ReferenceObservationError(
            "REFERENCE_DEVICE_INVENTORY_MALFORMED",
            "expected device inventory must be a non-empty exact dict",
        )
    result: dict[str, int] = {}
    for family, count in value.items():
        if (
            type(family) is not str
            or family not in _DEVICE_FAMILIES
            or type(count) is not int
            or count <= 0
        ):
            raise ReferenceObservationError(
                "REFERENCE_DEVICE_INVENTORY_MALFORMED",
                "device inventory requires canonical families and positive exact counts",
            )
        result[family] = count
    return result


def _topology_device_inventory(
    source_netlist: str,
    *,
    expected_top_cell: str,
) -> dict[str, int]:
    """Derive the exact family inventory from the bound topology bytes."""

    if type(source_netlist) is not str or not source_netlist.strip():
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_TOPOLOGY_INVENTORY_MISMATCH",
            "bound topology is empty or malformed",
    )
    from clay.frontend.reference_spatial_intent import device_master_map

    bodies: list[list[str]] = []
    active: list[str] | None = None
    for raw in source_netlist.splitlines():
        line = raw.strip()
        folded = line.casefold()
        if folded.startswith(".subckt"):
            tokens = line.split()
            if active is not None or len(tokens) < 2:
                raise ReferenceObservationError(
                    "REFERENCE_OBSERVATION_TOPOLOGY_CELL_MISMATCH",
                    "bound topology has malformed or nested subcircuit declarations",
                )
            active = [] if tokens[1] == expected_top_cell else ["__foreign__"]
            continue
        if folded.startswith(".ends"):
            if active is not None and active != ["__foreign__"]:
                bodies.append(active)
            active = None
            continue
        if active is not None and active != ["__foreign__"]:
            active.append(line)
    if active is not None or len(bodies) != 1:
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_TOPOLOGY_CELL_MISMATCH",
            "bound topology must contain one closed subcircuit matching the GDS top",
        )

    counts: dict[str, int] = {}
    seen_devices: set[str] = set()
    for line in bodies[0]:
        if not line or line.startswith("*") or line[0].casefold() not in "mrcq":
            continue
        mapped = device_master_map(line)
        if len(mapped) != 1:
            raise ReferenceObservationError(
                "REFERENCE_OBSERVATION_TOPOLOGY_INVENTORY_MISMATCH",
                f"topology device line has unsupported family syntax: {line!r}",
            )
        device_name, master = next(iter(mapped.items()))
        if device_name in seen_devices:
            raise ReferenceObservationError(
                "REFERENCE_OBSERVATION_TOPOLOGY_INVENTORY_MISMATCH",
                f"topology device identity is duplicated: {device_name!r}",
            )
        seen_devices.add(device_name)
        family = reference_device_family_from_cell_name(master)
        if family is None:
            raise ReferenceObservationError(
                "REFERENCE_OBSERVATION_TOPOLOGY_INVENTORY_MISMATCH",
                f"topology device family is unsupported: {master!r}",
            )
        counts[family] = counts.get(family, 0) + 1
    inventory = dict(sorted(counts.items()))
    if not inventory:
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_TOPOLOGY_INVENTORY_MISMATCH",
            "bound topology exposes no recognized device family",
        )
    return inventory


def _layermap_sha256(value: object) -> str:
    layermap = _exact_layermap(value)
    payload = json.dumps(
        [[token, list(layermap[token])] for token in sorted(layermap)],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return _sha256_bytes(payload)


def _implementation_state() -> dict[str, str]:
    repo_root = Path(__file__).resolve(strict=True).parents[2]
    state = {
        relative: _sha256_file((repo_root / relative).resolve(strict=True))
        for relative in _IMPLEMENTATION_PATHS
    }
    if tuple(state) != _IMPLEMENTATION_PATHS:
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_IMPLEMENTATION_CHANGED",
            "reference observer implementation closure is not canonical",
        )
    return state


def _bound_repository_file(
    root: Path,
    relative: object,
    *,
    suffix: str,
) -> Path:
    if (
        type(relative) is not str
        or not relative
        or "\\" in relative
        or relative != PurePosixPath(relative).as_posix()
    ):
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_PATH_MALFORMED",
            "receipt paths must be exact repository-relative POSIX paths",
        )
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.suffix.casefold() != suffix:
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_PATH_MALFORMED",
            f"receipt path must remain inside the repository and end in {suffix}",
        )
    try:
        resolved_root = root.resolve(strict=True)
        path = (resolved_root / Path(*pure.parts)).resolve(strict=True)
        path.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_INPUT_MISSING",
            f"bound input is unavailable: {relative}",
        ) from exc
    if not path.is_file():
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_INPUT_MISSING",
            f"bound input is not a regular file: {relative}",
        )
    return path


def reference_device_family_from_cell_name(name: object) -> str | None:
    """Return one semantic device family without using circuit identity or position."""

    if type(name) is not str or not name:
        return None
    folded = name.casefold()
    for pattern, family in _DEVICE_NAME_PATTERNS:
        if pattern.match(folded):
            return family
    return None


def _exact_device_family_from_cell_name(name: object) -> str | None:
    """Resolve one family and reject overlapping recognition rules explicitly."""

    if type(name) is not str or not name:
        return None
    matches = tuple(
        sorted(
            {
                family
                for pattern, family in _DEVICE_NAME_PATTERNS
                if pattern.match(name.casefold())
            }
        )
    )
    if len(matches) > 1:
        raise ReferenceObservationError(
            "REFERENCE_GDS_DEVICE_FAMILY_AMBIGUOUS",
            f"device master {name!r} matches multiple families: {matches}",
        )
    return matches[0] if matches else None


@dataclass(frozen=True, slots=True)
class ReferenceDeviceMasterOccurrenceRow:
    """Position-free occurrence count and intrinsic-artwork identity for one master."""

    master_name: str
    device_family: str
    expanded_occurrence_count: int
    intrinsic_artwork_sha256: str

    def __post_init__(self) -> None:
        if type(self.master_name) is not str or not self.master_name:
            raise TypeError("reference device master name must be exact")
        if self.device_family not in _DEVICE_FAMILIES:
            raise ValueError("reference device master family is unsupported")
        if (
            type(self.expanded_occurrence_count) is not int
            or self.expanded_occurrence_count <= 0
            or self.expanded_occurrence_count > _MAX_INSTANCE_OCCURRENCES
        ):
            raise ValueError("reference device master occurrence count is invalid")
        if (
            type(self.intrinsic_artwork_sha256) is not str
            or _SHA256_RE.fullmatch(self.intrinsic_artwork_sha256) is None
        ):
            raise ValueError("reference device master artwork digest is invalid")

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "master_name": self.master_name,
            "device_family": self.device_family,
            "expanded_occurrence_count": self.expanded_occurrence_count,
            "intrinsic_artwork_sha256": self.intrinsic_artwork_sha256,
        }


@dataclass(frozen=True, slots=True)
class ReferenceDeviceMasterOccurrenceObservation:
    """Bound GDS identity plus canonical, coordinate-free master rows."""

    gds_sha256: str
    top_cell: str
    rows: tuple[ReferenceDeviceMasterOccurrenceRow, ...]
    schema: str = "clay.reference-device-master-occurrence-observation/v1"

    def __post_init__(self) -> None:
        if self.schema != "clay.reference-device-master-occurrence-observation/v1":
            raise ValueError("reference device occurrence schema differs")
        if type(self.gds_sha256) is not str or _SHA256_RE.fullmatch(self.gds_sha256) is None:
            raise ValueError("reference device occurrence GDS digest is invalid")
        if type(self.top_cell) is not str or not self.top_cell:
            raise TypeError("reference device occurrence top cell is invalid")
        if type(self.rows) is not tuple or not self.rows:
            raise TypeError("reference device occurrence rows are incomplete")
        if self.rows != tuple(sorted(self.rows, key=lambda row: row.master_name)):
            raise ValueError("reference device occurrence rows are not canonical")
        if len({row.master_name for row in self.rows}) != len(self.rows):
            raise ValueError("reference device occurrence master identity is duplicated")
        for row in self.rows:
            if type(row) is not ReferenceDeviceMasterOccurrenceRow:
                raise TypeError("reference device occurrence row type is invalid")
            row.__post_init__()

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema": self.schema,
            "gds_sha256": self.gds_sha256,
            "top_cell": self.top_cell,
            "rows": [row.canonical_data() for row in self.rows],
        }


def _canonical_contour(
    points: object,
    *,
    origin: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    ring = tuple(
        (int(point.x) - origin[0], int(point.y) - origin[1])
        for point in points
    )
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if not ring:
        return ()
    rotations = tuple(ring[index:] + ring[:index] for index in range(len(ring)))
    reverse = tuple(reversed(ring))
    reverse_rotations = tuple(
        reverse[index:] + reverse[:index] for index in range(len(reverse))
    )
    return min(rotations + reverse_rotations)


def _intrinsic_master_artwork_sha256(
    layout: object,
    cell: object,
    *,
    layer_indices: Mapping[tuple[int, int], int],
) -> str:
    """Hash PDK-layer artwork after removing only a common local translation."""

    import klayout.db as kdb

    regions: dict[tuple[int, int], object] = {}
    artwork_bbox = None
    for key, index in sorted(layer_indices.items()):
        region = kdb.Region(cell.begin_shapes_rec(index)).merged()
        if region.is_empty():
            continue
        regions[key] = region
        box = region.bbox()
        artwork_bbox = box if artwork_bbox is None else artwork_bbox + box
    if artwork_bbox is None:
        raise ReferenceObservationError(
            "REFERENCE_GDS_EMPTY_PCELL",
            f"device master {cell.name!r} has no PDK-layer artwork",
        )
    origin = (int(artwork_bbox.left), int(artwork_bbox.bottom))
    records = []
    for (layer, datatype), region in sorted(regions.items()):
        polygons = []
        for polygon in region.each():
            hull = _canonical_contour(
                polygon.each_point_hull(),
                origin=origin,
            )
            holes = tuple(
                sorted(
                    _canonical_contour(
                        polygon.each_point_hole(index),
                        origin=origin,
                    )
                    for index in range(int(polygon.holes()))
                )
            )
            polygons.append((hull, holes))
        records.append((layer, datatype, tuple(sorted(polygons))))
    payload = {
        "dbu_um": format(float(layout.dbu), ".17g"),
        "layers": records,
    }
    return _sha256_bytes(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    )


def observe_reference_device_master_occurrences(
    gds_path: str | Path,
    *,
    expected_top_cell: str,
    pdk_layermap: Mapping[str, tuple[int, int]],
) -> ReferenceDeviceMasterOccurrenceObservation:
    """Observe resolved-master classes and expanded counts without coordinates."""

    if type(expected_top_cell) is not str or not expected_top_cell:
        raise ReferenceObservationError(
            "REFERENCE_GDS_TOP_CELL_MALFORMED",
            "expected top cell must be a non-empty exact string",
        )
    layermap = _exact_layermap(pdk_layermap)
    path = Path(gds_path)
    if not path.is_file() or path.suffix.casefold() != ".gds":
        raise ReferenceObservationError(
            "REFERENCE_GDS_MISSING", "reference GDS path is unavailable"
        )

    import klayout.db as kdb

    layout = kdb.Layout()
    try:
        layout.read(str(path))
    except Exception as exc:
        raise ReferenceObservationError(
            "REFERENCE_GDS_UNREADABLE", "reference GDS cannot be parsed"
        ) from exc
    tops = tuple(layout.top_cells())
    if len(tops) != 1:
        raise ReferenceObservationError(
            "REFERENCE_GDS_TOP_CELL_AMBIGUOUS",
            f"reference GDS has {len(tops)} top cells",
        )
    top = tops[0]
    if top.name != expected_top_cell:
        raise ReferenceObservationError(
            "REFERENCE_GDS_TOP_CELL_MISMATCH",
            "reference GDS top cell differs from the declared identity",
        )
    if layout.cells() <= 0 or layout.cells() > _MAX_CELLS:
        raise ReferenceObservationError(
            "REFERENCE_GDS_HIERARCHY_UNBOUNDED",
            f"reference GDS has {layout.cells()} cells",
        )

    layer_indices = {}
    for key in sorted(set(layermap.values())):
        index = layout.find_layer(*key)
        if index is not None and index >= 0:
            layer_indices[key] = int(index)
    counts: dict[str, int] = {}
    families: dict[str, str] = {}
    artwork: dict[str, str] = {}

    def array_multiplicity(instance: object) -> int:
        total = 0
        for _transform in instance.cell_inst.each_cplx_trans():
            total += 1
            if total > _MAX_INSTANCE_OCCURRENCES:
                raise ReferenceObservationError(
                    "REFERENCE_GDS_HIERARCHY_UNBOUNDED",
                    "one reference instance array exceeds the observation bound",
                )
        return total

    def visit(cell: object, inherited: int, ancestry: frozenset[int]) -> None:
        for instance in cell.each_inst():
            local_count = array_multiplicity(instance)
            expanded = inherited * local_count
            if local_count <= 0 or not (0 < expanded <= _MAX_INSTANCE_OCCURRENCES):
                raise ReferenceObservationError(
                    "REFERENCE_GDS_HIERARCHY_UNBOUNDED",
                    "expanded reference hierarchy exceeds the observation bound",
                )
            child = instance.cell
            child_index = int(child.cell_index())
            if child_index in ancestry:
                raise ReferenceObservationError(
                    "REFERENCE_GDS_HIERARCHY_UNBOUNDED",
                    "reference GDS contains a cyclic hierarchy",
                )
            family = _exact_device_family_from_cell_name(child.name)
            if family is None:
                visit(child, expanded, ancestry | frozenset((child_index,)))
                continue
            if child.name not in artwork:
                artwork[child.name] = _intrinsic_master_artwork_sha256(
                    layout,
                    child,
                    layer_indices=layer_indices,
                )
                families[child.name] = family
            elif families[child.name] != family:
                raise ReferenceObservationError(
                    "REFERENCE_GDS_DEVICE_FAMILY_AMBIGUOUS",
                    f"device master {child.name!r} changed family",
                )
            counts[child.name] = counts.get(child.name, 0) + expanded
            if counts[child.name] > _MAX_INSTANCE_OCCURRENCES:
                raise ReferenceObservationError(
                    "REFERENCE_GDS_HIERARCHY_UNBOUNDED",
                    "expanded reference master count exceeds the observation bound",
                )

    visit(top, 1, frozenset((int(top.cell_index()),)))
    if not counts:
        raise ReferenceObservationError(
            "REFERENCE_GDS_DEVICE_COVERAGE_MISSING",
            "reference GDS exposes no recognized device master",
        )
    rows = tuple(
        ReferenceDeviceMasterOccurrenceRow(
            master_name=name,
            device_family=families[name],
            expanded_occurrence_count=counts[name],
            intrinsic_artwork_sha256=artwork[name],
        )
        for name in sorted(counts)
    )
    return ReferenceDeviceMasterOccurrenceObservation(
        gds_sha256=_sha256_file(path),
        top_cell=top.name,
        rows=rows,
    )


@dataclass(frozen=True, slots=True)
class ReferenceGdsGeometryObservation:
    verdict: str
    gds_sha256: str
    top_cell: str
    cell_count: int
    physical_shape_count: int
    text_shape_count: int
    empty_leaf_count: int
    device_family_instances: Mapping[str, int]
    device_master_count: int
    routed_layer_tokens: tuple[str, ...]
    bbox_width_nm: int
    bbox_height_nm: int
    bbox_area_nm2: int

    def __post_init__(self) -> None:
        if self.verdict != "pass":
            raise ValueError("reference geometry observation cannot retain a non-pass row")
        if not re.fullmatch(r"[0-9a-f]{64}", self.gds_sha256):
            raise ValueError("reference geometry observation requires a GDS SHA-256")
        if type(self.top_cell) is not str or not self.top_cell:
            raise ValueError("reference geometry observation requires one top cell")
        for field_name in (
            "cell_count",
            "physical_shape_count",
            "device_master_count",
            "bbox_width_nm",
            "bbox_height_nm",
            "bbox_area_nm2",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"reference geometry {field_name} must be positive")
        if type(self.text_shape_count) is not int or self.text_shape_count < 0:
            raise ValueError("reference geometry text count must be nonnegative")
        if self.empty_leaf_count != 0:
            raise ValueError("complete reference geometry cannot retain empty leaves")
        if (
            type(self.device_family_instances) is not MappingProxyType
            or not self.device_family_instances
            or any(
                family not in _DEVICE_FAMILIES
                or type(count) is not int
                or count <= 0
                for family, count in self.device_family_instances.items()
            )
        ):
            raise ValueError("reference geometry device-family counts are not exact")
        if (
            type(self.routed_layer_tokens) is not tuple
            or not self.routed_layer_tokens
            or tuple(sorted(set(self.routed_layer_tokens)))
            != self.routed_layer_tokens
            or any(
                type(token) is not str or _METAL_TOKEN_RE.fullmatch(token) is None
                for token in self.routed_layer_tokens
            )
        ):
            raise ValueError("reference geometry routed layers are not canonical")

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.reference-gds-geometry-observation/v1",
            "verdict": self.verdict,
            "gds_sha256": self.gds_sha256,
            "top_cell": self.top_cell,
            "cell_count": self.cell_count,
            "physical_shape_count": self.physical_shape_count,
            "text_shape_count": self.text_shape_count,
            "empty_leaf_count": self.empty_leaf_count,
            "device_family_instances": dict(self.device_family_instances),
            "device_master_count": self.device_master_count,
            "routed_layer_tokens": list(self.routed_layer_tokens),
            "bbox_width_nm": self.bbox_width_nm,
            "bbox_height_nm": self.bbox_height_nm,
            "bbox_area_nm2": self.bbox_area_nm2,
        }


def _geometry_observation_from_data(
    data: object,
) -> ReferenceGdsGeometryObservation:
    expected = {
        "schema",
        "verdict",
        "gds_sha256",
        "top_cell",
        "cell_count",
        "physical_shape_count",
        "text_shape_count",
        "empty_leaf_count",
        "device_family_instances",
        "device_master_count",
        "routed_layer_tokens",
        "bbox_width_nm",
        "bbox_height_nm",
        "bbox_area_nm2",
    }
    if (
        type(data) is not dict
        or set(data) != expected
        or data.get("schema")
        != "clay.reference-gds-geometry-observation/v1"
        or type(data.get("device_family_instances")) is not dict
        or type(data.get("routed_layer_tokens")) is not list
    ):
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_RECEIPT_MALFORMED",
            "reference geometry payload is not exact",
        )
    result = ReferenceGdsGeometryObservation(
        verdict=data["verdict"],
        gds_sha256=data["gds_sha256"],
        top_cell=data["top_cell"],
        cell_count=data["cell_count"],
        physical_shape_count=data["physical_shape_count"],
        text_shape_count=data["text_shape_count"],
        empty_leaf_count=data["empty_leaf_count"],
        device_family_instances=MappingProxyType(
            dict(data["device_family_instances"])
        ),
        device_master_count=data["device_master_count"],
        routed_layer_tokens=tuple(data["routed_layer_tokens"]),
        bbox_width_nm=data["bbox_width_nm"],
        bbox_height_nm=data["bbox_height_nm"],
        bbox_area_nm2=data["bbox_area_nm2"],
    )
    if result.canonical_data() != data:
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_RECEIPT_MALFORMED",
            "reference geometry payload is not canonical",
        )
    return result


@dataclass(frozen=True, slots=True)
class ReferenceObservationReceipt:
    semantic_version: str
    circuit_id: str
    pdk_id: str
    gds_repository_path: str
    source_repository_path: str
    topology_repository_path: str
    expected_top_cell: str
    expected_device_inventory: Mapping[str, int]
    gds_sha256: str
    source_sha256: str
    topology_sha256: str
    implementation_sha256: Mapping[str, str]
    pdk_source_snapshot_sha256: str
    pdk_installed_authority_sha256: str
    pdk_layermap_sha256: str
    spatial_sha256: str
    geometry: ReferenceGdsGeometryObservation

    def __post_init__(self) -> None:
        if self.semantic_version != _SEMANTIC_VERSION:
            raise ValueError("reference observation semantic version is not canonical")
        for field_name in ("circuit_id", "pdk_id"):
            value = getattr(self, field_name)
            if type(value) is not str or _ID_RE.fullmatch(value) is None:
                raise ValueError(f"reference observation {field_name} is not canonical")
        for value, suffix in (
            (self.gds_repository_path, ".gds"),
            (self.source_repository_path, ".clay"),
            (self.topology_repository_path, ".cir"),
        ):
            pure = PurePosixPath(value)
            if (
                type(value) is not str
                or not value
                or "\\" in value
                or pure.is_absolute()
                or ".." in pure.parts
                or pure.as_posix() != value
                or pure.suffix.casefold() != suffix
            ):
                raise ValueError("reference observation path is not canonical")
        if type(self.expected_top_cell) is not str or not self.expected_top_cell:
            raise ValueError("reference observation top-cell identity is not exact")
        if (
            type(self.expected_device_inventory) is not MappingProxyType
        ):
            raise ValueError("reference observation device inventory is not canonical")
        inventory = _exact_inventory(dict(self.expected_device_inventory))
        if dict(self.expected_device_inventory) != dict(sorted(inventory.items())):
            raise ValueError("reference observation device inventory is not canonical")
        if (
            type(self.implementation_sha256) is not MappingProxyType
            or tuple(self.implementation_sha256) != _IMPLEMENTATION_PATHS
            or any(
                type(value) is not str or _SHA256_RE.fullmatch(value) is None
                for value in self.implementation_sha256.values()
            )
        ):
            raise ValueError(
                "reference observation implementation closure is not canonical"
            )
        for field_name in (
            "gds_sha256",
            "source_sha256",
            "topology_sha256",
            "pdk_source_snapshot_sha256",
            "pdk_installed_authority_sha256",
            "pdk_layermap_sha256",
            "spatial_sha256",
        ):
            if (
                type(getattr(self, field_name)) is not str
                or _SHA256_RE.fullmatch(getattr(self, field_name)) is None
            ):
                raise ValueError(f"reference observation {field_name} is not a SHA-256")
        if type(self.geometry) is not ReferenceGdsGeometryObservation:
            raise ValueError("reference observation lacks an exact geometry result")
        self.geometry.__post_init__()
        if (
            self.geometry.gds_sha256 != self.gds_sha256
            or self.geometry.top_cell != self.expected_top_cell
        ):
            raise ValueError("reference observation geometry identity is inconsistent")

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema": "clay.reference-observation/v1",
            "semantic_version": self.semantic_version,
            "circuit_id": self.circuit_id,
            "pdk_id": self.pdk_id,
            "gds_repository_path": self.gds_repository_path,
            "source_repository_path": self.source_repository_path,
            "topology_repository_path": self.topology_repository_path,
            "expected_top_cell": self.expected_top_cell,
            "expected_device_inventory": dict(self.expected_device_inventory),
            "gds_sha256": self.gds_sha256,
            "source_sha256": self.source_sha256,
            "topology_sha256": self.topology_sha256,
            "implementation_sha256": dict(self.implementation_sha256),
            "pdk_source_snapshot_sha256": self.pdk_source_snapshot_sha256,
            "pdk_installed_authority_sha256": (
                self.pdk_installed_authority_sha256
            ),
            "pdk_layermap_sha256": self.pdk_layermap_sha256,
            "spatial_sha256": self.spatial_sha256,
            "geometry": self.geometry.canonical_data(),
        }

    @property
    def receipt_sha256(self) -> str:
        payload = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return _sha256_bytes(payload)


def reference_observation_receipt_from_data(
    data: object,
) -> ReferenceObservationReceipt:
    expected = {
        "schema",
        "semantic_version",
        "circuit_id",
        "pdk_id",
        "gds_repository_path",
        "source_repository_path",
        "topology_repository_path",
        "expected_top_cell",
        "expected_device_inventory",
        "gds_sha256",
        "source_sha256",
        "topology_sha256",
        "implementation_sha256",
        "pdk_source_snapshot_sha256",
        "pdk_installed_authority_sha256",
        "pdk_layermap_sha256",
        "spatial_sha256",
        "geometry",
    }
    if (
        type(data) is not dict
        or set(data) != expected
        or data.get("schema") != "clay.reference-observation/v1"
        or type(data.get("expected_device_inventory")) is not dict
        or type(data.get("implementation_sha256")) is not dict
    ):
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_RECEIPT_MALFORMED",
            "reference observation receipt payload is not exact",
        )
    result = ReferenceObservationReceipt(
        semantic_version=data["semantic_version"],
        circuit_id=data["circuit_id"],
        pdk_id=data["pdk_id"],
        gds_repository_path=data["gds_repository_path"],
        source_repository_path=data["source_repository_path"],
        topology_repository_path=data["topology_repository_path"],
        expected_top_cell=data["expected_top_cell"],
        expected_device_inventory=MappingProxyType(
            dict(sorted(data["expected_device_inventory"].items()))
        ),
        gds_sha256=data["gds_sha256"],
        source_sha256=data["source_sha256"],
        topology_sha256=data["topology_sha256"],
        implementation_sha256=MappingProxyType(
            dict(data["implementation_sha256"])
        ),
        pdk_source_snapshot_sha256=data["pdk_source_snapshot_sha256"],
        pdk_installed_authority_sha256=data[
            "pdk_installed_authority_sha256"
        ],
        pdk_layermap_sha256=data["pdk_layermap_sha256"],
        spatial_sha256=data["spatial_sha256"],
        geometry=_geometry_observation_from_data(data["geometry"]),
    )
    if result.canonical_data() != data:
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_RECEIPT_MALFORMED",
            "reference observation receipt is not canonical",
        )
    return result


def build_reference_observation_receipt(
    repo_root: str | Path,
    *,
    circuit_id: str,
    pdk_id: str,
    gds_repository_path: str,
    source_repository_path: str,
    topology_repository_path: str,
    expected_top_cell: str,
    expected_device_inventory: Mapping[str, int],
    pdk_source_snapshot_sha256: str,
    pdk_installed_authority_sha256: str,
    pdk_layermap: Mapping[str, tuple[int, int]],
) -> ReferenceObservationReceipt:
    """Build one receipt and reread every authority after observation."""

    if type(circuit_id) is not str or _ID_RE.fullmatch(circuit_id) is None:
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_RECEIPT_MALFORMED",
            "circuit identity is not canonical",
        )
    if type(pdk_id) is not str or _ID_RE.fullmatch(pdk_id) is None:
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_RECEIPT_MALFORMED",
            "PDK identity is not canonical",
        )
    snapshot_sha256 = _exact_sha256(
        pdk_source_snapshot_sha256, field="PDK source snapshot"
    )
    installed_sha256 = _exact_sha256(
        pdk_installed_authority_sha256, field="installed PDK authority"
    )
    inventory = _exact_inventory(expected_device_inventory)
    layermap = _exact_layermap(pdk_layermap)
    root = Path(repo_root)
    gds = _bound_repository_file(root, gds_repository_path, suffix=".gds")
    source = _bound_repository_file(root, source_repository_path, suffix=".clay")
    topology = _bound_repository_file(
        root, topology_repository_path, suffix=".cir"
    )
    before = {
        "gds": _sha256_file(gds),
        "source": _sha256_file(source),
        "topology": _sha256_file(topology),
        "implementation": _implementation_state(),
    }
    try:
        topology_text = topology.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_TOPOLOGY_INVENTORY_MISMATCH",
            "bound topology is not strict UTF-8",
        ) from exc
    derived_inventory = _topology_device_inventory(
        topology_text,
        expected_top_cell=expected_top_cell,
    )
    if inventory != derived_inventory:
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_TOPOLOGY_INVENTORY_MISMATCH",
            "declared inventory differs from the bound topology: "
            f"declared={dict(sorted(inventory.items()))!r}, "
            f"derived={derived_inventory!r}",
        )
    geometry = observe_reference_gds_geometry(
        gds,
        expected_top_cell=expected_top_cell,
        expected_device_inventory=inventory,
        pdk_layermap=layermap,
    )
    from clay.frontend.reference_spatial_intent import (
        extract_reference_spatial_intent,
    )

    spatial = extract_reference_spatial_intent(
        str(gds),
        source_netlist=topology_text,
    )
    after = {
        "gds": _sha256_file(gds),
        "source": _sha256_file(source),
        "topology": _sha256_file(topology),
        "implementation": _implementation_state(),
    }
    if after != before:
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_INPUT_CHANGED_DURING_REPLAY",
            "a bound source, artifact, or implementation changed during observation",
        )

    return ReferenceObservationReceipt(
        semantic_version=_SEMANTIC_VERSION,
        circuit_id=circuit_id,
        pdk_id=pdk_id,
        gds_repository_path=gds_repository_path,
        source_repository_path=source_repository_path,
        topology_repository_path=topology_repository_path,
        expected_top_cell=expected_top_cell,
        expected_device_inventory=MappingProxyType(dict(sorted(inventory.items()))),
        gds_sha256=before["gds"],
        source_sha256=before["source"],
        topology_sha256=before["topology"],
        implementation_sha256=MappingProxyType(
            dict(before["implementation"])
        ),
        pdk_source_snapshot_sha256=snapshot_sha256,
        pdk_installed_authority_sha256=installed_sha256,
        pdk_layermap_sha256=_layermap_sha256(layermap),
        spatial_sha256=spatial.reference_scope_sha256,
        geometry=geometry,
    )


def replay_reference_observation_receipt(
    repo_root: str | Path,
    receipt: ReferenceObservationReceipt,
    *,
    pdk_source_snapshot_sha256: str,
    pdk_installed_authority_sha256: str,
    pdk_layermap: Mapping[str, tuple[int, int]],
) -> ReferenceObservationReceipt:
    """Reread all receipt authorities and reproduce the observation exactly."""

    if type(receipt) is not ReferenceObservationReceipt:
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_RECEIPT_MALFORMED",
            "receipt must be one exact typed row",
        )
    receipt.__post_init__()
    current_implementation = _implementation_state()
    if current_implementation != dict(receipt.implementation_sha256):
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_IMPLEMENTATION_CHANGED",
            "observer implementation no longer matches the receipt",
        )
    if (
        pdk_source_snapshot_sha256 != receipt.pdk_source_snapshot_sha256
        or pdk_installed_authority_sha256
        != receipt.pdk_installed_authority_sha256
        or _layermap_sha256(pdk_layermap) != receipt.pdk_layermap_sha256
    ):
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_PDK_CHANGED",
            "installed PDK authority no longer matches the receipt",
        )

    root = Path(repo_root)
    bound = (
        (
            "REFERENCE_OBSERVATION_GDS_CHANGED",
            receipt.gds_repository_path,
            ".gds",
            receipt.gds_sha256,
        ),
        (
            "REFERENCE_OBSERVATION_SOURCE_CHANGED",
            receipt.source_repository_path,
            ".clay",
            receipt.source_sha256,
        ),
        (
            "REFERENCE_OBSERVATION_TOPOLOGY_CHANGED",
            receipt.topology_repository_path,
            ".cir",
            receipt.topology_sha256,
        ),
    )
    for code, relative, suffix, expected_sha256 in bound:
        path = _bound_repository_file(root, relative, suffix=suffix)
        if _sha256_file(path) != expected_sha256:
            raise ReferenceObservationError(
                code, f"bound receipt input changed: {relative}"
            )

    replayed = build_reference_observation_receipt(
        root,
        circuit_id=receipt.circuit_id,
        pdk_id=receipt.pdk_id,
        gds_repository_path=receipt.gds_repository_path,
        source_repository_path=receipt.source_repository_path,
        topology_repository_path=receipt.topology_repository_path,
        expected_top_cell=receipt.expected_top_cell,
        expected_device_inventory=dict(receipt.expected_device_inventory),
        pdk_source_snapshot_sha256=pdk_source_snapshot_sha256,
        pdk_installed_authority_sha256=pdk_installed_authority_sha256,
        pdk_layermap=pdk_layermap,
    )
    if replayed.canonical_data() != receipt.canonical_data():
        raise ReferenceObservationError(
            "REFERENCE_OBSERVATION_REPLAY_MISMATCH",
            "replayed reference observation differs from the bound receipt",
        )
    return replayed


def observe_reference_gds_geometry(
    gds_path: str | Path,
    *,
    expected_top_cell: str,
    expected_device_inventory: Mapping[str, int],
    pdk_layermap: Mapping[str, tuple[int, int]],
) -> ReferenceGdsGeometryObservation:
    """Prove bounded, non-empty device and routing geometry in one reference GDS."""

    if type(expected_top_cell) is not str or not expected_top_cell:
        raise ReferenceObservationError(
            "REFERENCE_GDS_TOP_CELL_MALFORMED",
            "expected top cell must be a non-empty exact string",
        )
    inventory = _exact_inventory(expected_device_inventory)
    layermap = _exact_layermap(pdk_layermap)
    path = Path(gds_path)
    if not path.is_file() or path.suffix.casefold() != ".gds":
        raise ReferenceObservationError(
            "REFERENCE_GDS_MISSING", "reference GDS path is unavailable"
        )

    import klayout.db as kdb

    layout = kdb.Layout()
    try:
        layout.read(str(path))
    except Exception as exc:
        raise ReferenceObservationError(
            "REFERENCE_GDS_UNREADABLE", "reference GDS cannot be parsed"
        ) from exc
    top_cells = tuple(layout.top_cells())
    if len(top_cells) != 1:
        raise ReferenceObservationError(
            "REFERENCE_GDS_TOP_CELL_AMBIGUOUS",
            f"reference GDS has {len(top_cells)} top cells",
        )
    top = top_cells[0]
    if top.name != expected_top_cell:
        raise ReferenceObservationError(
            "REFERENCE_GDS_TOP_CELL_MISMATCH",
            "reference GDS top cell differs from the declared identity",
        )
    cell_count = int(layout.cells())
    if cell_count <= 0 or cell_count > _MAX_CELLS:
        raise ReferenceObservationError(
            "REFERENCE_GDS_HIERARCHY_UNBOUNDED",
            f"reference GDS has {cell_count} cells",
        )

    layer_indices: dict[tuple[int, int], int] = {}
    for key in set(layermap.values()):
        raw_index = layout.find_layer(*key)
        if raw_index is not None:
            layer_indices[key] = int(raw_index)

    def count_cell_physical_shapes(cell: object) -> int:
        total = 0
        for index in layer_indices.values():
            iterator = cell.begin_shapes_rec(index)
            while not iterator.at_end():
                if not iterator.shape().is_text():
                    total += 1
                iterator.next()
                if total > _MAX_SHAPES:
                    raise ReferenceObservationError(
                        "REFERENCE_GDS_GEOMETRY_UNBOUNDED",
                        "reference PCell geometry exceeds the observation bound",
                    )
        return total

    family_instances: dict[str, int] = {}
    device_masters: set[str] = set()

    def visit(
        cell: object,
        inherited_multiplicity: int,
        ancestry: frozenset[int],
    ) -> None:
        for instance in cell.each_inst():
            multiplicity = 0
            for _transform in instance.cell_inst.each_cplx_trans():
                multiplicity += 1
                if multiplicity > _MAX_INSTANCE_OCCURRENCES:
                    raise ReferenceObservationError(
                        "REFERENCE_GDS_HIERARCHY_UNBOUNDED",
                        "one reference instance array exceeds the observation bound",
                    )
            expanded_multiplicity = inherited_multiplicity * multiplicity
            if (
                multiplicity <= 0
                or expanded_multiplicity <= 0
                or expanded_multiplicity > _MAX_INSTANCE_OCCURRENCES
            ):
                raise ReferenceObservationError(
                    "REFERENCE_GDS_HIERARCHY_UNBOUNDED",
                    "expanded reference hierarchy exceeds the observation bound",
                )
            child = instance.cell
            child_index = int(child.cell_index())
            if child_index in ancestry:
                raise ReferenceObservationError(
                    "REFERENCE_GDS_HIERARCHY_UNBOUNDED",
                    "reference GDS contains a cyclic hierarchy",
                )
            family = reference_device_family_from_cell_name(child.name)
            if family is not None:
                if count_cell_physical_shapes(child) <= 0:
                    raise ReferenceObservationError(
                        "REFERENCE_GDS_EMPTY_PCELL",
                        f"device master {child.name!r} has no PDK-layer artwork",
                    )
                family_instances[family] = (
                    family_instances.get(family, 0) + expanded_multiplicity
                )
                device_masters.add(child.name)
            else:
                visit(
                    child,
                    expanded_multiplicity,
                    ancestry | frozenset((child_index,)),
                )

    visit(top, 1, frozenset((int(top.cell_index()),)))
    # The topology inventory counts extracted primitive devices, while this walk
    # counts hierarchical PCell occurrences.  Those domains are deliberately
    # recorded separately: one PCell may extract as multiple primitive devices
    # (for example, a segmented poly resistor).  Geometry can therefore prove
    # family coverage here, but it cannot compare the two multiplicities.
    missing = tuple(
        family for family in sorted(inventory) if family not in family_instances
    )
    if missing:
        raise ReferenceObservationError(
            "REFERENCE_GDS_DEVICE_COVERAGE_MISSING",
            "reference GDS lacks declared device-family coverage: "
            + ", ".join(missing),
        )

    all_shape_count = 0
    text_shape_count = 0
    for index in layout.layer_indices():
        iterator = top.begin_shapes_rec(index)
        while not iterator.at_end():
            all_shape_count += 1
            if iterator.shape().is_text():
                text_shape_count += 1
            iterator.next()
            if all_shape_count > _MAX_SHAPES:
                raise ReferenceObservationError(
                    "REFERENCE_GDS_GEOMETRY_UNBOUNDED",
                    "reference GDS shape inventory exceeds the observation bound",
                )

    physical_shape_count = 0
    physical_bbox: tuple[int, int, int, int] | None = None
    for index in set(layer_indices.values()):
        iterator = top.begin_shapes_rec(index)
        while not iterator.at_end():
            if not iterator.shape().is_text():
                physical_shape_count += 1
                box = iterator.shape().bbox().transformed(iterator.itrans())
                if not box.empty():
                    current = (
                        int(box.left),
                        int(box.bottom),
                        int(box.right),
                        int(box.top),
                    )
                    if physical_bbox is None:
                        physical_bbox = current
                    else:
                        physical_bbox = (
                            min(physical_bbox[0], current[0]),
                            min(physical_bbox[1], current[1]),
                            max(physical_bbox[2], current[2]),
                            max(physical_bbox[3], current[3]),
                        )
            iterator.next()
    if physical_shape_count <= 0:
        raise ReferenceObservationError(
            "REFERENCE_GDS_EMPTY", "reference GDS has no non-text geometry"
        )

    empty_leaf_count = 0
    for cell in layout.each_cell():
        if cell.child_instances() == 0 and count_cell_physical_shapes(cell) == 0:
            empty_leaf_count += 1
    if empty_leaf_count:
        raise ReferenceObservationError(
            "REFERENCE_GDS_EMPTY_PCELL",
            f"reference GDS retains {empty_leaf_count} empty or text-only leaf cells",
        )

    routed_tokens: set[str] = set()
    pending = [top]
    visited_nondevice_cells: set[int] = set()
    while pending:
        cell = pending.pop()
        cell_index = int(cell.cell_index())
        if cell_index in visited_nondevice_cells:
            continue
        visited_nondevice_cells.add(cell_index)
        for token, key in layermap.items():
            if _METAL_TOKEN_RE.fullmatch(token) is None:
                continue
            index = layer_indices.get(key)
            if index is None:
                continue
            if any(not shape.is_text() for shape in cell.shapes(index).each()):
                routed_tokens.add(token)
        for instance in cell.each_inst():
            child = instance.cell
            if reference_device_family_from_cell_name(child.name) is None:
                pending.append(child)
    routed_layer_tokens = tuple(sorted(routed_tokens))
    if not routed_layer_tokens:
        raise ReferenceObservationError(
            "REFERENCE_GDS_ROUTING_GEOMETRY_MISSING",
            "reference GDS has no non-text geometry on a PDK routing metal",
        )

    if physical_bbox is None:
        raise ReferenceObservationError(
            "REFERENCE_GDS_EMPTY", "reference GDS has no bounded physical envelope"
        )
    width_nm = int(
        round((physical_bbox[2] - physical_bbox[0]) * layout.dbu * 1000.0)
    )
    height_nm = int(
        round((physical_bbox[3] - physical_bbox[1]) * layout.dbu * 1000.0)
    )
    if width_nm <= 0 or height_nm <= 0:
        raise ReferenceObservationError(
            "REFERENCE_GDS_EMPTY", "reference GDS has a degenerate envelope"
        )

    result = ReferenceGdsGeometryObservation(
        verdict="pass",
        gds_sha256=_sha256_file(path),
        top_cell=top.name,
        cell_count=cell_count,
        physical_shape_count=physical_shape_count,
        text_shape_count=text_shape_count,
        empty_leaf_count=empty_leaf_count,
        device_family_instances=MappingProxyType(dict(sorted(family_instances.items()))),
        device_master_count=len(device_masters),
        routed_layer_tokens=routed_layer_tokens,
        bbox_width_nm=width_nm,
        bbox_height_nm=height_nm,
        bbox_area_nm2=width_nm * height_nm,
    )
    result.__post_init__()
    return result
