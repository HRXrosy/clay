"""Immutable canonical layout-model records."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from clay.layout.spatial.canonical_geometry import CanonicalPolygon


_MASK_SCHEMA = "clay.canonical-mask-view/v1"
_MASK_RECEIPT_SCHEMA = "clay.canonical-mask-extraction/v1"
_CONSTRUCTION_SCHEMA = "clay.canonical-construction-view/v1"
_LAYOUT_RECEIPT_SCHEMA = "clay.canonical-layout-extraction/v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class CanonicalLayoutModelError(ValueError):
    """A canonical layout record is incomplete or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class CanonicalLayerRegion:
    layer: int
    datatype: int
    polygons: tuple[CanonicalPolygon, ...]

    def __post_init__(self) -> None:
        if type(self.layer) is not int or type(self.datatype) is not int:
            raise TypeError("layer and datatype must be exact integers")
        if not 0 <= self.layer <= 65_535 or not 0 <= self.datatype <= 65_535:
            raise CanonicalLayoutModelError("layer and datatype must fit GDS fields")
        if type(self.polygons) is not tuple:
            raise TypeError("polygons must be an exact tuple")
        if not self.polygons or any(
            type(polygon) is not CanonicalPolygon for polygon in self.polygons
        ):
            raise TypeError("polygons must contain exact CanonicalPolygon values")
        object.__setattr__(self, "polygons", tuple(sorted(self.polygons)))

    @property
    def key(self) -> tuple[int, int]:
        return (self.layer, self.datatype)

    def canonical_data(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "datatype": self.datatype,
            "polygons": [
                {
                    "bbox_nm": list(polygon.bbox),
                    "contours_nm": [
                        [list(point) for point in contour]
                        for contour in polygon.contours
                    ],
                }
                for polygon in self.polygons
            ],
        }


@dataclass(frozen=True, slots=True)
class CanonicalMaskView:
    """Flattened merged mask geometry after removing one common translation."""

    quantum_nm: int
    layers: tuple[CanonicalLayerRegion, ...]

    def __post_init__(self) -> None:
        if type(self.quantum_nm) is not int or self.quantum_nm != 1:
            raise CanonicalLayoutModelError(
                "canonical mask v1 requires an exact 1 nm physical quantum"
            )
        if type(self.layers) is not tuple:
            raise TypeError("layers must be an exact tuple")
        if any(type(layer) is not CanonicalLayerRegion for layer in self.layers):
            raise TypeError("layers must contain exact CanonicalLayerRegion values")
        layers = tuple(sorted(self.layers, key=lambda layer: layer.key))
        if len({layer.key for layer in layers}) != len(layers):
            raise CanonicalLayoutModelError("canonical layer keys must be unique")
        if layers:
            x0 = min(polygon.bbox[0] for layer in layers for polygon in layer.polygons)
            y0 = min(polygon.bbox[1] for layer in layers for polygon in layer.polygons)
            if x0 != 0 or y0 != 0:
                raise CanonicalLayoutModelError(
                    "canonical mask must remove the common lower-left translation"
                )
        object.__setattr__(self, "layers", layers)

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _MASK_SCHEMA,
            "quantum_nm": self.quantum_nm,
            "layers": [layer.canonical_data() for layer in self.layers],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalMaskExtractionReceipt:
    """Source-byte provenance bound to the complete canonical mask bytes."""

    source_gds_sha256: str
    top_cell: str
    mask_view: CanonicalMaskView

    def __post_init__(self) -> None:
        if type(self.source_gds_sha256) is not str or _SHA256_RE.fullmatch(
            self.source_gds_sha256
        ) is None:
            raise CanonicalLayoutModelError("source_gds_sha256 must be lowercase SHA-256")
        if type(self.top_cell) is not str or not self.top_cell or self.top_cell != self.top_cell.strip():
            raise CanonicalLayoutModelError("top_cell must be a canonical string")
        if type(self.mask_view) is not CanonicalMaskView:
            raise TypeError("mask_view must be an exact CanonicalMaskView")

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _MASK_RECEIPT_SCHEMA,
            "source_gds_sha256": self.source_gds_sha256,
            "top_cell": self.top_cell,
            "mask_geometry_sha256": self.mask_view.sha256,
            "mask_view": self.mask_view.canonical_data(),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class CanonicalTransform:
    angle_microdegrees: int
    magnification_ppb: int
    mirror: bool
    displacement_nm: tuple[int, int]

    def __post_init__(self) -> None:
        if (
            type(self.angle_microdegrees) is not int
            or type(self.magnification_ppb) is not int
            or type(self.mirror) is not bool
            or type(self.displacement_nm) is not tuple
            or len(self.displacement_nm) != 2
            or any(type(value) is not int for value in self.displacement_nm)
        ):
            raise TypeError("construction transform fields must be exact integers/bool")
        if not 0 <= self.angle_microdegrees < 360_000_000:
            raise CanonicalLayoutModelError("transform angle must be normalized")
        if self.magnification_ppb <= 0:
            raise CanonicalLayoutModelError("transform magnification must be positive")

    def canonical_data(self) -> dict[str, object]:
        return {
            "angle_microdegrees": self.angle_microdegrees,
            "magnification_ppb": self.magnification_ppb,
            "mirror": self.mirror,
            "displacement_nm": list(self.displacement_nm),
        }


@dataclass(frozen=True, slots=True, order=True)
class CanonicalChildReference:
    child_content_sha256: str
    transform: CanonicalTransform
    array_a_nm: tuple[int, int] = (0, 0)
    array_b_nm: tuple[int, int] = (0, 0)
    array_na: int = 0
    array_nb: int = 0

    def __post_init__(self) -> None:
        if type(self.child_content_sha256) is not str or _SHA256_RE.fullmatch(
            self.child_content_sha256
        ) is None:
            raise CanonicalLayoutModelError("child content identity must be SHA-256")
        if type(self.transform) is not CanonicalTransform:
            raise TypeError("child transform must be an exact CanonicalTransform")
        for field in ("array_a_nm", "array_b_nm"):
            value = getattr(self, field)
            if (
                type(value) is not tuple
                or len(value) != 2
                or any(type(item) is not int for item in value)
            ):
                raise TypeError(f"{field} must be an exact integer pair")
        if type(self.array_na) is not int or type(self.array_nb) is not int:
            raise TypeError("array counts must be exact integers")
        if self.array_na < 0 or self.array_nb < 0:
            raise CanonicalLayoutModelError("array counts cannot be negative")
        is_array = self.array_na > 0 or self.array_nb > 0
        if is_array and (self.array_na <= 0 or self.array_nb <= 0):
            raise CanonicalLayoutModelError("AREF must have two positive dimensions")
        if not is_array and (self.array_a_nm != (0, 0) or self.array_b_nm != (0, 0)):
            raise CanonicalLayoutModelError("SREF cannot retain array vectors")

    def canonical_data(self) -> dict[str, object]:
        return {
            "child_content_sha256": self.child_content_sha256,
            "transform": self.transform.canonical_data(),
            "array_a_nm": list(self.array_a_nm),
            "array_b_nm": list(self.array_b_nm),
            "array_na": self.array_na,
            "array_nb": self.array_nb,
        }


@dataclass(frozen=True, slots=True)
class CanonicalMasterRecord:
    local_layers: tuple[CanonicalLayerRegion, ...]
    child_references: tuple[CanonicalChildReference, ...]

    def __post_init__(self) -> None:
        if type(self.local_layers) is not tuple or any(
            type(layer) is not CanonicalLayerRegion for layer in self.local_layers
        ):
            raise TypeError("master local_layers must contain exact layer regions")
        if type(self.child_references) is not tuple or any(
            type(ref) is not CanonicalChildReference for ref in self.child_references
        ):
            raise TypeError("master child_references must contain exact references")
        layers = tuple(sorted(self.local_layers, key=lambda layer: layer.key))
        if len({layer.key for layer in layers}) != len(layers):
            raise CanonicalLayoutModelError("master local layer keys must be unique")
        object.__setattr__(self, "local_layers", layers)
        object.__setattr__(self, "child_references", tuple(sorted(self.child_references)))

    def content_data(self) -> dict[str, object]:
        return {
            "local_layers": [layer.canonical_data() for layer in self.local_layers],
            "child_references": [
                ref.canonical_data() for ref in self.child_references
            ],
        }

    @property
    def content_sha256(self) -> str:
        payload = json.dumps(
            self.content_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    def canonical_data(self) -> dict[str, object]:
        return {
            "content_sha256": self.content_sha256,
            **self.content_data(),
        }


@dataclass(frozen=True, slots=True)
class CanonicalConstructionView:
    quantum_nm: int
    top_master_sha256: str
    masters: tuple[CanonicalMasterRecord, ...]

    def __post_init__(self) -> None:
        if type(self.quantum_nm) is not int or self.quantum_nm != 1:
            raise CanonicalLayoutModelError(
                "canonical construction v1 requires an exact 1 nm quantum"
            )
        if type(self.top_master_sha256) is not str or _SHA256_RE.fullmatch(
            self.top_master_sha256
        ) is None:
            raise CanonicalLayoutModelError("top master identity must be SHA-256")
        if type(self.masters) is not tuple or any(
            type(master) is not CanonicalMasterRecord for master in self.masters
        ):
            raise TypeError("masters must contain exact CanonicalMasterRecord values")
        by_sha = {master.content_sha256: master for master in self.masters}
        if len(by_sha) != len(self.masters):
            raise CanonicalLayoutModelError("construction masters must be content-unique")
        if self.top_master_sha256 not in by_sha:
            raise CanonicalLayoutModelError("top master is absent from construction view")
        object.__setattr__(
            self,
            "masters",
            tuple(by_sha[key] for key in sorted(by_sha)),
        )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _CONSTRUCTION_SCHEMA,
            "quantum_nm": self.quantum_nm,
            "top_master_sha256": self.top_master_sha256,
            "masters": [master.canonical_data() for master in self.masters],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalLayoutExtractionReceipt:
    """One source GDS bound to its independent mask and construction views."""

    source_gds_sha256: str
    top_cell: str
    mask_view: CanonicalMaskView
    construction_view: CanonicalConstructionView

    def __post_init__(self) -> None:
        if type(self.source_gds_sha256) is not str or _SHA256_RE.fullmatch(
            self.source_gds_sha256
        ) is None:
            raise CanonicalLayoutModelError("source_gds_sha256 must be lowercase SHA-256")
        if type(self.top_cell) is not str or not self.top_cell or self.top_cell != self.top_cell.strip():
            raise CanonicalLayoutModelError("top_cell must be a canonical string")
        if type(self.mask_view) is not CanonicalMaskView:
            raise TypeError("mask_view must be an exact CanonicalMaskView")
        if type(self.construction_view) is not CanonicalConstructionView:
            raise TypeError(
                "construction_view must be an exact CanonicalConstructionView"
            )

    @property
    def mask_geometry_sha256(self) -> str:
        return self.mask_view.sha256

    @property
    def construction_sha256(self) -> str:
        return self.construction_view.sha256

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _LAYOUT_RECEIPT_SCHEMA,
            "source_gds_sha256": self.source_gds_sha256,
            "top_cell": self.top_cell,
            "mask_geometry_sha256": self.mask_geometry_sha256,
            "construction_sha256": self.construction_sha256,
            "mask_view": self.mask_view.canonical_data(),
            "construction_view": self.construction_view.canonical_data(),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


__all__ = [
    "CanonicalLayerRegion",
    "CanonicalLayoutExtractionReceipt",
    "CanonicalLayoutModelError",
    "CanonicalChildReference",
    "CanonicalConstructionView",
    "CanonicalMasterRecord",
    "CanonicalMaskExtractionReceipt",
    "CanonicalMaskView",
    "CanonicalTransform",
]
