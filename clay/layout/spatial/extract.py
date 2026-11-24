"""KLayout-backed extraction into canonical layout-model views."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from clay.layout.spatial.canonical_geometry import (
    CanonicalPolygon,
    canonical_polygon,
)
from clay.layout.spatial.model import (
    CanonicalChildReference,
    CanonicalConstructionView,
    CanonicalLayerRegion,
    CanonicalLayoutExtractionReceipt,
    CanonicalLayoutModelError,
    CanonicalMasterRecord,
    CanonicalMaskExtractionReceipt,
    CanonicalMaskView,
    CanonicalTransform,
)


def _polygon_record(polygon: Any) -> CanonicalPolygon | None:
    return canonical_polygon(
        tuple(
            (int(point.x), int(point.y))
            for point in polygon.each_point_hull()
        ),
        tuple(
            tuple(
                (int(point.x), int(point.y))
                for point in polygon.each_point_hole(index)
            )
            for index in range(int(polygon.holes()))
        ),
    )


def _translated(
    polygon: CanonicalPolygon,
    *,
    dx: int,
    dy: int,
) -> CanonicalPolygon:
    contours = tuple(
        tuple((x - dx, y - dy) for x, y in contour)
        for contour in polygon.contours
    )
    x0, y0, x1, y1 = polygon.bbox
    return CanonicalPolygon(
        bbox=(x0 - dx, y0 - dy, x1 - dx, y1 - dy),
        contours=contours,
    )


def extract_canonical_mask_view(layout: Any, top_cell: str) -> CanonicalMaskView:
    """Flatten and merge non-text geometry under ``top_cell`` deterministically."""

    import klayout.db as kdb

    if not isinstance(layout, kdb.Layout):
        raise TypeError("mask extraction requires a KLayout Layout")
    if type(top_cell) is not str or not top_cell:
        raise TypeError("top_cell must be a non-empty exact string")
    if abs(float(layout.dbu) - 0.001) > 1e-12:
        raise CanonicalLayoutModelError(
            "canonical mask v1 requires a 1 nm KLayout database unit"
        )
    cell = layout.cell(top_cell)
    if cell is None:
        raise CanonicalLayoutModelError(f"top cell {top_cell!r} is absent")

    raw: list[tuple[int, int, tuple[CanonicalPolygon, ...]]] = []
    for layer_index in layout.layer_indexes():
        info = layout.get_info(layer_index)
        region = kdb.Region(cell.begin_shapes_rec(layer_index))
        region.merge()
        if region.is_empty():
            continue
        polygons: list[CanonicalPolygon] = []
        for polygon in region.each():
            record = _polygon_record(polygon)
            if record is None:
                raise CanonicalLayoutModelError(
                    f"merged mask contains degenerate geometry on {(info.layer, info.datatype)}"
                )
            polygons.append(record)
        if polygons:
            raw.append(
                (
                    int(info.layer),
                    int(info.datatype),
                    tuple(sorted(polygons)),
                )
            )

    if not raw:
        return CanonicalMaskView(quantum_nm=1, layers=())
    origin_x = min(
        polygon.bbox[0]
        for _layer, _datatype, polygons in raw
        for polygon in polygons
    )
    origin_y = min(
        polygon.bbox[1]
        for _layer, _datatype, polygons in raw
        for polygon in polygons
    )
    layers = tuple(
        CanonicalLayerRegion(
            layer=layer,
            datatype=datatype,
            polygons=tuple(
                _translated(polygon, dx=origin_x, dy=origin_y)
                for polygon in polygons
            ),
        )
        for layer, datatype, polygons in sorted(raw)
    )
    return CanonicalMaskView(quantum_nm=1, layers=layers)


def extract_canonical_mask_from_gds(
    gds_path: str | Path,
    *,
    top_cell: str,
    expected_source_sha256: str | None = None,
) -> CanonicalMaskExtractionReceipt:
    """Read one GDS once and bind its byte identity to canonical mask bytes."""

    import klayout.db as kdb

    path = Path(gds_path).resolve(strict=True)
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_source_sha256 is not None:
        if type(expected_source_sha256) is not str:
            raise TypeError("expected_source_sha256 must be an exact string or None")
        if source_sha256 != expected_source_sha256:
            raise ValueError("GDS source SHA-256 differs from its expected authority")
    layout = kdb.Layout()
    layout.read(str(path))
    mask_view = extract_canonical_mask_view(layout, top_cell)
    return CanonicalMaskExtractionReceipt(
        source_gds_sha256=source_sha256,
        top_cell=top_cell,
        mask_view=mask_view,
    )


def _local_layer_regions(layout: Any, cell: Any) -> tuple[CanonicalLayerRegion, ...]:
    import klayout.db as kdb

    layers: list[CanonicalLayerRegion] = []
    for layer_index in layout.layer_indexes():
        info = layout.get_info(layer_index)
        region = kdb.Region(cell.shapes(layer_index))
        region.merge()
        if region.is_empty():
            continue
        polygons: list[CanonicalPolygon] = []
        for polygon in region.each():
            record = _polygon_record(polygon)
            if record is None:
                raise CanonicalLayoutModelError(
                    f"master-local geometry is degenerate on {(info.layer, info.datatype)}"
                )
            polygons.append(record)
        if polygons:
            layers.append(
                CanonicalLayerRegion(
                    layer=int(info.layer),
                    datatype=int(info.datatype),
                    polygons=tuple(polygons),
                )
            )
    return tuple(sorted(layers, key=lambda layer: layer.key))


def _instance_transform(instance: Any) -> CanonicalTransform:
    transform = instance.cplx_trans
    return CanonicalTransform(
        angle_microdegrees=int(round(float(transform.angle) * 1_000_000.0))
        % 360_000_000,
        magnification_ppb=int(round(float(transform.mag) * 1_000_000_000.0)),
        mirror=bool(transform.is_mirror()),
        displacement_nm=(
            int(round(float(transform.disp.x))),
            int(round(float(transform.disp.y))),
        ),
    )


def _extract_canonical_construction(
    layout: Any,
    top_cell: str,
) -> tuple[CanonicalConstructionView, tuple[tuple[str, str], ...]]:
    """Return the content view and a diagnostic source-name binding."""

    import klayout.db as kdb

    if not isinstance(layout, kdb.Layout):
        raise TypeError("construction extraction requires a KLayout Layout")
    if type(top_cell) is not str or not top_cell:
        raise TypeError("top_cell must be a non-empty exact string")
    if abs(float(layout.dbu) - 0.001) > 1e-12:
        raise CanonicalLayoutModelError(
            "canonical construction v1 requires a 1 nm KLayout database unit"
        )
    top = layout.cell(top_cell)
    if top is None:
        raise CanonicalLayoutModelError(f"top cell {top_cell!r} is absent")

    by_cell_index: dict[int, CanonicalMasterRecord] = {}
    by_content: dict[str, CanonicalMasterRecord] = {}

    def build(cell: Any, ancestry: frozenset[int]) -> CanonicalMasterRecord:
        cell_index = int(cell.cell_index())
        cached = by_cell_index.get(cell_index)
        if cached is not None:
            return cached
        if cell_index in ancestry:
            raise CanonicalLayoutModelError("construction hierarchy contains a cycle")
        child_refs: list[CanonicalChildReference] = []
        for instance in cell.each_inst():
            child = build(instance.cell, ancestry | frozenset((cell_index,)))
            na = int(instance.na)
            nb = int(instance.nb)
            child_refs.append(
                CanonicalChildReference(
                    child_content_sha256=child.content_sha256,
                    transform=_instance_transform(instance),
                    array_a_nm=(int(instance.a.x), int(instance.a.y)),
                    array_b_nm=(int(instance.b.x), int(instance.b.y)),
                    array_na=na,
                    array_nb=nb,
                )
            )
        record = CanonicalMasterRecord(
            local_layers=_local_layer_regions(layout, cell),
            child_references=tuple(child_refs),
        )
        by_cell_index[cell_index] = record
        by_content.setdefault(record.content_sha256, record)
        return record

    top_record = build(top, frozenset())
    view = CanonicalConstructionView(
        quantum_nm=1,
        top_master_sha256=top_record.content_sha256,
        masters=tuple(by_content.values()),
    )
    name_bindings = tuple(
        sorted(
            (str(layout.cell(index).name), record.content_sha256)
            for index, record in by_cell_index.items()
        )
    )
    return view, name_bindings


def extract_canonical_construction_view(
    layout: Any,
    top_cell: str,
) -> CanonicalConstructionView:
    """Content-address reachable master-local geometry and child references."""

    return _extract_canonical_construction(layout, top_cell)[0]


def extract_named_master_content_sha256(
    layout: Any,
    top_cell: str,
) -> tuple[tuple[str, str], ...]:
    """Bind reachable source cell names to construction ids for diagnostics.

    Source names deliberately remain outside the canonical construction view;
    this adapter exists only to join external semantic authorities to the
    name-free content-addressed records.
    """

    return _extract_canonical_construction(layout, top_cell)[1]


def extract_canonical_layout_from_gds(
    gds_path: str | Path,
    *,
    top_cell: str,
    expected_source_sha256: str | None = None,
) -> CanonicalLayoutExtractionReceipt:
    """Read one GDS once and derive independent mask/construction authorities."""

    import klayout.db as kdb

    path = Path(gds_path).resolve(strict=True)
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_source_sha256 is not None:
        if type(expected_source_sha256) is not str:
            raise TypeError("expected_source_sha256 must be an exact string or None")
        if source_sha256 != expected_source_sha256:
            raise ValueError("GDS source SHA-256 differs from its expected authority")
    layout = kdb.Layout()
    layout.read(str(path))
    return CanonicalLayoutExtractionReceipt(
        source_gds_sha256=source_sha256,
        top_cell=top_cell,
        mask_view=extract_canonical_mask_view(layout, top_cell),
        construction_view=extract_canonical_construction_view(layout, top_cell),
    )


__all__ = [
    "extract_canonical_construction_view",
    "extract_canonical_layout_from_gds",
    "extract_canonical_mask_from_gds",
    "extract_canonical_mask_view",
    "extract_named_master_content_sha256",
]
