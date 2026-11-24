"""Shared exact-GDS read and layer-witness substrate.

The emitter's :class:`ExactEmittedArtifact` remains the digest authority.  This
module only opens that already-bound artifact once and exposes deterministic
merged layer regions to independent intent checkers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from clay.layout.spatial.canonical_geometry import (
    CanonicalPolygon,
    canonical_contour,
    canonical_polygon,
)


# Historical public name retained as an exact alias; there is one canonical
# polygon implementation, shared with the Canonical Layout Model.
GdsPolygonWitness = CanonicalPolygon


def rect_polygon_witness(
    bbox: tuple[int, int, int, int],
) -> GdsPolygonWitness:
    """Build the shared canonical witness for one exact integer rectangle."""

    if (
        type(bbox) is not tuple
        or len(bbox) != 4
        or any(type(value) is not int for value in bbox)
    ):
        raise TypeError("GDS rectangle witness requires one exact four-int bbox")
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError("GDS rectangle witness requires positive area")
    contour = canonical_contour(
        ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    )
    if contour is None:
        raise ValueError("GDS rectangle witness is degenerate")
    return GdsPolygonWitness(bbox=bbox, contours=(contour,))


def functional_active_layer_keys(tech: Any) -> tuple[tuple[int, int], ...]:
    """Resolve the installed PDK's functional active/OD drawing layers."""

    build_layer = getattr(tech, "build_layer", None)
    if not callable(build_layer):
        raise TypeError("functional active witness requires resolved layout tech")
    keys: set[tuple[int, int]] = set()
    for kind in ("diff", "ndiff", "pdiff"):
        raw = build_layer(kind)
        if raw is None:
            continue
        if (
            type(raw) is not tuple
            or len(raw) != 2
            or any(type(value) is not int or not 0 <= value <= 65_535 for value in raw)
        ):
            raise TypeError("installed functional active layer is not canonical")
        keys.add(raw)
    if not keys:
        raise ValueError("installed PDK exposes no functional active/OD layer")
    return tuple(sorted(keys))


def _polygon_witness(polygon: Any) -> GdsPolygonWitness | None:
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


def _shape_close(left: Any, right: Any, eps: int) -> bool:
    if any(
        abs(int(a) - int(b)) > eps for a, b in zip(left.bbox, right.bbox)
    ):
        return False
    same_vertex_walk = (
        len(left.contours) == len(right.contours)
        and all(
            len(left_contour) == len(right_contour)
            for left_contour, right_contour in zip(left.contours, right.contours)
        )
        and all(
            abs(int(left_value) - int(right_value)) <= eps
            for left_contour, right_contour in zip(left.contours, right.contours)
            for left_point, right_point in zip(left_contour, right_contour)
            for left_value, right_value in zip(left_point, right_point)
        )
    )
    if same_vertex_walk:
        return True
    if tuple(left.bbox) != tuple(right.bbox):
        return False
    import klayout.db as kdb

    def region(record: Any) -> Any:
        result = kdb.Region()
        polygon = kdb.Polygon(
            [kdb.Point(x, y) for x, y in record.contours[0]]
        )
        for contour in record.contours[1:]:
            polygon.insert_hole([kdb.Point(x, y) for x, y in contour])
        result.insert(polygon)
        result.merge()
        return result

    left_region = region(left)
    right_region = region(right)
    return (
        (left_region - right_region).is_empty()
        and (right_region - left_region).is_empty()
    )


class ExactGdsWitness:
    """One validated exact artifact, named top cell, and 1 nm GDS view."""

    __slots__ = ("_cell", "_layout", "gds_path", "gds_sha256", "top_cell")

    def __init__(
        self, path: Path, top_cell: str, gds_sha256: str, layout: Any, cell: Any
    ) -> None:
        self.gds_path = path
        self.top_cell = top_cell
        self.gds_sha256 = gds_sha256
        self._layout = layout
        self._cell = cell

    @classmethod
    def open(cls, gds_path: str | Path, top_cell: str, exact_artifact: Any):
        import klayout.db as kdb

        from clay.layout.route.exact_artifact_evidence import ExactEmittedArtifact

        if type(exact_artifact) is not ExactEmittedArtifact:
            raise TypeError("exact GDS witness requires ExactEmittedArtifact")
        exact_artifact.validate()
        path = Path(gds_path).resolve()
        if path != exact_artifact.exact_gds_path:
            from clay.layout.route.truth_manifest import sha256_path

            if sha256_path(path) != exact_artifact.scope.exact_gds_sha256:
                raise ValueError("GDS path differs from exact artifact authority")
        if type(top_cell) is not str or top_cell != exact_artifact.scope.top_cell:
            raise ValueError("top cell differs from exact artifact authority")

        layout = kdb.Layout()
        layout.read(str(path))
        if abs(float(layout.dbu) - 0.001) > 1e-12:
            raise ValueError("exact GDS witness requires a 1nm database unit")
        cell = layout.cell(top_cell)
        top_indexes = {item.cell_index() for item in layout.top_cells()}
        if cell is None or cell.cell_index() not in top_indexes:
            raise ValueError(f"named top cell {top_cell!r} is absent")
        return cls(path, top_cell, exact_artifact.scope.exact_gds_sha256, layout, cell)

    @property
    def dbu(self) -> float:
        return float(self._layout.dbu)

    def merged_region(self, layer_key: tuple[int, int]):
        import klayout.db as kdb

        if type(layer_key) is not tuple or len(layer_key) != 2:
            raise TypeError("GDS layer key must be one exact pair")
        region = kdb.Region(
            self._cell.begin_shapes_rec(self._layout.layer(*layer_key))
        )
        region.merge()
        return region

    def certified_route_regions(
        self,
        entries: Sequence[tuple[str, tuple[int, int], tuple[int, int, int, int]]],
    ) -> dict[str, dict[tuple[int, int], Any]]:
        """Replay an artifact-bound route manifest against the reread GDS.

        The caller still owns declaration semantics.  This shared substrate only
        proves that every signed route rectangle is present on the named layer of
        the exact top cell and returns deterministic per-net regions.
        """

        import klayout.db as kdb

        if type(entries) is not tuple:
            raise TypeError("exact route manifest must be an exact tuple")
        by_net: dict[str, dict[tuple[int, int], Any]] = {}
        for entry in entries:
            if type(entry) is not tuple or len(entry) != 3:
                raise TypeError("exact route manifest entry must be one triple")
            net, layer_key, bbox = entry
            if type(net) is not str or not net:
                raise TypeError("exact route manifest net must be non-empty")
            if (
                type(layer_key) is not tuple
                or len(layer_key) != 2
                or any(type(value) is not int for value in layer_key)
                or type(bbox) is not tuple
                or len(bbox) != 4
                or any(type(value) is not int for value in bbox)
                or bbox[0] >= bbox[2]
                or bbox[1] >= bbox[3]
            ):
                raise TypeError("exact route manifest geometry is not canonical")
            region = by_net.setdefault(net, {}).setdefault(
                layer_key, kdb.Region()
            )
            region.insert(kdb.Box(*bbox))
        for layers in by_net.values():
            for layer_key, region in layers.items():
                region.merge()
                if not (region - self.merged_region(layer_key)).is_empty():
                    raise ValueError(
                        f"signed route artwork is absent from exact GDS layer {layer_key}"
                    )
        return by_net

    def polygon_multiset(
        self,
        layer_keys: Sequence[tuple[int, int]] | None = None,
    ) -> dict[tuple[int, int], tuple[GdsPolygonWitness, ...]]:
        """Return deterministic raw polygon multiplicity from the bound top cell."""

        import klayout.db as kdb

        selected = None if layer_keys is None else {
            (int(key[0]), int(key[1])) for key in layer_keys
        }
        result: dict[tuple[int, int], tuple[GdsPolygonWitness, ...]] = {}
        for layer_index in self._layout.layer_indexes():
            info = self._layout.get_info(layer_index)
            key = (int(info.layer), int(info.datatype))
            if selected is not None and key not in selected:
                continue
            records: list[GdsPolygonWitness] = []
            iterator = self._cell.begin_shapes_rec(layer_index)
            while not iterator.at_end():
                shape = iterator.shape()
                transform = iterator.itrans()
                polygon = None
                if shape.is_box():
                    polygon = kdb.Polygon(shape.box).transformed(transform)
                elif shape.is_polygon():
                    polygon = shape.polygon.transformed(transform)
                elif shape.is_simple_polygon():
                    polygon = kdb.Polygon(shape.simple_polygon).transformed(transform)
                elif shape.is_path():
                    polygon = shape.path.polygon().transformed(transform)
                elif not shape.is_text():
                    raise ValueError(
                        f"unsupported exact GDS geometry on layer {key}"
                    )
                if polygon is not None:
                    record = _polygon_witness(polygon)
                    if record is None:
                        raise ValueError(
                            f"degenerate exact GDS polygon on layer {key}"
                        )
                    records.append(record)
                iterator.next()
            if records:
                result[key] = tuple(sorted(records))
        return result

    def claim_polygon_multiset(
        self,
        expected_by_owner: Mapping[
            str, Mapping[tuple[int, int], Sequence[Any]]
        ],
        *,
        eps: int,
        excluded_bboxes_by_owner: Mapping[
            str,
            Mapping[tuple[int, int], tuple[tuple[int, int, int, int], ...]],
        ] | None = None,
    ) -> dict[str, dict[tuple[int, int], tuple[GdsPolygonWitness, ...]]]:
        """Consume every expected owner polygon exactly once from the exact GDS."""

        if type(eps) is not int or not 0 <= eps <= 5:
            raise ValueError("exact GDS claim tolerance must be an integer in [0,5]")
        if not isinstance(expected_by_owner, Mapping) or not expected_by_owner:
            raise ValueError("exact GDS ownership inventory must be non-empty")
        layer_keys = tuple(
            sorted({key for layers in expected_by_owner.values() for key in layers})
        )
        unused = {
            key: list(records)
            for key, records in self.polygon_multiset(layer_keys).items()
        }
        if excluded_bboxes_by_owner is not None:
            import klayout.db as kdb

            if not isinstance(excluded_bboxes_by_owner, Mapping):
                raise TypeError("excluded exact-GDS ownership must be a mapping")

            def consume(
                bucket: list[GdsPolygonWitness],
                expected: Sequence[GdsPolygonWitness],
            ) -> list[GdsPolygonWitness] | None:
                trial = list(bucket)
                for target in sorted(expected):
                    index = next(
                        (
                            ordinal
                            for ordinal, actual in enumerate(trial)
                            if _shape_close(actual, target, eps)
                        ),
                        None,
                    )
                    if index is None:
                        return None
                    trial.pop(index)
                return trial

            for owner in sorted(excluded_bboxes_by_owner):
                layers = excluded_bboxes_by_owner[owner]
                if type(owner) is not str or not owner or not isinstance(layers, Mapping):
                    raise TypeError("excluded exact-GDS owner is not canonical")
                for key in sorted(layers):
                    bboxes = layers[key]
                    if (
                        type(key) is not tuple
                        or len(key) != 2
                        or any(type(value) is not int for value in key)
                        or type(bboxes) is not tuple
                        or not bboxes
                        or any(
                            type(box) is not tuple
                            or len(box) != 4
                            or any(type(value) is not int for value in box)
                            for box in bboxes
                        )
                    ):
                        raise TypeError(
                            "excluded exact-GDS ownership is not canonical"
                        )
                    if key not in layer_keys:
                        continue
                    raw_records = tuple(
                        _polygon_witness(kdb.Polygon(kdb.Box(*box)))
                        for box in bboxes
                    )
                    region = kdb.Region()
                    for box in bboxes:
                        region.insert(kdb.Box(*box))
                    region.merge()
                    merged_records = tuple(
                        _polygon_witness(polygon) for polygon in region.each()
                    )
                    variants = (merged_records, raw_records)
                    remaining = None
                    for variant in variants:
                        if any(record is None for record in variant):
                            raise ValueError(
                                f"excluded exact-GDS owner {owner!r} is degenerate"
                            )
                        remaining = consume(
                            unused.setdefault(key, []),
                            tuple(record for record in variant if record is not None),
                        )
                        if remaining is not None:
                            break
                    if remaining is None:
                        raise ValueError(
                            f"excluded exact-GDS owner {owner!r} does not replay "
                            f"on layer {key}"
                        )
                    unused[key] = remaining
        claimed: dict[
            str, dict[tuple[int, int], tuple[GdsPolygonWitness, ...]]
        ] = {}
        for owner in sorted(expected_by_owner):
            layers = expected_by_owner[owner]
            if not layers:
                raise ValueError(f"emitter owner {owner!r} has empty artwork")
            owner_claims: dict[tuple[int, int], tuple[GdsPolygonWitness, ...]] = {}
            for key in sorted(layers):
                bucket = unused.setdefault(key, [])
                matches: list[GdsPolygonWitness] = []
                for expected in sorted(layers[key]):
                    index = next(
                        (
                            ordinal
                            for ordinal, actual in enumerate(bucket)
                            if _shape_close(actual, expected, eps)
                        ),
                        None,
                    )
                    if index is None:
                        raise ValueError(
                            f"emitter owner {owner!r} artwork is missing "
                            f"from exact GDS layer {key}"
                        )
                    matches.append(bucket.pop(index))
                owner_claims[key] = tuple(matches)
            claimed[owner] = owner_claims
        return claimed

    @staticmethod
    def projected_union_region(
        claims: Mapping[
            str, Mapping[tuple[int, int], Sequence[GdsPolygonWitness]]
        ],
    ):
        """Project claimed structural polygons across layers into one XY region."""

        import klayout.db as kdb

        region = kdb.Region()
        for owner in sorted(claims):
            for key in sorted(claims[owner]):
                for record in claims[owner][key]:
                    polygon = kdb.Polygon(
                        [kdb.Point(x, y) for x, y in record.contours[0]]
                    )
                    for contour in record.contours[1:]:
                        polygon.insert_hole(
                            [kdb.Point(x, y) for x, y in contour]
                        )
                    region.insert(polygon)
        region.merge()
        return region

    def residual_region(
        self,
        layer_key: tuple[int, int],
        owned_bboxes: tuple[tuple[int, int, int, int], ...],
    ):
        """Remove an exact bbox inventory without erasing other owners.

        A merged GDS region loses multiplicity.  Exact ownership replay needs
        raw coverage depth when device artwork and route artwork occupy the
        same polygon.  This subtracts the supplied owner's coverage count, not
        merely its union, so duplicate route shapes still cannot stand in for
        missing device artwork.
        """
        import klayout.db as kdb

        if type(layer_key) is not tuple or len(layer_key) != 2:
            raise TypeError("GDS layer key must be one exact pair")
        if type(owned_bboxes) is not tuple or any(
            type(box) is not tuple
            or len(box) != 4
            or any(type(value) is not int for value in box)
            for box in owned_bboxes
        ):
            raise TypeError("owned GDS bboxes must be exact integer tuples")
        layer_index = self._layout.find_layer(*layer_key)
        if layer_index is None:
            return kdb.Region()
        raw = kdb.Region()
        iterator = self._cell.begin_shapes_rec(layer_index)
        while not iterator.at_end():
            shape = iterator.shape()
            transform = iterator.itrans()
            one = kdb.Region()
            if shape.is_box():
                one.insert(shape.box.transformed(transform))
            elif shape.is_polygon():
                one.insert(shape.polygon.transformed(transform))
            elif shape.is_path():
                one.insert(shape.path.polygon().transformed(transform))
            elif not shape.is_text():
                raise ValueError(
                    f"unsupported exact GDS geometry on layer {layer_key}"
                )
            if not one.is_empty():
                raw += one
            iterator.next()
        if not owned_bboxes:
            return raw.merged()

        owned = kdb.Region()
        for bbox in owned_bboxes:
            owned.insert(kdb.Box(*bbox))
        residual = raw.merged(False, 1) - owned.merged(False, 1)
        for depth in range(1, len(owned_bboxes) + 1):
            owned_at_depth = owned.merged(False, depth)
            if owned_at_depth.is_empty():
                break
            owned_deeper = owned.merged(False, depth + 1)
            owned_exactly_at_depth = owned_at_depth - owned_deeper
            raw_above_depth = raw.merged(False, depth + 1)
            if not raw_above_depth.is_empty():
                residual += owned_exactly_at_depth & raw_above_depth
        residual.merge()
        return residual


__all__ = [
    "ExactGdsWitness",
    "GdsPolygonWitness",
    "functional_active_layer_keys",
    "rect_polygon_witness",
]
