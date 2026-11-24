"""Adapter-bound BJT fixed-cell materialization for the canonical layout spine.

Unlike MOS devices, a foundry BJT is not reconstructed from guessed design rules.
The adapter must bind a digest-pinned, readable fixed GDS and identify its C/B/E
terminal labels.  OA-only cells remain unsupported until an authoritative stream
export is available.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple

from clay.layout.backend import LayerPolygon, Point, Polygon
from clay.layout.device_cell import Terminal
from clay.layout.unit_grid_pattern import UnitGridPattern


Bbox = Tuple[int, int, int, int]
LayerKey = Tuple[int, int]
_REPO_ROOT = Path(__file__).resolve().parents[2]


class BjtCellMaterializationError(RuntimeError):
    """A declared BJT cannot be lowered to exact foundry geometry."""


@dataclass(frozen=True, slots=True)
class BjtUnitOccurrenceGeometry:
    """Actual fixed units in the completed bank's local integer-nm frame."""

    kind: str
    source_sha256: str
    unit_count: int
    bank_extent_nm: Bbox
    unit_centers_nm: Tuple[Tuple[int, int], ...]
    unit_transforms: Tuple[str, ...]
    schema: str = "clay.bjt-unit-occurrence-geometry/v1"

    def __post_init__(self) -> None:
        if (type(self.schema) is not str or self.schema != "clay.bjt-unit-occurrence-geometry/v1"
                or type(self.kind) is not str or self.kind not in {"pnp", "npn"}
                or type(self.source_sha256) is not str or len(self.source_sha256) != 64
                or any(value not in "0123456789abcdef" for value in self.source_sha256)):
            raise ValueError("BJT unit geometry source identity or schema differs")
        box = self.bank_extent_nm
        if (type(box) is not tuple or len(box) != 4 or any(type(value) is not int for value in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError("BJT unit geometry requires a positive integer-nm bank extent")
        centers, transforms = self.unit_centers_nm, self.unit_transforms
        if (type(self.unit_count) is not int or self.unit_count < 1
                or type(centers) is not tuple or len(centers) != self.unit_count
                or type(transforms) is not tuple or len(transforms) != self.unit_count):
            raise ValueError("BJT unit geometry population differs from its unit count")
        if any(type(point) is not tuple or len(point) != 2
               or any(type(value) is not int for value in point) for point in centers):
            raise ValueError("BJT unit geometry centers must use integer nm")
        if len(set(centers)) != self.unit_count or any(
                not (box[0] <= x <= box[2] and box[1] <= y <= box[3]) for x, y in centers):
            raise ValueError("BJT unit geometry centers are duplicated or outside the bank")
        # build_bjt_cell and the canonical BJT placement path support R0 only.
        if any(type(value) is not str or value != "R0" for value in transforms):
            raise ValueError("BJT unit geometry orientation is not supported")


@dataclass(frozen=True)
class BjtCellGeom:
    polygons: List[LayerPolygon]
    terminals: Dict[str, Terminal]
    terminal_layers: Dict[str, LayerKey]
    terminal_accesses: Dict[str, Tuple[Terminal, ...]]
    unit_terminal_accesses: Dict[str, Tuple[Terminal, ...]]
    kind: str
    model: str
    source_path: str
    source_sha256: str
    unit_count: int = 1
    array_shape: Tuple[int, int] = (1, 1)
    internally_strapped: bool = False
    unit_occurrences: BjtUnitOccurrenceGeometry | None = None

    def extent(self) -> Bbox:
        boxes = []
        for lp in self.polygons:
            xs = [int(v.x) for v in lp.polygon.vertices]
            ys = [int(v.y) for v in lp.polygon.vertices]
            if xs:
                boxes.append((min(xs), min(ys), max(xs), max(ys)))
        if not boxes:
            raise BjtCellMaterializationError("BJT fixed cell contains no polygons")
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape_points(shape) -> List[object]:
    """Return one hole-free shape's hull points after hierarchy flattening."""
    if shape.is_box():
        box = shape.box
        return [
            type(box.p1)(box.left, box.bottom),
            type(box.p1)(box.right, box.bottom),
            type(box.p1)(box.right, box.top),
            type(box.p1)(box.left, box.top),
        ]
    if shape.is_path():
        polygon = shape.path.polygon()
    elif shape.is_polygon():
        polygon = shape.polygon
    elif shape.is_simple_polygon():
        polygon = shape.simple_polygon
    else:
        return []
    if hasattr(polygon, "holes") and polygon.holes():
        raise BjtCellMaterializationError(
            "fixed-cell polygon with holes cannot be represented losslessly"
        )
    return list(polygon.each_point_hull())


def _compact_array_shape(unit_count: int, width: int, height: int, spacing: int) -> Tuple[int, int]:
    """Choose a deterministic minimum-area, then minimum-perimeter bank grid."""
    candidates: List[Tuple[int, int, int, int]] = []
    for columns in range(1, unit_count + 1):
        rows = int(math.ceil(unit_count / columns))
        bank_w = columns * width + (columns - 1) * spacing
        bank_h = rows * height + (rows - 1) * spacing
        candidates.append((bank_w * bank_h, 2 * (bank_w + bank_h), columns, rows))
    _area, _perimeter, columns, rows = min(candidates)
    return columns, rows


def _routing_rect(
    key: LayerKey, x0: int, y0: int, x1: int, y1: int
) -> LayerPolygon:
    """Return one normalized Manhattan routing rectangle."""
    return LayerPolygon(
        layer=key[0],
        datatype=key[1],
        polygon=Polygon.rectangle(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
    )


def _point_is_covered(
    point: Tuple[int, int], polygon_points: List[Tuple[int, int]]
) -> bool:
    """Return whether an integer point lies on or inside a simple polygon."""
    px, py = point
    inside = False
    previous = polygon_points[-1]
    for current in polygon_points:
        x0, y0 = previous
        x1, y1 = current
        cross = (px - x0) * (y1 - y0) - (py - y0) * (x1 - x0)
        if cross == 0 and min(x0, x1) <= px <= max(x0, x1) and min(y0, y1) <= py <= max(y0, y1):
            return True
        if (y0 > py) != (y1 > py):
            intersection_x = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
            if px < intersection_x:
                inside = not inside
        previous = current
    return inside


def _ceil_sqrt(value: int) -> int:
    if value <= 0:
        return 0
    root = int(math.isqrt(int(value)))
    return root if root * root == value else root + 1


def _external_companion_tap(adapter, polygons, tap_binding):
    """Build one rule-sized external diffusion/contact/M1 companion island."""
    from clay.layout.tech import LayoutTech
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

    tech = LayoutTech.resolve(str(getattr(adapter, "pdk_id", "")))
    try:
        diffusion_key = tuple(tech.build_layer(tap_binding.diffusion_layer))
        implant_key = tuple(tech.build_layer(tap_binding.implant_layer))
        metal_key = tuple(tech.build_layer(tap_binding.metal_layer))
        contact_key = tuple(tech.build_layer(tap_binding.contact_via))
        # Resolve the excluded layer even though it is deliberately not drawn:
        # a misspelled adapter fact must fail closed before geometry emission.
        tuple(tech.build_layer(tap_binding.excluded_well_layer))
        cut = int(tech.via_size(tap_binding.contact_via))
        diff_enc = int(tech.enclosure(
            tap_binding.contact_via, tap_binding.diffusion_rule_layer
        ))
        metal_enc = int(tech.enclosure(
            tap_binding.contact_via, tap_binding.metal_layer
        ))
        diff_min = int(tech.min_width(tap_binding.diffusion_rule_layer) or 0)
        diff_area = int(tech.min_area(tap_binding.diffusion_rule_layer) or 0)
        metal_min = int(tech.min_width(tap_binding.metal_layer) or 0)
        metal_area = int(tech.min_area(tap_binding.metal_layer) or 0)
    except Exception as exc:
        raise BjtCellMaterializationError(
            f"{getattr(adapter, 'pdk_id', '<unknown>')}: invalid BJT companion-tap rule binding"
        ) from exc

    grid_nm = pdk_manufacturing_grid_nm(tech)

    def align_up(value: int, quantum: int) -> int:
        return ((int(value) + int(quantum) - 1) // int(quantum)) * int(quantum)

    def nearest_grid(value: int) -> int:
        lower = (int(value) // grid_nm) * grid_nm
        return lower if (int(value) - lower) * 2 < grid_nm else lower + grid_nm

    # A centered rectangle has on-grid lower/upper edges only when its dimension
    # is an even multiple of the manufacturing grid.  Grow rule minima upward;
    # never shrink a foundry enclosure/area obligation.  The contact cut has an
    # exact size, so an incompatible adapter fact is unsupported rather than
    # silently resizing a foundry cut.
    if grid_nm > 1:
        centred_quantum = 2 * grid_nm
        if cut % centred_quantum:
            raise BjtCellMaterializationError(
                f"{getattr(adapter, 'pdk_id', '<unknown>')}: BJT companion contact "
                f"size {cut}nm is incompatible with {grid_nm}nm manufacturing grid"
            )
        diff_enc = align_up(diff_enc, grid_nm)
        metal_enc = align_up(metal_enc, grid_nm)
        implant_enc = align_up(int(tap_binding.implant_enclosure_nm), grid_nm)
    else:
        centred_quantum = 1
        implant_enc = int(tap_binding.implant_enclosure_nm)

    diff_size = max(cut + 2 * diff_enc, diff_min, _ceil_sqrt(diff_area))
    metal_w = max(cut + 2 * metal_enc, metal_min)
    metal_h = max(metal_w, (metal_area + metal_w - 1) // metal_w if metal_w else 0)
    if grid_nm > 1:
        diff_size = align_up(diff_size, centred_quantum)
        metal_w = align_up(metal_w, centred_quantum)
        metal_h = align_up(metal_h, centred_quantum)

    base_boxes = []
    for poly in polygons:
        xs = [int(vertex.x) for vertex in poly.polygon.vertices]
        ys = [int(vertex.y) for vertex in poly.polygon.vertices]
        if xs:
            base_boxes.append((min(xs), min(ys), max(xs), max(ys)))
    bx0 = min(box[0] for box in base_boxes)
    by0 = min(box[1] for box in base_boxes)
    bx1 = max(box[2] for box in base_boxes)
    by1 = max(box[3] for box in base_boxes)

    implant_x0 = bx1 + int(tap_binding.fixed_cell_clearance_nm)
    if grid_nm > 1:
        implant_x0 = align_up(implant_x0, grid_nm)
    active_x0 = implant_x0 + implant_enc
    cx = active_x0 + diff_size // 2
    cy = nearest_grid((by0 + by1) // 2)
    active = (
        cx - diff_size // 2,
        cy - diff_size // 2,
        cx - diff_size // 2 + diff_size,
        cy - diff_size // 2 + diff_size,
    )
    implant = (
        active[0] - implant_enc,
        active[1] - implant_enc,
        active[2] + implant_enc,
        active[3] + implant_enc,
    )
    contact = (
        cx - cut // 2,
        cy - cut // 2,
        cx - cut // 2 + cut,
        cy - cut // 2 + cut,
    )
    metal = (
        cx - metal_w // 2,
        cy - metal_h // 2,
        cx - metal_w // 2 + metal_w,
        cy - metal_h // 2 + metal_h,
    )

    # One companion is permitted only while it reaches the complete fixed-cell
    # bank. A larger future bank stops rather than emitting an unmeasured weak tie.
    reach = int(tap_binding.max_distance_nm)
    if (
        implant[0] - bx0 > reach
        or implant[1] - by0 > reach
        or by1 - implant[3] > reach
    ):
        raise BjtCellMaterializationError(
            "BJT bank exceeds one declared companion tap's foundry max-distance envelope"
        )

    added = [
        _routing_rect(diffusion_key, *active),
        _routing_rect(implant_key, *implant),
        _routing_rect(contact_key, *contact),
        _routing_rect(metal_key, *metal),
    ]
    terminal = Terminal(role="tap", bbox=(cx, cy, cx, cy))
    return added, terminal, metal_key


def build_bjt_cell(
    adapter,
    semantic_kind: str,
    *,
    units: int = 1,
    tap_declaration=None,
    unit_pattern: UnitGridPattern | None = None,
) -> BjtCellGeom:
    """Load one digest-pinned fixed-GDS BJT and expose exact C/B/E access.

    Dispatch is adapter DATA only.  No circuit/device-name special case exists.
    Coordinates are normalized to the fixed cell's lower-left, in integer nm.
    """
    kind = str(semantic_kind).lower()
    if type(units) is not int or units < 1:
        raise BjtCellMaterializationError("BJT fixed-cell unit count must be a positive integer")
    if unit_pattern is not None:
        if type(unit_pattern) is not UnitGridPattern:
            raise BjtCellMaterializationError("BJT unit pattern must be an exact UnitGridPattern")
        unit_pattern.__post_init__()
        if unit_pattern.unit_count != units:
            raise BjtCellMaterializationError("BJT unit pattern count differs from declared units")
        if any(orientation != "R0" for _row, _column, orientation in unit_pattern.placements):
            raise BjtCellMaterializationError("BJT unit pattern currently supports only R0")
    binding = (getattr(adapter, "bjt_bindings", {}) or {}).get(kind)
    pdk_id = str(getattr(adapter, "pdk_id", "") or "<unknown>")
    if binding is None:
        raise BjtCellMaterializationError(
            f"{pdk_id}: no adapter BJT binding for {kind!r}"
        )
    if binding.geometry_provider != "fixed_gds":
        raise BjtCellMaterializationError(
            f"{pdk_id}: BJT geometry provider {binding.geometry_provider!r} is "
            "unsupported on the canonical open-tool materializer"
        )

    path = (_REPO_ROOT / binding.geometry_path).resolve()
    if not path.is_file():
        raise BjtCellMaterializationError(f"{pdk_id}: missing BJT GDS {path}")
    digest = _sha256(path)
    if digest != binding.geometry_sha256:
        raise BjtCellMaterializationError(
            f"{pdk_id}: BJT GDS digest mismatch ({digest})"
        )

    try:
        import klayout.db as kdb
    except Exception as exc:  # pragma: no cover - environment contract
        raise BjtCellMaterializationError("KLayout Python module unavailable") from exc

    layout = kdb.Layout()
    layout.read(str(path))
    top = layout.top_cell()
    if top is None:
        raise BjtCellMaterializationError(f"{pdk_id}: BJT GDS has no top cell")
    nm_per_dbu = float(layout.dbu) * 1000.0
    if abs(nm_per_dbu - round(nm_per_dbu)) > 1e-9:
        raise BjtCellMaterializationError(
            f"{pdk_id}: BJT GDS dbu={layout.dbu} is not integer-nm representable"
        )
    scale = int(round(nm_per_dbu))
    layout.flatten(top.cell_index(), -1, True)

    raw_polys: List[Tuple[int, int, List[Tuple[int, int]]]] = []
    label_layer = (
        tuple(binding.terminal_label_layer)
        if binding.terminal_label_layer is not None
        else None
    )
    labels: Dict[str, List[Tuple[int, int]]] = {"c": [], "b": [], "e": []}
    wanted = {str(text): role for role, text in binding.terminal_labels.items()}
    for layer_index in layout.layer_indices():
        info = layout.get_info(layer_index)
        layer_key = (int(info.layer), int(info.datatype))
        for shape in top.shapes(layer_index).each():
            if shape.is_text():
                if label_layer is not None and layer_key == label_layer:
                    text = shape.text
                    role = wanted.get(str(text.string))
                    if role is not None:
                        labels[role].append(
                            (int(text.trans.disp.x) * scale, int(text.trans.disp.y) * scale)
                        )
                continue
            points = _shape_points(shape)
            if len(points) >= 3:
                raw_polys.append(
                    (int(info.layer), int(info.datatype), [
                        (int(point.x) * scale, int(point.y) * scale)
                        for point in points
                    ])
                )
    if not raw_polys:
        raise BjtCellMaterializationError(f"{pdk_id}: BJT fixed cell has no polygons")
    explicit_points = dict(getattr(binding, "terminal_access_points_nm", {}) or {})
    missing = sorted(role for role, points in labels.items() if not points)
    if not explicit_points and missing:
        raise BjtCellMaterializationError(
            f"{pdk_id}: BJT fixed cell missing terminal labels {missing}"
        )

    x0 = min(x for _layer, _dt, points in raw_polys for x, _y in points)
    y0 = min(y for _layer, _dt, points in raw_polys for _x, y in points)
    x1 = max(x for _layer, _dt, points in raw_polys for x, _y in points)
    y1 = max(y for _layer, _dt, points in raw_polys for _x, y in points)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    unit_polygons = [
        LayerPolygon(
            layer=layer,
            datatype=datatype,
            polygon=Polygon([Point(x - x0, y - y0) for x, y in points]),
        )
        for layer, datatype, points in raw_polys
    ]
    unit_terminals: Dict[str, Terminal] = {}
    terminal_layers: Dict[str, LayerKey] = {}
    explicit_layers = dict(getattr(binding, "terminal_access_layers", {}) or {})
    for role in ("c", "b", "e"):
        if explicit_points:
            tx, ty = explicit_points[role]
            access_layer = tuple(explicit_layers[role])
            covered = any(
                (layer, datatype) == access_layer
                and _point_is_covered((tx, ty), points)
                for layer, datatype, points in raw_polys
            )
            if not covered:
                raise BjtCellMaterializationError(
                    f"{pdk_id}: BJT {role} access point is not covered by declared layer"
                )
        else:
            # Multiple equivalent labels are common on a symmetric fixed cell.  The
            # nearest-to-centroid access is deterministic and minimizes escape length.
            tx, ty = min(
                labels[role],
                key=lambda point: (
                    (point[0] - cx) ** 2 + (point[1] - cy) ** 2,
                    point[0],
                    point[1],
                ),
            )
            access_layer = label_layer
        tx -= x0
        ty -= y0
        unit_terminals[role] = Terminal(role=role, bbox=(tx, ty, tx, ty))
        terminal_layers[role] = access_layer

    spacing = int(getattr(binding, "array_spacing_nm", 0) or 0)
    width = x1 - x0
    height = y1 - y0
    columns, rows = _compact_array_shape(units, width, height, spacing)
    default_placements = tuple((index // columns, index % columns, "R0") for index in range(units))
    placements = default_placements if unit_pattern is None else unit_pattern.placements
    if unit_pattern is not None:
        columns, rows = unit_pattern.columns, unit_pattern.rows
    polygons: List[LayerPolygon] = []
    unit_centers: List[Tuple[int, int]] = []
    unit_transforms: List[str] = []
    terminal_accesses: Dict[str, List[Terminal]] = {role: [] for role in ("c", "b", "e")}
    for row, column, orientation in placements:
        dx = column * (width + spacing)
        dy = row * (height + spacing)
        unit_centers.append((width // 2 + dx, height // 2 + dy))
        unit_transforms.append(orientation)
        for lp in unit_polygons:
            polygons.append(
                LayerPolygon(
                    layer=lp.layer,
                    datatype=lp.datatype,
                    polygon=Polygon([
                        Point(int(vertex.x) + dx, int(vertex.y) + dy)
                        for vertex in lp.polygon.vertices
                    ]),
                )
            )
        for role, terminal in unit_terminals.items():
            tx0, ty0, tx1, ty1 = terminal.bbox
            terminal_accesses[role].append(
                Terminal(role=role, bbox=(tx0 + dx, ty0 + dy, tx1 + dx, ty1 + dy))
            )
    unit_accesses = {role: tuple(values) for role, values in terminal_accesses.items()}
    accesses = unit_accesses
    terminals = {role: values[0] for role, values in accesses.items()}
    internally_strapped = False

    if units > 1:
        landing_offsets = dict(getattr(binding, "array_landing_offsets_nm", {}) or {})
        if set(landing_offsets) != {"c", "b", "e"}:
            raise BjtCellMaterializationError(
                f"{pdk_id}: multi-unit BJT bank requires exact c/b/e metal landing offsets"
            )

        from clay.layout.bjt_array_interconnect import build_bjt_bank_interconnect
        routing, exposed, layers = build_bjt_bank_interconnect(
            pdk_id=pdk_id, landing_offsets=landing_offsets, placements=placements,
            width=width, height=height, spacing=spacing,
            legacy_layout=placements == default_placements)
        polygons.extend(routing)
        terminal_layers.update(layers)
        accesses = exposed
        terminals = {role: values[0] for role, values in accesses.items()}
        internally_strapped = True

    if tap_declaration is not None:
        tap_binding = getattr(binding, "tap_binding", None)
        if tap_binding is None:
            raise BjtCellMaterializationError(
                f"{pdk_id}: declared BJT @tap has no adapter realization"
            )
        where = str(getattr(tap_declaration, "where", "") or "")
        if where not in set(tap_binding.where):
            raise BjtCellMaterializationError(
                f"{pdk_id}: declared BJT @tap where={where!r} is unsupported"
            )
        if tap_binding.mode == "external_companion":
            added, tap_terminal, tap_layer = _external_companion_tap(
                adapter, polygons, tap_binding
            )
            polygons.extend(added)
            terminals["tap"] = tap_terminal
            terminal_layers["tap"] = tap_layer
            accesses = {**accesses, "tap": (tap_terminal,)}

    result = BjtCellGeom(
        polygons=polygons,
        terminals=terminals,
        terminal_layers=terminal_layers,
        kind=kind,
        model=str(binding.lvs_model),
        source_path=str(path),
        source_sha256=digest,
        terminal_accesses=accesses,
        unit_terminal_accesses=unit_accesses,
        unit_count=units,
        array_shape=(columns, rows),
        internally_strapped=internally_strapped,
    )
    return replace(result, unit_occurrences=BjtUnitOccurrenceGeometry(
        kind=kind, source_sha256=digest, unit_count=units,
        bank_extent_nm=result.extent(), unit_centers_nm=tuple(unit_centers),
        unit_transforms=tuple(unit_transforms)))


__all__ = [
    "BjtCellGeom",
    "BjtCellMaterializationError",
    "BjtUnitOccurrenceGeometry",
    "build_bjt_cell",
]
